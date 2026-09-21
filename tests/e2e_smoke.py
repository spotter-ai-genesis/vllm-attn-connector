"""End-to-end smoke test for the exact-attention connector.

Boots vLLM with the query probe installed and `AttnConnector` attached, then
validates the Flowcept records -- including that the recomputed attention is
real attention and not a plausible-looking artifact.

Run: python e2e_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import statistics as st

import numpy as np

from vllm_attn_connector import load_provenance

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_FILLER = (" ".join(f"Fact {i}: item {i} has weight {i * 7 % 13}." for i in range(40)))
PROMPTS = [
    f"{_FILLER}\nThe capital of France is Paris. The capital of Japan is Tokyo. "
    "Question: what is the capital of Japan? Answer:",
    f"{_FILLER}\nWrite a haiku about recursion in programming:",
    f"{_FILLER}\nList three prime numbers greater than ten:",
]
WORKFLOW_ID = "attn-connector-e2e-smoke"
# Longer than the 64-row initial step allocation, so the default unbounded path
# has to grow its buffers at least once during the run.
MAX_TOKENS = 80
# Deliberately uneven and not aligned to any chunk grid, so "variable" cannot
# accidentally coincide with "fixed".
RANGES = [[5, 40], [40, 41], [100, 260], [300, 505]]


def run(args) -> list[dict]:
    # `import vllm` first. Importing the connector pulls in
    # vllm.distributed.kv_transfer directly, which reaches torchvision's meta
    # registrations before vllm's own __init__ has run and dies with
    # "operator torchvision::nms does not exist". Importing the package proper
    # orders it correctly.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    import vllm_attn_connector

    # MUST precede engine construction: _cached_get_attn_backend is @cache'd, so
    # the backend class is resolved once and never re-read.
    assert vllm_attn_connector.install_probe(), "probe failed to install"

    from flowcept import Flowcept

    with Flowcept("vllm", workflow_id=WORKFLOW_ID, workflow_name="attn_e2e") as fc:
        llm = LLM(
            model=args.model,
            kv_transfer_config=KVTransferConfig(
                kv_connector="AttnConnector",
                kv_connector_module_path="vllm_attn_connector",
                kv_role="kv_producer",
                kv_connector_extra_config={"workflow_id": WORKFLOW_ID,
                                           "out_dir": args.out_dir,
                                           "max_steps": args.max_steps,
                                           "top_pct": args.top_pct,
                                           "chunk_size": args.chunk_size},
            ),
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        if args.ranges:
            # Per-request prompt ranges, vLLM's standard connector channel.
            sp.extra_args = {"kv_transfer_params": {"ranges": RANGES}}
        outs = llm.generate(PROMPTS, sp)
        print("\n=== generations ===")
        for o in outs:
            print(f"  {o.request_id}: {o.outputs[0].text.strip()[:60]!r}")
        llm.generate(["."], SamplingParams(temperature=0.0, max_tokens=1))
        del llm
        return list(fc.get_buffer())


def validate(buffer: list[dict], *, num_requests: int, top_pct: float,
             max_steps: int, chunk: int, ranges=None) -> int:
    print("\n=== validating ===")
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    wfs = [r for r in buffer if r.get("type") == "workflow" and r.get("conf")]
    tasks = [r for r in buffer if r.get("type") == "task"]
    check(len(wfs) == 1, f"one model workflow (got {len(wfs)})")
    check(bool(tasks), f"tasks emitted (got {len(tasks)})")
    if not tasks:
        return 1

    t = tasks[0]
    wfc = next((w.get("attention_config") for w in buffer
                if w.get("type") == "workflow" and w.get("attention_config")), None)
    check(wfc is not None, "attention_config recorded once on the workflow")
    if wfc is None:
        return 1
    desc = t["attention_stats"]
    check(desc.get("uri") is not None,
          f"statistics file written ({desc.get('error', 'no error reported')})")
    if desc.get("uri") is None:
        return 1
    # Everything below reads the arrays back out of the file. The record itself
    # carries only the reference, so a record no longer grows with the prompt.
    gen = {k: v.tolist() for k, v in load_provenance(t).items()}
    check(sorted(desc["tensors"]) == sorted(gen),
          "manifest lists exactly the tensors in the file")
    check(all(list(np.shape(gen[k])) == v["shape"] for k, v in desc["tensors"].items()),
          "manifest shapes match the arrays")
    # config lives on the workflow, per-request state on the task
    meta = {**wfc, **{k: v for k, v in desc.items() if k != "tensors"}}
    check(t["activity_id"] == "decode_attention",
          f"activity labels the connector (got {t['activity_id']!r})")
    check(meta["metric"] == "decode_attention", "metric label recorded")
    reqs = {x["task_id"].rsplit(":g", 1)[0] for x in tasks}
    check(num_requests <= len(reqs) <= num_requests + 1,
          f"one record set per prompt (got {len(reqs)})")

    expected = sorted(["attn_sum", "attn_peak", "segments", "topk_head", "topk_pos",
                       "topk_residual", "val_all_avg", "val_all_max",
                       "prompt_token_ids"])
    check(sorted(gen) == expected, f"tensors in the file (got {sorted(gen)})")
    if sorted(gen) != expected:
        return 1
    steps = len(gen["topk_pos"])
    width = len(gen["topk_pos"][0])
    n_keys = len(gen["attn_sum"])

    # The segment partition is the contract: everything else is indexed by it,
    # and it is emitted so a consumer never needs to know which mode produced it.
    segs = [tuple(r) for r in gen["segments"]]
    check(len(segs) == desc["tensors"]["segments"]["shape"][0],
          f"segments manifest agrees ({len(segs)})")
    check(segs[0][0] == 0 and segs[-1][1] == n_keys,
          f"segments cover [0, {n_keys}) (got {segs[0][0]}..{segs[-1][1]})")
    check(all(a[1] == b[0] for a, b in zip(segs, segs[1:])),
          "segments have no gaps or overlaps")
    check(all(lo < hi for lo, hi, _ in segs), "every segment is non-empty")
    eff_k = sum(keep for _, _, keep in segs)
    check(width == eff_k, f"k is the sum of per-segment quotas ({eff_k})")

    if ranges:
        check(meta["segment_mode"] == "variable",
              f"variable mode recorded (got {meta['segment_mode']})")
        # every declared range must appear verbatim as a segment; the gaps
        # around them are filled in, so len(segs) >= len(ranges)
        declared = {(a, min(b, n_keys)) for a, b in ranges if a < min(b, n_keys)}
        got = {(lo, hi) for lo, hi, _ in segs}
        check(declared <= got,
              f"every declared range is a segment (missing {sorted(declared - got)})")
        check(len(segs) >= len(declared), "gaps between ranges became segments")
    else:
        check(meta["segment_mode"] == "fixed",
              f"fixed mode recorded (got {meta['segment_mode']})")
        c = min(chunk, n_keys) if chunk else n_keys
        grid = [(i, min(i + c, n_keys)) for i in range(0, n_keys, c)]
        check([(lo, hi) for lo, hi, _ in segs] == grid,
              f"segments are the uniform {c}-token grid")
    # `chunk_size=1, top_pct=100` keeps every position: that is the full-capture
    # configuration that replaced dense mode, and selection is then a no-op.
    full = eff_k >= n_keys - 1          # -1: the sink is never selectable
    if full:
        check(eff_k == n_keys, f"full capture: k == prompt length ({eff_k})")
    else:
        check(eff_k < n_keys, f"k={eff_k} < {n_keys}, so selection truncates")
    check(meta["top_pct"] == top_pct, f"top_pct recorded ({meta['top_pct']})")
    check(steps > 1, f"more than one decode step captured ({steps})")
    if max_steps:
        check(steps <= max_steps, f"recorded steps respect the cap ({steps} <= {max_steps})")
    else:
        # Unbounded is the default: every decode token must be recorded, which
        # means the step-indexed buffers grew past their initial allocation
        # rather than silently truncating.
        check(meta["decode_steps_dropped"] == 0,
              f"unbounded: nothing dropped (dropped={meta['decode_steps_dropped']})")
        check(steps == MAX_TOKENS - 1,
              f"unbounded: every decode token recorded ({steps} of {MAX_TOKENS - 1})")
    check(len(gen["attn_sum"]) == n_keys, f"attn_sum spans the prompt ({n_keys})")
    # Attention is a softmax, so each scored decode step contributes exactly one
    # unit of mass across the prompt. Anything near zero means the scores were
    # never written -- the failure mode when the probe does not survive CUDA
    # graph replay, which passes every structural check below because a row of
    # zeros still "conserves mass" as 0 + 1 == 1.
    total = sum(gen["attn_sum"])
    check(abs(total - steps) < 0.05 * max(1, steps),
          f"attn_sum totals one softmax unit per decode step "
          f"({total:.4f} vs {steps} steps)")
    check(total > 0, "ATTENTION IS NON-ZERO -- if this fails the probe never ran; "
                     "check install_probe() preceded engine construction")

    check(width == eff_k, f"matrix width is k ({width})")
    for f in ("topk_head", "topk_pos", "topk_residual", "val_all_avg", "val_all_max"):
        check(len(gen[f]) == steps,
              f"{f}: one entry per decode step ({len(gen[f])} vs {steps})")
    check(all(len(r) == eff_k for r in gen["topk_pos"]), "topk_pos rows are k wide")
    sent = meta["no_entry_sentinel"]
    check(all(p == sent or 0 <= p < n_keys for r in gen["topk_pos"] for p in r),
          "every position is inside the prompt or the no-entry sentinel")
    real = [[p for p in r if p != sent] for r in gen["topk_pos"]]
    check(all(len(set(r)) == len(r) for r in real),
          "retained positions are distinct within a step")
    check(all(r == sorted(r) for r in real), "positions ascending within a step")

    # The point of segmenting: no region of the prompt goes unrepresented.
    import bisect
    starts = [lo for lo, _, _ in segs]
    worst, worst_step = len(segs), -1
    for i, r in enumerate(real):
        hit = {bisect.bisect_right(starts, p) - 1 for p in r}
        if len(hit) < worst:
            worst, worst_step = len(hit), i
    # segment 0 holds the sink, so it yields one slot fewer than its quota
    need = len(segs) - (1 if segs[0][2] == 1 else 0)
    check(worst >= need,
          f"every segment represented in every step "
          f"(worst step {worst_step}: {worst}/{len(segs)})")

    # val_all_avg holds a slice of a distribution, so retained + residual == 1.
    tot = [sum(v) + r[0] for v, r in zip(gen["val_all_avg"], gen["topk_residual"])]
    check(all(abs(x - 1.0) < 5e-3 for x in tot),
          f"retained mean mass + residual == 1 (min {min(tot):.5f}, max {max(tot):.5f})")

    pairs = [(lo, hi) for ra, rm in zip(gen["val_all_avg"], gen["val_all_max"])
             for lo, hi in zip(ra, rm)]
    check(all(lo <= hi + 1e-5 for lo, hi in pairs),
          f"val_all_avg <= val_all_max at all {len(pairs)} retained positions")
    check(sum(1 for lo, hi in pairs if lo < hi - 1e-6) > 0.5 * len(pairs),
          "the two aggregations differ at most positions, so both are informative")

    heads = gen["topk_head"]
    check(all(len(r) == eff_k for r in heads), "topk_head rows are k wide")
    nh = meta["num_query_heads"] * meta["num_layers_scored"]
    flat_h = [h for r in heads for h in r]
    check(all(h == -1 or 0 <= h < nh for h in flat_h),
          f"head codes are -1 or in [0, L*H={nh})")
    check(all(p != 0 for r in gen["topk_pos"] for p in r),
          "position 0 (attention sink) is never selected")

    # Attention must move between decode steps, or per-step capture would be
    # redundant with attn_sum. Under selection the direct evidence is that the
    # retained position set turns over; under full capture the sets are trivially
    # identical, so measure the distribution shift instead.
    jac = []
    for a, b in zip(real, real[1:]):
        sa, sb = set(a), set(b)
        if sa or sb:
            jac.append(len(sa & sb) / len(sa | sb))
    if full:
        # Each row is a *truncated* distribution (the sink is excluded), so
        # renormalise to its own mass before comparing -- otherwise the result
        # is not a TV distance and can exceed 1.
        norm = []
        for r in gen["val_all_avg"]:
            m = sum(r)
            norm.append([x / m for x in r] if m > 0 else r)
        tv = [1.0 - sum(min(x, y) for x, y in zip(norm[i], norm[i + 1]))
              for i in range(len(norm) - 1)]
        drift = st.mean(tv)
        check(0.0 <= drift <= 1.0, f"TV is a distance in [0,1] (got {drift:.4f})")
        check(drift > 0.01,
              f"attention shifts across steps (mean TV {drift:.4f})")
    else:
        check(jac and st.mean(jac) < 0.95,
              f"retained positions turn over across steps "
              f"(mean adjacent Jaccard {st.mean(jac):.3f})")

    frac = st.mean(sum(v) for v in gen["val_all_avg"])
    print(f"       {meta['segment_mode']}: {len(segs)} segments, k={eff_k} of "
          f"{n_keys} tokens ({eff_k/n_keys:.1%}), "
          f"retaining {frac:.1%} of mean mass"
          + (f", adjacent Jaccard {st.mean(jac):.3f}" if jac else ""))

    # The sink is a near-universal property of decoder LLMs; reproducing it is
    # evidence we are reading the model's real keys.
    s0 = gen["attn_sum"][0] / steps
    rest = sum(gen["attn_sum"][1:]) / steps / max(1, n_keys - 1)
    check(s0 > 20 * rest, f"attention sink at position 0 ({s0:.4f} vs {rest:.6f}, "
                          f"{s0/max(rest,1e-12):.0f}x)")

    print(f"\nFAILURES: {len(failures)}")
    for f_ in failures:
        print("  -", f_)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs; the default runs with them ON, "
                         "because that is how vllm serve runs and the probe has "
                         "to survive graph replay")
    ap.add_argument("--ranges", action="store_true",
                    help="declare per-request prompt ranges (variable segments)")
    ap.add_argument("--out-dir", default="./attention_provenance",
                    help="where the statistics files are written")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="decode steps recorded; 0 (default) = every one")
    ap.add_argument("--top-pct", type=float, default=10.0,
                    help="percent kept within each chunk; 0 = dense matrices")
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="prefill tokens per selection chunk; 0 = global selection")
    args = ap.parse_args()
    return validate(run(args), num_requests=len(PROMPTS), top_pct=args.top_pct,
                    max_steps=args.max_steps, chunk=args.chunk_size,
                    ranges=RANGES if args.ranges else None)


if __name__ == "__main__":
    raise SystemExit(main())
