"""Second end-to-end smoke test: a multi-round chain, answered by the model.

`e2e_smoke.py` checks the *provenance* against synthetic prompts. This one
checks the connector under the workload it was built for: a conversation where
each turn's answer depends on earlier turns, and the model's own replies -- not
ground truth -- are the history. Wrong answers therefore propagate, which is the
point: it exercises long, growing prompts, per-request declared ranges that move
with the conversation, and one capture per turn from a single engine.

The chain is `bio_example.csv`, five rounds of a synthetic phenotyping study from
the OPAL benchmark generator. Columns: chain_id, round_no, task_type, prompt,
tool_ranges, correct_answer, explanation.

Ranges declared per turn are every tool block of every round so far, plus each
prior assistant answer. That keeps the conversation history segmented instead of
collapsing it into one gap, so per-segment attention stays interpretable as the
prompt grows.

Two reasons `tool_ranges` from the CSV is NOT used, and `declare_ranges()`
recomputes the spans from the live prompt instead. Both are easy to get wrong
and neither fails loudly -- you simply get attention attributed to the wrong
text.

1. WRONG TOKENIZER. The CSV records spans under the generator's own regex
   tokenizer, which splits far more coarsely than a BPE vocabulary: on this
   chain the model produces 1.24x-1.43x more tokens for the same prompt. The
   error is not a constant factor either, because it depends on how much of the
   text is numbers and punctuation. Measured on this file, using the CSV indices
   verbatim puts a block's start between 15 and 143 tokens away from the block.

2. HISTORY SHIFTS EVERYTHING. CSV spans are relative to a round's own prompt.
   Once that prompt is turn N of a conversation, every index moves by the length
   of the system prompt, all previous rounds, all previous answers and the chat
   template's own markers -- +965 tokens by round 2 and +2969 by round 5 here.
   The shift also depends on WHOSE answers are in the history: replaying with
   the model's own replies rather than ground truth changed it by +1 token at
   round 3 and +6 at round 5, because the model's answers tokenize to different
   lengths. So the offset cannot be precomputed once per round; it has to be
   derived from the exact string handed to the tokenizer on that turn.

Run:
    ROOT=$(cd ../.. && pwd)
    export PYTHONPATH="$PWD/src:$ROOT/flowcept/src"
    export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    python tests/e2e_example.py

Exit status is the number of failed assertions. Answer accuracy is reported, not
asserted: the benchmark is hard and a low score is a fact about the model, not a
bug in the connector.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
from pathlib import Path

from vllm_attn_connector import load_provenance

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CSV_PATH = Path(__file__).with_name("bio_example.csv")
WORKFLOW_ID = "attn-connector-chain-smoke"
INSTRUCTION = ("You are answering a self-contained data-analysis question. Work only "
               "from the text given. Reply with exactly one line, starting 'ANSWER:'.")
ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def load_rounds() -> list[dict]:
    csv.field_size_limit(10 ** 7)
    rows = list(csv.DictReader(CSV_PATH.open()))
    rows.sort(key=lambda r: int(r["round_no"]))
    return rows


def parse_answer(text: str) -> str:
    hits = ANSWER_RE.findall(text or "")
    if hits:
        return hits[-1].strip().rstrip(".")
    body = (text or "").strip()
    return body.splitlines()[-1].strip() if body else ""


def ablations(explanation: str) -> dict[str, str]:
    """The wrong answers this row's `explanation` column predicts, by name.

    Each is what you get by ignoring one of the tool results, so a model that
    lands on one has not answered randomly -- it has taken a specific documented
    shortcut.

    The column is prose, so two things are lost in parsing and `matches_ablation`
    compensates for both: a composite value is cut at its first ";" ("1000;
    DISAGREES" arrives as "1000"), and some values trail explanatory words ("0 on
    5 plants").
    """
    out = {}
    for m in re.finditer(r"(?:^|; )(?:ablations: )?([a-z][^;]*?) gives ([^;)]+)",
                         explanation):
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def _head(value: str) -> str:
    """The comparable part of an answer or a parsed ablation value."""
    value = re.split(r"[;@]", value, 1)[0]
    value = re.split(r"\s+(?:on|which|is)\s", value, 1)[0]
    return value.strip()


def matches_ablation(got: str, value: str) -> bool:
    return same(got, value) or same(_head(got), _head(value))


def same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s).strip().strip(".").lower()
    return norm(a) == norm(b)


def declare_ranges(text: str, tok) -> list[dict]:
    """One range per tool block and per assistant answer, in token space."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = [(a, b) for a, b in enc["offset_mapping"] if b > a]

    def tspan(cs: int, ce: int):
        lo = next((i for i, (a, b) in enumerate(offs) if b > cs), None)
        hi = next((i for i, (a, b) in enumerate(offs) if a >= ce), len(offs))
        return lo, hi

    out = []
    for m in re.finditer(r"^>>> (\w+)", text, re.M):
        s = m.start()
        nxt = text.find("\n>>> ", s + 1)
        fmt = text.find("\nOUTPUT FORMAT", s)
        e = min([x for x in (nxt, fmt) if x > 0], default=len(text))
        lo, hi = tspan(s, e)
        if lo is not None and hi > lo:
            out.append({"kind": m.group(1), "lo": lo, "hi": hi})
    for m in re.finditer(r"<\|im_start\|>assistant\n(ANSWER: [^<]*)", text):
        lo, hi = tspan(m.start(1), m.end(1))
        if lo is not None and hi > lo:
            out.append({"kind": "answer", "lo": lo, "hi": hi})
    out.sort(key=lambda d: d["lo"])
    clean: list[dict] = []
    for d in out:                       # the connector truncates overlaps anyway
        if clean and d["lo"] < clean[-1]["hi"]:
            d["lo"] = clean[-1]["hi"]
        if d["hi"] > d["lo"]:
            clean.append(d)
    return clean


def run(args):
    import time

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    if not args.no_connector:
        import vllm_attn_connector
        assert vllm_attn_connector.install_probe(), "probe failed to install"

    from flowcept import Flowcept
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rounds = load_rounds()
    turns: list[dict] = []

    # `--no-connector` is the baseline arm of an overhead measurement: the same
    # workload, the same engine settings, no provenance capture at all.
    kv_cfg = None if args.no_connector else KVTransferConfig(
        kv_connector="AttnConnector",
        kv_connector_module_path="vllm_attn_connector",
        kv_role="kv_producer",
        kv_connector_extra_config={"workflow_id": WORKFLOW_ID,
                                   "top_pct": args.top_pct,
                                   "out_dir": args.out_dir},
    )
    gen_seconds = 0.0
    with Flowcept("vllm", workflow_id=WORKFLOW_ID, workflow_name="attn_chain") as fc:
        llm = LLM(
            model=args.model,
            kv_transfer_config=kv_cfg,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            # This chain is one request per turn, so capturing the full default
            # ladder of ~35 batch sizes buys nothing and its pools do not fit
            # alongside a 4B model on a small card. Both arms of an overhead
            # comparison get the same list, so it does not bias the result.
            compilation_config={"cudagraph_capture_sizes": args.cudagraph_sizes}
            if args.cuda_graphs else None,
            # Eager by default *here only*: this test loads a 4B model at high
            # utilisation, and CUDA graph pools push it out of memory on a 6 GB
            # card. The graph-replay path is covered by e2e_smoke.py, which runs
            # a 0.5B model with graphs on by default. Pass --cuda-graphs on a
            # larger card to exercise both together.
            enforce_eager=not args.cuda_graphs,
        )
        # The model's OWN answers are the history. Nothing here is repaired with
        # ground truth, so an early mistake stays in the context.
        msgs = [{"role": "system", "content": INSTRUCTION}]
        for row in rounds:
            msgs.append({"role": "user", "content": row["prompt"]})
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False)
            n_prompt = len(tok(text, add_special_tokens=False)["input_ids"])
            if n_prompt + args.max_tokens + 16 > args.max_model_len:
                print(f"  round {row['round_no']}: stopping, {n_prompt} prompt tokens "
                      f"exceeds the context window")
                msgs.pop()
                break
            ranges = declare_ranges(text, tok)
            sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                                extra_args={"kv_transfer_params":
                                            {"ranges": [[d["lo"], d["hi"]]
                                                        for d in ranges]}})
            _t0 = time.perf_counter()
            out = llm.generate([text], sp, use_tqdm=False)[0].outputs[0]
            gen_seconds += time.perf_counter() - _t0
            got = parse_answer(out.text)
            ok = same(got, row["correct_answer"])
            msgs.append({"role": "assistant", "content": out.text.strip()})
            shortcut = next((name for name, wrong
                             in ablations(row["explanation"]).items()
                             if matches_ablation(got, wrong)), None) if not ok else None
            turns.append({"round": int(row["round_no"]), "task": row["task_type"],
                          "got": got, "want": row["correct_answer"], "ok": ok,
                          "shortcut": shortcut,
                          "n_prompt": n_prompt, "n_ranges": len(ranges),
                          "ranges": ranges})
            print(f"  round {row['round_no']} {row['task_type']:<24} "
                  f"{'ok  ' if ok else 'WRONG'} got={got!r}"
                  + ("" if ok else f" want={row['correct_answer']!r}")
                  + f"   [{n_prompt} tok, {len(ranges)} ranges]")
        # a trailing request so the final turn's record is flushed before teardown
        llm.generate(["."], SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)
        del llm
        n_out = sum(args.max_tokens for _ in turns)
        print(f"TIMING generate_s={gen_seconds:.3f} turns={len(turns)} "
              f"prompt_tok={sum(t['n_prompt'] for t in turns)} "
              f"gen_tok={n_out} "
              f"connector={'off' if args.no_connector else 'on'} "
              f"graphs={'on' if args.cuda_graphs else 'off'}")
        return turns, list(fc.get_buffer())


def summarise(turns, buffer, *, top_pct: float) -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            failures.append(msg)

    tasks = [r for r in buffer if r.get("type") == "task"
             and r.get("attention_stats", {}).get("uri")]
    by_len = {r["used"]["num_prompt_tokens"]: r for r in tasks}

    print("\n=== connector ===")
    check(bool(tasks), f"attention records emitted (got {len(tasks)})")
    if not tasks:
        return 1
    covered = [t for t in turns if t["n_prompt"] in by_len]
    check(len(covered) >= len(turns) - 1,
          f"a record for each turn (got {len(covered)} of {len(turns)}; the last "
          f"may be lost to buffer flush)")
    wfc = next((w.get("attention_config") for w in buffer
                if w.get("type") == "workflow" and w.get("attention_config")), {})
    check(bool(wfc), "attention_config recorded on the workflow")
    # config is per-run, the rest per-request; merge so the checks below read
    # from one place
    meta = {**wfc, **{k: v for k, v in tasks[0]["attention_stats"].items()
                      if k != "tensors"}}
    check(meta["segment_mode"] == "variable",
          f"declared ranges were used (segment_mode={meta['segment_mode']!r})")
    check(meta["top_pct"] == top_pct, f"top_pct honoured ({meta['top_pct']})")

    grew = [t["n_prompt"] for t in turns]
    check(grew == sorted(grew) and len(set(grew)) == len(grew),
          f"prompt grows every turn ({grew[0]} -> {grew[-1]} tokens)")
    nr = [t["n_ranges"] for t in turns]
    check(nr == sorted(nr), f"declared ranges grow with the conversation ({nr})")

    for t in covered:
        rec = by_len[t["n_prompt"]]
        g = {k: v.tolist() for k, v in load_provenance(rec).items()}
        segs = g["segments"]
        declared = {(d["lo"], d["hi"]) for d in t["ranges"]}
        emitted = {(lo, hi) for lo, hi, _k in segs}
        check(declared <= emitted,
              f"round {t['round']}: all {len(declared)} declared ranges appear as "
              f"segments")
        resid = g["topk_residual"]
        step0 = resid[0][0] if isinstance(resid[0], list) else resid[0]
        mass = sum(g["val_all_avg"][0])
        check(abs(mass + step0 - 1.0) < 1e-3,
              f"round {t['round']}: kept mass + residual = 1 "
              f"({mass:.3f} + {step0:.3f})")

    print("\n=== accuracy, model's own answers as history ===")
    n_ok = sum(t["ok"] for t in turns)
    print(f"  rounds answered : {len(turns)}")
    print(f"  correct         : {n_ok}/{len(turns)}")
    first_bad = next((t["round"] for t in turns if not t["ok"]), None)
    print(f"  chain complete  : {'yes' if n_ok == len(turns) else 'no'}"
          + (f", first wrong at round {first_bad}" if first_bad else ""))
    if first_bad:
        after = [t for t in turns if t["round"] > first_bad]
        print(f"  after the first mistake: {sum(t['ok'] for t in after)}/{len(after)} "
              f"correct -- later rounds inherit it through the history")
    print("  per round:")
    for t in turns:
        note = ""
        if t.get("shortcut"):
            note = f"  <- documented shortcut: {t['shortcut']}"
        elif not t["ok"]:
            note = "  <- no documented shortcut matches"
        print(f"    r{t['round']} {t['task']:<24} {'ok' if t['ok'] else 'WRONG':<5} "
              f"{t['got'][:32]!r}{note}")
    named = sum(1 for t in turns if t.get("shortcut"))
    if named:
        print(f"  {named} of {len(turns) - n_ok} wrong answers are a shortcut the CSV "
              f"predicts, not noise")

    print("\n=== attention on the current round's ranges ===")
    for t in covered:
        rec = by_len[t["n_prompt"]]
        g = {k: v.tolist() for k, v in load_provenance(rec).items()}
        segs = [(lo, hi) for lo, hi, _k in g["segments"]]
        # peak val_all_max per segment at the final content token, the probe that
        # separates evidence from noise best on this workload
        step = max(0, len(g["topk_pos"]) - 2)
        peak = collections.defaultdict(float)
        for p, v in zip(g["topk_pos"][step], g["val_all_max"][step]):
            if p <= 0:
                continue
            for si, (lo, hi) in enumerate(segs):
                if lo <= p < hi:
                    peak[si] = max(peak[si], v)
                    break
        if not peak:
            continue
        top = max(peak, key=peak.get)
        lo, hi = segs[top]
        kind = next((d["kind"] for d in t["ranges"] if (d["lo"], d["hi"]) == (lo, hi)),
                    "prose gap")
        this_round = [si for si, (a, b) in enumerate(segs)
                      if a >= max((d["lo"] for d in t["ranges"]), default=0) - 1]
        print(f"  r{t['round']}: {len(segs)} segments, hottest = {kind} "
              f"[{lo},{hi})  peak {peak[top]:.3f}")

    print(f"\nFAILURES: {len(failures)}")
    for f_ in failures:
        print("  -", f_)
    return len(failures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--top-pct", type=float, default=10.0)
    ap.add_argument("--out-dir", default="./attention_provenance",
                    help="where the statistics files are written")
    ap.add_argument("--cudagraph-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="batch sizes to capture graphs for (single-request chain)")
    ap.add_argument("--no-connector", action="store_true",
                    help="baseline: run the identical workload with no capture")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="enable CUDA graphs; off by default here because this "
                         "test's 4B model plus graph pools needs >6 GB")
    args = ap.parse_args()
    print(f"chain: {CSV_PATH.name}  model: {args.model}")
    turns, buffer = run(args)
    if args.no_connector:
        print("\nbaseline run: no connector, nothing to validate")
        return 0
    return summarise(turns, buffer, top_pct=args.top_pct)


if __name__ == "__main__":
    raise SystemExit(main())
