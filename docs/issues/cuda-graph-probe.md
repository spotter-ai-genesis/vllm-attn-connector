# The query probe never fired on a replayed CUDA graph, so served runs recorded zeros

## Symptom

Serving with `vllm serve` and CUDA graphs left on, which is the default,
produced `decode_attention` records that passed every structural check and
carried no attention signal. `attn_sum`, `val_all_max` and `val_all_avg` summed
to exactly 0.0 on every record, `topk_pos` was populated (top-k over an all-zero
row), and `topk_residual` was 1.0 on every step. Nothing errored and nothing
warned.

Reported as https://github.com/spotter-ai-genesis/vllm-attn-connector/issues/2.
Observed on granite-4.2-30b (jobs 3164638, 3164747) and reproduced on
facebook/opt-125m. The same runs with `--enforce-eager` or
`--compilation-config '{"cudagraph_mode":"NONE"}'` gave correct, mass-conserving
values (jobs 3164843, 3164850, 3164857, 3164887).

Upstream closed the issue in 8282506 ("fix for issue #2 and added
e2e_example.py"). This branch is rebased on that commit and keeps its persistent
buffer; what it adds on top is the freshness rule that decides whether the
buffer holds the current step, and that is the subject of "What this branch adds
over 8282506" below.

## Cause

`_capture` ran from `_ProbedImpl.forward`, which `install()` registers as the
backend's impl class so that it sits inside the traced graph. That is what makes
the probe survive `torch.compile`, and the README said so. It does not make the
probe survive CUDA graphs. A graph records the kernels one execution of the
trace produced, and a replay runs those kernels and no Python. `REGISTRY.put`
is a plain dict write, so it ran once, while vLLM captured the graph, and never
again.

vLLM 0.28 defaults to `cudagraph_mode=FULL_AND_PIECEWISE`, which captures a full
graph per decode batch size, so under `vllm serve` every decode step is a
replay. `_WorkerSide.score_step` then looked the layer up with
`REGISTRY.get(layer_name)`, got `None`, and skipped the layer with a bare
`continue`. Every layer of every step took that branch, so each request still
emitted a complete record with a zero matrix in it.

The connector also cleared the registry in `wait_for_save` on every step, which
is correct for a live capture and exactly wrong for a recorded one.

`tests/e2e_smoke.py` built its engine with `enforce_eager=True`, so the test
suite never ran the path that was broken.

## What changed

Both 8282506 and this branch start from the same place: the probe no longer
tries to re-register a tensor per step. Each layer owns one
persistent buffer and `_capture` copies into it. Under capture that copy becomes
a recorded kernel, so every replay of that graph refreshes the buffer with no
Python involved. The buffer is never cleared between steps.

Freshness is tracked instead of being implied by "the registry was rebuilt this
step". The probe asks `torch.cuda.is_current_stream_capturing()`, so it knows
whether its copy is being recorded into a graph or executed now:

* A recorded copy re-runs on every replay of that graph, so it stays valid
  indefinitely. The probe remembers the largest decode batch it was recorded
  for, and a decode step of no more rows than that replayed a graph that
  rewrites the buffer.
* A live copy counts only for the step it ran on, via a generation counter the
  connector bumps in `wait_for_save` where the `clear()` used to be.

## What this branch adds over 8282506

8282506 makes the buffer persistent and stops clearing it, which is the right
mechanism and the part this branch keeps unchanged in spirit. It has no rule for
deciding whether the buffer holds the current step. `REGISTRY.get(layer, rows)`
returns the buffer whenever `rows` fits inside it, and the one cross-check it
keeps is guarded by `REGISTRY.ran_eagerly`, which is false on a graph replay.

The step that needs the check is a mixed prefill+decode batch. `_capture` skips
it deliberately, because `max_query_len != 1` means batch row *i* is no longer
request slot *i*. So no copy runs, `ran_eagerly` is false, the cross-check is
skipped, and `score_step` reads the previous pure-decode step's queries and
records them as this step. The resulting record is nonzero, conserves mass,
reproduces the attention sink and turns its retained positions over, because
those are real queries, just the wrong step's. Before 8282506 that step was
silently unscored, which was wrong but not misattributed.

Four differences follow from that, plus two the freshness work needed anyway:

1. **Freshness.** A live write counts only for the step it ran on, via a
   generation the connector bumps in `wait_for_save`. A recorded write counts
   whenever a decode graph ran, tracked by the largest decode batch the probe's
   recorded copies cover. Anything else resolves to `None` and the step is
   refused, not scored.
2. **The mixed-step guard.** `score_step` compares the batch's own token count
   against the decode requests the scheduler names, and refuses the step when
   they disagree. That is the check `ran_eagerly` cannot make.
3. **`decode_steps_unscored`.** A refused step is counted per request and
   warned about, rate limited to powers of two, instead of vanishing.
   `decode_steps_recorded + decode_steps_unscored` equals the decode tokens the
   request generated, so the accounting is exact.
4. **No read/write race on the persistent buffer.** Scoring takes a private copy
   of the rows it needs and holds the next forward only until that copy lands.
   With the per-step clones the original code made, there was no race; with one
   buffer that the next forward's recorded copy kernel rewrites in place, there
   is one, and the side stream is still reading when it happens.
5. **Allocation before capture.** The buffers are sized when the worker
   connector is built, from `max(max_num_seqs, max_cudagraph_capture_size)`.
   8282506 allocates on first use inside `_capture`, and that first call can land
   inside a graph capture, which would take the buffer from the graph's private
   memory pool. `put()` refuses to allocate while the stream is capturing and
   says so. `max_num_seqs` alone is also not enough: job 3166664 had
   `max_num_seqs=1024` against `max_cudagraph_capture_size=512`.
6. **`cudagraph_mode=FULL` is refused rather than read stale.** That mode
   captures only mixed-shaped graphs, so no copy is ever recorded; see "Limits
   of the fix".

The rest of 8282506 is kept as it stands: the persistent buffer itself, the
`enforce_eager` default flip in `tests/e2e_smoke.py` so graphs are on unless
asked otherwise, the `attn_sum` totals-one-softmax-unit assertion, and
`tests/e2e_example.py` with `bio_example.csv`.

## How the freshness signal was chosen

The first design keyed the buffers by the `(cudagraph_runtime_mode,
batch_descriptor)` pair vLLM's own `CudagraphDispatcher` uses, on the assumption
that the connector sees the same pair the probe saw at capture. It does not.
vLLM 0.28 runs opt-125m on `v1/worker/gpu/model_runner.py`, which calls
`post_forward`, and so `wait_for_save`, outside the `set_forward_context` block,
and on the FULL path calls `pre_forward` outside it too, so `get_forward_context`
is either unset or carries `cudagraph_runtime_mode = NONE` with no descriptor. In
job 3166664 the probe recorded 51 graph keys during capture and the connector
read `None` on every decode step. Asking whether a capture happened is a question
the probe can answer for itself, and it does not depend on the runner's plumbing.

That inference is sound for every cudagraph mode vLLM ships, because the
dispatcher never sends a pure decode batch to a graph the probe skipped.
`FULL_AND_PIECEWISE` and `FULL_DECODE_ONLY` capture a separate uniform-decode
graph per size, which is what the probe records. `PIECEWISE` leaves attention
outside the graph, so the probe runs live. A batch too wide for any graph runs
eager, so the probe runs live. Plain `cudagraph_mode=FULL` is the exception, in
"Limits of the fix" below.

One buffer per layer and not one per graph: the capture sizes sum to roughly
thirty times the largest one, so a clone per graph would have cost about 5 GB of
extra GPU memory on granite-4.2-30b for no benefit, since each graph writes only
the rows it owns and the connector reads only the rows the current step filled.
`_configure_registry` sizes the buffers when the worker connector is built,
which is before `compile_or_warm_up_model` runs, to the larger of
`max_cudagraph_capture_size` and `max_num_seqs`. The larger, not the smaller:
vLLM warms up at `max_num_seqs` rows after capture has finished, and a buffer
that grows at that point leaves every already-recorded copy writing into the old
one. Job 3166664 did exactly that, 512 against 1024, and lost all 51 graphs.

Three supporting changes:

* `score_step` resolves every layer's queries before it scores anything and
  calls `_warn_no_queries` when a decode step has none, rate limited to powers
  of two. The per-request count is emitted as `decode_steps_unscored`. The
  silent `continue` is what let this run for weeks, and a well-formed zero
  record can no longer be produced without a warning next to it.
* The decode-row sanity check no longer compares against the probe's row count.
  Under a replayed graph that number is the padded capture size and says
  nothing about the current step. It now compares the batch's own token count,
  from the scheduler metadata, against the decode requests the connector
  recognises.
* Scoring takes a private copy of the rows it needs and holds the next forward
  only until that copy lands. The buffers are persistent now, so without this
  the next forward would rewrite them in place while the side stream was still
  reading them.

`tests/e2e_smoke.py` runs with CUDA graphs on by default, as 8282506 made it,
and takes `--enforce-eager` for the graphs-off path; `--cuda-graphs` is kept as
an explicit way to force graphs on because the deploy scripts pass it.
`tests/test_aggregations.py` covers the freshness rules with no GPU: a live
write is fresh only for its own step, a recorded write stays fresh across steps
up to the rows it covers, a decode batch wider than any captured graph is
refused, a registry with no recorded copy refuses every later step, writes land
in the same buffer in place, and a buffer that has to grow drops the coverage it
invalidates and warns.

## Limits of the fix

`cudagraph_mode=FULL` without `PIECEWISE` captures one mixed prefill+decode
graph per size and dispatches pure decode batches to it. Those graphs are
captured with `max_query_len != 1`, so the probe skips them and no copy is
recorded. Decode steps under that setting are still unscored, but they now warn
and count instead of emitting zeros. `FULL_AND_PIECEWISE` (the default),
`FULL_DECODE_ONLY`, `PIECEWISE` and `NONE` all capture queries.

Mixed prefill+decode steps are still not scored, in any mode. `max_query_len !=
1` breaks the "batch row i is request slot i" identity the probe relies on, so
the probe skips them and the connector refuses them. That is unchanged by this
fix; what changed is that they are counted as `decode_steps_unscored` instead of
disappearing. Under concurrency it is a real cost: in the validation run six
simultaneous requests lost 13 of about 200 decode steps that way, concentrated
in the shortest requests, whose decodes overlap the others' prefills. Scoring
them would mean finding the decode prefix of a mixed batch from
`query_start_loc`, which a captured graph cannot do with a fixed row count.

## Evidence

Job 3166843, facebook/opt-125m on the served path, three CUDA-graph
configurations in one allocation. Each mode sends a reference prompt alone, then
five sequential prompts, then six concurrent completions of different lengths,
and every persisted record is checked, not a sample. The reference prompt is 373
tokens and generates 23 decode tokens, so `sum(attn_sum)` must be 23, one
softmax unit per step.

| mode | flags | records | ref `sum(attn_sum)` | value checks |
|---|---|---|---|---|
| default | none, `FULL_AND_PIECEWISE` | 12/12 | 22.999989 | 79 pass, 2 threshold |
| nograph | `cudagraph_mode=NONE` | 12/12 | 23.000006 | 80 pass, 1 threshold |
| enforce_eager | `--enforce-eager` | 12/12 | 22.999998 | 80 pass, 1 threshold |

The same quantity was exactly 0.0 in the default mode before the fix.

Graphs on against graphs off, same prompt at temperature 0, identical generated
text:

| comparison | max abs difference / max(attn_sum) | relative difference of totals |
|---|---|---|
| default vs enforce_eager | 1.053e-04 | 3.913e-07 |
| default vs nograph | 1.249e-04 | 7.391e-07 |

Per-record, default mode: retained mass plus residual equals 1 on every step
(worst 5.0e-06), the attention sink at position 0 is 713x to 1063x the mean of
the rest, and the mean adjacent Jaccard is 0.083 to 0.308, so the retained
positions turn over step to step. That last number is the direct check that the
replayed copy really does rewrite the buffer: a buffer frozen at capture would
give one query for the whole run and identical steps.

The concurrency burst drives the decode batch from six requests down to one, so
the connector logged scoring at widths 1, 2, 3, 4 and 5, padding to CUDA graph
sizes 1, 2, 4 and 8, with nine distinct `matrix_shape` values per mode.
`attn_connector: N decode step(s) unscored` never appeared in any of the three
runs.

The two threshold failures are the harness's own bound on
`decode_steps_unscored`, the counter this fix adds, not a value check. Mixed
prefill+decode steps have never been scored and are now counted rather than
silently dropped; under a six-way concurrent burst the shortest requests spend
several of their few decode steps in batches that are still prefilling someone
else. The default and nograph modes recorded the same 13 such steps and
enforce_eager 10, which is what shows they are scheduler behaviour and not a
graph artifact. On every record `decode_steps_recorded + decode_steps_unscored`
equals the decode tokens generated.

`tests/e2e_smoke.py` on Qwen/Qwen2.5-0.5B-Instruct: FAILURES 0 with
`enforce_eager=True` and FAILURES 0 with `--cuda-graphs`.
`pytest tests/test_aggregations.py` passes, including the new GPU-free coverage
of the registry's freshness rules.

Two earlier jobs in the same campaign failed, both caught by the warnings this
fix adds rather than by silence:

| job | what happened |
|---|---|
| 3166664 | every decode step unscored. The buffer was sized `min(max_cudagraph_capture_size, max_num_seqs)` = 512, and vLLM warms up at `max_num_seqs` = 1024 after capture finishes, so the buffer grew and all 51 graphs lost their copy. The same job also showed the connector reading cudagraph key `None` on every step while the probe had recorded 51 keys, which is what moved the freshness signal into the probe. |
| 3166820 | the validation script's own staleness check asserted a symbol the redesign had removed. |

### The production models (jobs 3166885, 3166949, 3167116, 3167117)

opt-125m and Qwen2.5-0.5B establish the mechanism. These are the two models the
connector is used on, served with CUDA graphs on and no compilation-config
override, which is the configuration that recorded zeros before the fix. Each
run sends a reference prompt of 300 tokens at temperature 0 and a six-way
concurrency burst of 150 to 400 tokens with `ignore_eos`.

| model | mode | records | value checks | ref `sum(attn_sum)` | unscored warnings | peak decode tok/s |
|---|---|---|---|---|---|---|
| granite-4.2-30b | default | 7/7 | 50 pass, 0 fail | 299.000048 | 0 | 77.1 |
| granite-4.2-30b | nograph | 7/7 | 50 pass, 0 fail | 298.999924 | 0 | 62.4 |
| Qwen3.8-27B | default | 7/7 | 74 pass, 0 fail | 298.999996 | 0 | 89.5 |
| Qwen3.8-27B | nograph | 7/7 | 74 pass, 0 fail | 298.999986 | 0 | 73.7 |

Every record is nonzero and mass-conserving, and on every one of them
`sum(attn_sum)` equals that record's recorded decode-step count to within 5e-5,
matching the 150 to 400 token burst exactly. `attn_connector: N decode step(s)
unscored` appears zero times in all four vLLM logs. granite reports one unscored
step per run, identical with graphs on and off, from a mixed prefill+decode
batch during the burst ramp.

Keeping graphs on is worth +23.6% decode throughput on granite and +21.4% on
Qwen at concurrency 6. Before this fix that throughput was only available by
recording zeros.

Qwen3.8-27B is the hybrid case, four KV cache groups with three of them Mamba.
Asserted on every record in both modes:

    PASS  kv_cache_group_id == 3 (got 3)
    PASS  num_groups == 4 (got 4)
    PASS  num_query_heads == 24 (got 24)
    attn_connector: 16 layers in 1 group(s), 24 query heads / 4 kv heads

so the block-table and head-count fixes hold on the replayed path too.

Comparing values across modes needs care. At temperature 0 the two
configurations use different fused kernels and eventually generate different
text, after which they are attending with different decode tokens: granite
matched for 724 characters of 1095 and agreed to 3.901e-04, Qwen diverged after
21 and came out at 9.342e-02, which measures the divergence and not the probe.
One decode step has no such freedom, so `--only-single-step` sends the reference
prompt with `max_tokens=2`, exactly one decode step:

| model | generated | sums | max abs difference / max(attn_sum) | total variation | top-20 overlap |
|---|---|---|---|---|---|
| granite-4.2-30b | `' Rome\n'` both | 0.999999, 1.000005 | 2.743e-04 | 0.00045 | 20/20 |
| Qwen3.8-27B | `' Rome.'` both | 0.999998, 1.000003 | 9.242e-03 | 0.00684 | 20/20 |

granite passes 1e-3 directly. Qwen's 9.242e-03 is that metric's denominator:
Qwen's attention is diffuse, its largest position holds 0.159 of the mass where
granite's sink holds 0.623, and the largest absolute difference is 0.001473 at a
position carrying about a quarter of a percent either way. The distributions
agree to a total variation distance of 0.7% with the same twenty top positions,
and the mass agrees to 5e-06. Qwen's head size is 256 against granite's 128, so
each bf16 logit sums twice as many terms, and `cudagraph_mode=NONE` leaves
`torch.compile` on, so the prefill that filled the KV cache ran piecewise in one
mode and eagerly in the other. A capture fault does not look like this: a stale
or misindexed query gives a different distribution, not the same one with 0.7%
of its mass moved among near-zero positions, and it does not sum to 1.
