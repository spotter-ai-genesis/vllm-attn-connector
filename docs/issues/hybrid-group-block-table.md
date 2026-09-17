# Hybrid models were scored against the wrong KV cache group's block table

## Symptom

On a hybrid model (linear attention or Mamba layers next to full attention),
the emitted records carry no attention. On Qwen3.8-27B every decode step
retains zero positions: `topk_pos` is the no-entry sentinel throughout,
`val_all_avg` is zero, `topk_residual` is 1.0. Nothing is reported as
non-finite and the record is otherwise well formed, so it takes a per-step
check to notice. The predecessor connector showed the same bug as NaN scores
past the first page, which depends on what happens to be in the blocks that get
read. Dense models are unaffected.

## Cause

`_WorkerSide.score_step` kept one block list per request:

    st.blocks.extend(step_req.new_block_ids[0])
    st.table = torch.tensor(st.blocks, ...)

and passed `st.table` to the kernel for every scored group.
`StepRequest.new_block_ids` is a tuple with one block-id list per KV cache
group, in `kv_cache_config.kv_cache_groups` order, so index 0 is the right
group only when the model has a single group.

`register_kv_caches` already handled the group layout correctly: it resolves a
layout per layer, skips groups whose spec has `num_kv_heads is None`, and
records the surviving groups in `self._scored` with their real group index
(`_Group.gid`). The block table was the one place the index was dropped.

## Evidence

Group dump of `Qwen/Qwen3.8-27B` on vLLM 0.28.0
(`deploy/runs/kvnorm/dump-groups-3044670.out`, job 3044670):

    NUM_GROUPS: 4
    GROUP 0: type=MambaSpec nlayers=16 block_size=784 num_kv_heads=None head_size=None
             first=language_model.model.layers.0.linear_attn
    GROUP 1: type=MambaSpec nlayers=16 block_size=784 num_kv_heads=None head_size=None
             first=language_model.model.layers.1.linear_attn
    GROUP 2: type=MambaSpec nlayers=16 block_size=784 num_kv_heads=None head_size=None
             first=language_model.model.layers.2.linear_attn
    GROUP 3: type=FullAttentionSpec nlayers=16 block_size=784 num_kv_heads=4 head_size=256
             first=language_model.model.layers.3.self_attn.attn
    STEPREQ 0-bcf54e43: n_groups=4 per_group_block_lens=[1, 1, 1, 1]

The 64 text layers alternate three GDN linear-attention layers to one full
attention layer (`full_attention_interval=4`), and vLLM splits the 48 GDN
layers into three Mamba groups.

Only group 3 resolves an attention layout, so `self._scored == [group 3]`, but
the table handed to the kernel came from `new_block_ids[0]`, which is a Mamba
state-block table. The attention layers were read at unrelated physical blocks.

The same bug was found and fixed in the retired predecessor `vllm-kvnorm`
(commit 2183ee2 on its `fix/hybrid-group-aware-block-table` branch). That
connector scored a single group, so recording one group index was enough. This
connector scores every group that resolves a layout, so it needs a table per
group.

## What changed

* `_group_block_ids(new_block_ids, gids)`: pure function mapping the scheduler
  tuple onto the scored group ids, in scored order. A gid past the end of the
  tuple yields an empty list.
* `_RequestState.blocks` / `.table` are now one list and one tensor per scored
  group, indexed by position in `_WorkerSide._scored`. Preemption restarts clear
  all of them.
* The scoring loop passes `st.table[gi]` to `decode_attention`.
* `n_keys` is clamped with each group's own `block_size` instead of a single
  `self._block_size`, which no longer exists. Nothing guarantees one block size
  across groups (on Qwen3.8-27B they happen to agree at 784), and the
  step-indexed buffers are shared across groups, so scoring is clamped to the
  prefix every scored group can address. Once prefill is over this is just the
  prompt length.
* `_emit` clamps the emitted `n_keys` to the length actually scored
  (`st.n_keys`) rather than to the width the `colsum` buffer happened to get,
  which is `max(256, n_keys)` and could exceed it.
* `kv_cache_group_id` in the emitted metadata was already `group.gid` per
  record and stays correct. It is now printed by `tests/e2e_smoke.py`.

## Validation

Before and after on Qwen3.8-27B, job 3164418, same node and same weights, only
`PYTHONPATH` differing. Before, every one of the 79 decode steps retained zero
positions: `topk_pos` is the no-entry sentinel throughout, `val_all_avg` is
zero and `topk_residual` is 1.0. The record looks well formed and carries no
data; nothing is flagged as non-finite. After, `FAILURES: 0`, k=54 of 575
tokens, 17.8% of the mean mass retained, adjacent Jaccard 0.222, and
`kv_cache_group_id=3`, `num_groups=4`, `scored_groups=1`.

Dense models are unchanged: Qwen2.5-0.5B-Instruct (jobs 3164288, 3164345, three
configurations each) and granite-4.2-30b (jobs 3164313, 3164394) report
`FAILURES: 0` with identical numbers before and after the change.

Reaching the hybrid model at all needed a second fix, in the kernel: every KV
cache group of Qwen3.8-27B pages at block_size=784 and `tl.arange` requires a
power-of-two extent, so `_qk_kernel` failed to compile and killed the whole
generate call on main and on this branch alike (job 3164330). That is a
separate commit.

Full job table in `deploy/ATTN_FIX_VALIDATION.md`.

## Found on the way

`_infer_query_heads` was returning 4 on Qwen3.8-27B, where the model has 24
query heads over 4 KV heads, so the Qwen numbers above are over 4 mismatched
heads. Fixed separately, see `query-head-count.md`; the corrected run is job
3164478.
