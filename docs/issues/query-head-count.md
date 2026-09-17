# Query head count was guessed from the HF config, and guessed wrong

## Symptom

On Qwen/Qwen3.8-27B the connector logged

    attn_connector: 16 layers in 1 group(s), 4 query heads / 4 kv heads

The model has 24 query heads over 4 KV heads. The emitted records looked
healthy, passed all 38 smoke properties, and covered 4 of the 24 query heads.

## Cause

`_infer_query_heads` had three paths:

1. `self._conf.get("num_attention_heads")`. Dead: `_model_conf` never puts that
   key in the dict it builds.
2. `AutoConfig.from_pretrained(model).num_attention_heads`. Missing on this
   model: Qwen3.8-27B is multimodal and keeps `num_attention_heads=24` and
   `num_key_value_heads=4` under `text_config`, with nothing at the top level.
3. `except: return num_kv_heads`, silently.

So the kernel got `num_query_heads=4`. It derives the GQA fan-out from that
(`group = num_query_heads // num_kv_heads`), which came out as 1 instead of 6,
so it scored only the first 4 of 24 query heads and paired query head h with KV
head h rather than h // 6. Three of the four scored heads were reading another
group's keys. Nothing downstream could tell: the row is still a softmax over
real keys, so it sums to 1 and still shows the attention sink.

Granite-4.2-30b was unaffected (32 / 8), since its config exposes the count at
the top level, which is why the bug survived the dense validation.

## Fix

`_model_conf` now asks vLLM, which has already resolved the architecture:

    conf["num_query_heads_per_rank"] = m.get_num_attention_heads(parallel_config)
    conf["num_kv_heads_per_rank"] = m.get_num_kv_heads(parallel_config)

Both are per TP rank already, so `_infer_query_heads` returns the value as is
and logs which path it took. The AutoConfig lookup is gone. The fallback to
`num_kv_heads` remains for the case where `ModelConfig` loses the accessor, but
it now logs a warning saying it is only correct for MHA.

## Evidence

Job 3164478, Qwen/Qwen3.8-27B, branch `fix/hybrid-group-aware-block-table`:

    attn_connector: 24 query head(s) per rank, from VllmConfig
    attn_connector: 16 layers in 1 group(s), 24 query heads / 4 kv heads
    FAILURES: 0
    fixed: 18 segments, k=54 of 575 tokens (9.4%), retaining 27.1% of mean mass,
    adjacent Jaccard 0.324

against the same run with 4 heads (job 3164418): `retaining 17.8% of mean mass,
adjacent Jaccard 0.222`. `k` is unchanged because the selection geometry does
not depend on the head count. The retained mean mass rises because the mean is
now over 24 heads including the ones that actually attend to the prompt, and
the per-step overlap rises with it.

Job 3164479, granite-4.2-30b: `32 query head(s) per rank`, `FAILURES: 0`, k=48
of 506, 9.6% of mean mass, Jaccard 0.202, all identical to the runs before this
change, which is the expected result for a model the old path already got
right.
