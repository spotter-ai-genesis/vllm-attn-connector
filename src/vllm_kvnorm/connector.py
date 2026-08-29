# SPDX-License-Identifier: Apache-2.0
"""``KVNormConnector`` -- score KV as it is written, not at finish.

The sibling :mod:`vllm_kvnorm.connector` reads a request's KV *after* it
finishes, which forces it to pin the blocks so nothing reallocates them first.
Pinning is correct but withholds memory: on a saturated pool it measurably
increases preemptions.

This connector removes the need for it. ``K`` and ``V`` for a token are written
once and never change, so scoring at write time gives an identical result. Each
step it scores only the tokens that step produced, on a side CUDA stream that
overlaps the next forward, and accumulates into a per-request buffer. By the
time a request finishes, every one of its tokens has already been scored -- the
blocks are never read again, so there is nothing to protect.

Per step:

    start_load_kv()   emit any request that finished, after waiting on the
                      event that covers its last scoring launch
    <forward>         writes this step's KV
    wait_for_save()   launch scoring for exactly this step's new tokens

``wait_for_save`` is skipped on the no-forward path, which is correct here: no
forward means no new tokens to score. Emission lives in ``start_load_kv``
because that hook always runs, including on that path.

Consequences:

* nothing is ever pinned, so capture cannot withhold memory from the pool;
* work is spread evenly across steps instead of spiking when a request ends;
* scoring overlaps the forward rather than preceding it;
* the cost is a growing score buffer per in-flight request, plus incremental
  block-table tracking.

The first KV cache group with supported attention layers is scored.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger

from .kernels import token_scores
from .layout import LayerLayout, UnsupportedLayout, resolve_layer_layout

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger("vllm.kvnorm")

FLOAT_DECIMALS = 6


def _model_conf(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Everything needed to interpret the records, especially to rebuild the
    tokenizer that produced ``prompt_token_ids``."""
    m = vllm_config.model_config
    conf = {
        "model": m.model,
        "tokenizer": m.tokenizer,
        "tokenizer_mode": m.tokenizer_mode,
        "trust_remote_code": m.trust_remote_code,
        "dtype": str(m.dtype),
        "max_model_len": m.max_model_len,
        "architectures": list(getattr(m, "architectures", []) or []),
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
    }
    # Revisions pin an exact tokenizer; omitted when unset to avoid nulls.
    for key in ("revision", "tokenizer_revision", "served_model_name"):
        value = getattr(m, key, None)
        if value:
            conf[key] = value if isinstance(value, str) else list(value)
    return conf


@dataclass
class StepRequest:
    """One request's slice of work for a single step."""

    request_id: str
    new_block_ids: tuple[list[int], ...]
    """Blocks added this step, appended to what the worker already tracks."""
    num_computed_tokens: int
    """Tokens already written before this step."""
    num_scheduled_tokens: int
    """Tokens this step writes."""
    prompt_token_ids: list[int] | None = None
    """Only on the request's first appearance."""


@dataclass
class StreamingMetadata(KVConnectorMetadata):
    """Scheduler -> worker, every step."""

    scheduled: list[StepRequest] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)


@dataclass
class StreamingWorkerMeta(KVConnectorWorkerMetadata):
    """Worker -> scheduler: requests fully emitted, so state can be dropped."""

    emitted: set[str] = field(default_factory=set)

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
        assert isinstance(other, StreamingWorkerMeta)
        self.emitted |= other.emitted
        return self


# --------------------------------------------------------------------------- #
# Scheduler side
# --------------------------------------------------------------------------- #


class _SchedulerSide:
    """Reports what each step wrote. Never touches the block pool."""

    def __init__(self) -> None:
        self._finished: list[str] = []
        self._seen: set[str] = set()

    def request_finished(self, request: "Request") -> tuple[bool, dict[str, Any] | None]:
        """Note the finish. Blocks are released immediately and normally.

        Returning ``False`` is not a compromise here: every token of this
        request was scored in the step that wrote it, so its KV is not needed.
        """
        if request.request_id in self._seen:
            self._finished.append(request.request_id)
        return False, None

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> StreamingMetadata:
        meta = StreamingMetadata(finished=self._finished)
        self._finished = []

        counts = scheduler_output.num_scheduled_tokens
        for new in scheduler_output.scheduled_new_reqs:
            n = counts.get(new.req_id, 0)
            if n:
                self._seen.add(new.req_id)
                meta.scheduled.append(
                    StepRequest(
                        request_id=new.req_id,
                        new_block_ids=tuple(list(g) for g in new.block_ids),
                        num_computed_tokens=new.num_computed_tokens,
                        num_scheduled_tokens=n,
                        prompt_token_ids=list(new.prompt_token_ids or ()),
                    )
                )

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            n = counts.get(req_id, 0)
            if not n:
                continue
            blocks = cached.new_block_ids[i]
            self._seen.add(req_id)
            meta.scheduled.append(
                StepRequest(
                    request_id=req_id,
                    new_block_ids=tuple(list(g) for g in blocks) if blocks else (),
                    num_computed_tokens=cached.num_computed_tokens[i],
                    num_scheduled_tokens=n,
                    # A resumed request restarts from scratch; the worker keys
                    # off num_computed_tokens so it re-scores what was lost.
                    prompt_token_ids=None,
                )
            )
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, StreamingWorkerMeta):
            self._seen -= meta.emitted

    def has_pending_push_work(self) -> bool:
        """Keep stepping until every finished request has been emitted."""
        return bool(self._finished)


# --------------------------------------------------------------------------- #
# Worker side
# --------------------------------------------------------------------------- #


class _RequestState:
    """Per-request scoring state, living only while the request is in flight."""

    __slots__ = ("blocks", "table", "scored", "total", "prompt_token_ids", "event", "started")

    def __init__(self, prompt_token_ids: list[int] | None):
        self.blocks: list[int] = []
        self.table: torch.Tensor | None = None
        self.scored: torch.Tensor | None = None  # (capacity,) float32, accumulated
        self.total = 0                           # tokens scored so far
        self.prompt_token_ids = prompt_token_ids or []
        self.event: torch.cuda.Event | None = None
        self.started = time()


class _WorkerSide:
    """Scores each step's new tokens on a side stream and emits on finish."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        conf: dict[str, Any],
        tp_size: int,
        workflow_id: str,
        parent_workflow_id: str | None,
    ) -> None:
        from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor

        self._interceptor = VLLMInterceptor.get_instance()
        self._interceptor.start(bundle_exec_id=workflow_id)
        self._workflow_id = workflow_id
        self._parent_workflow_id = parent_workflow_id
        self._conf = conf
        self._tp_size = tp_size
        self._tp_rank = 0

        self._groups = [
            (list(g.layer_names), g.kv_cache_spec) for g in kv_cache_config.kv_cache_groups
        ]
        self._layouts: dict[str, LayerLayout] = {}
        self._layers: list[LayerLayout] = []
        # Index into kv_cache_groups / StepRequest.new_block_ids of the group we
        # actually score. For dense models this is 0; for hybrid models the
        # attention group sits at a non-zero index (see score_step).
        self._score_group_idx = 0
        self._block_size = 0
        self._state: dict[str, _RequestState] = {}
        self._emitted: set[str] = set()
        self._stream: torch.cuda.Stream | None = None

    # -- registration ------------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        for layer_names, spec in self._groups:
            num_kv_heads = getattr(spec, "num_kv_heads", None)
            if num_kv_heads is None:
                continue
            for name in layer_names:
                cache = kv_caches.get(name)
                if cache is None:
                    continue
                try:
                    self._layouts[name] = resolve_layer_layout(
                        name, cache,
                        block_size=spec.block_size,
                        num_kv_heads=num_kv_heads,
                        head_size=spec.head_size,
                        head_size_v=getattr(spec, "head_size_v", None),
                    )
                except UnsupportedLayout as exc:
                    logger.warning("kvnorm: %s", exc)

        # Score the first KV cache group that has supported attention layers.
        # For hybrid models (e.g. Qwen3.8-27B with linear_attention + full_attention
        # groups), the first group may be Mamba/linear-attention with no num_kv_heads,
        # so we scan forward until we find a group with resolved layouts.
        # We must also *record* that group's index: StepRequest.new_block_ids is a
        # tuple with one block-id list per kv_cache_group, ordered identically to
        # kv_cache_groups. score_step indexes it by _score_group_idx so it applies
        # the attention group's own block table (not group 0's Mamba blocks).
        for gi, (layer_names, _spec) in enumerate(self._groups):
            candidate = [self._layouts[n] for n in layer_names if n in self._layouts]
            if candidate:
                self._layers = candidate
                self._score_group_idx = gi
                break
        if self._tp_size > 1:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            self._tp_rank = get_tensor_model_parallel_rank()

        if not self._layers:
            logger.warning("kvnorm: no supported attention layers found; capture disabled")
            return

        self._block_size = self._layers[0].block_size
        self._stream = torch.cuda.Stream()
        if self._tp_rank == 0:
            self._interceptor.send_model_workflow(
                self._workflow_id, self._conf, parent_workflow_id=self._parent_workflow_id
            )
        logger.info(
            "kvnorm: registered %d layers (layout=%s, block_size=%d, "
            "kv_heads=%d, tp_rank=%d/%d)",
            len(self._layers), self._layers[0].layout_kind, self._block_size,
            self._layers[0].num_kv_heads, self._tp_rank, self._tp_size,
        )

    # -- per-step scoring --------------------------------------------------- #

    def score_step(self, meta: StreamingMetadata) -> None:
        """Launch scoring for the tokens this step wrote."""
        if not self._layers or not meta.scheduled:
            return

        device = self._layers[0].k_view.device
        stream = self._stream
        assert stream is not None
        # The KV we are about to read was written by the forward on the default
        # stream, so the side stream must not start before that completes.
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            for step_req in meta.scheduled:
                st = self._state.get(step_req.request_id)
                if st is None:
                    st = _RequestState(step_req.prompt_token_ids)
                    self._state[step_req.request_id] = st
                elif step_req.prompt_token_ids:
                    st.prompt_token_ids = step_req.prompt_token_ids

                if step_req.new_block_ids:
                    # new_block_ids has one block-id list per kv_cache_group, in
                    # the same order as kv_cache_groups. We score the attention
                    # group at _score_group_idx, so we must extend from that
                    # group's block table -- not group 0, which for hybrid models
                    # is a Mamba/linear-attention block table and would apply the
                    # wrong physical blocks to the attention layers (-> NaN).
                    gi = self._score_group_idx
                    if gi < len(step_req.new_block_ids):
                        st.blocks.extend(step_req.new_block_ids[gi])
                        st.table = torch.tensor(st.blocks, dtype=torch.int32, device=device)
                if st.table is None:
                    continue

                target = step_req.num_computed_tokens + step_req.num_scheduled_tokens
                target = min(target, len(st.blocks) * self._block_size)
                # A preempted request restarts at 0; re-score what it lost.
                start = min(st.total, step_req.num_computed_tokens)
                if target <= start:
                    continue

                st.scored = _ensure_capacity(st.scored, target, device)
                # token_scores() *stores* into `out`, it does not accumulate, so
                # the per-layer sum has to be taken explicitly -- exactly as the
                # pinning connector does.
                scratch = torch.empty(target, dtype=torch.float32, device=device)
                total = torch.zeros(target - start, dtype=torch.float32, device=device)
                for layout in self._layers:
                    token_scores(
                        layout.k_view, layout.v_view, st.table, target,
                        out=scratch, start_token=start,
                    )
                    total += scratch[start:target]
                st.scored[start:target] = total / len(self._layers)
                st.total = target

            event = torch.cuda.Event()
            event.record(stream)
        for step_req in meta.scheduled:
            st = self._state.get(step_req.request_id)
            if st is not None:
                st.event = event

    # -- emission ----------------------------------------------------------- #

    def emit_finished(self, meta: StreamingMetadata) -> None:
        """Emit every finished request, waiting on its last scoring launch."""
        for req_id in meta.finished:
            st = self._state.pop(req_id, None)
            self._emitted.add(req_id)
            if st is None or st.scored is None or st.total == 0:
                continue
            try:
                if self._tp_rank == 0 or self._tp_size > 1:
                    self._emit(req_id, st)
            except Exception:
                logger.exception("kvnorm: failed to emit %s", req_id)

    def _emit(self, req_id: str, st: _RequestState) -> None:
        if st.event is not None:
            # Step 3 of the design: the request finished but its scoring may
            # still be in flight on the side stream.
            st.event.synchronize()

        score = st.scored[: st.total]
        if self._tp_size > 1:
            from vllm.distributed.communication_op import tensor_model_parallel_all_reduce

            score = tensor_model_parallel_all_reduce(score) / self._tp_size
        if self._tp_rank != 0:
            return

        self._interceptor.capture_request(
            workflow_id=self._workflow_id,
            request_id=f"{req_id}:g0",
            scores=score.double().round(decimals=FLOAT_DECIMALS).tolist(),
            prompt_token_ids=st.prompt_token_ids,
            started_at=st.started,
            metadata={
                "metric": "pagedeviction_v_over_k_l2",
                "metric_reference": "arXiv:2509.04377 Algorithm 1",
                "kv_cache_group_id": self._score_group_idx,
                "num_layers": len(self._layers),
                "num_kv_heads": self._layers[0].num_kv_heads,
                "num_tokens": st.total,
            },
        )

    def take_emitted(self) -> StreamingWorkerMeta | None:
        if not self._emitted:
            return None
        meta = StreamingWorkerMeta(emitted=self._emitted)
        self._emitted = set()
        return meta

    def shutdown(self) -> None:
        self._interceptor.stop()


def _ensure_capacity(
    buf: torch.Tensor | None, needed: int, device: torch.device
) -> torch.Tensor:
    """Grow the per-request score buffer geometrically."""
    if buf is not None and buf.numel() >= needed:
        return buf
    capacity = max(256, needed * 2)
    grown = torch.zeros(capacity, dtype=torch.float32, device=device)
    if buf is not None:
        grown[: buf.numel()] = buf
    return grown


# --------------------------------------------------------------------------- #
# Connector facade
# --------------------------------------------------------------------------- #


class KVNormConnector(KVConnectorBase_V1, SupportsHMA):
    """Pin-free KV importance capture: scores each token as it is written.

    Configure with::

        --kv-transfer-config '{
          "kv_connector": "KVNormConnector",
          "kv_connector_module_path": "vllm_kvnorm",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {"workflow_id": "my-workflow"}
        }'
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}

        self._scheduler: _SchedulerSide | None = None
        self._worker: _WorkerSide | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _SchedulerSide()
        else:
            parent = extra.get("workflow_id")
            workflow_id = f"vllm-kvnorm-{uuid.uuid4().hex[:12]}"
            logger.info(
                "kvnorm: workflow_id=%s parent=%s", workflow_id, parent
            )
            self._worker = _WorkerSide(
                kv_cache_config,
                conf=_model_conf(vllm_config),
                tp_size=vllm_config.parallel_config.tensor_parallel_size,
                workflow_id=workflow_id,
                parent_workflow_id=parent,
            )

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return None

    # -- scheduler side ----------------------------------------------------- #

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        assert self._scheduler is not None
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self._scheduler is not None
        return self._scheduler.request_finished(request)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        assert self._scheduler is not None
        self._scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        assert self._scheduler is not None
        return self._scheduler.has_pending_push_work()

    # -- worker side -------------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Emit finished requests. Runs on every path, including no-forward."""
        assert self._worker is not None
        meta = self._get_connector_metadata()
        if isinstance(meta, StreamingMetadata):
            self._worker.emit_finished(meta)

    def wait_for_save(self) -> None:
        """Score this step's new tokens. Only called when a forward happened."""
        assert self._worker is not None
        meta = self._get_connector_metadata()
        if isinstance(meta, StreamingMetadata):
            self._worker.score_step(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer, attn_metadata, **kwargs: Any) -> None:
        return

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self._worker is not None
        return self._worker.take_emitted()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()
