"""Attention-backend override that stashes decode queries. The sensor half.

Queries never enter the KV cache -- they are transient activations, consumed and
freed every step. A ``KVConnector`` therefore cannot see them, which is why
capturing exact attention needs a second extension point.

vLLM provides one: ``register_backend(AttentionBackendEnum.CUSTOM)``
(``v1/attention/backends/registry.py``), documented for exactly this. We
subclass the selected backend's ``Impl``, copy the decode queries, and delegate
everything else to ``super()``. **No vLLM source is patched and no kernel is
touched** -- FlashAttention runs unmodified.

Why the impl and not a module hook
----------------------------------
``Attention.forward`` is traced through by ``torch.compile``; only the custom op
``unified_attention_with_output`` survives as a runtime node, and it dispatches
``self.impl.forward(...)`` dynamically. Overriding the *impl* is therefore the
only hook that is still present in the traced graph; an ``nn.Module`` forward
hook is erased by the trace.

Why that is not enough under CUDA graphs
---------------------------------------
Surviving the trace is not the same as surviving graph replay. A CUDA graph
records the kernels one execution of the trace produced, and a replay runs only
those kernels: no Python. So ``_capture`` runs once, while vLLM captures the
graph, and never again for that graph -- which is every decode step of a served
run, since ``vllm serve`` captures a full graph per decode batch size by
default. A registry that a step-scoped ``clear()`` empties is therefore empty
on every replayed step (issue #2: attn_sum == 0 on every served record).

What does survive replay is the recorded *copy*. So the probe owns one
persistent buffer per layer, allocated outside any capture, and ``_capture``
copies into it. During capture that copy becomes a recorded kernel; every
replay of that graph re-runs it and the buffer holds the current step's
queries with no Python involved. The buffer is never cleared -- clearing it
would be exactly the bug -- so the connector instead asks whether the write it
is about to read is fresh *for this step*:

* a live write is fresh only for the step it ran on, tracked by a step
  generation the connector bumps once per step.
* a recorded write is fresh whenever a decode graph ran, and on a pure decode
  step one always does. ``_capture`` asks
  ``torch.cuda.is_current_stream_capturing()``, so it knows directly whether
  its copy is being recorded rather than executed, and remembers the largest
  decode batch it was recorded for. A decode step of no more rows than that
  replayed a graph that writes this buffer.

Why not key the buffer by the graph vLLM is about to replay: the connector
cannot see which one that is. ``wait_for_save`` runs from ``post_forward`` in
``v1/worker/gpu/model_runner.py``, which is outside the ``set_forward_context``
block, and on the FULL path ``pre_forward`` is outside it too, so
``get_forward_context`` is either unset or carries ``cudagraph_runtime_mode =
NONE`` with no batch descriptor. Observed in job 3166664: the probe recorded 51
graph keys during capture and the connector read ``None`` on every decode step.
Asking whether a capture happened, which the probe can answer for itself, does
not depend on the runner's plumbing.

The inference is sound for every cudagraph mode vLLM ships, because the
dispatcher never sends a pure decode batch to a graph the probe skipped.
``FULL_AND_PIECEWISE`` and ``FULL_DECODE_ONLY`` capture a separate uniform
decode graph per size, which is what the probe records; ``PIECEWISE`` leaves
attention outside the graph, so the probe runs live; a batch too large for any
graph runs eager, so the probe runs live. Plain ``cudagraph_mode=FULL`` captures
only mixed-shaped graphs, which the probe skips, so it records nothing and the
connector refuses every step with a warning rather than reading a stale buffer.

Sizing: one buffer per layer, ``max_rows x heads x head_size``, where
``max_rows`` covers the largest captured decode batch. Not one buffer per
graph: that would be ~30x the memory (the capture sizes sum to far more than
the largest one), and it buys nothing, since every graph writes the rows it
owns and the connector only ever reads the rows the current step filled.

Why decode only
---------------
vLLM v1 orders decode requests first in the batch and exposes
``num_decode_tokens``. In decode each request contributes exactly one query
token, so batch row *i* is request slot *i* -- no ``query_start_loc`` parsing,
no token-to-request mapping. Prefill queries are skipped: capturing them is
O(T^2) work and O(T*H*D) memory per layer.

The copy is small (``num_decode_reqs x heads x head_size``) but it must happen
on the current stream: ``query`` is a buffer vLLM reuses next step, so a
deferred copy could read it after ``q_proj`` has overwritten it.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.attn_connector")
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


class QueryRegistry:
    """Decode queries, written by the probe, read by the connector in
    ``wait_for_save`` -- which runs after the forward, so the handoff needs no
    synchronisation beyond ordinary step ordering.

    One persistent buffer per layer, keyed by layer name because that is the
    one identifier both halves share. See the module docstring for why the
    buffer persists across steps and how freshness is tracked instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: dict[str, torch.Tensor] = {}
        # Largest decode batch this layer's copy was *recorded* for, over every
        # CUDA graph capture seen. A permanent property of the graphs, so a
        # running max and not a last-value: replaying any of them rewrites the
        # buffer.
        self._graph_rows: dict[str, int] = {}
        # Step generation at which the probe last ran live for this layer, and
        # the rows that write covered.
        self._live_gen: dict[str, int] = {}
        self._live_rows: dict[str, int] = {}
        self._gen = 0
        self._max_rows = 0
        self.enabled = False
        self.layers_seen: set[str] = set()

    # -- configuration ------------------------------------------------------ #

    def configure(self, max_rows: int) -> None:
        """Rows to allocate per layer, from the largest decode batch vLLM will
        capture a graph for. Must be called before graph capture: a buffer that
        has to grow afterwards leaves already-recorded kernels writing into the
        old one, which costs those graphs their capture (see ``put``).
        """
        self._max_rows = max(self._max_rows, int(max_rows))

    # -- probe side --------------------------------------------------------- #

    def put(self, layer_name: str, q: torch.Tensor, n: int) -> None:
        """Copy ``q[:n]`` into this layer's buffer and record how it got there.

        ``is_current_stream_capturing`` is the whole distinction: True and the
        copy is being recorded into a graph that will re-run it on every replay,
        False and it is running now, for this step only.
        """
        recorded = _capturing()
        with self._lock:
            buf = self._buf.get(layer_name)
            if (buf is None or buf.shape[0] < n or buf.shape[1:] != q.shape[1:]
                    or buf.dtype != q.dtype or buf.device != q.device):
                if recorded:
                    # Allocating here would take the buffer from the graph's
                    # private pool, and the graph would be the only thing
                    # keeping it addressable. Skip instead: the key is not
                    # recorded, so the connector reports the layer unscored
                    # rather than reading a buffer nobody refreshes.
                    logger.warning(
                        "attn_connector: %s needs a %d-row query buffer but the "
                        "stream is capturing a CUDA graph; this graph will not "
                        "capture queries. Call REGISTRY.configure() before "
                        "capture.", layer_name, n)
                    return
                stale = self._graph_rows.pop(layer_name, 0)
                if stale:
                    logger.warning(
                        "attn_connector: %s query buffer grew to %d rows after "
                        "CUDA graphs were captured against the old one (up to "
                        "%d rows); those graphs can no longer capture queries.",
                        layer_name, max(n, self._max_rows), stale)
                buf = torch.empty((max(n, self._max_rows), *q.shape[1:]),
                                  dtype=q.dtype, device=q.device)
                self._buf[layer_name] = buf
            # Under capture this becomes a recorded kernel; every replay of the
            # graph re-runs it, which is what keeps the buffer current.
            buf[:n].copy_(q[:n])
            if recorded:
                self._graph_rows[layer_name] = max(
                    self._graph_rows.get(layer_name, 0), n)
            else:
                self._live_gen[layer_name] = self._gen
                self._live_rows[layer_name] = n
            self.layers_seen.add(layer_name)

    # -- connector side ----------------------------------------------------- #

    def get(self, layer_name: str, n_rows: int) -> torch.Tensor | None:
        """This layer's queries for a decode step of ``n_rows`` requests, or
        None if nothing wrote them this step.

        None is the signal that the step cannot be scored. Returning a stale
        buffer instead would produce records that look right and are not.
        """
        buf = self._buf.get(layer_name)
        if buf is None or buf.shape[0] < n_rows:
            return None
        if (self._live_gen.get(layer_name, -1) == self._gen
                and self._live_rows.get(layer_name, 0) >= n_rows):
            return buf
        if n_rows <= self._graph_rows.get(layer_name, 0):
            return buf
        return None

    def graph_rows(self, layer_name: str) -> int:
        """Largest decode batch a captured graph writes this layer for, or 0 if
        no graph does. Diagnostics only."""
        return self._graph_rows.get(layer_name, 0)

    def end_step(self) -> None:
        """Retire live writes. Buffers deliberately survive: under CUDA graphs
        the probe never runs again, so dropping them is the bug this replaced.
        """
        self._gen += 1

    def clear(self) -> None:
        """Forget everything, buffers included. Not part of the step loop."""
        with self._lock:
            self._buf.clear()
            self._graph_rows.clear()
            self._live_gen.clear()
            self._live_rows.clear()


REGISTRY = QueryRegistry()


def _capture(layer: Any, query: torch.Tensor, attn_metadata: Any) -> None:
    """Copy this step's decode queries, if any."""
    if not REGISTRY.enabled or attn_metadata is None:
        return
    # `num_decode_tokens` exists on the dataclass but is only populated on some
    # paths (it was 0 on every call in testing). `max_query_len` is always set,
    # and == 1 is exactly the condition we need: one query token per request,
    # so batch row i is request slot i. Anything else -- prefill, chunked
    # prefill, speculative decoding -- breaks that identity, so skip it rather
    # than mis-attribute. It also keeps mixed prefill+decode graphs out of the
    # registry, so the connector sees them as unscorable instead of reading a
    # buffer their replay never writes.
    if getattr(attn_metadata, "max_query_len", 0) != 1:
        return
    # `query` is padded out to the CUDA-graph batch size; only the first
    # num_actual_tokens rows are real. Under capture that is the whole padded
    # batch (`_dummy_run` runs the capture size unpadded), so the recorded copy
    # covers every row any replay of this graph can fill.
    n = getattr(attn_metadata, "num_actual_tokens", 0)
    if not n:
        return
    name = getattr(layer, "layer_name", None)
    if name is None:
        return
    REGISTRY.put(name, query, n)


def install(backend: str | None = None) -> bool:
    """Register a query-capturing subclass of the active attention backend.

    Must run before the engine is built: ``_cached_get_attn_backend`` is
    ``@cache``-decorated, so the class is resolved once and never re-read.

    Returns True if the override was installed.
    """
    try:
        from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    except ImportError as exc:  # pragma: no cover
        logger.warning("attn_connector: attention backend registry unavailable: %s", exc)
        return False

    name = backend or "FLASH_ATTN"
    try:
        member = AttentionBackendEnum[name]
    except KeyError:
        logger.warning("attn_connector: unknown attention backend %r; probe not installed", name)
        return False

    try:
        base_cls = member.get_class()
    except Exception as exc:  # pragma: no cover
        logger.warning("attn_connector: cannot resolve %s: %s", name, exc)
        return False

    base_impl = base_cls.get_impl_cls()

    class _ProbedImpl(base_impl):  # type: ignore[misc, valid-type]
        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, *args, **kwargs):
            _capture(layer, query, attn_metadata)
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, *args, **kwargs
            )

    class _ProbedBackend(base_cls):  # type: ignore[misc, valid-type]
        @staticmethod
        def get_impl_cls():
            return _ProbedImpl

    # __qualname__ matters as much as __name__: vLLM's platform re-derives the
    # class path from the class object itself (`_backend_cls_path`, cuda.py:423)
    # rather than reusing the string handed to register_backend. A class defined
    # inside a function has qualname "install.<locals>._ProbedBackend", which
    # resolves to a module path that does not exist.
    for cls, base in ((_ProbedImpl, base_impl), (_ProbedBackend, base_cls)):
        cls.__name__ = f"Probed{base.__name__}"
        cls.__qualname__ = cls.__name__
        cls.__module__ = __name__
        globals()[cls.__name__] = cls
    register_backend(member, f"{__name__}.{_ProbedBackend.__name__}")
    REGISTRY.enabled = True
    logger.info("attn_connector: query probe installed over %s", base_cls.__name__)
    return True
