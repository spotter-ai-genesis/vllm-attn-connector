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
``self.impl.forward(...)`` dynamically. Overriding the *impl* therefore fires
during tracing; an ``nn.Module`` forward hook would not.

Why the copy goes into a persistent buffer
------------------------------------------
Being on the traced path is necessary but not sufficient once CUDA graphs are
on -- which is the default for ``vllm serve``. A graph records the kernels a
trace emits *once*, and replay re-runs only those kernels. No Python executes on
replay, so a ``dict[name] = tensor`` inside ``_capture`` fires at capture time
and never again.

What *does* re-run is the copy kernel. So the probe allocates one buffer per
layer, sized to the engine's ``max_num_seqs``, and copies into a prefix of it.
Every graph -- whatever batch size it was captured for -- records a copy into
that same buffer, so after any replay the first ``m`` rows hold the current
queries. The connector reads the buffer rather than a per-step dict entry.
Eager is unaffected: ``_capture`` still runs each step and copies eagerly.

The registry is therefore **never cleared between steps**. Freshness is
established by the copy kernel having run, not by the entry's existence, so the
connector has to know which buffer this step wrote -- see ``ran_eagerly``.

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


class QueryRegistry:
    """One decode-query buffer per layer, written by the probe, read by the
    connector.

    Sized to ``max_rows`` (the engine's ``max_num_seqs``) and allocated once,
    before any graph is captured. Every CUDA graph, whatever batch size it was
    captured for, records a copy into a prefix of that same buffer, so on replay
    the first ``m`` rows always hold the current step's queries and there is
    nothing to key or guess.

    Keeping one buffer per *size* instead would also work but costs far more:
    vLLM captures a graph per batch size, and the sizes sum to several times the
    largest, which measured 182 MB on a 24-layer 0.5B model and would exceed a
    gigabyte on a 36-layer 4B one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buf: dict[str, torch.Tensor] = {}
        self._eager_rows = 0
        self.max_rows = 0        # set by the connector from max_num_seqs
        self.enabled = False
        self.layers_seen: set[str] = set()

    def buffer(self, layer_name: str, rows: int, like: torch.Tensor) -> torch.Tensor | None:
        """The persistent destination for a layer, allocating on first use.

        Returns None if ``rows`` exceeds what was allocated, which would mean a
        decode batch larger than ``max_num_seqs``. Growing instead is not an
        option: graphs already captured hold the old address and would keep
        writing there.
        """
        buf = self._buf.get(layer_name)
        if buf is None:
            cap = max(self.max_rows, rows)
            buf = torch.zeros((cap, *like.shape[1:]), dtype=like.dtype,
                              device=like.device)
            with self._lock:
                self._buf[layer_name] = buf
                self.layers_seen.add(layer_name)
        if rows > buf.shape[0]:
            return None
        return buf

    def begin_step(self) -> None:
        """Called before each forward, to reset the did-Python-run marker."""
        self._eager_rows = 0

    def note_ran(self, rows: int) -> None:
        self._eager_rows = rows

    @property
    def ran_eagerly(self) -> bool:
        """True when ``_capture`` executed during this step's forward.

        False means the forward was a CUDA graph replay: the copy kernel ran but
        no Python did. The buffer is still current either way; this only tells
        the connector whether it has a probe-side row count to cross-check.
        """
        return self._eager_rows > 0

    @property
    def num_decode_reqs(self) -> int:
        return self._eager_rows

    def get(self, layer_name: str, rows: int) -> torch.Tensor | None:
        buf = self._buf.get(layer_name)
        if buf is None or rows > buf.shape[0]:
            return None
        return buf

    def clear(self) -> None:
        """Drop the buffers. For teardown only.

        Deliberately *not* called between steps: under CUDA graphs these buffers
        are the only thing the replayed copy kernels write into, and dropping
        them would leave every later step reading nothing.
        """
        with self._lock:
            self._buf.clear()
            self._eager_rows = 0


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
    # than mis-attribute.
    if getattr(attn_metadata, "max_query_len", 0) != 1:
        return
    # `query` is padded out to the CUDA-graph batch size; only the first
    # num_actual_tokens rows are real.
    n = getattr(attn_metadata, "num_actual_tokens", 0)
    if not n:
        return
    name = getattr(layer, "layer_name", None)
    if name is None:
        return
    # Copy into a persistent buffer rather than allocating a fresh clone: the
    # copy is what survives into a CUDA graph, the allocation is not. A view
    # would not do either -- `query` is reused by the next step.
    buf = REGISTRY.buffer(name, n, query)
    if buf is None:
        return
    buf[:n].copy_(query[:n].detach())
    REGISTRY.note_ran(n)


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
