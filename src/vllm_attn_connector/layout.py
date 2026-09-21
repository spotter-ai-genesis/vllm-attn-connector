"""Model-agnostic resolution of paged KV cache tensors into canonical K/V views.

VENDORED -- third copy (vllm-kvnorm, vllm-expected-attn, here), identical apart
from the logger name. Out-of-tree connectors must be installable independently
so none may import another. Three copies is past the point where this should be
a shared ``vllm-kv-layout`` package; extract it before adding a fourth.

vLLM allocates one tensor per attention layer whose *logical* shape is exactly
``AttentionBackend.get_kv_cache_shape(...)``; the NHD/HND choice only permutes
strides (``_reshape_attention_kv_cache`` ends with ``.permute(*inv_order)``).
That means we can dispatch on the logical shape and let strides be handled
generically downstream.

This module maps whatever layout a backend chose onto a pair of canonical views

    ``(num_blocks, block_size, num_kv_heads, head_size)``

built purely with ``transpose``/slicing, so no data is copied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

try:  # see connector.py for why this uses vLLM's logger namespace
    from vllm.logger import init_logger

    logger = init_logger("vllm.attn_connector")
except ImportError:  # importable without vLLM, for unit tests
    logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class UnsupportedLayout(Exception):
    """Raised when a layer's KV cache cannot be mapped to canonical K/V views."""


@dataclass(frozen=True)
class LayerLayout:
    """Canonical, copy-free views of one layer's paged K and V caches."""

    layer_name: str
    k_view: torch.Tensor
    """``(num_blocks, block_size, num_kv_heads, head_size)``."""
    v_view: torch.Tensor
    """``(num_blocks, block_size, num_kv_heads, head_size_v)``."""
    block_size: int
    num_kv_heads: int
    head_size: int
    head_size_v: int
    layout_kind: str
    """Diagnostic label for the matched layout pattern."""

    @property
    def num_blocks(self) -> int:
        return self.k_view.shape[0]


def _canonical(t: torch.Tensor) -> torch.Tensor:
    """Assert a view is already ``(B, N, H, D)``-ordered."""
    assert t.ndim == 4, f"expected 4-D canonical view, got {t.shape}"
    return t


def resolve_layer_layout(
    layer_name: str,
    kv_cache: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int | None = None,
) -> LayerLayout:
    """Map one layer's KV cache tensor onto canonical K/V views.

    Args:
        layer_name: Name of the attention layer (diagnostics only).
        kv_cache: The tensor handed to ``KVConnector.register_kv_caches``.
        block_size: Tokens per page, from the layer's ``KVCacheSpec``.
        num_kv_heads: KV heads per page, from the layer's ``AttentionSpec``.
        head_size: Key head dimension.
        head_size_v: Value head dimension; defaults to ``head_size``.

    Raises:
        UnsupportedLayout: if the tensor is not a block-indexed dense K/V cache
            (e.g. Mamba state, MLA latent, quantised cache).
    """
    head_size_v = head_size if head_size_v is None else head_size_v

    if not isinstance(kv_cache, torch.Tensor):
        # Mamba / linear-attention layers hand over a list of state tensors that
        # are not block-indexed at all.
        raise UnsupportedLayout(
            f"{layer_name}: KV cache is {type(kv_cache).__name__}, not a Tensor "
            "(non-attention state layer)"
        )

    if kv_cache.dtype not in _SUPPORTED_DTYPES:
        raise UnsupportedLayout(
            f"{layer_name}: unsupported KV cache dtype {kv_cache.dtype} "
            "(quantised caches need scale plumbing, not implemented)"
        )

    shape = tuple(kv_cache.shape)
    n, h, d, dv = block_size, num_kv_heads, head_size, head_size_v
    fused = d + dv

    # --- 4-D fused K|V layouts (K and V packed along the last dim) ------------
    if len(shape) == 4 and shape[3] == fused:
        head_major = shape[1] == h and shape[2] == n
        token_major = shape[1] == n and shape[2] == h

        if head_major and token_major and h != n:
            # Unreachable, kept for clarity.
            raise UnsupportedLayout(f"{layer_name}: contradictory match on {shape}")

        if head_major and token_major:
            # block_size == num_kv_heads: genuinely ambiguous. FlashAttention's
            # head-major logical shape is the overwhelmingly common case.
            logger.warning(
                "attn_connector: %s has block_size == num_kv_heads == %d; assuming "
                "head-major (num_blocks, num_kv_heads, block_size, 2*head_size)",
                h,
                layer_name,
            )
            head_major, token_major = True, False

        if head_major:
            # (num_blocks, num_kv_heads, block_size, head_size + head_size_v)
            # This is FlashAttention's layout; mirrors flash_attn.py:
            #   kv_cache.transpose(1, 2).split(head_size, dim=-1)
            t = kv_cache.transpose(1, 2)
            return LayerLayout(
                layer_name=layer_name,
                k_view=_canonical(t[..., :d]),
                v_view=_canonical(t[..., d:]),
                block_size=n,
                num_kv_heads=h,
                head_size=d,
                head_size_v=dv,
                layout_kind="fused_head_major",
            )

        if token_major:
            # (num_blocks, block_size, num_kv_heads, head_size + head_size_v)
            return LayerLayout(
                layer_name=layer_name,
                k_view=_canonical(kv_cache[..., :d]),
                v_view=_canonical(kv_cache[..., d:]),
                block_size=n,
                num_kv_heads=h,
                head_size=d,
                head_size_v=dv,
                layout_kind="fused_token_major",
            )

    # --- 5-D split K/V layouts (leading dim of 2 selects K or V) --------------
    if len(shape) == 5 and shape[0] == 2 and shape[4] == d and d == dv:
        if shape[2] == n and shape[3] == h:
            # (2, num_blocks, block_size, num_kv_heads, head_size)
            return LayerLayout(
                layer_name=layer_name,
                k_view=_canonical(kv_cache[0]),
                v_view=_canonical(kv_cache[1]),
                block_size=n,
                num_kv_heads=h,
                head_size=d,
                head_size_v=dv,
                layout_kind="split_token_major",
            )
        if shape[2] == h and shape[3] == n:
            # (2, num_blocks, num_kv_heads, block_size, head_size)
            return LayerLayout(
                layer_name=layer_name,
                k_view=_canonical(kv_cache[0].transpose(1, 2)),
                v_view=_canonical(kv_cache[1].transpose(1, 2)),
                block_size=n,
                num_kv_heads=h,
                head_size=d,
                head_size_v=dv,
                layout_kind="split_head_major",
            )

    raise UnsupportedLayout(
        f"{layer_name}: cannot map KV cache of shape {shape} onto canonical "
        f"(num_blocks, block_size={n}, num_kv_heads={h}, head_size={d}/{dv}) views. "
        "MLA and other exotic layouts are not supported."
    )
