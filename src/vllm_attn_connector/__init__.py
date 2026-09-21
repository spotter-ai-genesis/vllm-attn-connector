"""Exact per-decode-step attention capture for vLLM, out-of-tree.

Two extension points, both public:
  * ``install_probe()`` registers an attention-backend override that copies the
    decode queries. Must run BEFORE the engine is constructed.
  * ``AttnConnector`` is a ``KVConnector`` that recomputes q.K^T against the
    paged prompt keys, writes the statistics to a SafeTensors file, and records
    a reference to it as Flowcept provenance.
  * ``load_provenance(task_record)`` reads those arrays back.
"""

from .connector import AttnConnector
from .store import load as load_provenance, open_lazy
from .kernels import decode_attention, decode_attention_torch
from .probe import REGISTRY, install as install_probe

__all__ = [
    "AttnConnector",
    "load_provenance",
    "open_lazy",
    "install_probe",
    "REGISTRY",
    "decode_attention",
    "decode_attention_torch",
]
