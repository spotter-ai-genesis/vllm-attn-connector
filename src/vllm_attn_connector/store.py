"""Attention statistics on disk, referenced from provenance.

SafeTensors because it is typed, shaped, mmap-able, has no pickle in it, allows
reading one tensor without the rest, and carries a string header we can make
the file self-describing with. It is not appendable, which is fine: a request's
statistics are complete before anything is written.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.attn_connector")
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

FORMAT = "safetensors"


def write(out_dir: Path, workflow_id: str, request_id: str, group_id: int,
          tensors: dict[str, np.ndarray], header: dict[str, str],
          checksum: bool = True) -> dict[str, Any]:
    """Write one request's arrays and return the descriptor to record.

    Written to a temporary name and renamed, so a reader tailing the directory
    never sees a half-written file.

    Never raises. Provenance capture must not be able to fail a generation
    request, so a write error comes back as a descriptor carrying `error`,
    which makes the gap visible in the data rather than silent.
    """
    rel = f"{workflow_id}/{request_id}_g{group_id}.{FORMAT}"
    path = out_dir / rel
    try:
        from safetensors.numpy import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{FORMAT}.tmp")
        save_file(tensors, str(tmp), metadata=header)
        os.replace(tmp, path)
        desc: dict[str, Any] = {
            "uri": path.resolve().as_uri(),
            "format": FORMAT,
            "bytes": path.stat().st_size,
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                        for k, v in tensors.items()},
        }
        if checksum:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            desc["sha256"] = h.hexdigest()
        return desc
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("attn_connector: could not write %s: %s", path, exc)
        return {"uri": None, "format": FORMAT, "error": f"{type(exc).__name__}: {exc}"}


def load(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """Read back the arrays a task record points at.

    Takes the whole task record -- the thing a Flowcept query returns -- rather
    than a path, so callers do not have to know where the descriptor lives.

        arrays = load(task)
        arrays["val_all_max"]        # [decode steps, k]
        arrays["segments"]           # [n_segments, 3] as (lo, hi, keep)
    """
    from safetensors.numpy import load_file

    desc = record.get("attention_stats") if isinstance(record, dict) else None
    if not desc or not desc.get("uri"):
        raise ValueError(
            "record has no attention_stats.uri"
            + (f" (error: {desc['error']})" if desc and desc.get("error") else ""))
    uri = desc["uri"]
    path = uri[len("file://"):] if uri.startswith("file://") else uri
    return load_file(path)


def open_lazy(record: dict[str, Any]):
    """`safe_open` handle, for reading one tensor out of a large file.

    Use when the file is big and only part of it is wanted::

        with open_lazy(task) as f:
            segs = f.get_tensor("segments")
    """
    from safetensors import safe_open

    desc = record["attention_stats"]
    uri = desc["uri"]
    path = uri[len("file://"):] if uri.startswith("file://") else uri
    return safe_open(path, framework="numpy")

