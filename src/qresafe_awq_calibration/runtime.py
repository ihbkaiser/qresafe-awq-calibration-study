from __future__ import annotations

import json
import inspect
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QRESAFE_ROOT = PROJECT_ROOT.parent / "Qresafe" / "quant-without-ft"


def qresafe_root_from_arg(path: str | None = None) -> Path:
    root = Path(path or os.environ.get("QRESAFE_ROOT", DEFAULT_QRESAFE_ROOT)).resolve()
    if not (root / "auto.py").exists():
        raise FileNotFoundError(
            f"Q-resafe quant-without-ft checkout not found at {root}; "
            "set --qresafe-root to the official checkout"
        )
    return root


def qresafe_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def import_qresafe_auto(root: Path):
    install_qresafe_triton_compat()
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from auto import AutoAWQForCausalLM

    return AutoAWQForCausalLM


def install_qresafe_triton_compat() -> dict[str, object]:
    """Provide the legacy Triton helper required by Q-resafe's AWQ kernel.

    Q-resafe's official ``quant-without-ft`` kernel calls ``tl.interleave``.
    The Triton bundled with the pinned Torch 2.1.2 image does not expose that
    helper, although it does expose the primitives needed to express the same
    operation.  This shim is applied only when the helper is absent; it keeps
    the upstream Q-resafe AWQ kernel and quantization path unchanged.
    """

    try:
        import triton
        import triton.language as tl
    except ImportError:
        return {"available": False, "patched": False}

    # The fallback is a Triton JIT function.  Keep these names in this module's
    # globals so Triton's dependency resolver can compile the helper when it
    # is called from the upstream Q-resafe kernel.
    globals()["triton"] = triton
    globals()["tl"] = tl

    if hasattr(tl, "interleave"):
        return {
            "available": True,
            "patched": False,
            "triton": getattr(triton, "__version__", "unknown"),
        }

    @triton.jit
    def _qresafe_interleave(a, b):
        # Q-resafe invokes this with (x, x) on 2-D blocks.  Expand and
        # duplicate the last dimension, then view it as [x0, x0, x1, x1, ...].
        expanded = tl.expand_dims(a, 2)
        expanded = tl.broadcast_to(expanded, (a.shape[0], a.shape[1], 2))
        return tl.view(expanded, (a.shape[0], a.shape[1] * 2))

    tl.interleave = _qresafe_interleave
    # Triton 2.1 exposes reshape but its frontend intentionally rejects it;
    # Q-resafe's kernel uses it for shape-only transformations.
    if hasattr(tl, "view") and hasattr(tl, "reshape"):
        tl.reshape = tl.view
    if hasattr(tl, "tensor") and hasattr(tl, "view"):
        def _qresafe_tensor_reshape(self, shape, _builder=None):
            if isinstance(shape, int) or type(shape).__name__ == "constexpr":
                shape = (shape,)
            return tl.view(self, shape, _builder=_builder)

        tl.tensor.reshape = _qresafe_tensor_reshape

    # Q-resafe's GEMM kernel uses the newer ``tl.dot(a, b, acc,
    # out_dtype=...)`` API.  Triton 2.1 exposes the older
    # ``tl.dot(a, b, allow_tf32=True, out_dtype=...)`` API instead, so the
    # third positional argument is otherwise misinterpreted as a boolean.
    # Adapt only that legacy signature by computing the dot and adding the
    # supplied accumulator, which is the behavior expected by Q-resafe.
    try:
        dot_parameters = inspect.signature(tl.dot).parameters
    except (TypeError, ValueError, AttributeError):
        dot_parameters = {}
    if "allow_tf32" in dot_parameters and "acc" not in dot_parameters:
        legacy_dot = tl.dot
        tl.dot_qresafe_legacy = legacy_dot

        @triton.jit
        def _qresafe_dot(input_value, other_value, accumulator, out_dtype):
            return tl.dot_qresafe_legacy(
                input_value,
                other_value,
                allow_tf32=True,
                out_dtype=out_dtype,
            ) + accumulator

        tl.dot = _qresafe_dot
    return {
        "available": True,
        "patched": True,
        "triton": getattr(triton, "__version__", "unknown"),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
