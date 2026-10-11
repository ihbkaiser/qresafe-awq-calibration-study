from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qresafe_awq_calibration.config import ExperimentConfig  # noqa: E402
from qresafe_awq_calibration.runtime import (  # noqa: E402
    dump_json,
    import_qresafe_auto,
    qresafe_commit,
    qresafe_root_from_arg,
    set_seed,
)


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_fp16_model(model_id: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


def load_awq_model(model_path: str, qresafe_root: Path, device_map: str):
    AutoAWQForCausalLM = import_qresafe_auto(qresafe_root)
    return AutoAWQForCausalLM.from_quantized(
        model_path,
        fuse_layers=False,
        device_map=device_map,
        trust_remote_code=True,
    )


def model_device(model):
    return next(model.parameters()).device


def underlying_model(model):
    return getattr(model, "model", model)


def _reference_awq_forward(module, x):
    """Evaluate one Q-resafe AWQ GEMM layer with its cached dense weight."""

    import torch

    out_shape = x.shape[:-1] + (module.out_features,)
    input_dtype = x.dtype
    if input_dtype != torch.float16:
        x = x.to(torch.float16)
    out = torch.matmul(x, module._qresafe_reference_weight)
    if module.bias is not None:
        out = out + module.bias
    if input_dtype != torch.float16:
        out = out.to(input_dtype)
    return out.reshape(out_shape)


def enable_reference_awq_backend(model) -> int:
    """Use Q-resafe's own dequantizer as a Triton-independent eval backend.

    Quantization and packed weights remain unchanged. This only replaces the
    failing Triton GEMM call during inference with a cached dense matmul.
    """

    import types

    import torch
    from modules.linear.gemm import WQLinear_GEMM
    from utils.packing_utils import dequantize_gemm

    patched = 0
    for module in underlying_model(model).modules():
        if not isinstance(module, WQLinear_GEMM):
            continue
        dense_weight = dequantize_gemm(
            module.qweight,
            module.qzeros,
            module.scales,
            module.w_bit,
            module.group_size,
        ).to(dtype=torch.float16).contiguous()
        module.register_buffer(
            "_qresafe_reference_weight",
            dense_weight,
            persistent=False,
        )
        module.forward = types.MethodType(_reference_awq_forward, module)
        patched += 1

    if patched == 0:
        raise RuntimeError("No Q-resafe WQLinear_GEMM modules found for reference backend")
    return patched
