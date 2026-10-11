from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_QRESAFE_ROOT = LOCAL_PROJECT_ROOT.parent / "Qresafe"
PROJECT_REMOTE_ROOT = "/root/project"
QRESAFE_REMOTE_ROOT = "/root/Qresafe/quant-without-ft"
ARTIFACT_REMOTE_ROOT = "/root/artifacts"
GPU_TYPE = "L40S"
DATASET_RUNTIME_PACKAGES = ("zstandard==0.23.0",)
CONDITION_CALIBRATION_SPECS = {
    "pile": {
        "max_calib_samples": 128,
        "max_calib_seq_len": 512,
        "calibration_block_length_policy": "qresafe_default_512",
    },
    "risk1": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "common_risk_128",
    },
    "risk2": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "common_risk_128",
    },
    "risk3": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "common_risk_128",
    },
}
TOKEN_MATCHED_CONDITION_CALIBRATION_SPECS = {
    "risk1": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "token_matched_common_384",
        "token_match_target_tokens": 384,
    },
    "risk2": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "token_matched_common_384",
        "token_match_target_tokens": 384,
    },
    "risk3": {
        "max_calib_samples": 10,
        "max_calib_seq_len": 128,
        "calibration_block_length_policy": "token_matched_common_384",
        "token_match_target_tokens": 384,
    },
}


def _calibration_metadata_matches(
    metadata: dict, condition: str, protocol: str = "standard"
) -> bool:
    if protocol == "standard":
        expected = CONDITION_CALIBRATION_SPECS[condition]
    elif protocol == "token_matched":
        expected = TOKEN_MATCHED_CONDITION_CALIBRATION_SPECS[condition]
    else:
        raise ValueError(f"unsupported calibration protocol: {protocol}")
    return all(metadata.get(key) == value for key, value in expected.items())


def _condition_artifacts_complete(
    artifact_root: Path, condition: str, protocol: str = "standard"
) -> bool:
    """Return whether a condition has a complete, minimally valid checkpoint."""

    if condition not in {"fp16", "pile", "risk1", "risk2", "risk3"}:
        raise ValueError(f"unsupported condition: {condition}")
    if protocol == "token_matched" and condition not in {"risk1", "risk2", "risk3"}:
        raise ValueError("token-matched protocol only supports Risk conditions")

    stem = condition if condition == "fp16" else f"awq_{condition}"
    summary_path = artifact_root / f"{stem}.summary.json"
    ppl_path = artifact_root / f"{stem}.ppl.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ppl = json.loads(ppl_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False

    if (
        summary.get("condition") != condition
        or not isinstance(summary.get("n"), int)
        or summary["n"] <= 0
        or "asr_heldout_refusal_heuristic" not in summary
        or not isinstance(ppl.get("ppl"), (int, float))
    ):
        return False

    if condition == "fp16":
        return True

    metadata_path = artifact_root / "models" / f"awq_{condition}" / "quantization_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("condition") == condition
        and metadata.get("bits") == 4
        and _calibration_metadata_matches(metadata, condition, protocol)
    )


def _quantized_model_complete(
    artifact_root: Path, condition: str, protocol: str = "standard"
) -> bool:
    """Return whether a saved AWQ model can be reused for evaluation."""

    if condition not in {"pile", "risk1", "risk2", "risk3"}:
        raise ValueError(f"unsupported AWQ condition: {condition}")
    if protocol == "token_matched" and condition not in {"risk1", "risk2", "risk3"}:
        raise ValueError("token-matched protocol only supports Risk conditions")

    model_dir = artifact_root / "models" / f"awq_{condition}"
    required = ("model.safetensors", "config.json", "tokenizer.json", "quantization_metadata.json")
    if not all((model_dir / filename).is_file() for filename in required):
        return False
    try:
        metadata = json.loads(
            (model_dir / "quantization_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("condition") == condition
        and metadata.get("bits") == 4
        and _calibration_metadata_matches(metadata, condition, protocol)
    )


def _retire_condition_evaluation_outputs(artifact_root: Path, condition: str) -> None:
    """Move outputs from a legacy model aside before evaluating a new model."""

    if condition == "fp16":
        stem = "fp16"
    elif condition in {"pile", "risk1", "risk2", "risk3"}:
        stem = f"awq_{condition}"
    else:
        raise ValueError(f"unsupported condition: {condition}")

    for path in (
        artifact_root / f"{stem}.jsonl",
        artifact_root / f"{stem}.summary.json",
        artifact_root / f"{stem}.ppl.json",
    ):
        if not path.exists():
            continue
        candidate = path.with_name(path.name + ".stale")
        counter = 1
        while candidate.exists():
            candidate = path.with_name(path.name + f".stale.{counter}")
            counter += 1
        path.rename(candidate)


def _ignore_project_path(path: Path) -> bool:
    excluded = {"remote_artifacts", "outputs", "__pycache__", ".pytest_cache", ".git"}
    return any(part in excluded for part in path.parts)


def _ignore_qresafe_path(path: Path) -> bool:
    return any(part in {"__pycache__", ".git"} for part in path.parts)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        extra_options="--index-url https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "huggingface_hub==0.27.0",
        "numpy==1.26.4",
        "protobuf==4.25.3",
        "safetensors==0.4.5",
        "scipy==1.14.1",
        "tokenizers==0.21.0",
        "tqdm==4.67.1",
        "transformers==4.47.1",
        "pytest>=8.0",
    )
    .pip_install(*DATASET_RUNTIME_PACKAGES)
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT),
        remote_path=PROJECT_REMOTE_ROOT,
        copy=True,
        ignore=_ignore_project_path,
    )
    .add_local_dir(
        str(LOCAL_QRESAFE_ROOT),
        remote_path="/root/Qresafe",
        copy=True,
        ignore=_ignore_qresafe_path,
    )
)

hf_cache = modal.Volume.from_name("qresafe-awq-hf-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("qresafe-awq-artifacts", create_if_missing=True)
app = modal.App("qresafe-awq-calibration")


def _remote_imports():
    scripts = f"{PROJECT_REMOTE_ROOT}/scripts"
    src = f"{PROJECT_REMOTE_ROOT}/src"
    for path in (scripts, src, QRESAFE_REMOTE_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from qresafe_awq_calibration.runtime import install_qresafe_triton_compat

    compat = install_qresafe_triton_compat()
    print("MODAL_TRITON_COMPAT_JSON=" + json.dumps(compat), flush=True)


def _login_huggingface():
    """Authenticate Hub clients without printing or persisting the raw token in code."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing inside Modal")
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, ARTIFACT_REMOTE_ROOT: artifacts},
)
def smoke():
    """Validate the Modal image, L40S allocation, and upstream AWQ path."""

    _remote_imports()
    import torch
    from auto import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Modal smoke did not receive a CUDA GPU")
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    print("MODAL_ENV_JSON=" + json.dumps(env), flush=True)

    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        safetensors=False,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
        calib_data=[
            "hello world " * 24,
            "calibration text " * 24,
            "third calibration example " * 18,
        ],
        max_calib_samples=3,
        max_calib_seq_len=128,
    )
    model.model.to("cuda:0")
    prompt = tokenizer("hello", return_tensors="pt")
    device = next(model.parameters()).device
    prompt = {key: value.to(device) for key, value in prompt.items()}
    with torch.inference_mode():
        generated = model.generate(
            **prompt,
            do_sample=False,
            max_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(
        "MODAL_AWQ_FORWARD_JSON="
        + json.dumps({"input_tokens": int(prompt["input_ids"].shape[-1]), "output_shape": list(generated.shape)}),
        flush=True,
    )
    summary = {"status": "ok", "model": model_id, "path": "Qresafe/quant-without-ft AWQ"}
    print("MODAL_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    return {"env": env, **summary}


_full_function_options = {
    "image": image,
    "gpu": GPU_TYPE,
    "timeout": 24 * 60 * 60,
    "volumes": {"/root/.cache/huggingface": hf_cache, ARTIFACT_REMOTE_ROOT: artifacts},
    # Keep the object graph identical in local and remote environments.
    # The secret itself is never copied into the image or project files.
    "secrets": [modal.Secret.from_name("huggingface")],
}
_prepare_function_options = dict(_full_function_options)
_prepare_function_options.pop("gpu")
_prepare_function_options["timeout"] = 60 * 60


@app.function(**_prepare_function_options)
def prepare_data():
    """Reconstruct locked calibration/eval data inside the persistent Volume."""

    _remote_imports()
    _login_huggingface()

    from argparse import Namespace

    from prepare_calib import prepare

    data_dir = Path(ARTIFACT_REMOTE_ROOT) / "data"
    args = Namespace(
        ultrachat_dataset="HuggingFaceH4/ultrachat_200k",
        ultrachat_config=None,
        ultrachat_split="train_sft",
        advbench_dataset="AlignmentResearch/AdvBench",
        advbench_config=None,
        advbench_split="train",
        risk3_response_file=None,
        risk3_response_column="proxy_gen_target",
        output_dir=data_dir,
    )
    prepare(args)

    summary = {
        "status": "ok",
        "risk3_source": "AlignmentResearch/AdvBench",
        "risk3_response_column": "proxy_gen_target",
        "risk3_samples": 10,
        "evaluation_file": str(data_dir / "advbench_eval.jsonl"),
        "note": "Risk-III is reconstructed; it is not the exact unreleased Q-resafe selection.",
    }
    print("MODAL_DATA_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    artifacts.commit()
    return summary


@app.function(**_prepare_function_options)
def token_matched_preflight(model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Check the real Llama tokenizer and token-matched budgets before GPU use."""

    _remote_imports()
    _login_huggingface()
    from common import load_tokenizer
    from qresafe_awq_calibration.calibration import (
        require_complete_block,
        token_match_calibration,
    )
    from qresafe_awq_calibration.data import read_jsonl

    tokenizer = load_tokenizer(model_id)
    data_dir = Path(ARTIFACT_REMOTE_ROOT) / "data"
    metrics = {}
    for condition in ("risk1", "risk2", "risk3"):
        records = read_jsonl(data_dir / f"{condition}.jsonl")
        _, condition_metrics = token_match_calibration(
            records, tokenizer, target_tokens=384, max_seq_len=128
        )
        require_complete_block(condition_metrics, condition, 128)
        metrics[condition] = {
            "source_examples": condition_metrics["source_examples"],
            "original_tokens": condition_metrics["original_total_tokens_before_matching"],
            "selected_tokens": condition_metrics["total_tokens_before_blocking"],
            "blocks": condition_metrics["blocks"],
        }
    summary = {"status": "ok", "protocol": "token_matched_common_384", "metrics": metrics}
    print("MODAL_TOKEN_MATCHED_PREFLIGHT_JSON=" + json.dumps(summary), flush=True)
    return summary


@app.function(**_prepare_function_options)
def check_model_access(model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Check gated-model access before allocating an L40S for quantization."""

    _remote_imports()
    _login_huggingface()
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_id, "config.json")
    summary = {"status": "ok", "model_id": model_id, "config_cached": bool(config_path)}
    print("MODAL_MODEL_ACCESS_JSON=" + json.dumps(summary), flush=True)
    return summary


@app.function(**_full_function_options)
def full_experiment(model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Run FP16 + Pile + Risk-I/II/III sequentially on one L40S."""

    _remote_imports()
    if model_id.startswith("meta-llama/") and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "HF_TOKEN is missing inside Modal. Set MODAL_HF_SECRET_NAME locally "
            "to the name of a Modal secret containing HF_TOKEN."
        )
    _login_huggingface()
    from argparse import Namespace

    from common import qresafe_root_from_arg, set_seed
    from collect_results import collect
    from eval_safety import evaluate as evaluate_safety
    from eval_utility import evaluate as evaluate_utility
    from quantize_model import quantize_condition
    from write_environment import main as write_environment

    data_dir = Path(ARTIFACT_REMOTE_ROOT) / "data"
    eval_file = data_dir / "advbench_eval.jsonl"
    if not eval_file.exists():
        raise FileNotFoundError(
            f"{eval_file} is missing; run the prepare stage before the full experiment"
        )
    qresafe_root = qresafe_root_from_arg(QRESAFE_REMOTE_ROOT)
    artifact_root = Path(ARTIFACT_REMOTE_ROOT)
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_environment(artifact_root / "environment.txt", QRESAFE_REMOTE_ROOT)
    set_seed(42)

    def commit_checkpoint():
        artifacts.commit()

    # This keeps every condition on the same image/GPU and persists all outputs
    # to the mounted Modal Volume.
    results = []
    eval_specs = [("fp16", None)] + [
        (condition, artifact_root / "models" / f"awq_{condition}")
        for condition in ("pile", "risk1", "risk2", "risk3")
    ]
    for condition, model_path in eval_specs:
        if _condition_artifacts_complete(artifact_root, condition):
            results.append(condition)
            print(
                "MODAL_CONDITION_JSON="
                + json.dumps({"condition": condition, "status": "resumed"}),
                flush=True,
            )
            continue
        if condition != "fp16":
            if _quantized_model_complete(artifact_root, condition):
                print(
                    "MODAL_QUANTIZATION_JSON="
                    + json.dumps({"condition": condition, "status": "resumed"}),
                    flush=True,
                )
            else:
                quantize_condition(
                    condition,
                    model_id,
                    model_path,
                    data_dir,
                    qresafe_root,
                    "auto",
                )
                _retire_condition_evaluation_outputs(artifact_root, condition)
                # Persist the quantized model and its metadata before any
                # expensive safety/utility evaluation starts.
                commit_checkpoint()
        safety_args = Namespace(
            condition=condition,
            model_id=model_id,
            model_path=model_path,
            eval_file=eval_file,
            output=artifact_root / f"{condition if condition == 'fp16' else 'awq_' + condition}.jsonl",
            qresafe_root=QRESAFE_REMOTE_ROOT,
            device_map="auto",
            max_samples=None,
            checkpoint_callback=commit_checkpoint,
            checkpoint_every=10,
        )
        utility_args = Namespace(
            condition=condition,
            model_id=model_id,
            model_path=model_path,
            qresafe_root=QRESAFE_REMOTE_ROOT,
            device_map="auto",
            seqlen=2048,
            max_tokens=None,
            output=artifact_root / f"{condition if condition == 'fp16' else 'awq_' + condition}.ppl.json",
        )
        evaluate_safety(safety_args)
        evaluate_utility(utility_args)
        commit_checkpoint()
        results.append(condition)
        print("MODAL_CONDITION_JSON=" + json.dumps({"condition": condition, "status": "ok"}), flush=True)

    collect(artifact_root, model_id, artifact_root / "results.csv")
    summary = {"status": "ok", "gpu": GPU_TYPE, "model_id": model_id, "conditions": results}
    print("MODAL_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    artifacts.commit()
    return summary


@app.function(**_full_function_options)
def token_matched_experiment(model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
    """Run the token-matched R1/R2/R3 confirmation without overwriting R6."""

    _remote_imports()
    if model_id.startswith("meta-llama/") and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is missing inside Modal")
    _login_huggingface()
    from argparse import Namespace

    from common import qresafe_root_from_arg, set_seed
    from collect_results import collect
    from eval_safety import evaluate as evaluate_safety
    from eval_utility import evaluate as evaluate_utility
    from quantize_model import quantize_condition
    from write_environment import main as write_environment

    data_dir = Path(ARTIFACT_REMOTE_ROOT) / "data"
    eval_file = data_dir / "advbench_eval.jsonl"
    if not eval_file.exists():
        raise FileNotFoundError(f"{eval_file} is missing; run preparation first")

    qresafe_root = qresafe_root_from_arg(QRESAFE_REMOTE_ROOT)
    artifact_root = Path(ARTIFACT_REMOTE_ROOT) / "token_matched"
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_environment(artifact_root / "environment.txt", QRESAFE_REMOTE_ROOT)
    set_seed(42)

    def commit_checkpoint():
        artifacts.commit()

    results = []
    for condition in ("risk1", "risk2", "risk3"):
        model_path = artifact_root / "models" / f"awq_{condition}"
        if _condition_artifacts_complete(artifact_root, condition, protocol="token_matched"):
            results.append(condition)
            print(
                "MODAL_TOKEN_MATCHED_CONDITION_JSON="
                + json.dumps({"condition": condition, "status": "resumed"}),
                flush=True,
            )
            continue
        if _quantized_model_complete(artifact_root, condition, protocol="token_matched"):
            print(
                "MODAL_TOKEN_MATCHED_QUANTIZATION_JSON="
                + json.dumps({"condition": condition, "status": "resumed"}),
                flush=True,
            )
        else:
            quantize_condition(
                condition,
                model_id,
                model_path,
                data_dir,
                qresafe_root,
                "auto",
                token_matched=True,
            )
            _retire_condition_evaluation_outputs(artifact_root, condition)
            commit_checkpoint()

        stem = f"awq_{condition}"
        safety_args = Namespace(
            condition=condition,
            model_id=model_id,
            model_path=model_path,
            eval_file=eval_file,
            output=artifact_root / f"{stem}.jsonl",
            qresafe_root=QRESAFE_REMOTE_ROOT,
            device_map="auto",
            max_samples=None,
            checkpoint_callback=commit_checkpoint,
            checkpoint_every=10,
        )
        utility_args = Namespace(
            condition=condition,
            model_id=model_id,
            model_path=model_path,
            qresafe_root=QRESAFE_REMOTE_ROOT,
            device_map="auto",
            seqlen=2048,
            max_tokens=None,
            output=artifact_root / f"{stem}.ppl.json",
        )
        evaluate_safety(safety_args)
        evaluate_utility(utility_args)
        commit_checkpoint()
        results.append(condition)
        print(
            "MODAL_TOKEN_MATCHED_CONDITION_JSON="
            + json.dumps({"condition": condition, "status": "ok"}),
            flush=True,
        )

    collect(
        artifact_root,
        model_id,
        artifact_root / "results.csv",
        selected_conditions=("risk1", "risk2", "risk3"),
    )
    summary = {
        "status": "ok",
        "protocol": "token_matched_common_384",
        "gpu": GPU_TYPE,
        "model_id": model_id,
        "conditions": results,
    }
    print("MODAL_TOKEN_MATCHED_SUMMARY_JSON=" + json.dumps(summary), flush=True)
    artifacts.commit()
    return summary


@app.local_entrypoint()
def main(mode: str = "smoke", model_id: str = "meta-llama/Llama-2-7b-chat-hf"):
    if mode == "smoke":
        result = smoke.remote()
    elif mode == "prepare":
        result = prepare_data.remote()
    elif mode == "check":
        result = check_model_access.remote(model_id=model_id)
    elif mode == "full":
        prepare_result = prepare_data.remote()
        print(json.dumps({"prepare": prepare_result}, indent=2))
        result = full_experiment.remote(model_id=model_id)
    elif mode == "token_matched":
        prepare_result = prepare_data.remote()
        print(json.dumps({"prepare": prepare_result}, indent=2))
        preflight_result = token_matched_preflight.remote(model_id=model_id)
        print(json.dumps({"preflight": preflight_result}, indent=2))
        result = token_matched_experiment.remote(model_id=model_id)
    else:
        raise ValueError("mode must be smoke, prepare, check, full, or token_matched")
    print(json.dumps(result, indent=2))
