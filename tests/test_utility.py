import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from eval_utility import utility_forward_model  # noqa: E402


class _CausalLM:
    pass


def test_fp16_utility_forwards_through_causal_lm_wrapper():
    causal_lm = _CausalLM()
    model = SimpleNamespace(model=SimpleNamespace(base=causal_lm))

    assert utility_forward_model(model, "fp16") is model


def test_awq_utility_forwards_through_official_awq_inner_model():
    awq_inner_model = _CausalLM()
    model = SimpleNamespace(model=awq_inner_model)

    assert utility_forward_model(model, "risk3") is awq_inner_model
