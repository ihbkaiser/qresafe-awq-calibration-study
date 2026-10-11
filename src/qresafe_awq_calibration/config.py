from dataclasses import dataclass, field
from typing import Any, Dict


DEFAULT_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_RISK_MAX_CALIB_SEQ_LEN = 128
DEFAULT_TOKEN_MATCHED_TARGET_TOKENS = 384
DEFAULT_QUANT_CONFIG = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Locked configuration for the first AWQ calibration-data experiment."""

    model_id: str = DEFAULT_MODEL_ID
    seed: int = 42
    max_calib_samples: int = 128
    max_calib_seq_len: int = 512
    # Q-resafe floors the concatenated custom stream to complete blocks. The
    # ten reconstructed Risk-II/Risk-III pairs are too short to form a 512
    # token block, so all three Risk conditions use the same 128-token block
    # length. Pile keeps the unmodified 512-token Q-resafe default.
    risk_max_calib_seq_len: int = DEFAULT_RISK_MAX_CALIB_SEQ_LEN
    token_matched_target_tokens: int = DEFAULT_TOKEN_MATCHED_TARGET_TOKENS
    risk_samples: int = 10
    quant_config: Dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_QUANT_CONFIG)
    )

    def __post_init__(self) -> None:
        expected = DEFAULT_QUANT_CONFIG
        if self.quant_config != expected:
            raise ValueError(
                "This first experiment is locked to the Q-resafe INT4 AWQ "
                f"configuration: {expected}"
            )
        if self.seed != 42:
            raise ValueError("The first experiment is locked to seed=42")
        if self.max_calib_samples != 128 or self.max_calib_seq_len != 512:
            raise ValueError(
                "The first experiment is locked to max_calib_samples=128 and "
                "max_calib_seq_len=512"
            )
        if self.risk_max_calib_seq_len != DEFAULT_RISK_MAX_CALIB_SEQ_LEN:
            raise ValueError(
                "The first experiment is locked to a common 128-token block "
                "length for Risk-I/Risk-II/Risk-III"
            )
        if self.token_matched_target_tokens != DEFAULT_TOKEN_MATCHED_TARGET_TOKENS:
            raise ValueError(
                "The token-matched confirmation is locked to 384 tokens "
                "(three complete 128-token blocks)"
            )
        if self.token_matched_target_tokens % self.risk_max_calib_seq_len:
            raise ValueError(
                "The token-matched target must be divisible by the Risk block length"
            )
        if self.risk_samples != 10:
            raise ValueError("The first experiment is locked to 10 risk examples")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "seed": self.seed,
            "max_calib_samples": self.max_calib_samples,
            "max_calib_seq_len": self.max_calib_seq_len,
            "risk_max_calib_seq_len": self.risk_max_calib_seq_len,
            "token_matched_target_tokens": self.token_matched_target_tokens,
            "risk_samples": self.risk_samples,
            "quant_config": dict(self.quant_config),
        }
