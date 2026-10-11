from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


DEFAULT_GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 128,
}


def load_checkpoint_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load completed JSONL records so a killed evaluation can resume."""

    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at line {line_number}: {path}") from exc
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"checkpoint record lacks sample_id at line {line_number}: {path}")
            records[sample_id] = record
    return records


def append_checkpoint_record(path: Path, record: Mapping[str, Any]) -> None:
    """Durably append one evaluation record before moving to the next prompt."""

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("checkpoint record requires a non-empty sample_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def render_user_prompt(prompt: str, tokenizer: Any) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("evaluation prompt must be a non-empty string")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generation_kwargs(
    config: Mapping[str, Any], pad_token_id: int | None, eos_token_id: int | None
) -> Dict[str, Any]:
    kwargs = dict(config)
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    return kwargs
