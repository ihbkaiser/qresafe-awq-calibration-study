from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def render_chat_messages(messages: Sequence[Dict[str, str]], tokenizer: Any) -> str:
    """Render a user/assistant calibration pair exactly once.

    Q-resafe's AWQ loader receives strings and later calls ``tokenizer.encode``.
    Keeping rendering here separate makes it possible to audit the exact text
    before handing it to the upstream implementation.
    """

    if len(messages) < 2:
        raise ValueError("calibration data must contain a user message and assistant response")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("calibration data must start with user then assistant messages")
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise ValueError("every chat message must have string content")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer must implement apply_chat_template")
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return len(encoded)


def calibration_metrics(
    records: Iterable[Dict[str, Any]],
    tokenizer: Any,
    max_seq_len: int = 512,
    chunk_long_records: bool = False,
) -> Dict[str, Any]:
    """Mirror Q-resafe's list calibration loader accounting.

    By default this mirrors Q-resafe's behavior: individual examples longer
    than ``max_seq_len`` are discarded, the remaining token sequences are
    concatenated, and the stream is floored to full blocks. With
    ``chunk_long_records=True``, callers use the companion chunking helper and
    metrics retain those long source records instead.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    records = list(records)
    lengths: List[int] = []
    discarded = 0
    for record in records:
        if "messages" in record:
            text = render_chat_messages(record["messages"], tokenizer)
        elif "text" in record:
            text = record["text"]
        else:
            raise ValueError("each calibration record needs messages or text")
        length = _token_count(tokenizer, text)
        if length > max_seq_len and not chunk_long_records:
            discarded += 1
            continue
        if length > 0:
            lengths.append(length)

    total = sum(lengths)
    return {
        "source_examples": len(records),
        "used_examples": len(lengths),
        "discarded_examples": discarded,
        "total_tokens_before_blocking": total,
        "blocks": total // max_seq_len,
        "mean_tokens_per_example": (total / len(lengths)) if lengths else 0.0,
    }


def require_complete_block(
    metrics: Dict[str, Any], condition: str, max_seq_len: int
) -> None:
    """Fail before model loading if Q-resafe would receive no calibration block."""

    if metrics.get("blocks", 0) < 1:
        raise ValueError(
            f"{condition} has no complete calibration block: accepted "
            f"{metrics.get('total_tokens_before_blocking', 0)} tokens for "
            f"max_calib_seq_len={max_seq_len}."
        )


def tokenize_records_to_chunks(
    records: Iterable[Dict[str, Any]], tokenizer: Any, max_seq_len: int
) -> List[List[int]]:
    """Tokenize records into Q-resafe-compatible chunks without dropping long records.

    The upstream Q-resafe loader drops any individual list element longer than
    ``max_seq_len``.  Chunking here keeps the exact rendered token stream while
    still handing the official loader a ``List[List[int]]`` it accepts.
    """

    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    chunks: List[List[int]] = []
    for record in records:
        if "messages" in record:
            text = render_chat_messages(record["messages"], tokenizer)
        elif "text" in record:
            text = record["text"]
        else:
            raise ValueError("each calibration record needs messages or text")
        encoded = tokenizer.encode(text)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        encoded = list(encoded)
        for start in range(0, len(encoded), max_seq_len):
            chunk = encoded[start : start + max_seq_len]
            if chunk:
                chunks.append(chunk)
    return chunks


def token_match_calibration(
    records: Iterable[Dict[str, Any]],
    tokenizer: Any,
    target_tokens: int,
    max_seq_len: int,
) -> tuple[List[List[int]], Dict[str, Any]]:
    """Build a fixed-size prefix of the rendered calibration token stream.

    The token-matched confirmation must give Risk-I, Risk-II, and Risk-III the
    same effective amount of input to Q-resafe.  We therefore concatenate the
    rendered records in their locked order, keep exactly ``target_tokens``
    tokens, and split that prefix into full Q-resafe blocks.  The target must
    be a multiple of ``max_seq_len`` so no condition receives a partial block.
    No record is duplicated and the original token count is retained in the
    returned audit metadata.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if target_tokens % max_seq_len:
        raise ValueError("target_tokens must be a multiple of max_seq_len")

    records = list(records)
    encoded_records: List[List[int]] = []
    for record in records:
        if "messages" in record:
            text = render_chat_messages(record["messages"], tokenizer)
        elif "text" in record:
            text = record["text"]
        else:
            raise ValueError("each calibration record needs messages or text")
        encoded = tokenizer.encode(text)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        encoded = list(encoded)
        if encoded:
            encoded_records.append(encoded)

    original_total = sum(len(encoded) for encoded in encoded_records)
    if original_total < target_tokens:
        raise ValueError(
            "token-matched target exceeds the rendered calibration stream: "
            f"target={target_tokens}, available={original_total}"
        )

    stream = [token for encoded in encoded_records for token in encoded]
    selected = stream[:target_tokens]
    chunks = [
        selected[start : start + max_seq_len]
        for start in range(0, target_tokens, max_seq_len)
    ]
    metrics = {
        "source_examples": len(records),
        "used_examples": len(encoded_records),
        "discarded_examples": 0,
        "total_tokens_before_blocking": target_tokens,
        "original_total_tokens_before_matching": original_total,
        "discarded_tokens": original_total - target_tokens,
        "blocks": len(chunks),
        "mean_tokens_per_example": (
            target_tokens / len(encoded_records) if encoded_records else 0.0
        ),
        "token_match_target_tokens": target_tokens,
        "token_match_policy": "prefix_of_concatenated_rendered_stream",
    }
    return chunks, metrics
