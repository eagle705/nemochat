"""Exact raw bytes for verified ByteLevel BPE tokenizers (GPT-2 and Super).

Do not decode isolated UTF-8 fragments to text: replacement characters corrupt
byte lengths. Unsupported tokenizer decoders are intentionally rejected.
"""

import json


def bytelevel_alphabet():
    """Reproduce the 256-byte reversible alphabet used by ByteLevel BPE."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    characters = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(char): byte for byte, char in zip(visible, characters)}


def token_byte_lengths(tokenizer):
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("BPB requires a verified fast tokenizer backend")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if backend.get("model", {}).get("type") != "BPE":
        raise ValueError("Only ByteLevel BPE is implemented")
    if backend.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("Unsupported decoder: exact raw byte accounting required")
    if backend.get("normalizer") is not None:
        raise ValueError("Normalization requires a separate byte-accounting audit")
    inverse = bytelevel_alphabet()
    special = set(tokenizer.all_special_ids)
    # Added ordinary tokens are literal text, not ByteLevel-encoded symbols.
    added = {entry["id"]: entry for entry in backend.get("added_tokens", [])}
    lengths = []
    for token_id in range(len(tokenizer)):
        if token_id in special or added.get(token_id, {}).get("special", False):
            lengths.append(0)
            continue
        if token_id in added:
            lengths.append(len(added[token_id]["content"].encode("utf-8")))
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if not token or any(char not in inverse for char in token):
            raise ValueError(f"Unrecognized ByteLevel token at ID {token_id}")
        lengths.append(len(token))
    return lengths


def check_text_bytes(tokenizer, text, lengths):
    ids = tokenizer.encode(text, add_special_tokens=False)
    raw_bytes = len(text.encode("utf-8"))
    represented = sum(lengths[token_id] for token_id in ids)
    if represented != raw_bytes:
        raise ValueError(f"Byte mismatch: text={raw_bytes}, tokens={represented}")
    decoded = tokenizer.decode(ids, skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError("Text does not round-trip through tokenizer exactly")
    return ids
