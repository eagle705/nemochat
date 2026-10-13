"""Single-process HF evaluation. This is NOT a Megatron distributed adapter."""

import torch
import torch.nn.functional as F

from nemochat.eval.metrics import LossTotals
from nemochat.eval.token_bytes import check_text_bytes


@torch.inference_mode()
def next_token_outputs(model, input_ids):
    """loss[i] and prediction[i] refer to input_ids[i + 1]."""
    if len(input_ids) < 2:
        raise ValueError("Need a prefix and at least one target token")
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite model logits in next-token evaluation")
    losses = F.cross_entropy(logits, ids[0, 1:], reduction="none")
    if not torch.isfinite(losses).all().item():
        raise ValueError("Nonfinite token NLL in next-token evaluation")
    predictions = logits.argmax(dim=-1)
    return losses.cpu().tolist(), predictions.cpu().tolist()


def evaluate_texts(model, tokenizer, texts, byte_lengths, *, context_length=128):
    """Score document-local chunks with a BOS/EOS prefix and explicit resets.

    Every ordinary document token is a target exactly once. A reset prefix is
    inserted for each chunk; there is no context shared across documents.
    Fix this protocol/context for checkpoint comparisons.
    """
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    prefix = tokenizer.bos_token_id
    if prefix is None:
        prefix = tokenizer.eos_token_id
    if prefix is None:
        raise ValueError("A declared BOS or EOS reset-prefix token is required")
    prior_training = model.training
    model.eval()
    totals = LossTotals()
    records = []
    try:
        for index, text in enumerate(texts):
            ids = check_text_bytes(tokenizer, text, byte_lengths)
            if not ids:
                continue
            doc_totals = LossTotals()
            for start in range(0, len(ids), context_length - 1):
                targets = ids[start:start + context_length - 1]
                losses, _ = next_token_outputs(model, [prefix] + targets)
                doc_totals.add(losses, [byte_lengths[t] for t in targets])
            totals.merge(doc_totals)
            records.append({"document_index": index, **doc_totals.report()})
    finally:
        model.train(prior_training)
    return {"aggregate": totals.report(), "documents": records,
            "protocol": "document-local BOS-prefixed disjoint chunks",
            "context_length": context_length}
