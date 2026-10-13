"""A readable port of nanochat's pinned CORE task scoring semantics.

Reference: karpathy/nanochat, MIT, snapshot
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd, core_eval.py/base_eval.py.
No nanochat model class is required; inference is single-process HF here.
"""

import random

from nemochat.eval.hf_eval import next_token_outputs
from nemochat.eval.metrics import core_score


def common_length(sequences, *, suffix=False):
    if not sequences:
        raise ValueError("Sequences must not be empty")
    for offset in range(min(map(len, sequences))):
        index = -1 - offset if suffix else offset
        if len({sequence[index] for sequence in sequences}) != 1:
            return offset
    return min(map(len, sequences))


def render_prompts(item, task_type, delimiter=" ", examples=()):
    if task_type == "multiple_choice":
        demonstrations = [ex["query"] + delimiter + ex["choices"][ex["gold"]]
                          for ex in examples]
        prefix = "\n\n".join(demonstrations)
        prefix = prefix + "\n\n" if prefix else ""
        return [prefix + item["query"] + delimiter + choice
                for choice in item["choices"]]
    if task_type == "schema":
        demonstrations = [ex["context_options"][ex["gold"]] + delimiter + ex["continuation"]
                          for ex in examples]
        prefix = "\n\n".join(demonstrations)
        prefix = prefix + "\n\n" if prefix else ""
        return [prefix + context + delimiter + item["continuation"]
                for context in item["context_options"]]
    if task_type == "language_modeling":
        demonstrations = [ex["context"].strip() + delimiter + ex["continuation"]
                          for ex in examples]
        prefix = "\n\n".join(demonstrations)
        prefix = prefix + "\n\n" if prefix else ""
        without = prefix + item["context"].strip() + delimiter
        return [without.strip(), without + item["continuation"]]
    raise ValueError(f"Unknown CORE task type: {task_type}")


def tokenize_spans(tokenizer, prompts, task_type):
    bos = tokenizer.bos_token_id
    if bos is None:
        raise ValueError("CORE requires an explicit BOS token")
    sequences = [[bos] + tokenizer.encode(prompt, add_special_tokens=False)
                 for prompt in prompts]
    if task_type == "multiple_choice":
        start = common_length(sequences)
        if len(sequences) < 2:
            raise ValueError("Multiple choice requires at least two options")
        spans = [(start, len(seq)) for seq in sequences]
    elif task_type == "schema":
        size = common_length(sequences, suffix=True)
        if len(sequences) < 2:
            raise ValueError("Schema requires at least two contexts")
        spans = [(len(seq) - size, len(seq)) for seq in sequences]
    elif task_type == "language_modeling":
        prefix, full = sequences
        if full[:len(prefix)] != prefix:
            raise ValueError("CORE answer boundary is not a token prefix")
        sequences = [full]
        spans = [(len(prefix), len(full))]
    else:
        raise ValueError(f"Unknown CORE task type: {task_type}")
    if any(start < 1 or end <= start for start, end in spans):
        raise ValueError("CORE continuation must be nonempty with a causal prefix")
    return sequences, spans


def evaluate_item(model, tokenizer, item, task_type, *, delimiter=" ", examples=(),
                  context_length=2048):
    prompts = render_prompts(item, task_type, delimiter, examples)
    sequences, spans = tokenize_spans(tokenizer, prompts, task_type)
    option_losses = []
    match = None
    cropped = 0
    for sequence, (start, end) in zip(sequences, spans):
        crop = max(0, len(sequence) - context_length)
        if start - crop < 1:
            raise ValueError("Context truncation would remove answer or causal prefix")
        cropped += crop
        sequence = sequence[crop:]
        start, end = start - crop, end - crop
        losses, predictions = next_token_outputs(model, sequence)
        selected = losses[start - 1:end - 1]
        option_losses.append(sum(selected) / len(selected))
        if task_type == "language_modeling":
            match = predictions[start - 1:end - 1] == sequence[start:end]
    if task_type == "language_modeling":
        correct = match
        predicted_option = None
    else:
        gold = item["gold"]
        if not isinstance(gold, int) or not 0 <= gold < len(option_losses):
            raise ValueError("Invalid gold option")
        predicted_option = min(range(len(option_losses)), key=option_losses.__getitem__)
        correct = predicted_option == gold
    return {"correct": bool(correct), "mean_token_nll": option_losses,
            "predicted_option": predicted_option, "cropped_tokens": cropped}


def evaluate_core_tasks(model, tokenizer, tasks, data_by_task, baselines, *,
                        max_per_task=100, context_length=2048):
    if max_per_task != -1 and max_per_task <= 0:
        raise ValueError("max_per_task must be positive or -1 for full")
    if context_length < 2:
        raise ValueError("CORE context_length must be at least 2")
    model_training = model.training
    model.eval()
    raw, counts, accuracies = [], {}, {}
    try:
        for task in tasks:
            label = task["label"]
            # nanochat shuffles with 1337 BEFORE few-shot selection on capped data.
            data = list(data_by_task[label])
            random.Random(1337).shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]
            shots = task["num_fewshot"][0]
            if shots < 0 or len(data) <= shots:
                raise ValueError(f"Too few examples for {label}: {len(data)}, shots={shots}")
            results = []
            for index, item in enumerate(data):
                available = [i for i in range(len(data)) if i != index]
                chosen = random.Random(1234 + index).sample(available, shots)
                result = evaluate_item(
                    model, tokenizer, item, task["icl_task_type"],
                    delimiter=task.get("continuation_delimiter", " "),
                    examples=[data[i] for i in chosen], context_length=context_length,
                )
                results.append(result["correct"])
                raw.append({"task": label, "shuffled_index": index,
                            "fewshot_indices": chosen, "item": item, **result})
            counts[label] = len(results)
            accuracies[label] = sum(results) / len(results)
    finally:
        model.train(model_training)
    selected_baselines = {task: baselines[task] for task in accuracies}
    return {**core_score(accuracies, selected_baselines), "counts": counts,
            "max_per_task": max_per_task, "context_length": context_length,
            "protocol": "nanochat-compatible CORE; single-process HF",
            "raw_results": raw}
