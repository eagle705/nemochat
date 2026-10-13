"""Deterministic token-budget plans, not a trainer or Bridge integration."""
import json
from pathlib import Path
from .io import write_json


LONG = 65536
SHORT = 4096


def make_plan(long_only_steps=2, alternating_steps=4, effective_tokens=131072,
              micro_batch_size=1, world_size=1):
    values = locals().copy()
    for name, value in values.items():
        if type(value) is not int or value < (0 if name.endswith("steps") else 1):
            raise ValueError(f"{name} must be an integer with a valid positive budget/nonnegative step count")
    if not long_only_steps + alternating_steps:
        raise ValueError("the plan must contain at least one step")
    for length in (LONG, SHORT):
        tokens_per_microstep = length * micro_batch_size * world_size
        if effective_tokens % tokens_per_microstep:
            raise ValueError(f"effective_tokens must be divisible by {tokens_per_microstep} for length {length}")
    return {"version": 1, **values}


def step_at(plan, step):
    validated = make_plan(**{key: value for key, value in plan.items() if key != "version"})
    if plan != validated:
        raise ValueError("unsupported or malformed plan")
    if type(step) is not int or not 0 <= step < plan["long_only_steps"] + plan["alternating_steps"]:
        raise ValueError("step outside plan")
    long_phase = step < plan["long_only_steps"]
    length = LONG if long_phase or (step - plan["long_only_steps"]) % 2 == 0 else SHORT
    batch = plan["micro_batch_size"] * plan["world_size"]
    return {"step": step, "phase": "long_only" if long_phase else "alternating",
            "sequence_length": length, "micro_batch_size": plan["micro_batch_size"],
            "world_size": plan["world_size"],
            "gradient_accumulation_steps": plan["effective_tokens"] // (length * batch),
            "effective_sequences": plan["effective_tokens"] // length,
            "effective_tokens": plan["effective_tokens"]}


def load_cursor(path, plan):
    """Missing file starts at zero. Existing plan mismatch is an error, never reset."""
    step_at(plan, 0)
    if not Path(path).exists():
        return {"version": 1, "plan": plan, "next_step": 0}
    with open(path, encoding="utf-8") as stream:
        cursor = json.load(stream)
    position = cursor.get("next_step")
    total = plan["long_only_steps"] + plan["alternating_steps"]
    if (cursor.get("version") != 1 or cursor.get("plan") != plan or
            type(position) is not int or not 0 <= position <= total):
        raise ValueError("cursor is malformed or belongs to a different plan")
    return cursor


def advance_cursor(cursor, count=1):
    """Return (emitted records, new cursor); caller checkpoints explicitly."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    plan = cursor["plan"]
    start = cursor["next_step"]
    total = plan["long_only_steps"] + plan["alternating_steps"]
    if type(start) is not int or not 0 <= start <= total or cursor.get("version") != 1:
        raise ValueError("invalid cursor")
    # Validate even an exhausted cursor.
    step_at(plan, 0)
    end = min(start + count, total)
    return [step_at(plan, step) for step in range(start, end)], {**cursor, "next_step": end}


def save_cursor(path, cursor):
    advance_cursor(cursor, 1)
    write_json(path, cursor)
