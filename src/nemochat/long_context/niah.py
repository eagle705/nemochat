"""Tokenizer-measured fixtures and real completion-endpoint predictions.

Fixtures are synthetic inputs, never claims about model retrieval performance.
The endpoint must serve the same tokenizer/BOS policy: usage is checked strictly.
"""
import hashlib
import ipaddress
import json
import math
import os
import random
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse


MAX_RESPONSE_BYTES = 1_048_576  # HTTP JSON body cap; read cap + 1 to detect overflow.


DEFAULT_HAYSTACK = "The archive contains ordinary notes about weather, books, and gardens.\n"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, stream, code, message, headers, new_url):
        # Refuse redirects rather than forward credentials to a different host.
        raise urllib.error.HTTPError(request.full_url, code, message, headers, stream)


def open_request(request, timeout):
    handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
    if request.type == "http" and request.has_header("Authorization"):
        # run_predictions permits plaintext authentication only on loopback.
        # Environment proxies would bypass that trust boundary and receive the key.
        handlers.append(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers).open(request, timeout=timeout)


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=True))


def generate_fixtures(tokenizer, context_lengths, depths, samples=3, seed=1729,
                      generation_tokens=32, haystack=DEFAULT_HAYSTACK, random_depths=0):
    """Assemble decoded haystack slices around an intact textual needle.

    Depth is relative to haystack token slices, not the full instruction prompt.
    The final entire prompt is encoded with special tokens. Filler shrinks until
    prompt + requested generation fits; exact filling is not promised. Decode
    must preserve the entire needle; normalizers that change it are rejected.
    """
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError("samples must be positive and seed an integer")
    if type(generation_tokens) is not int or generation_tokens < 1:
        raise ValueError("generation_tokens must be positive")
    if type(random_depths) is not int or random_depths < 0:
        raise ValueError("random_depths must be a nonnegative integer")
    lengths, depths = list(context_lengths), list(depths)
    rng = random.Random(seed)
    depths += [rng.random() for _ in range(random_depths)]
    if not lengths or not depths or len(set(lengths)) != len(lengths) or len(set(depths)) != len(depths):
        raise ValueError("nonempty unique lengths/depths required")
    if any(type(n) is not int or n <= generation_tokens for n in lengths):
        raise ValueError("context length must exceed generation reservation")
    if any(not isinstance(d, (int, float)) or not math.isfinite(d) or not 0 <= d <= 1 for d in depths):
        raise ValueError("depth must be finite in [0, 1]")
    source = list(tokenizer.encode(haystack, add_special_tokens=False))
    if not source:
        raise ValueError("haystack must encode to nonempty tokens")
    rows = []
    for length in lengths:
        for depth in depths:
            for sample in range(samples):
                secret = f"{rng.getrandbits(128):032x}"
                needle = f"The secret retrieval code is {secret}."
                prefix = "Read the archive and retrieve its secret retrieval code.\nArchive:\n"
                suffix = "\nEnd archive.\nReturn only the secret retrieval code, without punctuation or explanation.\nCode:"
                budget = length - generation_tokens
                filler_count = budget
                while True:
                    filler = (source * ((filler_count + len(source) - 1) // len(source)))[:filler_count]
                    split = round(filler_count * depth)
                    before = tokenizer.decode(filler[:split], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
                    after = tokenizer.decode(filler[split:], skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                    prompt = prefix + before + "\n" + needle + "\n" + after + suffix
                    ids = encode(tokenizer, prompt)
                    if len(ids) <= budget:
                        break
                    if filler_count == 0:
                        raise ValueError(f"context {length} cannot hold instructions, needle and generation reservation")
                    filler_count = max(0, filler_count - max(1, len(ids) - budget))
                decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if needle not in prompt or needle not in decoded:
                    raise ValueError("tokenizer round-trip does not preserve the intact needle")
                rows.append({"id": f"{seed}:{length}:{depth}:{sample}", "data_kind": "fixture",
                             "seed": seed, "sample": sample, "context_length": length,
                             "depth": depth, "haystack_tokens": filler_count,
                             "needle_offset_haystack_tokens": split,
                             "actual_haystack_depth": split / filler_count if filler_count else None,
                             "haystack_sha256": hashlib.sha256(haystack.encode()).hexdigest(),
                             "prompt": prompt, "expected": secret, "needle": needle,
                             "prompt_tokens": len(ids), "prompt_token_sha256": token_hash(ids),
                             "generation_tokens": generation_tokens,
                             "tokenizer": getattr(tokenizer, "name_or_path", type(tokenizer).__name__)})
    return rows


def validate_fixture(row, tokenizer):
    for field in ("context_length", "generation_tokens", "prompt_tokens"):
        if type(row.get(field)) is not int or row[field] < 1:
            raise ValueError(f"invalid fixture {field}")
    for field in ("prompt", "needle", "expected", "prompt_token_sha256"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"invalid fixture {field}")
    ids = encode(tokenizer, row["prompt"])
    if len(ids) != row["prompt_tokens"] or token_hash(ids) != row["prompt_token_sha256"]:
        raise ValueError("fixture tokenizer mismatch: prompt token IDs changed")
    if len(ids) + row["generation_tokens"] > row["context_length"]:
        raise ValueError("fixture exceeds complete prompt/generation budget")
    if row["needle"] not in row["prompt"] or row["expected"] not in row["needle"]:
        raise ValueError("fixture needle missing")
    if row["needle"] not in tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        raise ValueError("fixture needle does not survive tokenizer decoding")
    return len(ids)


def run_predictions(fixtures, tokenizer, endpoint, model, timeout=60,
                    api_key_env="OPENAI_API_KEY", client=None):
    """Call a FULL /v1/completions URL, or client.completions.create.

    This is a plain-text completion protocol, not chat templating. Only responses
    with matching reported prompt token usage become scored measurements. A
    hosted OpenAI-compatible client is optional; no client dependency is required.
    Errors are recorded and excluded from accuracy, never replaced with guesses.
    """
    if not model or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("model and positive finite timeout required")
    key = None
    if client is None:
        key = os.environ.get(api_key_env)
        try:
            url = urlparse(endpoint)
            hostname = url.hostname
            url.port  # Validate malformed/out-of-range ports before any request.
        except ValueError:
            raise ValueError("endpoint must be a valid HTTP(S) URL") from None
        if (url.scheme not in ("http", "https") or not hostname
                or any(char.isspace() for char in hostname) or url.username or url.password):
            raise ValueError("endpoint must be an HTTP(S) URL without embedded credentials")
        if key and url.scheme == "http":
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = hostname == "localhost"
            if not loopback:
                raise ValueError("authenticated endpoints require HTTPS except for loopback hosts")
    # Freeze and validate the COMPLETE collection before the first paid request.
    fixtures = list(fixtures)
    measured = []
    seen = set()
    for row in fixtures:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("invalid fixture ID")
        if row["id"] in seen:
            raise ValueError("duplicate fixture ID")
        seen.add(row["id"])
        depth = row.get("depth")
        if type(depth) not in (int, float) or not math.isfinite(depth) or not 0 <= depth <= 1:
            raise ValueError("invalid fixture depth")
        measured.append(validate_fixture(row, tokenizer))
    results = []
    for row, measured_tokens in zip(fixtures, measured):
        result = {**row, "data_kind": "measured", "model": model, "endpoint": endpoint,
                  "status": "error", "prediction": None}
        payload = {"model": model, "prompt": row["prompt"], "max_tokens": row["generation_tokens"],
                   "temperature": 0, "stream": False}
        try:
            if client is not None:
                reply = client.completions.create(**payload, timeout=timeout)
                response = reply.model_dump() if hasattr(reply, "model_dump") else reply
            else:
                headers = {"Content-Type": "application/json"}
                if key:
                    headers["Authorization"] = "Bearer " + key
                request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
                with open_request(request, timeout=timeout) as stream:
                    content = stream.read(MAX_RESPONSE_BYTES + 1)
                    if len(content) > MAX_RESPONSE_BYTES:
                        raise ValueError("HTTP response exceeds byte limit")
                    response = json.loads(content)
            if not isinstance(response, dict) or response.get("error") is not None:
                raise ValueError("endpoint returned error/non-object response")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("endpoint returned missing/malformed choices")
            prediction = choices[0].get("text")
            if not isinstance(prediction, str):
                raise ValueError("endpoint did not return completion text")
            usage = response.get("usage")
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            result["reported_prompt_tokens"] = prompt_tokens
            result["prediction"] = prediction
            if type(prompt_tokens) is not int:
                result["status"] = "tokenization_unverified"
            elif prompt_tokens != measured_tokens:
                result["status"] = "tokenization_mismatch"
            else:
                result["status"] = "ok"
        except urllib.error.HTTPError as exc:
            result["status"] = "unsupported" if exc.code in (400, 404, 413, 422, 501) else "error"
            result["error"] = f"HTTP {exc.code}"  # Do not store possibly sensitive server bodies.
            exc.close()
        except (TimeoutError, socket.timeout):
            result["status"] = "timeout"
            result["error"] = "request timed out"
        except urllib.error.URLError as exc:
            result["status"] = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "error"
            result["error"] = "network request failed"
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            result["error"] = f"invalid response: {type(exc).__name__}"
        except Exception as exc:
            # Hosted client SDK exception types vary; avoid leaking keys or bodies.
            result["status"] = "timeout" if "timeout" in type(exc).__name__.lower() else "error"
            result["error"] = f"client failure: {type(exc).__name__}"
        results.append(result)
    return results


def exact_match(prediction, expected):
    """Whitespace normalization ONLY. Case, punctuation and extra words matter."""
    if not isinstance(prediction, str) or not isinstance(expected, str):
        raise ValueError("exact-match inputs must be strings")
    return " ".join(prediction.split()) == " ".join(expected.split())


def wilson(successes, count):
    if count == 0:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0., center - radius), min(1., center + radius)


def score_predictions(rows):
    """Group cells; only status=ok enters denominator. No mixed provenance/model."""
    if not rows:
        raise ValueError("no rows to score")
    kinds = {row.get("data_kind") for row in rows}
    if len(kinds) != 1 or not kinds <= {"measured", "fixture"}:
        raise ValueError("rows must have one explicit data_kind: measured or fixture")
    if len({row.get("model") for row in rows}) > 1 or len({row.get("endpoint") for row in rows}) > 1:
        raise ValueError("score each model/endpoint separately")
    groups, seen = {}, set()
    statuses = {"ok", "timeout", "unsupported", "error", "tokenization_mismatch", "tokenization_unverified", "fixture"}
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate fixture ID")
        seen.add(row["id"])
        status = row.get("status", "fixture" if row["data_kind"] == "fixture" else None)
        if status not in statuses or (row["data_kind"] == "measured" and status == "fixture"):
            raise ValueError("invalid prediction status")
        length, depth = row["context_length"], row["depth"]
        if type(length) is not int or length < 1 or type(depth) not in (int, float) or not math.isfinite(depth) or not 0 <= depth <= 1:
            raise ValueError("invalid scoring context length/depth")
        key = (length, depth)
        group = groups.setdefault(key, {"context_length": key[0], "depth": key[1],
                                        "total": 0, "count": 0, "successes": 0,
                                        "statuses": {}, "data_kind": row["data_kind"]})
        group["total"] += 1
        group["statuses"][status] = group["statuses"].get(status, 0) + 1
        if status == "ok":
            group["count"] += 1
            group["successes"] += int(exact_match(row["prediction"], row["expected"]))
    for group in groups.values():
        group["accuracy"] = group["successes"] / group["count"] if group["count"] else None
        group["wilson_low"], group["wilson_high"] = wilson(group["successes"], group["count"])
        group["excluded"] = group["total"] - group["count"]
    return [groups[key] for key in sorted(groups)]
