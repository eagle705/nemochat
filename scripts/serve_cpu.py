"""Loopback-only real HF CPU completion server for NIAH integration smoke.

Not a production server, not vLLM, and not evidence of long-context capability.
The small control model is expected to fail retrieval; no answers are fabricated.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


REQUEST_TIMEOUT_SECONDS = 5.0  # Socket inactivity limit, not a generation deadline.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--revision", default="5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                             trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                trust_remote_code=False, weights_only=True).eval()
    limit = model.config.max_position_embeddings

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(REQUEST_TIMEOUT_SECONDS)
            super().setup()

        def handle(self):
            try:
                super().handle()
            except (TimeoutError, ConnectionError):
                # A timed-out/disconnected peer must not escape to server traceback
                # handling or hold up the next request on this serial smoke server.
                self.close_connection = True

        def reply(self, status, body):
            content = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "model": args.model,
                                 "kind": "CPU smoke control", "context_limit": limit})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/v1/completions":
                self.reply(404, {"error": "Not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 512_000:
                    raise ValueError("Request size out of bounds")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Request must be a JSON object")
                if payload.get("model") != args.model or not isinstance(payload.get("prompt"), str):
                    raise ValueError("Model/prompt mismatch")
                count = payload.get("max_tokens", 32)
                if type(count) is not int or not 1 <= count <= 64:
                    raise ValueError("max_tokens must be in [1, 64]")
                if payload.get("stream", False) or payload.get("temperature", 0) != 0:
                    raise ValueError("Smoke server supports only nonstreaming greedy decoding")
                inputs = tokenizer(payload["prompt"], return_tensors="pt", add_special_tokens=True)
                prompt_tokens = inputs["input_ids"].shape[1]
                if prompt_tokens < 1 or prompt_tokens + count > limit:
                    raise ValueError("Context capacity exceeded")
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=count,
                                            pad_token_id=tokenizer.eos_token_id)
                new_ids = output[0, prompt_tokens:]
                text = tokenizer.decode(new_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                self.reply(200, {"model": args.model, "choices": [{"text": text}],
                                 "usage": {"prompt_tokens": prompt_tokens,
                                           "completion_tokens": len(new_ids),
                                           "total_tokens": prompt_tokens + len(new_ids)}})
            except TimeoutError:
                self.close_connection = True
                self.reply(408, {"error": "Request timed out"})
            except (ValueError, KeyError, TypeError):
                self.reply(400, {"error": "Invalid request or unsupported context"})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"CPU control ready at http://127.0.0.1:{args.port}/health", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
