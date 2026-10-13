"""Pure stdlib fixtures; runnable with unittest or pytest, no HF/network imports."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.data.climbmix import SOURCE, prepare_data, validate_config


class TinyTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.file = self.root / "fixture.jsonl"
        self.file.write_text("".join(json.dumps({"text": t}) + "\n" for t in ["", "abc", "abc", "test", "한글", "later"]), encoding="utf-8")
        self.config = {"mode": "local_jsonl", "files": [str(self.file)], "max_shards": 1,
                       "token_quota": 13, "split": "train", "purpose": "smoke",
                       "rights_acknowledged": True, "local_license": "CC0-1.0",
                       "local_attribution": "self-authored test fixture", "tokenizer": "gpt2",
                       "reserved_hashes": {"train": [], "test": [hashlib.sha256(b"test").hexdigest()], "validation": []}}
        self.output = self.root / "pack"

    def tearDown(self):
        self.temp.cleanup()

    def run_pack(self):
        return prepare_data(self.config, self.output, TinyTokenizer())

    def test_exact_quota_eos_hashes_offsets_and_dedup(self):
        manifest = self.run_pack()
        counters = manifest["counters"]
        self.assertEqual(counters["accepted_tokens"], 13)
        self.assertEqual(counters["accepted_text_tokens"], 10)
        self.assertEqual(counters["accepted_eos_tokens"], 3)
        self.assertEqual(counters["consumed_training_tokens"], 0)
        self.assertEqual(counters["duplicate"], 1)
        self.assertEqual(counters["reserved"], 1)
        self.assertEqual(counters["empty"], 1)
        self.assertEqual(counters["truncated_records"], 1)
        records = json.loads((self.output / "records.json").read_text())
        self.assertEqual([r["token_offset"] for r in records], [0, 4, 11])
        data = (self.output / "tokens.u32").read_bytes()
        self.assertEqual(len(data), 52)
        self.assertEqual(manifest["tokens_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(struct.unpack("<13I", data)[3], 0)
        for record in records:
            start = record["token_offset"] * 4
            payload = data[start:start + record["token_count"] * 4]
            self.assertEqual(hashlib.sha256(payload).hexdigest(), record["token_content_sha256"])

    def test_prefix_quota(self):
        self.config["token_quota"] = 3
        manifest = self.run_pack()
        self.assertEqual(struct.unpack("<3I", (self.output / "tokens.u32").read_bytes()), (97, 98, 0))
        self.assertEqual(manifest["counters"]["truncated_records"], 1)
        self.assertEqual(manifest["record_hashes"], [hashlib.sha256(b"abc").hexdigest()])

    def test_complete_document_underfill(self):
        self.config.update(token_quota=6, quota_policy="complete")
        manifest = self.run_pack()
        self.assertEqual(manifest["counters"]["accepted_tokens"], 4)
        self.assertFalse(manifest["quota_reached"])
        self.assertGreater(manifest["counters"]["oversized_for_quota"], 0)

    def test_one_token_remainder_stops_before_later_documents_and_files(self):
        self.file.write_text('{"text":"abc"}\nnot json\n')
        missing = self.root / "unvisited.jsonl"
        self.config.update(token_quota=5, quota_policy="complete",
                           files=[str(self.file), str(missing)], max_shards=2)
        tokenizer = TinyTokenizer()
        with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
            manifest = prepare_data(self.config, self.output, tokenizer)
        encode.assert_called_once_with("abc", add_special_tokens=False)
        self.assertFalse(manifest["quota_reached"])
        self.assertEqual(manifest["counters"], {
            "rows_seen": 1, "empty": 0, "duplicate": 0, "reserved": 0,
            "oversized_for_quota": 0, "tokenized_text_tokens": 3,
            "accepted_text_tokens": 3, "accepted_eos_tokens": 1,
            "accepted_tokens": 4, "consumed_training_tokens": 0,
            "truncated_records": 0,
        })
        self.assertEqual(len(manifest["inputs"]), 1)
        self.assertEqual(struct.unpack("<4I", (self.output / "tokens.u32").read_bytes()),
                         (97, 98, 99, 0))

    def test_one_token_remainder_does_not_download_next_shard(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        config = {**self.config, "mode": "climbmix", "source": SOURCE,
                  "shards": ["shard_00000.parquet", "shard_00001.parquet"],
                  "max_shards": 2, "max_download_bytes": 100,
                  "token_quota": 5, "quota_policy": "complete"}

        class Response:
            headers = {"Content-Length": "1"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"x"

        class Parquet:
            schema = SimpleNamespace(names=["text"])

            def __init__(self, path):
                pass

            def iter_batches(self, **kwargs):
                yield SimpleNamespace(column=lambda index: SimpleNamespace(to_pylist=lambda: ["abc"]))
                raise AssertionError("later documents must not be consumed")

        get = Mock(return_value=Response())
        parquet = SimpleNamespace(ParquetFile=Parquet)
        modules = {"requests": SimpleNamespace(get=get),
                   "pyarrow": SimpleNamespace(parquet=parquet), "pyarrow.parquet": parquet}
        with patch.dict(sys.modules, modules):
            manifest = prepare_data(config, self.output, TinyTokenizer())
        get.assert_called_once()
        self.assertTrue(get.call_args.args[0].endswith("/shard_00000.parquet"))
        self.assertFalse(manifest["quota_reached"])
        self.assertEqual(manifest["counters"]["rows_seen"], 1)
        self.assertEqual(manifest["counters"]["accepted_tokens"], 4)
        self.assertEqual(manifest["inputs"][0]["shard"], "shard_00000.parquet")
        self.assertEqual(len(manifest["inputs"]), 1)

    def test_malformed_cleanup_and_retry(self):
        self.file.write_text('{"text":"ok"}\nnot json\n')
        with self.assertRaisesRegex(ValueError, "malformed"):
            self.run_pack()
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["fixture.jsonl"])
        self.file.write_text('{"text":"works"}\n')
        self.run_pack()

    def test_no_overwrite(self):
        self.run_pack()
        before = (self.output / "tokens.u32").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_pack()
        self.assertEqual(before, (self.output / "tokens.u32").read_bytes())

    def test_reserved_manifest_blocks_cross_split(self):
        original = self.run_pack()
        self.config["split"] = "test"
        self.config["reserved_manifests"] = [str(self.output / "manifest.json")]
        second = prepare_data(self.config, self.root / "test", TinyTokenizer())
        self.assertFalse(set(original["record_hashes"]) & set(second["record_hashes"]))
        self.assertGreater(second["counters"]["reserved"], 0)

    def test_fail_closed(self):
        for key, value in [("token_quota", 0), ("token_quota", True), ("max_shards", -1),
                           ("local_license", "MIT"), ("rights_acknowledged", False),
                           ("reserved_hashes", {}), ("tokenizer", "arbitrary"),
                           ("tokenizer_revision", "main"), ("purpose", "training")]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_shard_bounds_validation_rights(self):
        config = {**self.config, "mode": "climbmix", "source": SOURCE, "max_download_bytes": 1000}
        for shard in ["shard_06542.parquet", "shard_06543.parquet", "../shard_00000.parquet", "unknown.parquet"]:
            with self.subTest(shard=shard), self.assertRaises(ValueError):
                validate_config({**config, "shards": [shard]})
        validate_config({**config, "shards": ["shard_00000.parquet"]})
        with self.assertRaises(ValueError):
            validate_config({**config, "source": {**SOURCE, "license": "MIT"}, "shards": ["shard_00000.parquet"]})

    def test_tokenizer_bad_ids_cleanup(self):
        class Bad(TinyTokenizer):
            def encode(self, *args, **kwargs):
                return [-1]
        with self.assertRaises(ValueError):
            prepare_data(self.config, self.output, Bad())
        self.assertFalse(self.output.exists())

    def test_empty_only_and_nonstring_cleanup(self):
        for line in ['{"text":" "}\n', '{"text":12}\n', '{"other":"text"}\n']:
            self.file.write_text(line)
            with self.assertRaises(ValueError):
                self.run_pack()
            self.assertFalse(self.output.exists())

    def test_one_token_quota_cannot_accept_text_plus_eos(self):
        self.config["token_quota"] = 1
        with patch("nemochat.data.climbmix.iter_records") as iterate, \
                patch("nemochat.data.climbmix.load_tokenizer") as load:
            with self.assertRaisesRegex(ValueError, "no accepted"):
                prepare_data(self.config, self.output)
        iterate.assert_not_called()
        load.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["fixture.jsonl"])

    def test_input_and_document_bounds(self):
        for key in ["max_input_bytes", "max_document_bytes"]:
            self.config[key] = 1
            with self.assertRaises(ValueError):
                self.run_pack()
            self.config.pop(key)

    def test_broken_symlink_not_overwritten(self):
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            self.run_pack()
        self.assertTrue(self.output.is_symlink())

    def test_remote_http_and_byte_budget_cleanup(self):
        from types import SimpleNamespace
        config = {**self.config, "mode": "climbmix", "source": SOURCE,
                  "shards": ["shard_00000.parquet"], "max_download_bytes": 3}

        class Response:
            def __init__(self, headers, blocks, failure=False):
                self.headers, self.blocks, self.failure = headers, blocks, failure

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                if self.failure:
                    raise OSError("HTTP 503")

            def iter_content(self, chunk_size):
                return iter(self.blocks)

        for response in [Response({"Content-Length": "4"}, []),
                         Response({}, [b"ab", b"cd"]), Response({}, [], True)]:
            requests = SimpleNamespace(get=lambda *args, **kwargs: response)
            parquet = SimpleNamespace()
            modules = {"requests": requests, "pyarrow": SimpleNamespace(parquet=parquet),
                       "pyarrow.parquet": parquet}
            with self.subTest(headers=response.headers), patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "acquisition failed"):
                    prepare_data(config, self.output, TinyTokenizer())
            self.assertFalse(self.output.exists())
            self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["fixture.jsonl"])

    def test_network_failure_cleans_stage(self):
        # Iterator-level fault injection requires no optional packages.
        def broken(*args):
            yield "shard_00000.parquet", 0, "ok"
            raise ValueError("network failed")
        with patch("nemochat.data.climbmix.iter_records", broken):
            with self.assertRaisesRegex(ValueError, "network failed"):
                self.run_pack()
        self.assertFalse(self.output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["fixture.jsonl"])


if __name__ == "__main__":
    unittest.main()
