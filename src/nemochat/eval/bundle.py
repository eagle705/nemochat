"""Load the pinned evaluation bundle without extracting untrusted ZIP paths."""

import csv
import hashlib
import io
import json
import urllib.request
import zipfile

BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
BUNDLE_SHA256 = "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
MAX_BUNDLE_BYTES = 30_000_000


def read_core_bundle(path=None, *, labels=None):
    import yaml

    if path is None:
        with urllib.request.urlopen(BUNDLE_URL, timeout=60) as response:
            content = response.read(MAX_BUNDLE_BYTES + 1)
    else:
        with open(path, "rb") as stream:
            content = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(content) > MAX_BUNDLE_BYTES:
        raise ValueError("Evaluation bundle exceeds bounded download size")
    if hashlib.sha256(content).hexdigest() != BUNDLE_SHA256:
        raise ValueError("Evaluation bundle SHA256 does not match pinned version")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        tasks = yaml.safe_load(archive.read("eval_bundle/core.yaml"))["icl_tasks"]
        if labels is not None:
            selected = set(labels)
            if not selected or selected - {task["label"] for task in tasks}:
                raise ValueError("Unknown or empty task selection")
            tasks = [task for task in tasks if task["label"] in selected]
        metadata = archive.read("eval_bundle/eval_meta_data.csv").decode("utf-8")
        baselines = {row["Eval Task"]: float(row["Random baseline"]) / 100
                     for row in csv.DictReader(io.StringIO(metadata))}
        data = {}
        for task in tasks:
            raw = archive.read("eval_bundle/eval_data/" + task["dataset_uri"])
            data[task["label"]] = [json.loads(line) for line in raw.decode("utf-8").splitlines()
                                   if line.strip()]
    return tasks, data, baselines
