"""Synthetic controls; real pinned 16k integration is separately receipted."""
import hashlib
import json
import numpy as np
from nemochat.data import nemotron_sft as data

LONG = 'nvidia/Nemotron-SFT-Long-Context-Transduction-v1'


def test_explicit_exclusion_skips_even_short_long_context_rows_and_preserves_reservation(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    raw.mkdir()
    group = hashlib.sha256(b'reserved prompt').hexdigest()
    reserved = {group} | {hashlib.sha256(str(i).encode()).hexdigest() for i in range(13)}
    calls = []
    def encode(record, dataset, tokenizer, sequence_length):
        calls.append((dataset, sequence_length))
        norm = data.normalize_record(record, dataset)
        return dict(norm, tokens=[1] * (sequence_length+1), loss_mask=[0]*(sequence_length-1)+[1],
                    nonpadding_tokens=9000, supervised_tokens=1, official_render_sha256='test')
    monkeypatch.setattr(data, 'encode_record', encode)
    original_pins = dict(data.PINS)
    for index, (name, revision) in enumerate(data.PINS.items()):
        dataset = 'nvidia/' + name
        row = dict(uuid=str(index), messages=[dict(role='user', content='reserved prompt'),
                    dict(role='assistant', content=name)], metadata={'train_turns_mask':[False, True]})
        path = raw / (str(index)+'.jsonl')
        path.write_text(json.dumps(row)+'\n')
        receipt = dict(dataset=dataset, revision=revision, key=name, path='data/train.jsonl',
                       raw_file=path.name, rows=1, retained_bytes=path.stat().st_size,
                       purpose='reserved_rl_seed' if 'Seed' in name else 'sft')
        (raw/(str(index)+'.receipt.json')).write_text(json.dumps(receipt))
    report = data.prepare_subset(raw, tmp_path/'data', None, set(), sequence_length=16384,
        excluded_datasets=[LONG], reserved_prompt_groups=reserved,
        reserved_prompt_group_datasets={g:[LONG] for g in reserved if g != group})
    assert report['status'] == 'passed'
    assert report['excluded_datasets'] == [LONG]
    assert report['source_selection_policy']['excluded_datasets'] == [LONG]
    assert len(report['sources']) == 5 and data.PINS == original_pins
    assert len(calls) == 3 and all(d != LONG and n == 16384 for d,n in calls)
    assert report['train_dialogues'] == 0 and report['heldout_dialogues'] == 3
    assert np.load(tmp_path/'data/heldout-tokens.npy').shape == (3,16385)
    assert np.load(tmp_path/'data/heldout-masks.npy').shape == (3,16384)
    assert report['prior_reserved_prompt_groups'] == 14
    assert report['reservation_presence']['physically_present'] == 1
    assert report['reservation_presence']['not_present_in_selected_corpus'] == 13
    assert report['reservation_presence']['inactive_due_excluded_dataset'] == 13
    excluded = report['per_source'][LONG.removeprefix('nvidia/')]
    assert excluded['excluded'] is True and excluded['exclusion_reason'] == 'deferred_long_context_stage'
    assert excluded['train'] == excluded['heldout'] == 0
    assert len(report['active_training_source_keys']) == 3
    assert report['array_shapes']['heldout_tokens'] == [3,16385]


def test_prepare_cli_exposes_explicit_sequence_and_dataset_policy():
    import subprocess, sys
    result = subprocess.run([sys.executable, 'sft/data/nemotron_sft.py', 'prepare', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--sequence-length' in result.stdout
    assert '--exclude-dataset' in result.stdout
