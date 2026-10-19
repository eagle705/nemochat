"""Fixed C-accepted input replay. Any new fault is fatal, never reselected."""
import hashlib
import json
from pathlib import Path

KEYS = ('tokens', 'labels', 'loss_mask', 'position_ids')

def digest(sample):
    h = hashlib.sha256()
    for key in KEYS:
        h.update(sample[key].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def durable(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def validate_mapping(mapping):
    if len(mapping) != 1000 or mapping != sorted(set(mapping)) or mapping[0] != 0 or mapping[-1] != 1090:
        raise ValueError('exact 1000-call C accepted mapping required')

def validate_sample(sample, expected, physical, micro, rank):
    index = physical * 128 + micro * 8 + rank
    if (expected['iteration'], expected['micro'], expected['rank']) != (physical, physical*16+micro, rank):
        raise ValueError('historical input identity disagreement')
    for key in KEYS:
        if tuple(sample[key].shape) != (4096,) or sample[key].device.type != 'cpu':
            raise ValueError('native input shape/device disagreement')
    actual = digest(sample)
    count = int(sample['loss_mask'].sum())
    if actual != expected['sha256'] or count != expected['masked_tokens']:
        raise ValueError(json.dumps(dict(error='SourceInputDigestMismatch', physical=physical,
            micro=micro, rank=rank, source_index=index, expected_sha256=expected['sha256'],
            observed_sha256=actual, expected_masked_tokens=expected['masked_tokens'], observed_masked_tokens=count)))
    return dict(rank=rank, physical_iteration=physical, micro=physical*16+micro,
                source_index=index, sha256=actual, masked_tokens=count)

class Replay:
    def __init__(self, source, mapping, expected, rank, gather, out):
        validate_mapping(mapping)
        if len(source) < 468736:
            raise ValueError('full native sampler budget required')
        self.source, self.mapping, self.expected = source, mapping, expected
        self.rank, self.gather, self.out = rank, gather, Path(out)
        self.calls, self.samples, self.used = 0, None, 0

    def begin(self, logical):
        if logical != self.calls or self.samples is not None or logical >= 1000:
            raise ValueError('native replay cursor disagreement')
        physical = self.mapping[logical]
        samples, proofs = [], []
        try:
            for micro in range(16):
                index = physical*128 + micro*8 + self.rank
                # Own immutable snapshots; never retain mmap document views.
                original = self.source[index]
                sample = {k: (v.clone() if hasattr(v, 'clone') else v) for k,v in original.items()}
                proofs.append(validate_sample(sample, self.expected[physical*16+micro], physical, micro, self.rank))
                samples.append(sample)
            local = dict(rank=self.rank, ok=True, logical=logical, physical=physical)
        except Exception as exc:
            local = dict(rank=self.rank, ok=False, logical=logical, physical=physical,
                         error_type=type(exc).__name__, error=str(exc))
        ranks = self.gather(local)
        if len(ranks) != 8 or any(not r['ok'] or r['logical'] != logical or r['physical'] != physical for r in ranks):
            durable(self.out/f'preforward-failure-rank{self.rank}-logical{logical}.json', dict(ranks=ranks))
            raise RuntimeError('all-rank pre-forward replay input failure; no skip/rebase/retry')
        self.samples, self.used = samples, 0
        with (self.out/f'input-preflight-rank{self.rank}.jsonl').open('a') as f:
            for proof in proofs:
                f.write(json.dumps(dict(proof, logical_iteration=logical))+'\n')
        return physical

    def actual(self, batch, logical, micro):
        physical = self.mapping[logical]
        expected = self.expected[physical*16+micro]
        actual, count = None, None
        try:
            actual = digest(batch)
            count = int(batch['loss_mask'].sum())
            local = dict(rank=self.rank, logical=logical, micro=micro,
                         ok=actual == expected['sha256'] and count == expected['masked_tokens'],
                         expected_sha256=expected['sha256'], observed_sha256=actual)
        except Exception as exc:
            local = dict(rank=self.rank, logical=logical, micro=micro, ok=False,
                         error_type=type(exc).__name__, error=str(exc),
                         expected_sha256=expected['sha256'], observed_sha256=actual)
        ranks = self.gather(local)
        if any(not r['ok'] or r['logical'] != logical or r['micro'] != micro for r in ranks):
            durable(self.out/f'actual-preforward-failure-rank{self.rank}-logical{logical}-micro{micro}.json',dict(ranks=ranks))
            raise RuntimeError('actual tensor mutation/collation mismatch before any rank forward')
        row = dict(rank=self.rank, micro=physical*16+micro, iteration=physical,
                   physical_iteration=physical, logical_iteration=logical,
                   logical_micro=logical*16+micro, source_index=physical*128+micro*8+self.rank,
                   sha256=actual, masked_tokens=count)
        with (self.out/f'student-rank{self.rank}-samples.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        self.used += 1

    def complete(self):
        if self.used != 16:
            raise ValueError('native microbatch count disagreement')
        self.calls += 1
        self.samples = None

    def snapshot(self):
        return dict(completed=self.calls, committed_mapping=self.mapping[:self.calls],
                    next_physical=self.mapping[self.calls] if self.calls < 1000 else None,
                    accepted_limit=1000, replay_only=True)

def binding_for_save(replay, step, manifest_sha256, checkpoint):
    checkpoint = Path(checkpoint)
    if replay.calls != step or replay.samples is not None:
        raise ValueError('save must bind completed replay cursor')
    if not (checkpoint/'.metadata').is_file() or len(list(checkpoint.glob('*.distcp'))) != 8:
        raise ValueError('native synchronous save lacks metadata/all8 rank shards')
    return dict(next_iteration=step, consumed_samples=step*128,
                replay_state=replay.snapshot(), replay_manifest_sha256=manifest_sha256)

class ReplayDataset:
    def __init__(self, replay):
        self.replay = replay
    def __len__(self):
        return len(self.replay.source)
    def __getitem__(self, index):
        logical, offset = divmod(index,128)
        r = self.replay
        if logical != r.calls or offset%8 != r.rank or r.samples is None:
            raise IndexError('wrong native sampler order/rank')
        return r.samples[offset//8]
