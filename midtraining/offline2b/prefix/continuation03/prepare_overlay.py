"""Build the new read-only cache view only after complete parent codec census.

No shard is copied over, modified, skipped, or synthesized. Corrupt global
iterations are replaced by validated fresh all-rank frozen-teacher artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('/raid/yoon/nemochat')
OUT = REPO/'results/midtraining/offline2b-prefixtrain-01/continuation-03'
SOURCE = REPO/'results/midtraining/offline2b-production1920-01'
sys.path.insert(0, str(HERE))
from repair_contract import build_overlay, repair_plan
from contract import authenticate_shard, segment_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-audit', type=Path, required=True, help='completed parent shards.jsonl')
    args = parser.parse_args()
    audit = [json.loads(line) for line in args.parent_audit.read_text().splitlines()]
    expected = {f'cp0_dp{rank}__{iteration}.tar' for rank in range(8) for iteration in range(1143)}
    if len(audit) != 9144 or {row['name'] for row in audit} != expected:
        raise ValueError('complete exact 9144-shard parent codec audit required')
    request = json.loads((OUT/'repair-request.json').read_text())
    selected = {row['iteration'] for row in repair_plan(request['iterations'])}
    bad = {row['iteration'] for row in audit if not row['ok']}
    if selected != bad or not request['parent_approved']:
        raise ValueError('repair set must exactly equal all corrupt global iterations')
    final = json.loads((OUT/'repair-engine-final.json').read_text())
    if final['state']['Running'] or final['state']['ExitCode'] != 0:
        raise ValueError('exact native sparse repair must have clean exit')
    ranks = [json.loads((OUT/f'repair-rank{rank}-complete.json').read_text()) for rank in range(8)]
    if any(set(row['iterations']) != selected or row['optimizer_calls'] or not row['all_microbatch_digests_match'] for row in ranks):
        raise ValueError('native all-rank frozen repair and sample proof incomplete')
    config = json.loads((SOURCE/'run-config.json').read_text())
    plan = segment_plan(config,1143)
    if any(row['cache_id'] != plan['cache_id'] for row in ranks) or request['cache_id'] != plan['cache_id']:
        raise ValueError('original cache ID required')
    rows = build_overlay(SOURCE/'cache', OUT/'repair-cache', OUT/'cache-overlay', sorted(selected),
        repaired_names={row['name'] for row in audit if not row['ok']})
    for row in rows:
        if row['regenerated']:
            Path(row['source']).chmod(0o444)
    shards = [authenticate_shard(OUT/'cache-overlay'/row['name'], config, int(row['name'].split('__')[1].split('.')[0])) for row in rows]
    manifest = dict(cache_id=plan['cache_id'], coverage=[0,1142], full_teacher_ready=False,
        producer_intentionally_stopped=True,sampler_budget=plan['sampler_budget'],shards=shards,
        source=str(SOURCE),config=config,closed_payload_ownership_authenticated=True,
        overlay=rows, regenerated_iterations=sorted(selected), parent_codec_audit=str(args.parent_audit.resolve()),
        parent_codec_audit_sha256=hashlib.sha256(args.parent_audit.read_bytes()).hexdigest())
    for name, value in [('prefix-manifest',manifest),('run-config',config),('segment-plan',plan)]:
        with (OUT/(name+'.json')).open('x') as stream: json.dump(value,stream,indent=2)
    print(json.dumps(dict(status='overlay_built_waiting_parent_acceptance',shards=len(shards),
        regenerated_iterations=sorted(selected), original_cache_preserved=True,student_started=False)))


if __name__ == '__main__':
    main()
