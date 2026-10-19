"""Default inert. Only a parent-reviewed exact source pin can start a supervisor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
REPO=Path('/raid/yoon/nemochat');sys.path.insert(0,str(REPO))
from midtraining.offline2b.continue2000.contract import OUT,plan,sha
HERE=Path(__file__).resolve().parent
UNIT='nemochat-offline2b-2000-01.service'

def command(review,review_sha):
    python=str(REPO/'.venv/bin/python');worker=str(HERE/'supervisor.py')
    return ['systemd-run','--user','--unit='+UNIT,'--property=Type=exec',
        '--property=WorkingDirectory='+str(REPO),'--property=RuntimeMaxSec=86400',
        '--property=TimeoutStopSec=90','--property=KillMode=control-group',
        '--property=StandardOutput=append:'+str(OUT/'supervisor.log'),
        '--property=StandardError=append:'+str(OUT/'supervisor.log'),
        '--property=ExecStopPost='+python+' '+worker+' --cleanup',
        python,'-u',worker,'--review',str(review),'--review-sha256',review_sha]

def verify_review(path,expected_sha):
    if path is None or expected_sha is None:raise ValueError('parent source review and exact review SHA required')
    path=Path(path).resolve()
    if sha(path)!=expected_sha:raise ValueError('parent review SHA changed')
    review=json.loads(path.read_text());pins=OUT/'preparation/source-pins.json'
    if review.get('status')!='parent_approved' or review.get('source_pins_sha256')!=sha(pins):
        raise ValueError('parent source review does not bind current frozen source')
    frozen=json.loads(pins.read_text())
    from midtraining.offline2b.continue2000.prepare import OWN_FILES,DEPENDENCIES,SNAPSHOTS
    expected={str((HERE/n).relative_to(REPO)) for n in OWN_FILES}|set(DEPENDENCIES)
    if set(frozen['source_sha256'])!=expected:raise ValueError('exact reviewed source allowlist required')
    snapshot=Path(json.loads((OUT/'preparation/prepared.json').read_text())['snapshot'])
    if set(frozen['snapshot_sha256'])!={str(snapshot/n) for n in SNAPSHOTS}:raise ValueError('exact reviewed snapshot allowlist required')
    for name,digest in frozen['source_sha256'].items():
        if sha(REPO/name)!=digest:raise ValueError('reviewed source hash mismatch: '+name)
    for name,digest in frozen['snapshot_sha256'].items():
        if sha(name)!=digest:raise ValueError('reviewed SDK snapshot changed: '+name)
    for name,digest in frozen['ancestry_sha256'].items():
        if sha(name)!=digest:raise ValueError('reviewed ancestor changed: '+name)
    return review

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--review',type=Path);p.add_argument('--review-sha256');a=p.parse_args()
    cmd=command(a.review or OUT/'parent-source-review.json',a.review_sha256 or 'PARENT_REVIEW_SHA_REQUIRED')
    result=dict(mode='INERT_PREVIEW',gpu_launched=False,plan=plan(),systemd_argv=cmd,
        source_pins_path=str(OUT/'preparation/source-pins.json'),data_ready=False,
        checkpoints_restored=False,optimizer_calls_in_this_generation=0)
    if not a.execute:print(json.dumps(result,indent=2));return
    verify_review(a.review,a.review_sha256)
    if (OUT/'controller-started.json').exists():raise ValueError('no implicit rerun/retry')
    subprocess.run(cmd,check=True)
    print(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState','-p','ExecStart'],text=True))
if __name__=='__main__':main()
