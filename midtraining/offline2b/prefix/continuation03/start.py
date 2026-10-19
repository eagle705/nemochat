"""Freeze and launch continuation03 under a user-systemd owner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = Path('/raid/yoon/nemochat')
OUT = REPO/'results/midtraining/offline2b-prefixtrain-01/continuation-03'
sys.path.insert(0, str(REPO/'midtraining/offline2b'))
from launch import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['repair', 'student'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); stage = args.stage
    unit = f'nemochat-offline2b-prefixtrain-01-continuation03-{stage}'+('02' if stage == 'repair' else '')
    command = ['systemd-run', '--user', '--unit='+unit, '--property=Type=exec',
        '--property=WorkingDirectory='+str(REPO), '--property=RuntimeMaxSec=43800',
        '--property=StandardOutput=append:'+str(OUT/f'{stage}-supervisor.log'),
        '--property=StandardError=append:'+str(OUT/f'{stage}-supervisor.log'),
        '--property=ExecStopPost='+str(REPO/'.venv/bin/python')+' '+str(HERE/'supervisor.py')+' '+stage+' --cleanup',
        str(REPO/'.venv/bin/python'), '-u', str(HERE/'supervisor.py'), stage]
    expected = dict(prefix={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}, original=freeze())
    print(json.dumps(dict(command=command, source_freeze=expected), indent=2))
    if args.execute:
        with (OUT/f'{stage}-source-freeze.json').open('x') as stream: json.dump(expected, stream, indent=2)
        subprocess.run(command, check=True)
        print(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState','-p','ExecStart'],text=True))


if __name__ == '__main__':
    main()
