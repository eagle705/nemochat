"""Genuine small CPU capture, catalog and pinned transport tests."""
import json
import os
import socket
import tempfile
import uuid
from pathlib import Path
from . import holder as h
from midtraining.sft_after_kd.sealed_snapshot03 import transport


def main():
    report={'status':'BLOCKED','GPU_allocated':False};fds={};server=None
    directory=Path('/run/user/1001')/('ncr02fx-'+uuid.uuid4().hex[:10]);directory.mkdir(mode=0o700)
    try:
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as d:
            p=Path(d)/'genuine-short-eof';gold=bytes(range(256))*49+b'short-end'
            p.write_bytes(gold);e={'bytes':len(gold),'sha256':h.old.sha(gold)};capture={}
            fd=h.capture(p,e,capture);fds['fixture']=fd
            report['capture']=capture
            fail={}
            try:h.capture(p,dict(e,sha256='0'*64),fail)
            except ValueError as error:h.require(str(error)=='whole_sha256','unexpected_fixture_rejection')
            else:raise ValueError('bad_expected_digest_accepted')
            h.require(len(fail['views'])==3 and fail['vote_evidence']['counts']['fastpath_chunks']>0,'three_real_views')
            report['bad_expected_digest']=fail
            prior=json.loads(h.PRIOR.read_bytes())
            ready={k:prior[k] for k in ('schema','status','sft_manifest_sha256','c_manifest_sha256')}
            ready.update(snapshot_id=uuid.uuid4().hex,holder_identity=h.old.process_identity(),producer_source_freeze_sha256='fixture',socket_path=str(directory/'s'),files={'fixture':h.old.file_metadata('fixture',e,fd)})
            code=h.CODE/'fixture_code';code.mkdir()
            for name in ('client.py','consumer.py'):(code/name).write_bytes((h.old.CODE/name).read_bytes())
            server=transport.Server(ready,fds);server.start()
            with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as sock:
                sock.settimeout(20);sock.connect(ready['socket_path']);sock.send(json.dumps(h.request_header(ready,'fixture')).encode())
                try:h.require('peer_not_owned_pinned_container' in json.loads(sock.recv(65536))['denied'],'host_auth')
                except ConnectionResetError:h.require(any('peer_not_owned_pinned_container' in x['error'] for x in server.errors),'auth_denial_evidence')
            report['host_rejected']=True
            for label,desc in [('init',False),('descendant',True)]:
                report[label]=transport.consume(ready,code,{},timeout=120,descendant_probe=desc)
            report['peer_receipts']=server.receipts
            report['status']='PASS'
    finally:
        if server:server.close()
        for fd in fds.values():os.close(fd)
        directory.rmdir()
        h.old.save(h.OUT/'FIXTURE.json',report)
    print(json.dumps({'status':report['status'],'receipt':str(h.OUT/'FIXTURE.json')}))

if __name__=='__main__':main()
