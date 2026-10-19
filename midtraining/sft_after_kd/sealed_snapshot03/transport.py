"""Owned pinned-image Docker peers only; no Docker socket in clients."""
import array
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import threading
import time
import urllib.parse
import uuid
try:
    from .client import require, request_header, validate_descriptor
except ImportError:
    from client import require, request_header, validate_descriptor

IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'

class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout); self.sock.connect('/var/run/docker.sock')

def api(method,path,body=None,raw=False,missing=False):
    connection = UnixHTTP('localhost',timeout=30)
    try:
        connection.request(method,'/v1.52'+path,None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json'})
        response=connection.getresponse(); payload=response.read()
        if missing and response.status == 404: return None
        if response.status >= 300: raise RuntimeError((method,path,response.status,payload[:1500]))
        return payload if raw else (json.loads(payload) if payload else None)
    finally: connection.close()

def proc_record(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'ppid':int(fields[1]),'start_ticks':int(fields[19])}

def descendant(pid,init):
    seen=set()
    for _ in range(128):
        if pid in seen or pid <= 0: return False
        seen.add(pid)
        if pid == init: return True
        pid=proc_record(pid)['ppid']
    return False

class Server:
    def __init__(self,ready,fds):
        self.ready,self.fds=ready,fds; self.stop_event=threading.Event(); self.receipts=[];self.errors=[]
        self.image_id=api('GET','/images/'+urllib.parse.quote(IMAGE,safe='')+'/json')['Id']
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET)
        self.sock.bind(ready['socket_path']);self.sock.listen(64);self.sock.settimeout(.25)
        self.thread=threading.Thread(target=self.run,name='sealed03-peer-server',daemon=True)
    def authenticate(self,peer):
        pid,uid,gid=struct.unpack('3i',peer.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
        require(uid in (1001,0),'peer_uid')
        first=proc_record(pid)
        filters=urllib.parse.quote(json.dumps({'label':['nemochat.sealed03=v1']}),safe='')
        candidates=api('GET','/containers/json?filters='+filters)
        for candidate in candidates:
            if not any(n.lstrip('/').startswith('nemochat-sft-sealed03-') for n in candidate['Names']):continue
            info=api('GET','/containers/'+candidate['Id']+'/json')
            if not (info['State']['Running'] and info['Image']==self.image_id and info['Config']['Image']==IMAGE and info['Config']['Labels'].get('nemochat.sealed03')=='v1'):continue
            init=info['State']['Pid'];init_record=proc_record(init)
            if descendant(pid,init):
                require(proc_record(pid)==first and proc_record(init)==init_record,'peer_pid_reuse')
                again=api('GET','/containers/'+candidate['Id']+'/json')
                require(again['State']['Running'] and again['State']['Pid']==init,'container_lifetime')
                return {'peer_pid':pid,'peer_uid':uid,'peer_gid':gid,'peer_start_ticks':first['start_ticks'],'container_init_pid':init,'container_id':candidate['Id'],'container_name':info['Name'],'image_id':info['Image'],'verified_init_or_descendant':True}
        raise ValueError('peer_not_owned_pinned_container')
    def handle(self,peer):
        peer.settimeout(10)
        receipt=self.authenticate(peer)
        data,ancillary,flags,_=peer.recvmsg(65536,socket.CMSG_SPACE(64),socket.MSG_CMSG_CLOEXEC)
        unexpected=[]
        for level,kind,raw in ancillary:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                ints=array.array('i');ints.frombytes(raw);unexpected.extend(ints)
        for fd in unexpected:os.close(fd)
        require(not ancillary and not flags & (socket.MSG_TRUNC|socket.MSG_CTRUNC),'request_truncation_ancillary')
        request=json.loads(data);file_id=request['file']['file_id']
        require(request==request_header(self.ready,file_id),'request_header_identity')
        fd=self.fds[file_id];validate_descriptor(fd,self.ready['files'][file_id])
        peer.sendmsg([json.dumps(request,sort_keys=True).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[fd]))])
        receipt.update(file_id=file_id,time=time.time());self.receipts.append(receipt)
    def run(self):
        while not self.stop_event.is_set():
            try: peer,_=self.sock.accept()
            except socket.timeout:continue
            except OSError:
                if self.stop_event.is_set():return
                raise
            with peer:
                try:self.handle(peer)
                except Exception as error:
                    self.errors.append({'error':repr(error),'time':time.time()})
                    try:peer.send(json.dumps({'denied':str(error)}).encode())
                    except OSError:pass
    def start(self):self.thread.start()
    def close(self):
        self.stop_event.set();self.thread.join(timeout=12);self.sock.close()
        require(not self.thread.is_alive(),'server_shutdown')
        Path(self.ready['socket_path']).unlink(missing_ok=True)

def demux(data):
    text=b''
    while data:
        require(len(data)>=8,'docker_log_header');n=int.from_bytes(data[4:8],'big')
        require(len(data)>=8+n,'docker_log_size');text+=data[8:8+n];data=data[8+n:]
    return text.decode(errors='replace')

def consume(ready,code,report,timeout=1200,negative='',descendant_probe=False):
    name='nemochat-sft-sealed03-cpu-'+uuid.uuid4().hex[:12]
    labels={'nemochat.sealed03':'v1','nemochat.sealed03.instance':name}
    directory=str(Path(ready['socket_path']).parent)
    command=['python3','/sealed03/consumer.py']
    if descendant_probe:command=['python3','-c','import subprocess,sys;sys.exit(subprocess.call(["python3","/sealed03/consumer.py"]))']
    payload={'Image':IMAGE,'User':'1001:1001','Entrypoint':command,'Cmd':[],'WorkingDir':'/','Env':['SNAPSHOT_READY='+json.dumps(ready,sort_keys=True),'EXPECTED_REJECTION='+negative,'NVIDIA_VISIBLE_DEVICES=void','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1'],'Labels':labels,'NetworkDisabled':True,'HostConfig':{'Binds':[directory+':'+directory+':ro',str(code)+':/sealed03:ro'],'NetworkMode':'none','CapDrop':['ALL'],'SecurityOpt':['no-new-privileges:true'],'ReadonlyRootfs':True,'Memory':16*1024**3,'MemorySwap':16*1024**3,'NanoCpus':4*10**9,'PidsLimit':64,'DeviceRequests':[],'Devices':[],'AutoRemove':False,'Runtime':'runc'}}
    cid=None;report.update(container_name=name,labels=labels,create_attempted=True)
    try:
        cid=api('POST','/containers/create?name='+name,payload)['Id'];report['container_id']=cid
        info=api('GET','/containers/'+cid+'/json');hc=info['HostConfig']
        require(info['Config']['Image']==IMAGE and info['Config']['User']=='1001:1001','container_identity')
        require(hc['Memory']==hc['MemorySwap']==16*1024**3 and hc['NanoCpus']==4*10**9 and hc['PidsLimit']==64,'container_bounds')
        require(hc['NetworkMode']=='none' and hc['ReadonlyRootfs'] and hc['CapDrop']==['ALL'] and hc['SecurityOpt']==['no-new-privileges:true'] and not hc['Devices'] and not hc['DeviceRequests'] and not hc['Privileged'],'sandbox')
        require(len(info['Mounts'])==2 and all(not m['RW'] for m in info['Mounts']) and all(m['Destination'] in (directory,'/sealed03') for m in info['Mounts']),'mounts')
        require(all(info['Config']['Labels'].get(k)==v for k,v in labels.items()),'owned_labels')
        report.update(image_id=info['Image'],sandbox={k:hc[k] for k in ('Memory','MemorySwap','NanoCpus','PidsLimit','NetworkMode','ReadonlyRootfs','CapDrop','SecurityOpt','Devices','DeviceRequests','Runtime')},mounts=info['Mounts'])
        api('POST','/containers/'+cid+'/start'); deadline=time.monotonic()+timeout
        while True:
            state=api('GET','/containers/'+cid+'/json')['State']
            if not state['Running']:break
            if time.monotonic()>=deadline:raise TimeoutError('CPU_consumer_deadline')
            time.sleep(.25)
        report['state']=state;report['logs']=demux(api('GET','/containers/'+cid+'/logs?stdout=1&stderr=1',raw=True))
        require(state['ExitCode']==0 and not state['OOMKilled'],'consumer_exit:'+report['logs'][-3000:])
        report['result']=json.loads(report['logs'].strip().splitlines()[-1])
        require(report['result']['status']==('EXPECTED_REJECTION' if negative else 'PASS'),'consumer_status')
    finally:
        if cid is None:
            info=api('GET','/containers/'+name+'/json',missing=True)
            if info:
                require(all(info['Config']['Labels'].get(k)==v for k,v in labels.items()),'cleanup_ownership');cid=info['Id']
        if cid:
            info=api('GET','/containers/'+cid+'/json',missing=True)
            if info:
                require(all(info['Config']['Labels'].get(k)==v for k,v in labels.items()),'cleanup_ownership')
                api('DELETE','/containers/'+cid+'?force=1')
            require(api('GET','/containers/'+cid+'/json',missing=True) is None,'cleanup_readback')
            report['cleanup_verified_absent']=True
    return report
