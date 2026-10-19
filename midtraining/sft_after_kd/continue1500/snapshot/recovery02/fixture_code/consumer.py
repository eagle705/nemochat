"""Finite CPU-only full stream + mmap verification of received descriptors."""
import hashlib
import json
import mmap
import os
import sys
from client import CHUNK, open_snapshot, digest_fd, require

def main():
    ready=json.loads(os.environ['SNAPSHOT_READY']); expected=os.environ.get('EXPECTED_REJECTION','')
    opened={};result={'status':'PASS','snapshot_id':ready['snapshot_id'],'files':{},'uid':os.getuid(),'pid_namespace_pid':os.getpid()}
    try:
        opened=open_snapshot(ready,list(ready['files']))
        for file_id,obj in opened.items():
            meta=ready['files'][file_id]
            stream=digest_fd(obj.fd,meta['bytes']);require(stream==meta['sha256'],'stream_sha256')
            with mmap.mmap(obj.fd,0,access=mmap.ACCESS_READ) as mm:
                h=hashlib.sha256()
                for offset in range(0,meta['bytes'],CHUNK):h.update(mm[offset:offset+CHUNK])
                mapped=h.hexdigest();require(mapped==meta['sha256'],'mmap_sha256')
                require(memoryview(mm).readonly,'mmap_readonly')
                try:mm[0]=0
                except (ValueError,TypeError):pass
                else:raise ValueError('mmap_mutation_allowed')
            denied={}
            for label,action in [('write',lambda:os.pwrite(obj.fd,b'x',0)),('grow',lambda:os.ftruncate(obj.fd,meta['bytes']+1)),('shrink',lambda:os.ftruncate(obj.fd,0))]:
                try:action()
                except OSError as error:denied[label]=error.errno
                else:raise ValueError('fd_mutation_allowed')
            if obj.array is not None:require(not obj.array.flags.writeable,'array_writeable')
            result['files'][file_id]={'bytes':meta['bytes'],'stream_sha256':stream,'mmap_sha256':mapped,'readonly':True,'seals':15,'mutation_denials':denied}
        require(not expected,'negative_unexpected_pass')
        result['file_count']=len(result['files'])
        print(json.dumps(result,sort_keys=True),flush=True)
    except ValueError as error:
        if str(error)!=expected:raise
        print(json.dumps({'status':'EXPECTED_REJECTION','reason':str(error)}),flush=True)
    finally:
        for obj in opened.values():obj.close()

if __name__=='__main__':main()
