"""Fresh generation transport proof, not a repeated full corpus audit."""
import hashlib
import json
import mmap
import os
from client import open_snapshot, require, row_digest

def main():
    ready=json.loads(os.environ['SNAPSHOT_READY']);negative=os.environ.get('EXPECTED_REJECTION')
    if negative:
        try:open_snapshot(ready,[next(iter(ready['files']))])
        except (ValueError,ConnectionResetError) as error:
            print(json.dumps({'status':'EXPECTED_REJECTION','error':repr(error)}));return
        raise ValueError('wrong_snapshot_accepted')
    opened=open_snapshot(ready,list(ready['files']))
    result={'status':'PASS','file_count':len(opened),'original_sft_files':sum(k.startswith('sft:') for k in opened),'files':{},'sample_rows':{},'sample_row_sha256':{}}
    try:
        for key,obj in opened.items():
            size=ready['files'][key]['bytes']
            # Actual reads of both boundary regions of every received descriptor.
            count=min(4096,size)
            first=os.pread(obj.fd,count,0);last=os.pread(obj.fd,count,size-count)
            require(len(first)==len(last)==count,'boundary_short_read')
            with mmap.mmap(obj.fd,0,access=mmap.ACCESS_READ) as view:
                require(view[:count]==first and view[size-count:]==last,'boundary_stream_mmap')
            result['files'][key]={'header_seals_mode_size_validated':True,'boundary_stream_mmap_equal':True,'first_sha256':hashlib.sha256(first).hexdigest(),'last_sha256':hashlib.sha256(last).hexdigest(),'full_original_sha256_rescanned':False}
        for split,indices in [('train',[0,115839,115840,4369279]),('heldout',[0,1,45089])]:
            tokens=opened['sft:'+split+'-tokens.npy'].array;masks=opened['sft:'+split+'-masks.npy'].array;catalog=opened['rowhash:'+split].array
            result['sample_rows'][split]=indices;result['sample_row_sha256'][split]={}
            for index in indices:
                digest=row_digest(index,tokens[index],masks[index])
                require(digest==bytes(catalog[index]),'selected_row_digest')
                require(bool((masks[index]<=1).all()) and bool((tokens[index]<131072).all()),'sample_bounds_mask')
                require(tokens[index,:-1].shape==tokens[index,1:].shape==masks[index].shape,'shift_shapes')
                result['sample_row_sha256'][split][str(index)]=digest.hex()
        print(json.dumps(result,sort_keys=True))
    finally:
        tokens=masks=catalog=None
        for obj in opened.values():obj.close()

if __name__=='__main__':main()
