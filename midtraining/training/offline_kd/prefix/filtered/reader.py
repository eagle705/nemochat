"""Task-local subclass of frozen native tar dataset; codec and metadata intact."""
import hashlib
from pathlib import Path
from .mapping import INVERSE,PHYSICAL

def physical_url(url):return int(Path(url).stem.split('__')[1])
def select_urls(urls,processed,start_logical):
    result=[]
    for url in urls:
        physical=physical_url(url)
        if not 0<=physical<len(INVERSE):raise ValueError('unknown physical cache iteration')
        logical=INVERSE[physical]
        if url not in processed and logical is not None and logical>=start_logical:result.append(url)
    return sorted(result,key=physical_url)
def remap_decoded(decoded):
    physical,values,indices=decoded
    if not 0<=physical<len(INVERSE) or INVERSE[physical] is None:raise ValueError('excluded/unknown physical cache consumed')
    return INVERSE[physical],values,indices

def install(native,expected_sha256):
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected_sha256:
        raise RuntimeError('native reader source changed')
    original=native.TeacherTarDataset
    class FilteredTeacherTarDataset(original):
        def _discover_shards(self,already_processed,dp_rank):
            prefix=native.batched_tar_prefix(self.cp_rank,dp_rank)
            urls=native.storage_glob_with_caching(self.logprobs_dir,f'{prefix}*.tar',cached=False)
            return select_urls(urls,already_processed,self.start_iteration)
        def _iter_decoded_entries(self,pool,url):
            # Native codec sees original member iteration and original full hash.
            entries=native.iter_logprobs_tar_entries(url,start_iteration=PHYSICAL[self.start_iteration],expected_hash=self._expected_hash)
            decoded=(self._decode_entry(e) for e in entries) if pool is None else self._iter_entries_parallel(pool,entries)
            for item in decoded:
                if item[0]!=physical_url(url):raise ValueError('tar member/physical URL iteration mismatch')
                yield remap_decoded(item)
    native.TeacherTarDataset=FilteredTeacherTarDataset
    return original
