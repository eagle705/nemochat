"""All-rank pre-forward selection. No model, optimizer or schedule operations."""
import hashlib
import json
import os

class CacheRejected(Exception):
    def __init__(self, observation):
        self.observation = observation
        super().__init__(str(observation))


def durable(path, value):
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

class Broker:
    def __init__(self, rank, reader, gather, out, limit=1143, historical=(20,86,98,121,122,126)):
        self.rank=rank; self.reader=reader; self.gather=gather; self.out=out
        self.limit=limit; self.historical=set(historical)
        self.scan=0; self.mapping=[]; self.exclusions=[]
        self.current=None; self.lookahead=None; self.calls=0; self.consumed=False

    def _select(self):
        while self.scan<self.limit:
            physical=self.scan; self.scan+=1
            data=None
            local=dict(rank=self.rank, source_iteration=physical, status='valid')
            if physical in self.historical:
                local.update(status='historical_quarantine',error='historically observed rejection; no reread')
                local.update(getattr(self,'historical_details',{}).get((self.rank,physical),{}))
            else:
                try:data=self.reader(physical)
                except CacheRejected as exc:local.update(status='quarantine',**exc.observation)
                except Exception as exc:local.update(status='fatal',error_type=type(exc).__name__,error=str(exc))
            ranks=self.gather(local)
            if (len(ranks)!=8 or [r['rank'] for r in ranks]!=list(range(8))
                or any(r['source_iteration']!=physical for r in ranks)):
                raise RuntimeError('allrank source agreement failed')
            if any(r['status']=='fatal' for r in ranks):
                raise RuntimeError(f'nonquarantinable preflight failure: {ranks}')
            if any(r['status']!='valid' for r in ranks):
                self.exclusions.append(dict(source_iteration=physical,ranks=ranks,
                    optimizer_calls=self.calls, no_forward=True, no_optimizer_or_lr_advance=True))
                if self.rank==0:durable(self.out/'exclusion-ledger.json',self.exclusions)
                # Do not release any forward until durable rank-zero commit completes.
                ack=self.gather(dict(rank=self.rank,source_iteration=physical,status='ledger_committed'))
                if any(r['status']!='ledger_committed' for r in ack):raise RuntimeError('ledger commit failed')
                continue
            return (physical,data)
        return None

    def prime(self):
        if self.current is not None or self.mapping:raise RuntimeError('broker already primed')
        self.current=self._select()
        self.lookahead=self._select() if self.current is not None else None

    def begin(self, logical):
        if logical!=self.calls or self.current is None or len(self.mapping)!=self.calls:
            raise RuntimeError('logical/native optimizer cursor mismatch')
        self.mapping.append(self.current[0]);self.consumed=False
        return self.current[0]

    def take(self):
        if self.current is None or self.consumed or len(self.mapping)!=self.calls+1:
            raise RuntimeError('native cached reader requested unbound batch')
        self.consumed=True
        return self.current[1]

    def complete(self):
        if not self.consumed:raise RuntimeError('optimizer completed without native cache consumption')
        self.calls+=1
        self.current=self.lookahead
        self.lookahead=None
        if self.current is not None:self.lookahead=self._select()

    @property
    def is_last(self):
        return self.current is not None and self.lookahead is None

    def restore(self, state):
        expected=dict(state); digest=expected.pop('mapping_sha256')
        if hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()!=digest:
            raise ValueError('dynamic checkpoint mapping identity mismatch')
        mapping=state['committed_mapping']; calls=state['optimizer_calls']
        if (state['physical_limit']!=self.limit or len(mapping)!=calls
            or mapping!=sorted(set(mapping))
            or any(p in state['quarantined_physical'] for p in mapping)
            or (mapping and state['next_unconsumed_physical']<=mapping[-1])):
            raise ValueError('dynamic checkpoint cursor inconsistent')
        self.calls=calls; self.mapping=list(mapping);self.exclusions=list(state['exclusions'])
        self.historical.update(state['quarantined_physical'])
        self.scan=state['next_unconsumed_physical']
        self.current=self._select()
        self.lookahead=self._select() if self.current is not None else None

    def snapshot(self):
        value=dict(optimizer_calls=self.calls,committed_mapping=self.mapping[:self.calls],
            next_unconsumed_physical=self.current[0] if self.current is not None else self.limit,
            prefetched_unconsumed=[x[0] for x in (self.current,self.lookahead) if x is not None],
            scanned_cursor=self.scan,quarantined_physical=[r['source_iteration'] for r in self.exclusions],
            exclusions=self.exclusions,physical_limit=self.limit)
        value['mapping_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return value
