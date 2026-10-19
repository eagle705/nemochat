"""Reuse accepted collective preflight; restore explicit extended-state only."""
from midtraining.offline2b.prefix.dynamic.restart01.broker import Broker as OriginalBroker, CacheRejected, durable

class Broker(OriginalBroker):
    def prime(self):
        # Restored broker has already re-read both prefetched source batches.
        if self.calls:
            if len(self.mapping)!=self.calls:raise RuntimeError('restore cursor invalid')
            return
        super().prime()
