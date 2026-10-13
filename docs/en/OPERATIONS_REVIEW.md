# Operations Extension Review

Status: local CPU/offline regression gates passed after review repairs. Remote HF round trip, online W&B and GPU stage validation remain pending. The earlier pre-commit approval is historical and applies to its earlier scope.

## Independent findings and parent repairs

- Budget fit/slack previously discarded sunk overruns. Fit now compares projected totals against original limits, preserving negative slack; capacity alone uses clamped remaining availability.
- Peak storage previously ignored an initial checkpoint count exceeding retention. Initial footprint and the next write before pruning are now included. The projection explicitly assumes pruning after completed writes; it is not a bound for delayed/no pruning or simultaneous uploads/copies.
- Optimizer update counts used float division for integer targets. Integer ceiling division now retains exact counts beyond float precision. Wall-clock arithmetic remains floating-point, not arbitrary-precision financial arithmetic.
- Hub SDK excludes `.git` and `.cache/huggingface` namespaces. Packaging now rejects these anywhere in inventory names before committing. Existing read-back fails closed but cannot undo an incomplete remote commit; preflight avoids this known SDK-filter failure.
- Added `tests/test_operations_review.py`: regressions for sunk overruns, initial storage, initial write before pruning, large integer updates and reserved namespaces. Parent targeted run: **90 passed** across operations-review, planning and HF tests.

## Statistical interpretation

A supplied representative-sample file must be **self-weighting** for the unweighted population × mean estimator to apply. Merely describing unequal inclusion probabilities in provenance does not make it valid; weighted samples require a separate design-aware weighted estimator that is not implemented. The reservoir mode scans the complete declared local population and selects a bounded sample; it is bounded tokenization, not sublinear I/O over those selected files. Neither mode establishes random selection of source shards. Reported uncertainty remains conditional on declared populations and documented sampling design.

## W&B and remote boundaries

W&B review confirmed real offline SDK files and exact CPU saved-state equality. Config schema and exception-preservation repairs have landed; parent rerun: **358 passed, 1 skipped, 25 subtests passed**. Actual offline SDK tests are included. Dependency check, compileall, wheel build, local doc links/companions and JSON checks passed. CUDA RNG, multi-rank correctness and automatic checkpoint/run-ID continuity are not established. Key-based secret filtering is not value-based DLP: never pass credential-bearing URLs/paths or arbitrary auth material in metadata.

No HF remote authentication/transfer or W&B online run has been exercised. Native bundle inventory checks do not prove model deserialization or optimizer resume. Synthetic transport fixture bytes are deliberately not valid model state.
