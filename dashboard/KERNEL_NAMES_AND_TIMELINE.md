# Raw kernel-name references and recorded iteration timeline

See [한국어](KERNEL_NAMES_AND_TIMELINE.ko.md). Preserve exact symbols as primary identities. Token explanations are separate reference annotations, **not Python caller, shape, runtime algorithm or bottleneck proof**.

## NCCL example

`ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)`

- `ncclDevKernel`: generated device kernel entry point.
- `AllGather`: collective specialization reference: gather rank-ordered inputs on every rank.
- `RING`: logical rank-ring algorithm; neighboring ranks forward data chunks. Not a claim of physical circular wiring.
- `LL`: Low Latency protocol reference. Algorithm and protocol are different choices. NVIDIA describes LL as latency-oriented, Simple as bandwidth-oriented and LL128 as a compromise. None implies this captured run achieved low latency.
- `ncclDevKernelArgsStorage<capacity>` / `4096`: byte capacity of kernel argument storage, not tensor elements, tokens or communicated payload size. The demangled unsigned-long cast describes the C++ template argument, not tensor dtype.

**Important correction:** the pinned NCCL v2.26.2-1 generator maps AllGather/Broadcast variants to RING_LL specialization entries. The device body calls a function table when the actual work function ID differs. Therefore the symbol alone does **not** prove the work executed Ring+LL. Confirm actual library version and work/API/communication evidence before tuning a protocol. This version is a verified source reference, not proof of the capture's installed NCCL version.

Primary references, also linked from decoded UI rows:
- [argument storage](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/include/device.h)
- [generator specialization](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/device/generate.py)
- [runtime dispatch](https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/device/common.h)
- [NCCL environment selection](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/env.html)
- [NVIDIA tuning explanation](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/)

## NVJET example

`nvjet_sm103_tss_128x128_64x9_2x2_2cta_h_badd_NTT`

- `nvjet`: NVIDIA employee confirms an internal kernel backend; not Jetson-specific.
- `sm103`: inferred SM10.3/`sm_103` target convention; not an official explanation of this proprietary token or proof of execution GPU.
- `128x128`: inferred tile candidate, not actual input M/N/K.
- `2cta`: inferred connection to two-CTA cooperation, not two kernel launches, physical cluster size2 or proof of an instruction used.
- `tss`, `64x9`, `2x2`, `h`, `badd`, `NTT`: exact private meanings unverified. Do not fabricate staging counts, transposition order, FP16 dtype, bias/add epilogue or cluster geometry.

The parent independently inspected two real whole-capture launches of this exact symbol: grid128x1x1, block256x1x1, physical cluster4x1x1,255registers/thread,229760bytes dynamic shared memory. These are separate recorded launch observations, **not** decoding rules or measured-update/global performance claims. In particular `2cta` did not establish a physical cluster size2 and `2x2` did not establish physical cluster dimensions2x2.

- [NVIDIA maintainer explanation](https://forums.developer.nvidia.com/t/purpose-of-these-kernels/328273/2)
- [PTX target / CTA-group concepts](https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html)
- [general cuBLAS tiling concepts](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

## Timeline UI

Nsight panel → measured scope → Rank0 → Update3/4. Leave HOST phase at whole update to see one complete selected iteration, or filter by a recorded phase. Device/context/stream kernel lanes, exact CPU-globalTid API lanes and host NVTX phases share a relative time axis. GPU timestamps and CPU API timestamps are separate. Change GPU-start/CPU-enqueue ordering, use100/10/1ms windows and time pan, select a bar or any chronological row for exact ns/ms/correlation metadata. All returned kernels, including sub-microsecond launches, remain in the100-row paginated list. Lane pagination prevents unbounded canvas size. Canvas uses the bundled Korean font and has no remote dependency.

This is a bounded recorded kernel/API/phase viewer, **not** a full Nsight Systems replacement. It omits uncorrelated CPU APIs, memcpy/memset, other processes, OS scheduling, dependencies and exact native Python frames. GPU-start timestamp ordering is not cross-stream dependency ordering; gaps are not proved idle. Minimum1px marks are visibility aids, not duration measurements. Duplicate CPU APIs associated with multiple kernels are drawn once; the table contains kernel events.

## Executed parent verification

- Parent independently regenerated `results/dashboard_timeline/parent-4092da896b57/inspector.json` through the existing read-only CLI,8.885s,1331517bytes; old outputs/capture/source files preserved.
- Actual selected updates3/4:7025/6834events, no truncation. Phase counts3:data8/forward2433/backward3579/optimizer1005;4:data8/forward2362/backward3459/optimizer1005.
- Actual6streams and2CPUthreads; captured API names retained. Importer regenerates fixed-source decoded annotations rather than trusting arbitrary producer explanations/URLs. Events validated against types, safe integers, dictionaries, ordering, counts and scope.
- Parent related232tests passed,32subtests passed; JUnit232testcases reconciled. Node app/timeline syntax, Python compilation and diff check exit0. Raw `verification.log`, `tests.xml`, `execution-receipt.json`, `live-readback.json` in the parent artifact directory.
- Actual HTTP200/SSE-backed UI, canvas,100-row listing, sort/page/select/window/pan, exact raw name/token references and desktop1440/mobile390 no document overflow checked. Screenshot `results/dashboard-iteration-timeline.png`.
- Current snapshot includes cached timelines and measured ~4.05MB HTTP payload. Collection sample ~423ms, cached GET ~44ms; these are host monitoring observations, **not proof of no training slowdown**. SQL is offline once, not per browser/poll. Repeating payload is a known bandwidth cost; lazy timeline transport remains a possible future optimization, not a deployed feature.
- No trainer callback/writer, new GPU job/capture/attach, remote collector, dependency install, commit or push. Current coverage remains an independent rank0 diagnostic, not selected Bridge pilot/all-rank/later-stage captures.
