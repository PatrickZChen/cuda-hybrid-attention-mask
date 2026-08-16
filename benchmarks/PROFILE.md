# Baseline profiling

This report separates profiler observations from interpretations. No kernel
optimization was performed.

## Measured facts

### Nsight Compute availability and limitation

- `ncu` is installed at `/usr/local/cuda-12.5/bin/ncu`, version 2024.2.1.0.
- A single-launch `medium-alternating` pass requested `SpeedOfLight`,
  `LaunchStats`, `Occupancy`, `WarpStateStats`, and `SourceCounters`.
- The NVIDIA driver rejected collection with `ERR_NVGPUCTRPERM`: the user does
  not have permission to access GPU performance counters.
- A second pass requesting only `LaunchStats` failed with the same driver
  error. No competing GPU process was present.
- A non-interactive privileged retry was not possible because `sudo` requires
  a password. Driver settings were not changed.

Consequently, this run has no NCU values for DRAM-throughput percentage,
SM-throughput percentage, achieved occupancy, or branch efficiency. Reporting
numbers for those counters would be fabrication.

Nsight Systems 2024.2.3 was also tried on one `medium-alternating` and one
`large-alternating` launch. It recorded CUDA API activity but exported no CUDA
GPU kernel activity on this WSL2 environment. The installed `nvprof` rejected
the Ada GPU because it does not support compute capability 8.0 or newer. No
binary profiler report is stored in the repository.

### Rate and peak definitions

`Effective output GiB/s` is calculated as
`output_elements × 4 / median_seconds / 2^30`. It describes the logical
float32 output produced per unit time; it is not measured DRAM bandwidth and
does not include all physical memory traffic.

The comparison peak is a nominal value derived from the CUDA-reported 9001 MHz
memory clock and 192-bit bus:
`2 × 9.001×10^9 clocks/s × (192 bits / 8 bits/byte) = 432.048×10^9
bytes/s`. The factor of two is for double-data-rate memory. The result is
432.048 GB/s in decimal units (`10^9` bytes/s), or 402.376 GiB/s in binary
units (`2^30` bytes/s). The reported 97.70% is
`393.141 GiB/s / 402.376 GiB/s`; it is a unit-consistent contextual ratio, not
a measured DRAM-throughput percentage.

### Device-level utilization fallback

As a non-NCU fallback, `nvidia-smi dmon -s pucm -d 1 -c 10` sampled a sustained
`large-alternating` run with 1,000 warmups and 12,000 timed launches. The
benchmark itself reported a 635.904 us median and 393.141 effective output
GiB/s; as defined above, the latter is a logical output rate rather than a
DRAM byte-rate measurement.

After excluding the initial idle sample and final partial sample, all eight
full active one-second samples reported:

| Metric | Observed value |
| --- | ---: |
| `mem` utilization | 100% in every active sample |
| `sm` utilization | 91–94% |
| Board power | 153–174 W |
| SM clock | 2370–2475 MHz |
| Memory clock | 9100 MHz |
| GPU temperature | 72–79 °C |

These are coarse, GPU-wide NVIDIA-SMI samples. In particular, `mem=100%`
means full reported memory activity, not a direct byte-rate counter, and
`sm=91–94%` is not achieved occupancy.

### Launch and compiled-resource characteristics

The launch geometry below is derived exactly from the existing launcher, which
uses 256 threads per block and caps the one-dimensional grid at 65,535 blocks:

| Size | Output elements | Grid blocks | Threads/block | Elements per launched thread |
| --- | ---: | ---: | ---: | ---: |
| Small | 262,144 | 1,024 | 256 | 1 |
| Medium | 8,388,608 | 32,768 | 256 | 1 |
| Large | 67,108,864 | 65,535 | 256 | 4 for most threads, 5 for 1,024 threads |

The launch requests zero dynamic shared memory. `cuobjdump
--dump-resource-usage` reports 29 registers, zero shared memory, zero local
memory, and zero stack for the embedded `sm_52` cubin. The default build also
contains PTX, so the Ada driver JITs code for `sm_89`; the cubin register count
must not be presented as the runtime register count. Runtime theoretical and
achieved occupancy remain unmeasured without NCU access.

For branch behavior, the high-sample baseline has identical small medians
(6.144 us) and identical large medians (635.904 us) across all-full,
all-sliding, and alternating modes. Medium ordering changed on the independent
repeat, so it does not establish a consistent mode advantage. Hardware branch
counters remain unavailable.

## Interpretation and hypotheses

### Large-case bottleneck inference

The strongest evidence-based inference is that dense output stores are likely
the dominant constraint for the large baseline case. The supporting
observations are:

1. The 256 MiB output is over five times the 48 MiB reported L2 size.
2. The logical effective output rate is 393.141 GiB/s, 97.70% of the 402.376
   GiB/s calculated nominal peak when both use binary units.
3. Every full sustained sample reports 100% memory activity.
4. Changing head patterns does not materially change large-case time.

This is a strong inference rather than a directly measured NCU conclusion.
Neither effective output GiB/s nor NVIDIA-SMI `mem` utilization measures the
achieved DRAM byte rate. After GPU-counter permissions are enabled, a short NCU
rerun should test the inference by measuring DRAM throughput and achieved
occupancy before optimizing.

The small case finishes in an approximately 6.1 us CUDA-event interval and
reaches only 158.946 effective output GiB/s, which is consistent with a
fixed-cost regime. The benchmark did not include a control that isolates
launch overhead, so the interval cannot establish launch or dispatch overhead
as the dominant cost. The current evidence also does not support branch
specialization as a high-priority change, and it is insufficient to call
occupancy a bottleneck.

## Ranked Milestone 4 kernel candidates

These experiments preserve the current dense float32 output contract and
should be evaluated one at a time against the recorded correctness suite and
benchmark matrix:

1. Reduce repeated 64-bit division, modulo, and indexing work. The flat path
   reconstructs key, query, head, and batch coordinates for every output
   element. Map work by output rows or tiles, or advance coordinates
   incrementally, so key iteration stays coalesced while row-level coordinates
   are computed once and reused.
2. Reuse row-level values such as head mode, absolute query position, and
   sliding-window bounds rather than recomputing or reloading them for each
   key. Measure this separately from the indexing change where practical.
3. Test store-oriented row/tile shapes, modest unrolling, and vectorized stores
   while retaining aligned, coalesced writes and exact `0`/`-inf` semantics.
4. Revisit block size, grid shape, and mode specialization after NCU counters
   are available. Current measurements do not identify occupancy or branching
   as primary constraints, so these rank below the code-visible indexing work.

## Architectural alternatives

These options change the produced representation or its consumer and therefore
are not kernel-level optimizations of the benchmarked contract:

- Eliminate mask materialization by fusing the mask predicate into the
  attention consumer.
- Use a packed representation that reduces mask storage and traffic, with the
  corresponding unpacking or direct-consumption support downstream.

Either alternative requires a new interface, correctness contract, and
end-to-end benchmark; its result should not be compared as if it were a faster
implementation of the current dense float32 materialization benchmark.
