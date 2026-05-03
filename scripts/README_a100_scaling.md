# A100 Scaling Tests

This benchmark plan is for Grace GPU nodes with one NVIDIA A100. It uses the simulator's `--stats` mode so runs produce daily aggregate CSV rows instead of full replay frames.

## Build

The runner compiles directly with `nvcc` for A100:

```bash
module load CUDA/12.0
bash scripts/run_a100_scaling.sh
```

The compile target is `build_a100/sim_a100`. By default the script tries `icc`, then `icx`, then `gcc` as the `nvcc -ccbin` host compiler. Override with:

```bash
HOST_COMPILER=/path/to/icc bash scripts/run_a100_scaling.sh
```

For CUDA debugging builds:

```bash
DEBUG=1 SUITES=capacity CAPACITY_CASES="10000" REPEATS=1 bash scripts/run_a100_scaling.sh
```

## Submit

```bash
sbatch scripts/grace_a100_scaling.sbatch
```

Useful overrides:

```bash
SUITES=capacity REPEATS=5 sbatch scripts/grace_a100_scaling.sbatch
SUITES="grid radius" TICKS=192 sbatch scripts/grace_a100_scaling.sbatch
SUITES=multiday MULTIDAY_CAPACITY=80000 MULTIDAY_TICKS=960 sbatch scripts/grace_a100_scaling.sbatch
```

Results are written under `experiments/a100_scaling/<timestamp>/`.

## Test Suites

`capacity`: one simulated day at increasing capacity. Defaults:

```text
10000 20000 40000 80000 120000 160000 200000
```

This is the main weak-scaling curve. Initial populations are fixed fractions of capacity: bluegill 20%, minnow 25%, bass 1%, food 33%, cover 5%.

`grid`: fixed 100,000 capacity while sweeping `--spatial-grid`:

```text
24 32 48 64 96 128
```

This isolates spatial partition overhead versus nearby-cell candidate count.

`radius`: fixed 100,000 capacity and default grid while sweeping detection radius:

```text
0.12 0.20 0.28 0.35
```

This shows how neighborhood density affects the optimized search path.

`multiday`: a longer dynamics/stability run. Default is 50,000 capacity for 960 ticks, which is 10 simulated days at 96 ticks/day.

## Outputs

`results.csv` has one row per run with:

```text
suite, case_id, repeat, seed, capacity, initial counts, grid/radius settings,
program_ms, kernel timing, wall_seconds, final daily counts, status
```

Each run also keeps its raw daily stats CSV, simulator summary, and `/usr/bin/time` output. `metadata.txt` records host, CUDA, `nvidia-smi`, and run configuration.

For report plots, use `results.csv` to compute mean and standard deviation over repeats for `wall_seconds`, `program_ms`, and `kernel_total_ms`.
