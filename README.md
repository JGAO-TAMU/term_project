# CUDA Pond Artificial Life Prototype

This branch simulates three species plus a limited food resource:

- `food`: limited particles that bluegill and minnows can detect, move toward, and eat for energy. Consumed food can respawn later at a new random position based on a simple CPU-side respawn chance. Initial food placement can use `grid`, `random`, `cluster`, `ring`, `dense-circle`, or `dense-rect` patterns.
- `cover`: static habitat patches with per-patch intensity. Cover is streamed to the visualizer as its own heat layer, and prey inside stronger cover zones are harder for bass to catch. Initial cover placement can use `grid`, `random`, `cluster`, `ring`, `dense-circle`, or `dense-rect` patterns.
- `predation`: bass predation is now resolved with a simple abundance-based saturating formula. Predation pressure rises with predator and prey abundance, levels off through a half-saturation term, and is reduced by prey cover through one refuge multiplier with a floor so prey never become completely untouchable.
- `bluegill`: prey. Bluegill lose a small maintenance cost, seek nearby food for energy, seek eligible mates when they have enough energy, and reproduce into available agent capacity. Reproduction now happens in simple species-level pulses spaced in simulated days rather than effectively continuous spawning.
- `minnow`: smaller prey. Minnows eat the same food as bluegill, have lower energy drain for higher vitality, start smaller, and use the same pulse framework with a shorter interval and larger litters.
- `bass`: predator. Bass seek nearby bluegill or minnows when hungry, stop eating when full, and gain energy when they catch prey. Predation success can be set per prey species with `--bluegill-predation-success` and `--minnow-predation-success`, with minnows easier to catch by default. Static cover reduces effective catch success depending on local cover intensity, so the default predation success was nudged upward to compensate. Bass use the same pulse-spawning framework with slower pulses and smaller effective reproduction.
- `crowding`: reproduction now includes a simple soft crowding penalty. Each species gets a soft cap, and the multiplier `max(0, 1 - population / soft_cap)` reduces pulse reproduction success and juvenile survival as that species becomes dense.
- `size`: fish grow when energy is abundant and shrink when energy is scarce, within species-specific size ranges. Starting ranges are bass `0.8-10.0`, bluegill `0.1-0.5`, and minnow `0.05-0.1`. Larger prey gives bass more energy when eaten, which makes size a useful hook for future traits.
- `death_energy`: each fish has a small random death threshold. A small configurable share of bass can receive a negative threshold, letting them survive a brief food shortage without forcing population rescue.

Simulation time is now interpreted as ticks and days. `--ticks` is the number of ticks to simulate, `--ticks-per-day` defines how many ticks make up one simulated day, and the visualizer shows the current day directly.

The agent and food data are still stored in Structure of Arrays form on the GPU. The CUDA kernel handles per-tick micro behavior such as movement, wandering, detection, chasing, predation targeting, and prey food consumption. Agent neighbor searches use a species-partitioned spatial grid, while food searches use a separate food grid; `--spatial-grid` controls the grid resolution. CPU-side setup, runtime options, memory ownership, and simulation flow are wrapped in a small `SimulationApp` class, while CPU cleanup handles food respawning, predation resolution, and reproduction events to keep the model easy to debug.

Each run also writes a `sim_report.txt` file in the project directory with program timing, aggregate kernel timing, final simulation counts, and a small block of debug/config values.

## Source Layout

- `main.cu`: small entry point.
- `simulation_app.h/.cu`: command-line options, memory ownership, CUDA transfers, and simulation loop.
- `kernels.cuh/.cu`: CUDA kernel and device helpers.
- `cpu_rules.h/.cpp`: CPU-side ecology rules such as food respawn, predation, and reproduction.
- `output.h/.cpp`: ASCII debug output and stream output for Python visualization.
- `sim_types.h`, `sim_math.h`, `cuda_utils.h`: shared types and small helpers.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

If `cmake` is not on your `PATH`, use `C:\Program Files\CMake\bin\cmake.exe` instead.

## Run

```powershell
.\build\Release\sim.exe
```

Stream data for Python visualization:

```powershell
.\build\Release\sim.exe --stream --ticks 50
```

Daily aggregate stats mode for long runs:

```powershell
.\build\Release\sim.exe --stats --ticks 5760
```

At the end of each run, the simulator prints only a short timing summary with total program time, aggregate kernel time, average kernel launch time, max kernel launch time, final day/tick, and final species counts. The full per-tick ASCII/debug trace is written to `sim_report.txt` instead of being dumped to the terminal. In `--stream` mode this summary is written to `stderr` so the CSV-style stream on `stdout` stays clean for `vis.py`.

The same run also writes `sim_report.txt` in the project folder. That report includes:

- program timing
- kernel timing totals / average / max / launch count
- final tick and simulated day
- initial and final species counts
- active food count and total cover count
- debug/config values such as detection radius, predation radius, predation half-saturation, food respawn chance, food gain, cover radius, cover protection, refuge floor, predation success, species soft caps, and reproduction pulse intervals

Useful runtime options include `--ticks`, `--ticks-per-day`, `--agent-capacity`, `--bluegill`, `--minnow`, `--bass`, `--food`, `--food-pattern`, `--cover`, `--cover-pattern`, `--bluegill-gain`, `--bluegill-drain`, `--bluegill-max-energy`, `--bluegill-soft-cap`, `--bluegill-refuge-scale`, `--minnow-gain`, `--minnow-drain`, `--minnow-max-energy`, `--minnow-soft-cap`, `--minnow-refuge-scale`, `--minnow-reproduce-energy`, `--minnow-reproduce-cost`, `--minnow-reproduce-cooldown-days`, `--minnow-child-energy`, `--minnow-max-children`, `--bass-drain`, `--bass-gain`, `--bass-max-energy`, `--bass-soft-cap`, `--death-energy-variation`, `--bass-starvation-resistant`, `--bass-starvation-reserve`, `--bass-reproduce-energy`, `--bass-reproduce-cost`, `--bass-reproduce-cooldown-days`, `--bass-child-energy`, `--bass-max-children`, `--detection-radius`, `--predation-radius`, `--predation-success`, `--bluegill-predation-success`, `--minnow-predation-success`, `--predation-half-sat`, `--move-step`, `--reproduce-energy`, `--reproduce-cost`, `--reproduce-cooldown-days`, `--mate-radius`, `--child-energy`, `--reproduce-chance`, `--max-children`, `--food-gain`, `--food-eat-radius`, `--food-respawn-chance`, `--cover-radius`, `--cover-protection`, `--min-refuge-multiplier`, `--size-growth-threshold`, `--size-shrink-threshold`, `--size-growth-rate`, `--size-shrink-rate`, `--bluegill-min-size`, `--bluegill-max-size`, `--minnow-min-size`, `--minnow-max-size`, `--bass-min-size`, `--bass-max-size`, and `--stream`. Pattern values are `grid`, `random`, `cluster`, `ring`, `dense-circle`, and `dense-rect`.

Example with tighter food supply and mixed prey:

```powershell
.\build\Release\sim.exe --ticks 120 --agent-capacity 320 --bluegill 45 --minnow 60 --bass 10 --food 240 --cover 90 --food-gain 0.24 --food-respawn-chance 6 --cover-radius 0.07 --cover-protection 0.60 --reproduce-energy 1.35 --reproduce-cost 0.80 --mate-radius 0.08 --max-children 5 --minnow-drain 0.006 --minnow-reproduce-energy 1.20 --minnow-reproduce-cost 0.55 --minnow-child-energy 0.70 --minnow-max-children 9 --bass-reproduce-energy 1.95 --bass-reproduce-cost 1.55 --bass-max-children 3 --bluegill-predation-success 45 --minnow-predation-success 70 --bass-gain 1.0 --bluegill-min-size 0.10 --bluegill-max-size 0.50 --minnow-min-size 0.05 --minnow-max-size 0.10 --bass-min-size 0.80 --bass-max-size 10.0 --size-growth-threshold 1.30 --size-shrink-threshold 0.70 --size-growth-rate 0.015 --size-shrink-rate 0.020
```

Big run:
```
& C:\Users\jobog\AppData\Local\Programs\Python\Python310\python.exe "c:/Users/jobog/Documents/Homework/ECEN 489/term_project/vis.py" --ticks 400 --vis-skip 20 --vis-pause 0.03 --vis-autoplay --agent-capacity 15000 --bluegill 1200 --minnow 1700 --bass 80 --food 3500 --food-respawn-chance 5 --food-gain 0.22 --food-eat-radius 0.032 --bluegill-gain 0.0 --bluegill-drain 0.006 --bluegill-max-energy 2.0 --reproduce-energy 1.30 --reproduce-cost 0.80 --max-children 5 --minnow-gain 0.0 --minnow-drain 0.0065 --minnow-max-energy 1.75 --minnow-reproduce-energy 1.25 --minnow-reproduce-cost 0.70 --minnow-child-energy 0.65 --minnow-max-children 7 --detection-radius 0.28 --mate-radius 0.075 --predation-radius 0.024 --bluegill-predation-success 40 --minnow-predation-success 70 --bass-reproduce-energy 2.0 --bass-reproduce-cost 1.70 --bass-max-children 2 --bass-gain 1.0 --bass-drain 0.013 --bass-max-energy 2.0 --size-growth-threshold 1.30 --size-shrink-threshold 0.70 --size-growth-rate 0.015 --size-shrink-rate 0.020 --bluegill-min-size 0.10 --bluegill-max-size 0.50 --minnow-min-size 0.05 --minnow-max-size 0.10 --bass-min-size 0.80 --bass-max-size 10.0
```
## Python Visualization

```powershell
python .\vis.py
```

`vis.py` launches `sim --stream`, loads the streamed ticks into memory, and opens a Matplotlib replay viewer. Bluegill are blue circles, minnows are small cyan circles, bass are red triangles, newly spawned prey are stars for one frame, newly spawned bass are bright red triangles for one frame, active food is shown as a green heat map layer, and static cover is shown as a blue heat map layer. Fish marker size scales with the simulation `size` value. The viewer now tracks simulated day directly and plots history against days rather than raw ticks.

For longer headless or near-headless runs, `vis.py --vis-stats` switches to the simulator's daily aggregate stats mode and shows a simple population-over-time graph instead of loading per-agent replay frames.

The replay viewer includes play/pause, previous/next frame buttons, a restart button, a frame slider, and a delay slider for playback speed. Keyboard shortcuts are also available: Space toggles play/pause, Left/Right move backward/forward one tick, Home jumps to the start, and End jumps to the final frame.

If you want to archive a run, `vis.py` can also save the streamed replay rows into a timestamped CSV inside a `replay` folder.

Stats plots can be saved without opening the replay window:

```powershell
python .\vis.py --vis-stats --vis-save-plot experiments\stats.png --vis-no-show --ticks 5760
```

Any extra arguments passed to `vis.py` are forwarded to the simulation:

```powershell
python .\vis.py --ticks 50 --vis-skip 2
```

Save a replay CSV while loading the viewer:

```powershell
python .\vis.py --ticks 200 --vis-skip 4 --vis-save-replay
```

Use the lightweight stats plot for long runs:

```powershell
python .\vis.py --vis-stats --ticks 5760
```

Visualizer-only options include `--vis-skip N` to store every Nth simulation tick, `--vis-pause X` to set the initial playback delay, `--vis-autoplay` to start replay automatically after loading frames, `--vis-stats` to use daily aggregate plotting mode, `--vis-save-replay` to save a replay CSV, and `--vis-replay-dir PATH` to choose the replay output folder.
