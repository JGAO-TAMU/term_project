# CUDA Pond Artificial Life Prototype

This branch simulates three species plus a limited food resource:

- `food`: limited particles that bluegill and minnows can detect, move toward, and eat for energy. Consumed food can respawn later at a new random position based on a simple CPU-side respawn chance.
- `bluegill`: prey. Bluegill lose a small maintenance cost, seek nearby food for energy, seek eligible mates when they have enough energy, and reproduce into available agent capacity. Reproduction now uses cooldowns measured in simulated days rather than effectively continuous spawning.
- `minnow`: smaller prey. Minnows eat the same food as bluegill, have lower energy drain for higher vitality, start smaller, and use their own day-based reproduction cooldown and litter settings.
- `bass`: predator. Bass seek nearby bluegill or minnows when hungry, stop eating when full, and gain energy when they catch prey. Predation success can be set per prey species with `--bluegill-predation-success` and `--minnow-predation-success`, with minnows easier to catch by default. Bass also use the same tick/day reproduction architecture with their own slower cooldown and smaller litters.
- `size`: fish grow when energy is abundant and shrink when energy is scarce, within species-specific size ranges. Starting ranges are bass `0.8-10.0`, bluegill `0.1-0.5`, and minnow `0.05-0.1`. Larger prey gives bass more energy when eaten, which makes size a useful hook for future traits.

Simulation time is now interpreted as ticks and days. `--ticks` is the number of ticks to simulate, `--ticks-per-day` defines how many ticks make up one simulated day, and the visualizer shows the current day directly.

The agent and food data are still stored in Structure of Arrays form on the GPU. The CUDA kernel handles per-tick micro behavior such as movement, wandering, detection, chasing, predation targeting, and prey food consumption. CPU-side setup, runtime options, memory ownership, and simulation flow are wrapped in a small `SimulationApp` class, while CPU cleanup handles food respawning, predation resolution, and reproduction events to keep the model easy to debug.

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

At the end of each run, the simulator prints only a short timing summary with total program time, aggregate kernel time, average kernel launch time, max kernel launch time, final day/tick, and final species counts. The full per-tick ASCII/debug trace is written to `sim_report.txt` instead of being dumped to the terminal. In `--stream` mode this summary is written to `stderr` so the CSV-style stream on `stdout` stays clean for `vis.py`.

The same run also writes `sim_report.txt` in the project folder. That report includes:

- program timing
- kernel timing totals / average / max / launch count
- final tick and simulated day
- initial and final species counts
- active food count
- debug/config values such as detection radius, predation radius, food respawn chance, food gain, predation success, and reproduction cooldowns

Useful runtime options include `--ticks`, `--ticks-per-day`, `--agent-capacity`, `--bluegill`, `--minnow`, `--bass`, `--food`, `--bluegill-gain`, `--bluegill-drain`, `--bluegill-max-energy`, `--minnow-gain`, `--minnow-drain`, `--minnow-max-energy`, `--minnow-reproduce-energy`, `--minnow-reproduce-cost`, `--minnow-reproduce-cooldown-days`, `--minnow-child-energy`, `--minnow-max-children`, `--bass-drain`, `--bass-gain`, `--bass-max-energy`, `--bass-reproduce-energy`, `--bass-reproduce-cost`, `--bass-reproduce-cooldown-days`, `--bass-child-energy`, `--bass-max-children`, `--detection-radius`, `--predation-radius`, `--predation-success`, `--bluegill-predation-success`, `--minnow-predation-success`, `--move-step`, `--reproduce-energy`, `--reproduce-cost`, `--reproduce-cooldown-days`, `--mate-radius`, `--child-energy`, `--reproduce-chance`, `--max-children`, `--food-gain`, `--food-eat-radius`, `--food-respawn-chance`, `--size-growth-threshold`, `--size-shrink-threshold`, `--size-growth-rate`, `--size-shrink-rate`, `--bluegill-min-size`, `--bluegill-max-size`, `--minnow-min-size`, `--minnow-max-size`, `--bass-min-size`, `--bass-max-size`, and `--stream`.

Example with tighter food supply and mixed prey:

```powershell
.\build\Release\sim.exe --ticks 120 --agent-capacity 320 --bluegill 45 --minnow 60 --bass 10 --food 240 --food-gain 0.24 --food-respawn-chance 6 --reproduce-energy 1.35 --reproduce-cost 0.80 --mate-radius 0.08 --max-children 5 --minnow-drain 0.006 --minnow-reproduce-energy 1.20 --minnow-reproduce-cost 0.55 --minnow-child-energy 0.70 --minnow-max-children 9 --bass-reproduce-energy 1.95 --bass-reproduce-cost 1.55 --bass-max-children 3 --bluegill-predation-success 45 --minnow-predation-success 70 --bass-gain 1.0 --bluegill-min-size 0.10 --bluegill-max-size 0.50 --minnow-min-size 0.05 --minnow-max-size 0.10 --bass-min-size 0.80 --bass-max-size 10.0 --size-growth-threshold 1.30 --size-shrink-threshold 0.70 --size-growth-rate 0.015 --size-shrink-rate 0.020
```

Big run:
```
& C:\Users\jobog\AppData\Local\Programs\Python\Python310\python.exe "c:/Users/jobog/Documents/Homework/ECEN 489/term_project/vis.py" --ticks 400 --vis-skip 20 --vis-pause 0.03 --vis-autoplay --agent-capacity 15000 --bluegill 1200 --minnow 1700 --bass 80 --food 3500 --food-respawn-chance 5 --food-gain 0.22 --food-eat-radius 0.032 --bluegill-gain 0.0 --bluegill-drain 0.006 --bluegill-max-energy 2.0 --reproduce-energy 1.30 --reproduce-cost 0.80 --max-children 5 --minnow-gain 0.0 --minnow-drain 0.0065 --minnow-max-energy 1.75 --minnow-reproduce-energy 1.25 --minnow-reproduce-cost 0.70 --minnow-child-energy 0.65 --minnow-max-children 7 --detection-radius 0.28 --mate-radius 0.075 --predation-radius 0.024 --bluegill-predation-success 40 --minnow-predation-success 70 --bass-reproduce-energy 2.0 --bass-reproduce-cost 1.70 --bass-max-children 2 --bass-gain 1.0 --bass-drain 0.013 --bass-max-energy 2.0 --size-growth-threshold 1.30 --size-shrink-threshold 0.70 --size-growth-rate 0.015 --size-shrink-rate 0.020 --bluegill-min-size 0.10 --bluegill-max-size 0.50 --minnow-min-size 0.05 --minnow-max-size 0.10 --bass-min-size 0.80 --bass-max-size 10.0
```
## Python Visualization

```powershell
python .\vis.py
```

`vis.py` launches `sim --stream`, loads the streamed ticks into memory, and opens a Matplotlib replay viewer. Bluegill are blue circles, minnows are small cyan circles, bass are red triangles, newly spawned prey are stars for one frame, newly spawned bass are bright red triangles for one frame, and active food is shown as a green heat map layer. Fish marker size scales with the simulation `size` value. The viewer now tracks simulated day directly and plots history against days rather than raw ticks.

The replay viewer includes play/pause, previous/next frame buttons, a restart button, a frame slider, and a delay slider for playback speed. Keyboard shortcuts are also available: Space toggles play/pause, Left/Right move backward/forward one tick, Home jumps to the start, and End jumps to the final frame.

Any extra arguments passed to `vis.py` are forwarded to the simulation:

```powershell
python .\vis.py --ticks 50 --vis-skip 2
```

Visualizer-only options include `--vis-skip N` to store every Nth simulation tick, `--vis-pause X` to set the initial playback delay, and `--vis-autoplay` to start replay automatically after loading frames.
