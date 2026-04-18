# Minimal CUDA Artificial Life Simulation

This project contains a minimal CUDA C++ artificial life prototype. Ten agents start alive, with extra allocated capacity for spawned agents. The capacity is a GPU memory/resource limit, not a biological population rule. Agents store position, energy, alive state, and movement direction in a Structure of Arrays layout. A single CUDA kernel performs deterministic movement, energy drain, mate seeking, food seeking, and simple food consumption against 24 fixed food particles. Consumed food can occasionally respawn on the CPU at new deterministic pseudo-random positions. Reproduction and death cleanup are handled on the CPU after each CUDA step: agents can spawn children into available capacity, and dead agents become food at their last position.

The Python visualizer reads the simulation state directly from the executable through a stdout pipe, so no CSV file is needed.

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

If your generator is single-config, the executable may be at `.\build\sim.exe`.

Runtime parameters can be adjusted from the command line:

```powershell
.\build\Release\sim.exe --steps 100 --move-step 0.02 --detection-radius 0.20 --respawn-chance 50
```

Available options include `--steps`, `--agent-capacity`, `--max-agents`, `--seed`, `--energy-drain`, `--food-energy`, `--detection-radius`, `--eat-radius`, `--move-step`, `--decay`, `--noise`, `--respawn-chance`, `--reproduce-energy`, `--reproduce-cost`, `--mate-radius`, `--child-energy`, `--reproduce-chance`, and `--stream`.

To make reproduction easy to see, lower the required reproduction energy:

```powershell
.\build\Release\sim.exe --steps 20 --agent-capacity 30 --reproduce-energy 0.9 --reproduce-cost 0.1 --mate-radius 0.1
```

Agents that have enough energy to reproduce look for the nearest eligible mate within `--detection-radius`. If no mate is detected, they look for food as before.

## Python Visualization

```powershell
python .\vis.py
```

`vis.py` launches `sim --stream` and renders the streamed state live with Matplotlib, including an agent status table with position and energy.

Any extra arguments passed to `vis.py` are forwarded to the simulation:

```powershell
python .\vis.py --steps 100 --noise 0.10
```
