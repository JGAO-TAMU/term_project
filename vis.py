import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def find_sim_executable():
    candidates = [
        Path("build/Release/sim.exe"),
        Path("build/sim.exe"),
        Path("build/Release/sim"),
        Path("build/sim"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find the sim executable. Build the project first.")


def draw_step(plot_ax, table_ax, step, rows, seen_agent_ids):
    alive_agents_x = []
    alive_agents_y = []
    new_child_x = []
    new_child_y = []
    dead_agents_x = []
    dead_agents_y = []
    food_x = []
    food_y = []
    agent_rows = []
    current_agent_ids = set()
    total_agents = 0
    alive_agents = 0
    total_food = 0
    active_food = 0

    for row in rows:
        x = float(row["x"])
        y = float(row["y"])
        if row["type"] == "agent":
            total_agents += 1
            agent_id = int(row["id"])
            current_agent_ids.add(agent_id)
            energy = float(row["energy"])
            active = int(row["active"])
            is_new_child = step > 0 and agent_id not in seen_agent_ids
            if active == 0:
                dead_agents_x.append(x)
                dead_agents_y.append(y)
            elif is_new_child:
                alive_agents += 1
                new_child_x.append(x)
                new_child_y.append(y)
            else:
                alive_agents += 1
                alive_agents_x.append(x)
                alive_agents_y.append(y)
            agent_rows.append(
                (
                    agent_id,
                    x,
                    y,
                    energy,
                    "new child" if is_new_child and active == 1 else "alive" if active == 1 else "dead",
                )
            )
        elif row["type"] == "food":
            total_food += 1
            if int(row["active"]) == 1:
                active_food += 1
                food_x.append(x)
                food_y.append(y)

    agent_rows.sort(key=lambda item: item[0])

    plot_ax.clear()
    plot_ax.scatter(alive_agents_x, alive_agents_y, label="Alive Agents", s=60)
    if new_child_x:
        plot_ax.scatter(
            new_child_x,
            new_child_y,
            label="New Children",
            s=140,
            marker="*",
            color="orange",
            edgecolors="black",
        )
    if dead_agents_x:
        plot_ax.scatter(dead_agents_x, dead_agents_y, label="Dead Agents", s=60, color="gray")
    plot_ax.scatter(food_x, food_y, label="Food", s=30)
    plot_ax.set_xlim(0, 1)
    plot_ax.set_ylim(0, 1)
    plot_ax.set_title(
        f"Step {step} | Agents {alive_agents}/{total_agents} | New {len(new_child_x)} | Food {active_food}/{total_food}"
    )
    plot_ax.legend(loc="upper right")

    table_ax.clear()
    table_ax.axis("off")
    table_data = [
        [str(agent_id), f"{x:.2f}", f"{y:.2f}", f"{energy:.2f}", status]
        for agent_id, x, y, energy, status in agent_rows
    ]
    table = table_ax.table(
        cellText=table_data,
        colLabels=["Agent", "X", "Y", "Energy", "Status"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    table_ax.set_title(f"Agent Status | Alive {alive_agents} | New {len(new_child_x)} | Food {active_food}")

    seen_agent_ids.update(current_agent_ids)

    plt.pause(0.3)


def main():
    sim_executable = find_sim_executable()
    sim_command = [str(sim_executable), "--stream"] + sys.argv[1:]
    print("Launching simulation:")
    print(" ".join(sim_command))

    process = subprocess.Popen(
        sim_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture simulation stdout.")

    reader = csv.DictReader(process.stdout)
    plt.ion()
    fig, (plot_ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 6),
        gridspec_kw={"width_ratios": [3, 2]},
    )
    fig.tight_layout()

    current_step = None
    rows = []
    seen_agent_ids = set()

    for row in reader:
        step = int(row["step"])

        if current_step is None:
            current_step = step

        if step != current_step:
            draw_step(plot_ax, table_ax, current_step, rows, seen_agent_ids)
            rows = []
            current_step = step

        rows.append(row)

    if rows:
        draw_step(plot_ax, table_ax, current_step, rows, seen_agent_ids)

    stderr_output = ""
    if process.stderr is not None:
        stderr_output = process.stderr.read()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Simulation exited with code {return_code}.\n{stderr_output}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
