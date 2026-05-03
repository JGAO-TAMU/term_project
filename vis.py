import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider


FOOD_GRID_SIZE = 36
ACTIVE_VIEWER = None
COVER_CMAP = LinearSegmentedColormap.from_list(
    "cover_blocks",
    ["#eef5ff", "#5c85d6", "#1f4aa8", "#0a2463"],
)
COVER_SPLAT_RADIUS = 2


def add_cover_heat(cover_heat, x, y, intensity):
    center_col = min(FOOD_GRID_SIZE - 1, max(0, int(x * FOOD_GRID_SIZE)))
    center_row = min(FOOD_GRID_SIZE - 1, max(0, int(y * FOOD_GRID_SIZE)))

    for row_idx in range(center_row - COVER_SPLAT_RADIUS, center_row + COVER_SPLAT_RADIUS + 1):
        if row_idx < 0 or row_idx >= FOOD_GRID_SIZE:
            continue
        for col in range(center_col - COVER_SPLAT_RADIUS, center_col + COVER_SPLAT_RADIUS + 1):
            if col < 0 or col >= FOOD_GRID_SIZE:
                continue

            row_diff = row_idx - center_row
            col_diff = col - center_col
            distance = math.sqrt(float(row_diff * row_diff + col_diff * col_diff))
            if distance > float(COVER_SPLAT_RADIUS):
                continue

            weight = 1.0 - distance / float(COVER_SPLAT_RADIUS + 1)
            cover_heat[row_idx][col] += intensity * weight


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


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vis-skip", type=int, default=1)
    parser.add_argument("--vis-pause", type=float, default=0.05)
    parser.add_argument("--vis-autoplay", action="store_true")
    parser.add_argument("--vis-stats", action="store_true")
    parser.add_argument("--vis-save-replay", action="store_true")
    parser.add_argument("--vis-replay-dir", default="replay")
    parser.add_argument("--vis-save-plot")
    parser.add_argument("--vis-no-show", action="store_true")
    parser.add_argument("--help-vis", action="store_true")
    args, sim_args = parser.parse_known_args()

    if args.help_vis:
        print("Visualizer-only options:")
        print("  --vis-skip N       Store every Nth simulation tick for replay. Default: 1")
        print("  --vis-pause X      Initial replay delay in seconds. Default: 0.05")
        print("  --vis-autoplay     Start replay automatically after loading frames.")
        print("  --vis-stats        Use daily aggregate stats mode instead of full replay mode.")
        print("  --vis-save-replay  Save streamed replay CSV to the replay folder.")
        print("  --vis-replay-dir   Replay output folder. Default: replay")
        print("  --vis-save-plot P  Save stats plot image to path P.")
        print("  --vis-no-show      Do not open the Matplotlib window after saving.")
        print("All other options are forwarded to sim.exe.")
        sys.exit(0)

    if args.vis_skip < 1:
        args.vis_skip = 1
    if args.vis_pause < 0.001:
        args.vis_pause = 0.001

    return args, sim_args


def create_replay_path(replay_dir):
    replay_root = Path(replay_dir)
    replay_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return replay_root / f"replay_{timestamp}.csv"


def empty_history():
    return {"tick": [], "day": [], "bluegill": [], "minnow": [], "bass": [], "food": [], "new": []}


def build_frame(tick, day, rows, seen_agent_ids):
    food_heat = [[0 for _ in range(FOOD_GRID_SIZE)] for _ in range(FOOD_GRID_SIZE)]
    cover_heat = [[0.0 for _ in range(FOOD_GRID_SIZE)] for _ in range(FOOD_GRID_SIZE)]
    frame = {
        "tick": tick,
        "day": day,
        "bluegill_x": [],
        "bluegill_y": [],
        "bluegill_size": [],
        "minnow_x": [],
        "minnow_y": [],
        "minnow_size": [],
        "bass_x": [],
        "bass_y": [],
        "bass_size": [],
        "new_child_x": [],
        "new_child_y": [],
        "new_child_size": [],
        "new_minnow_x": [],
        "new_minnow_y": [],
        "new_minnow_size": [],
        "new_bass_x": [],
        "new_bass_y": [],
        "new_bass_size": [],
        "dead_agents_x": [],
        "dead_agents_y": [],
        "dead_agents_size": [],
        "food_heat": food_heat,
        "cover_heat": cover_heat,
        "total_agents": 0,
        "alive_agents": 0,
        "bluegill_count": 0,
        "minnow_count": 0,
        "bass_count": 0,
        "active_food_count": 0,
        "total_food_count": 0,
        "total_cover_count": 0,
        "avg_cover_intensity": 0.0,
        "avg_bluegill_energy": 0.0,
        "avg_minnow_energy": 0.0,
        "avg_bass_energy": 0.0,
        "avg_bluegill_size": 0.0,
        "max_bluegill_size": 0.0,
        "avg_minnow_size": 0.0,
        "max_minnow_size": 0.0,
        "avg_bass_size": 0.0,
        "max_bass_size": 0.0,
    }

    bluegill_energy_total = 0.0
    minnow_energy_total = 0.0
    bass_energy_total = 0.0
    bluegill_size_total = 0.0
    minnow_size_total = 0.0
    bass_size_total = 0.0
    cover_intensity_total = 0.0

    for row in rows:
        if row["type"] == "food":
            frame["total_food_count"] += 1
            if int(row["active"]) != 0:
                x = float(row["x"])
                y = float(row["y"])
                col = min(FOOD_GRID_SIZE - 1, max(0, int(x * FOOD_GRID_SIZE)))
                row_idx = min(FOOD_GRID_SIZE - 1, max(0, int(y * FOOD_GRID_SIZE)))
                food_heat[row_idx][col] += 1
                frame["active_food_count"] += 1
            continue

        if row["type"] == "cover":
            x = float(row["x"])
            y = float(row["y"])
            intensity = float(row.get("intensity", 0.0))
            add_cover_heat(cover_heat, x, y, intensity)
            frame["total_cover_count"] += 1
            cover_intensity_total += intensity
            continue

        if row["type"] != "agent":
            continue

        x = float(row["x"])
        y = float(row["y"])
        agent_id = int(row["id"])
        species = row["species"]
        energy = float(row["energy"])
        active = int(row["active"])
        fish_size = float(row.get("size", 1.0))
        is_new_child = tick > 0 and agent_id not in seen_agent_ids

        frame["total_agents"] += 1

        if active == 0:
            frame["dead_agents_x"].append(x)
            frame["dead_agents_y"].append(y)
            frame["dead_agents_size"].append(fish_size)
            continue

        frame["alive_agents"] += 1
        if species == "bluegill":
            frame["bluegill_count"] += 1
            bluegill_energy_total += energy
            bluegill_size_total += fish_size
            frame["max_bluegill_size"] = max(frame["max_bluegill_size"], fish_size)
            if is_new_child:
                frame["new_child_x"].append(x)
                frame["new_child_y"].append(y)
                frame["new_child_size"].append(fish_size)
            else:
                frame["bluegill_x"].append(x)
                frame["bluegill_y"].append(y)
                frame["bluegill_size"].append(fish_size)
        elif species == "minnow":
            frame["minnow_count"] += 1
            minnow_energy_total += energy
            minnow_size_total += fish_size
            frame["max_minnow_size"] = max(frame["max_minnow_size"], fish_size)
            if is_new_child:
                frame["new_minnow_x"].append(x)
                frame["new_minnow_y"].append(y)
                frame["new_minnow_size"].append(fish_size)
            else:
                frame["minnow_x"].append(x)
                frame["minnow_y"].append(y)
                frame["minnow_size"].append(fish_size)
        elif species == "bass":
            frame["bass_count"] += 1
            bass_energy_total += energy
            bass_size_total += fish_size
            frame["max_bass_size"] = max(frame["max_bass_size"], fish_size)
            if is_new_child:
                frame["new_bass_x"].append(x)
                frame["new_bass_y"].append(y)
                frame["new_bass_size"].append(fish_size)
            else:
                frame["bass_x"].append(x)
                frame["bass_y"].append(y)
                frame["bass_size"].append(fish_size)

    if frame["bluegill_count"] > 0:
        frame["avg_bluegill_energy"] = bluegill_energy_total / frame["bluegill_count"]
        frame["avg_bluegill_size"] = bluegill_size_total / frame["bluegill_count"]
    if frame["minnow_count"] > 0:
        frame["avg_minnow_energy"] = minnow_energy_total / frame["minnow_count"]
        frame["avg_minnow_size"] = minnow_size_total / frame["minnow_count"]
    if frame["bass_count"] > 0:
        frame["avg_bass_energy"] = bass_energy_total / frame["bass_count"]
        frame["avg_bass_size"] = bass_size_total / frame["bass_count"]
    if frame["total_cover_count"] > 0:
        frame["avg_cover_intensity"] = cover_intensity_total / frame["total_cover_count"]

    return frame


def remember_agent_ids(rows, seen_agent_ids):
    for row in rows:
        if row["type"] == "agent":
            seen_agent_ids.add(int(row["id"]))


def collect_frames(sim_command, vis_skip, replay_path=None):
    print("Launching simulation:")
    print(" ".join(sim_command))
    print("Loading frames...")

    process = subprocess.Popen(
        sim_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture simulation stdout.")

    frames = []
    history = empty_history()
    seen_agent_ids = set()
    current_tick = None
    current_day = 0.0
    rows = []
    saved_replay = None

    header_line = process.stdout.readline()
    if not header_line:
        stderr_output = ""
        if process.stderr is not None:
            stderr_output = process.stderr.read()
        process.wait()
        raise RuntimeError(f"Simulation did not produce stream output.\n{stderr_output}")

    fieldnames = next(csv.reader([header_line.strip()]))
    replay_file = None
    if replay_path is not None:
        replay_file = replay_path.open("w", newline="", encoding="utf-8")
        replay_file.write(header_line)
        saved_replay = replay_path

    def finish_tick(tick, day, tick_rows):
        if tick is None:
            return

        if tick % vis_skip == 0:
            frame = build_frame(tick, day, tick_rows, seen_agent_ids)
            frames.append(frame)
            history["tick"].append(tick)
            history["day"].append(day)
            history["bluegill"].append(frame["bluegill_count"])
            history["minnow"].append(frame["minnow_count"])
            history["bass"].append(frame["bass_count"])
            history["food"].append(frame["active_food_count"])
            history["new"].append(len(frame["new_child_x"]) +
                                  len(frame["new_minnow_x"]) +
                                  len(frame["new_bass_x"]))

        remember_agent_ids(tick_rows, seen_agent_ids)

    for line in process.stdout:
        if replay_file is not None:
            replay_file.write(line)

        stripped = line.strip()
        if not stripped:
            continue

        values = next(csv.reader([stripped]))
        if len(values) != len(fieldnames):
            continue

        row = dict(zip(fieldnames, values))
        tick = int(row.get("tick", row.get("step", "0")))
        day = float(row.get("day", tick))

        if current_tick is None:
            current_tick = tick
            current_day = day

        if tick != current_tick:
            finish_tick(current_tick, current_day, rows)
            rows = []
            current_tick = tick
            current_day = day

        rows.append(row)

    finish_tick(current_tick, current_day, rows)

    stderr_output = ""
    if process.stderr is not None:
        stderr_output = process.stderr.read()

    if replay_file is not None:
        replay_file.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Simulation exited with code {return_code}.\n{stderr_output}")
    if not frames:
        raise RuntimeError("No frames were captured from the simulation.")

    print(f"Loaded {len(frames)} replay frames.")
    if saved_replay is not None:
        print(f"Saved replay to: {saved_replay}")
    return frames, history


def collect_daily_stats(sim_command, replay_path=None):
    print("Launching simulation:")
    print(" ".join(sim_command))
    print("Loading daily stats...")

    process = subprocess.Popen(
        sim_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("Failed to capture simulation stdout.")

    header_line = process.stdout.readline()
    if not header_line:
        stderr_output = ""
        if process.stderr is not None:
            stderr_output = process.stderr.read()
        process.wait()
        raise RuntimeError(f"Simulation did not produce stats output.\n{stderr_output}")

    fieldnames = next(csv.reader([header_line.strip()]))
    rows = []
    saved_replay = None
    replay_file = None
    if replay_path is not None:
        replay_file = replay_path.open("w", newline="", encoding="utf-8")
        replay_file.write(header_line)
        saved_replay = replay_path

    for line in process.stdout:
        if replay_file is not None:
            replay_file.write(line)

        stripped = line.strip()
        if not stripped:
            continue

        values = next(csv.reader([stripped]))
        if len(values) != len(fieldnames):
            continue
        rows.append(dict(zip(fieldnames, values)))

    stderr_output = ""
    if process.stderr is not None:
        stderr_output = process.stderr.read()

    if replay_file is not None:
        replay_file.close()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Simulation exited with code {return_code}.\n{stderr_output}")
    if not rows:
        raise RuntimeError("No daily stats were captured from the simulation.")

    print(f"Loaded {len(rows)} daily summaries.")
    if saved_replay is not None:
        print(f"Saved replay to: {saved_replay}")
    return rows


def show_stats_plot(rows, output_path=None, show_window=True):
    days = [float(row["day"]) for row in rows]
    bluegill = [int(row["bluegill"]) for row in rows]
    minnow = [int(row["minnow"]) for row in rows]
    bass = [int(row["bass"]) for row in rows]
    food = [int(row["food"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(days, bluegill, label="Bluegill", color="royalblue")
    ax.plot(days, minnow, label="Minnow", color="deepskyblue")
    ax.plot(days, bass, label="Bass", color="crimson")
    ax.plot(days, food, label="Food", color="darkolivegreen", alpha=0.8)
    ax.set_title("Daily Population Stats")
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
        print(f"Saved stats plot to: {output_path}")
    if show_window:
        plt.show()
    else:
        plt.close(fig)


def draw_frame(plot_ax, stats_ax, summary_ax, frames, history, frame_index):
    frame = frames[frame_index]

    def jitter_points(xs, ys, jitter_seed, scale):
        jittered_x = []
        jittered_y = []
        for idx, (x, y) in enumerate(zip(xs, ys)):
            angle = 0.37 * float(jitter_seed) + 1.91 * float(idx)
            jitter_x = scale * math.sin(angle)
            jitter_y = scale * math.cos(angle * 1.27)
            jittered_x.append(min(1.0, max(0.0, x + jitter_x)))
            jittered_y.append(min(1.0, max(0.0, y + jitter_y)))
        return jittered_x, jittered_y

    plot_ax.clear()
    if frame["total_cover_count"] > 0:
        plot_ax.imshow(
            frame["cover_heat"],
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap=COVER_CMAP,
            alpha=0.78,
            interpolation="bilinear",
            vmin=0,
        )
    if frame["active_food_count"] > 0:
        plot_ax.imshow(
            frame["food_heat"],
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="YlGn",
            alpha=0.45,
            interpolation="nearest",
            vmin=0,
        )
    if frame["bluegill_x"]:
        bluegill_x, bluegill_y = jitter_points(frame["bluegill_x"],
                                               frame["bluegill_y"],
                                               frame["tick"] + 11,
                                               0.0035)
        plot_ax.scatter(
            bluegill_x,
            bluegill_y,
            label="Bluegill",
            s=[18 + 60 * value for value in frame["bluegill_size"]],
            color="royalblue",
            alpha=0.75,
        )
    if frame["minnow_x"]:
        minnow_x, minnow_y = jitter_points(frame["minnow_x"],
                                           frame["minnow_y"],
                                           frame["tick"] + 29,
                                           0.0045)
        plot_ax.scatter(
            minnow_x,
            minnow_y,
            label="Minnow",
            s=[8 + 80 * value for value in frame["minnow_size"]],
            color="deepskyblue",
            alpha=0.65,
        )
    if frame["bass_x"]:
        bass_x, bass_y = jitter_points(frame["bass_x"],
                                       frame["bass_y"],
                                       frame["tick"] + 53,
                                       0.0030)
        plot_ax.scatter(
            bass_x,
            bass_y,
            label="Bass",
            s=[32 + 8 * value for value in frame["bass_size"]],
            color="crimson",
            marker="^",
        )
    if frame["new_child_x"]:
        plot_ax.scatter(
            frame["new_child_x"],
            frame["new_child_y"],
            label="New Bluegill",
            s=[70 + 60 * value for value in frame["new_child_size"]],
            marker="*",
            color="orange",
            edgecolors="black",
        )
    if frame["new_minnow_x"]:
        plot_ax.scatter(
            frame["new_minnow_x"],
            frame["new_minnow_y"],
            label="New Minnow",
            s=[45 + 80 * value for value in frame["new_minnow_size"]],
            marker="*",
            color="cyan",
            edgecolors="black",
        )
    if frame["new_bass_x"]:
        plot_ax.scatter(
            frame["new_bass_x"],
            frame["new_bass_y"],
            label="New Bass",
            s=[65 + 8 * value for value in frame["new_bass_size"]],
            marker="^",
            color="red",
            edgecolors="black",
        )
    if frame["dead_agents_x"]:
        plot_ax.scatter(
            frame["dead_agents_x"],
            frame["dead_agents_y"],
            label="Dead",
            s=[10 + 12 * value for value in frame["dead_agents_size"]],
            color="gray",
            alpha=0.35,
        )

    plot_ax.set_xlim(0, 1)
    plot_ax.set_ylim(0, 1)
    plot_ax.set_aspect("equal", adjustable="box")
    plot_ax.set_title(
        f"Tick {frame['tick']} | Day {frame['day']:.2f} | Bluegill {frame['bluegill_count']} | "
        f"Minnow {frame['minnow_count']} | "
        f"Bass {frame['bass_count']} | Food {frame['active_food_count']} | Cover {frame['total_cover_count']}"
    )
    plot_ax.legend(loc="upper right")

    stats_ax.clear()
    stats_ax.plot(history["day"], history["bluegill"], label="Bluegill", color="royalblue")
    stats_ax.plot(history["day"], history["minnow"], label="Minnow", color="deepskyblue")
    stats_ax.plot(history["day"], history["bass"], label="Bass", color="crimson")
    stats_ax.plot(history["day"], history["food"], label="Food", color="darkolivegreen", alpha=0.8)
    stats_ax.axvline(frame["day"], color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    stats_ax.set_title("Population / Food History")
    stats_ax.set_xlabel("Day")
    stats_ax.set_ylabel("Count")
    stats_ax.legend(loc="upper left")

    status_summary = (
        f"Replay frame: {frame_index + 1}/{len(frames)}\n"
        f"Day: {frame['day']:.2f} | Tick: {frame['tick']}\n"
        f"Alive agents: {frame['alive_agents']}/{frame['total_agents']}\n"
        f"Active food: {frame['active_food_count']}/{frame['total_food_count']}\n"
        f"Cover zones: {frame['total_cover_count']} | Avg intensity: {frame['avg_cover_intensity']:.2f}\n"
        f"New bluegill this frame: {len(frame['new_child_x'])}\n"
        f"New minnow this frame: {len(frame['new_minnow_x'])}\n"
        f"New bass this frame: {len(frame['new_bass_x'])}\n"
        f"Avg bluegill energy: {frame['avg_bluegill_energy']:.2f}\n"
        f"Avg minnow energy: {frame['avg_minnow_energy']:.2f}\n"
        f"Avg bass energy: {frame['avg_bass_energy']:.2f}"
    )
    size_summary = (
        f"Bluegill size avg/max: {frame['avg_bluegill_size']:.2f}/{frame['max_bluegill_size']:.2f}\n"
        f"Minnow size avg/max: {frame['avg_minnow_size']:.2f}/{frame['max_minnow_size']:.2f}\n"
        f"Bass size avg/max: {frame['avg_bass_size']:.2f}/{frame['max_bass_size']:.2f}"
    )

    summary_ax.clear()
    summary_ax.axis("off")
    summary_ax.text(
        0.02,
        0.98,
        status_summary,
        transform=summary_ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    summary_ax.text(
        0.54,
        0.98,
        size_summary,
        transform=summary_ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )


class ReplayViewer:
    def __init__(self, frames, history, initial_pause, autoplay):
        self.frames = frames
        self.history = history
        self.index = 0
        self.is_playing = autoplay
        self.slider_is_updating = False

        self.fig = plt.figure(figsize=(12, 7))
        grid = self.fig.add_gridspec(
            2,
            2,
            height_ratios=[3, 1],
            width_ratios=[3, 2],
            hspace=0.25,
            wspace=0.28,
        )
        self.plot_ax = self.fig.add_subplot(grid[:, 0])
        self.stats_ax = self.fig.add_subplot(grid[0, 1])
        self.summary_ax = self.fig.add_subplot(grid[1, 1])

        self.fig.subplots_adjust(bottom=0.22)
        self.fig.suptitle("Controls: Space = Play/Pause | Left/Right = Tick | Home = Start | End = Finish")

        tick_slider_ax = self.fig.add_axes([0.15, 0.12, 0.70, 0.03])
        speed_slider_ax = self.fig.add_axes([0.15, 0.07, 0.70, 0.03])
        prev_ax = self.fig.add_axes([0.15, 0.015, 0.12, 0.04])
        play_ax = self.fig.add_axes([0.30, 0.015, 0.16, 0.04])
        next_ax = self.fig.add_axes([0.49, 0.015, 0.12, 0.04])
        restart_ax = self.fig.add_axes([0.64, 0.015, 0.12, 0.04])

        max_index = max(0, len(frames) - 1)
        self.tick_slider = Slider(
            tick_slider_ax,
            "Frame",
            0,
            max_index,
            valinit=0,
            valstep=1,
        )
        self.speed_slider = Slider(
            speed_slider_ax,
            "Delay",
            0.001,
            0.5,
            valinit=initial_pause,
        )
        self.prev_button = Button(prev_ax, "Prev")
        self.play_button = Button(play_ax, "Pause" if autoplay else "Play")
        self.next_button = Button(next_ax, "Next")
        self.restart_button = Button(restart_ax, "Restart")

        self.tick_slider.on_changed(self.on_slider_changed)
        self.prev_button.on_clicked(self.on_prev)
        self.play_button.on_clicked(self.on_play_pause)
        self.next_button.on_clicked(self.on_next)
        self.restart_button.on_clicked(self.on_restart)
        self.key_connection = self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.timer = self.fig.canvas.new_timer(interval=self.delay_ms())
        self.timer.add_callback(self.on_timer)

        self.redraw()
        if self.is_playing:
            self.timer.start()

    def delay_ms(self):
        return max(1, int(self.speed_slider.val * 1000))

    def redraw(self):
        draw_frame(self.plot_ax, self.stats_ax, self.summary_ax, self.frames, self.history, self.index)
        self.fig.canvas.draw_idle()

    def set_index(self, new_index):
        max_index = len(self.frames) - 1
        self.index = min(max_index, max(0, int(new_index)))

        self.slider_is_updating = True
        self.tick_slider.set_val(self.index)
        self.slider_is_updating = False

        self.redraw()

    def on_slider_changed(self, value):
        if self.slider_is_updating:
            return
        self.index = int(value)
        self.redraw()

    def on_prev(self, _event):
        self.pause()
        self.set_index(self.index - 1)

    def on_next(self, _event):
        self.pause()
        self.set_index(self.index + 1)

    def on_restart(self, _event):
        self.pause()
        self.set_index(0)

    def on_play_pause(self, _event):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def on_key_press(self, event):
        if event.key in (" ", "space"):
            self.on_play_pause(event)
        elif event.key == "right":
            self.on_next(event)
        elif event.key == "left":
            self.on_prev(event)
        elif event.key == "home":
            self.pause()
            self.set_index(0)
        elif event.key == "end":
            self.pause()
            self.set_index(len(self.frames) - 1)

    def play(self):
        self.is_playing = True
        self.play_button.label.set_text("Pause")
        self.timer.interval = self.delay_ms()
        self.timer.start()

    def pause(self):
        self.is_playing = False
        self.play_button.label.set_text("Play")
        self.timer.stop()

    def on_timer(self):
        if not self.is_playing:
            return True

        if self.index >= len(self.frames) - 1:
            self.pause()
            return True

        self.set_index(self.index + 1)
        self.timer.interval = self.delay_ms()
        return True


def main():
    global ACTIVE_VIEWER

    vis_args, sim_args = parse_args()
    sim_executable = find_sim_executable()
    replay_path = create_replay_path(vis_args.vis_replay_dir) if vis_args.vis_save_replay else None
    if vis_args.vis_stats:
        stats_command = [str(sim_executable), "--stats"] + sim_args
        rows = collect_daily_stats(stats_command, replay_path)
        show_stats_plot(rows, vis_args.vis_save_plot, not vis_args.vis_no_show)
    else:
        sim_command = [str(sim_executable), "--stream"] + sim_args
        frames, history = collect_frames(sim_command, vis_args.vis_skip, replay_path)
        ACTIVE_VIEWER = ReplayViewer(frames, history, vis_args.vis_pause, vis_args.vis_autoplay)
        plt.show()


if __name__ == "__main__":
    main()
