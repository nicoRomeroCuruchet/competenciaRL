#!/usr/bin/env python3
"""Plot MountainCar greedy action maps for all submissions.

For each policy.onnx under submissions/<team>/policy.onnx, this script computes
argmax(Q_values, axis=-1) on a regular grid over [position, velocity], then
saves one action map per team.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = ROOT / "submissions"
MOUNTAINCAR_POS_RANGE = (-1.2, 0.6)
MOUNTAINCAR_VEL_RANGE = (-0.07, 0.07)
GRID_PAD_POS = 0.05
GRID_PAD_VEL = 0.005
START_POS_RANGE = (-0.6, -0.4)
START_VEL = 0.0
TERMINAL_POS = 0.5
FORCE = 0.001
GRAVITY = 0.0025
MAX_STEPS = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MountainCar argmax(Q) action maps for each submission"
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=SUBMISSIONS_DIR,
        help="Directory containing submissions/<team>/policy.onnx",
    )
    parser.add_argument(
        "--grid-position",
        type=int,
        default=220,
        help="Number of points along position axis",
    )
    parser.add_argument(
        "--grid-velocity",
        type=int,
        default=220,
        help="Number of points along velocity axis",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Batch size for ONNX inference over grid points",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "action_maps",
        help="Output directory for generated figures",
    )
    return parser.parse_args()


def discover_policies(submissions_dir: Path) -> Dict[str, Path]:
    policies: Dict[str, Path] = {}
    for policy_path in sorted(submissions_dir.glob("*/policy.onnx")):
        team = policy_path.parent.name
        if team.startswith("_"):
            continue
        policies[team] = policy_path
    if not policies:
        raise SystemExit(f"No policy.onnx found under {submissions_dir}")
    return policies


def mountaincar_grid(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_min = MOUNTAINCAR_POS_RANGE[0] - GRID_PAD_POS
    pos_max = MOUNTAINCAR_POS_RANGE[1] + GRID_PAD_POS
    vel_min = MOUNTAINCAR_VEL_RANGE[0] - GRID_PAD_VEL
    vel_max = MOUNTAINCAR_VEL_RANGE[1] + GRID_PAD_VEL

    position = np.linspace(pos_min, pos_max, nx, dtype=np.float32)
    velocity = np.linspace(vel_min, vel_max, ny, dtype=np.float32)
    pos_grid, vel_grid = np.meshgrid(position, velocity, indexing="xy")
    states = np.column_stack((pos_grid.ravel(), vel_grid.ravel())).astype(np.float32)
    return pos_grid, vel_grid, states


def batched(iterable: np.ndarray, n: int) -> Iterable[np.ndarray]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def mountaincar_step(position: float, velocity: float, action: int) -> Tuple[float, float]:
    velocity_next = velocity + (action - 1) * FORCE - GRAVITY * np.cos(3.0 * position)
    velocity_next = float(
        np.clip(velocity_next, MOUNTAINCAR_VEL_RANGE[0], MOUNTAINCAR_VEL_RANGE[1])
    )
    position_next = position + velocity_next
    position_next = float(
        np.clip(position_next, MOUNTAINCAR_POS_RANGE[0], MOUNTAINCAR_POS_RANGE[1])
    )
    return position_next, velocity_next


def greedy_action(session: ort.InferenceSession, obs: np.ndarray) -> int:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    obs = np.asarray(obs, dtype=np.float32)
    try:
        q_values = np.asarray(session.run([output_name], {input_name: obs})[0])
    except Exception:
        q_values = np.asarray(session.run([output_name], {input_name: obs.reshape(1, -1)})[0])
    q_values = np.squeeze(q_values)
    if q_values.ndim != 1:
        q_values = q_values.reshape(-1)
    return int(np.argmax(q_values))


def build_example_starts(start_vel_halfband: float) -> np.ndarray:
    x0, x1 = START_POS_RANGE
    y0 = START_VEL - start_vel_halfband
    y1 = START_VEL + start_vel_halfband
    xc = 0.5 * (x0 + x1)
    yc = START_VEL
    starts = np.array(
        [
            [x0, y0],
            [x0, y1],
            [x1, y0],
            [x1, y1],
            [xc, yc],
        ],
        dtype=np.float32,
    )
    return starts


def rollout_trajectories(session: ort.InferenceSession, starts: np.ndarray) -> Iterable[np.ndarray]:
    trajectories = []
    for start in starts:
        position = float(start[0])
        velocity = float(start[1])
        pts = [(position, velocity)]
        for _ in range(MAX_STEPS):
            action = greedy_action(session, np.array([position, velocity], dtype=np.float32))
            position, velocity = mountaincar_step(position, velocity, action)
            pts.append((position, velocity))
            if position >= TERMINAL_POS:
                break
        trajectories.append(np.asarray(pts, dtype=np.float32))
    return trajectories


def compute_action_grid(
    session: ort.InferenceSession,
    states: np.ndarray,
    grid_shape: Tuple[int, int],
    chunk_size: int,
    progress_desc: str,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    actions = np.empty((states.shape[0],), dtype=np.int64)
    n_chunks = int(np.ceil(states.shape[0] / chunk_size))
    start = 0
    batch_mode = True

    def infer_single(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        try:
            out = session.run([output_name], {input_name: obs})[0]
        except Exception:
            out = session.run([output_name], {input_name: obs.reshape(1, -1)})[0]
        out = np.asarray(out)
        out = np.squeeze(out)
        if out.ndim != 1:
            out = out.reshape(-1)
        return out

    for batch in tqdm(
        batched(states, chunk_size),
        total=n_chunks,
        desc=progress_desc,
        leave=False,
    ):
        if batch_mode:
            try:
                q_values = np.asarray(session.run([output_name], {input_name: batch})[0])
                if q_values.ndim == 1:
                    q_values = q_values.reshape(1, -1)
            except Exception:
                batch_mode = False
                q_values = np.vstack([infer_single(obs) for obs in batch]).astype(np.float32)
        else:
            q_values = np.vstack([infer_single(obs) for obs in batch]).astype(np.float32)

        end = start + batch.shape[0]
        actions[start:end] = np.argmax(q_values, axis=1)
        start = end

    return actions.reshape(grid_shape)


def plot_action_map(
    ax: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    actions: np.ndarray,
    title: str,
    trajectories: Iterable[np.ndarray],
) -> None:
    cmap = ListedColormap(["#3b82f6", "#9ca3af", "#ef4444"])
    extent = (
        float(x_grid.min()),
        float(x_grid.max()),
        float(y_grid.min()),
        float(y_grid.max()),
    )
    im = ax.imshow(
        actions,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )
    xmin = float(x_grid.min())
    xmax = float(x_grid.max())
    ymin = float(y_grid.min())
    ymax = float(y_grid.max())

    start_vel_halfband = 0.003
    start_rect = Rectangle(
        (START_POS_RANGE[0], START_VEL - start_vel_halfband),
        START_POS_RANGE[1] - START_POS_RANGE[0],
        2 * start_vel_halfband,
        fill=False,
        edgecolor="white",
        linestyle=":",
        linewidth=1.8,
    )
    state_rect = Rectangle(
        (MOUNTAINCAR_POS_RANGE[0], MOUNTAINCAR_VEL_RANGE[0]),
        MOUNTAINCAR_POS_RANGE[1] - MOUNTAINCAR_POS_RANGE[0],
        MOUNTAINCAR_VEL_RANGE[1] - MOUNTAINCAR_VEL_RANGE[0],
        fill=False,
        edgecolor="gray",
        linestyle=":",
        linewidth=0.9,
        alpha=1.0,
    )
    terminal_rect = Rectangle(
        (TERMINAL_POS, ymin),
        max(0.0, xmax - TERMINAL_POS),
        ymax - ymin,
        fill=False,
        edgecolor="white",
        linestyle=":",
        linewidth=1.8,
    )
    ax.add_patch(start_rect)
    ax.add_patch(terminal_rect)
    ax.add_patch(state_rect)

    for traj in trajectories:
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="white",
            linewidth=1.1,
            alpha=0.6,
            zorder=4,
        )
        ax.plot(
            traj[0, 0],
            traj[0, 1],
            marker="o",
            markersize=2.8,
            color="white",
            alpha=0.75,
            zorder=5,
        )

    ax.text(
        START_POS_RANGE[0],
        START_VEL + start_vel_halfband + 0.002,
        "start",
        color="white",
        fontsize=9,
    )
    ax.text(
        TERMINAL_POS + 0.01,
        ymax - 0.01,
        "terminal",
        color="white",
        fontsize=9,
    )

    fig = ax.get_figure()
    if fig is not None:
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["0 left", "1 coast", "2 right"])

    ax.set_title(title)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policies = discover_policies(args.submissions_dir)
    pos_grid, vel_grid, states = mountaincar_grid(args.grid_position, args.grid_velocity)
    grid_shape = vel_grid.shape

    print(f"Found {len(policies)} policies in {args.submissions_dir}")
    print(f"Grid: {args.grid_position}x{args.grid_velocity} = {states.shape[0]} states")

    failed_teams = []
    for team in tqdm(sorted(policies.keys()), desc="Generating action maps"):
        session = ort.InferenceSession(
            str(policies[team]), providers=["CPUExecutionProvider"]
        )
        try:
            action_grid = compute_action_grid(
                session,
                states,
                grid_shape,
                args.chunk_size,
                progress_desc=f"Computing {team}",
            )
            start_vel_halfband = 0.003
            starts = build_example_starts(start_vel_halfband)
            trajectories = rollout_trajectories(session, starts)
        except Exception as exc:
            failed_teams.append((team, str(exc)))
            print(f"[warn] Skipping {team}: {exc}")
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8.4, 6.2), constrained_layout=True)
        plot_action_map(
            ax,
            pos_grid,
            vel_grid,
            action_grid,
            title=f"argmax(Q) action map: {team}",
            trajectories=trajectories,
        )

        out_file = args.out_dir / f"{team}_argmax_action.png"
        fig.savefig(out_file, dpi=220)
        plt.close(fig)

    print(f"Saved figures in: {args.out_dir}")
    if failed_teams:
        print("\nTeams skipped due to inference errors:")
        for team, reason in failed_teams:
            print(f"- {team}: {reason}")


if __name__ == "__main__":
    main()
