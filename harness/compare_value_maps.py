#!/usr/bin/env python3
"""Plot MountainCar Q-value extrema maps for all submissions.

For each policy.onnx under submissions/<team>/policy.onnx, this script computes
on a regular grid over [position, velocity], then saves:
1) V_max(s) = max_a Q(s, a),
2) V_min(s) = min_a Q(s, a).
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
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
COLORBAR_MIN = -70.0
COLORBAR_MAX = 5.0
SHARED_CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MountainCar max(Q) and min(Q) maps for each submission"
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
        "--plot",
        choices=["imshow", "contourf"],
        default="imshow",
        help="Plot style",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=40,
        help="Number of contour levels when --plot contourf",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "value_maps",
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

    position = np.linspace(
        pos_min, pos_max, nx, dtype=np.float32
    )
    velocity = np.linspace(
        vel_min, vel_max, ny, dtype=np.float32
    )
    pos_grid, vel_grid = np.meshgrid(position, velocity, indexing="xy")
    states = np.column_stack((pos_grid.ravel(), vel_grid.ravel())).astype(np.float32)
    return pos_grid, vel_grid, states


def batched(iterable: np.ndarray, n: int) -> Iterable[np.ndarray]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def compute_q_extrema_grid(
    session: ort.InferenceSession,
    states: np.ndarray,
    grid_shape: Tuple[int, int],
    chunk_size: int,
    progress_desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    values_max = np.empty((states.shape[0],), dtype=np.float32)
    values_min = np.empty((states.shape[0],), dtype=np.float32)
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
        values_max[start:end] = np.max(q_values, axis=1)
        values_min[start:end] = np.min(q_values, axis=1)
        start = end

    # meshgrid(shape) is [n_velocity, n_position]
    return values_max.reshape(grid_shape), values_min.reshape(grid_shape)


def plot_map(
    ax: Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    title: str,
    mode: str,
    cmap: str,
    levels: int,
    norm: Normalize,
) -> None:
    if norm.vmin is None or norm.vmax is None:
        raise ValueError("Expected fixed normalization bounds for plotting")
    vmin = float(norm.vmin)
    vmax = float(norm.vmax)

    if mode == "contourf":
        contour_levels = np.linspace(vmin, vmax, levels)
        ax.contourf(x_grid, y_grid, z, levels=contour_levels, cmap=cmap, norm=norm)
    else:
        extent = (
            float(x_grid.min()),
            float(x_grid.max()),
            float(y_grid.min()),
            float(y_grid.max()),
        )
        ax.imshow(
            z,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    if float(z.min()) <= 0.0 <= float(z.max()):
        ax.contour(
            x_grid,
            y_grid,
            z,
            levels=[0.0],
            colors="black",
            linestyles=":",
            linewidths=1.2,
            alpha=0.95,
        )

    xmin, xmax = float(x_grid.min()), float(x_grid.max())
    ymin, ymax = float(y_grid.min()), float(y_grid.max())

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
    terminal_rect = Rectangle(
        (TERMINAL_POS, ymin),
        max(0.0, xmax - TERMINAL_POS),
        ymax - ymin,
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
    ax.add_patch(start_rect)
    ax.add_patch(terminal_rect)
    ax.add_patch(state_rect)

    ax.text(START_POS_RANGE[0], START_VEL + start_vel_halfband + 0.002, "start", color="white", fontsize=9)
    ax.text(TERMINAL_POS + 0.01, ymax - 0.01, "terminal", color="white", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policies = discover_policies(args.submissions_dir)

    pos_grid, vel_grid, states = mountaincar_grid(
        args.grid_position, args.grid_velocity
    )
    grid_shape = vel_grid.shape

    print(f"Found {len(policies)} policies in {args.submissions_dir}")
    print(
        f"Grid: {args.grid_position}x{args.grid_velocity} = {states.shape[0]} states"
    )

    ordered_teams = sorted(policies.keys())
    failed_teams = []
    norm = Normalize(vmin=COLORBAR_MIN, vmax=COLORBAR_MAX)

    for team in tqdm(ordered_teams, desc="Comparing policies"):
        session = ort.InferenceSession(
            str(policies[team]), providers=["CPUExecutionProvider"]
        )
        try:
            team_vmax, team_vmin = compute_q_extrema_grid(
                session,
                states,
                grid_shape,
                args.chunk_size,
                progress_desc=f"Computing {team}",
            )
        except Exception as exc:
            failed_teams.append((team, str(exc)))
            print(f"[warn] Skipping {team}: {exc}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
        plot_map(
            axes[0],
            pos_grid,
            vel_grid,
            team_vmax,
            title=f"max(Q) map: {team}",
            mode=args.plot,
            cmap=SHARED_CMAP,
            levels=args.levels,
            norm=norm,
        )
        plot_map(
            axes[1],
            pos_grid,
            vel_grid,
            team_vmin,
            title=f"min(Q) map: {team}",
            mode=args.plot,
            cmap=SHARED_CMAP,
            levels=args.levels,
            norm=norm,
        )

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=SHARED_CMAP)
        fig.colorbar(mappable, ax=axes, location="right", shrink=0.92, label="Q value")

        out_file = args.out_dir / f"{team}_q_extrema.png"
        fig.savefig(out_file, dpi=220)
        plt.close(fig)

    print(f"Saved figures in: {args.out_dir}")
    if failed_teams:
        print("\nTeams skipped due to inference errors:")
        for team, reason in failed_teams:
            print(f"- {team}: {reason}")


if __name__ == "__main__":
    main()
