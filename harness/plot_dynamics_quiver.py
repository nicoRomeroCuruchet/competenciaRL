#!/usr/bin/env python3
"""Generate quiver maps for MountainCar dynamics for each action.

Creates 3 panels (action 0, 1, 2), where each arrow is the one-step state delta:
    (d_position, d_velocity) = (position_next - position, velocity_next - velocity)
using the same clipped transition equations as the environment.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot MountainCar one-step dynamics quiver maps for all 3 actions"
    )
    parser.add_argument("--grid-position", type=int, default=41, help="Grid points on position axis")
    parser.add_argument("--grid-velocity", type=int, default=41, help="Grid points on velocity axis")
    parser.add_argument(
        "--arrow-multiplier",
        type=float,
        default=25.0,
        help="Visual multiplier for arrow length (for readability only)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output image DPI")
    parser.add_argument("--fig-w", type=float, default=15.0, help="Figure width")
    parser.add_argument("--fig-h", type=float, default=4.8, help="Figure height")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "dynamics_quiver_actions.png",
        help="Output image path",
    )
    return parser.parse_args()


def mountaincar_step(position: np.ndarray, velocity: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    force = 0.001
    gravity = 0.0025

    velocity_next = velocity + (action - 1) * force - gravity * np.cos(3.0 * position)
    velocity_next = np.clip(velocity_next, -max_speed, max_speed)
    position_next = position + velocity_next
    position_next = np.clip(position_next, min_position, max_position)
    return position_next, velocity_next


def main() -> None:
    args = parse_args()

    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5

    pos = np.linspace(min_position, max_position, args.grid_position)
    vel = np.linspace(-max_speed, max_speed, args.grid_velocity)
    pos_grid, vel_grid = np.meshgrid(pos, vel, indexing="xy")

    fig, axes = plt.subplots(1, 3, figsize=(args.fig_w, args.fig_h), constrained_layout=True)
    action_titles = {
        0: "action 0: push left",
        1: "action 1: no push",
        2: "action 2: push right",
    }

    for action, ax in enumerate(axes):
        pos_next, vel_next = mountaincar_step(pos_grid, vel_grid, action)
        d_pos = pos_next - pos_grid
        d_vel = vel_next - vel_grid

        d_pos_plot = d_pos * args.arrow_multiplier
        d_vel_plot = d_vel * args.arrow_multiplier
        quiv = ax.quiver(
            pos_grid,
            vel_grid,
            d_pos_plot,
            d_vel_plot,
            color="black",
            angles="xy",
            scale_units="xy",
            scale=None,
            width=0.0032,
            headwidth=4.0,
            headlength=5.2,
            headaxislength=4.5,
            alpha=0.98,
        )

        env_rect = Rectangle(
            (min_position, -max_speed),
            max_position - min_position,
            2 * max_speed,
            fill=False,
            edgecolor="gray",
            linestyle=":",
            linewidth=1.0,
            alpha=1.0,
        )
        ax.add_patch(env_rect)
        ax.axvline(goal_position, color="white", linestyle="--", linewidth=1.0, alpha=0.9)

        ax.set_title(action_titles[action])
        ax.set_xlabel("position")
        ax.set_ylabel("velocity")
        ax.set_aspect("auto")
        ax.grid(alpha=0.15, linestyle=":")
        # Keep all arrows same color and rely on direction/length for interpretation.
        _ = quiv

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved quiver map figure to: {args.out}")


if __name__ == "__main__":
    main()
