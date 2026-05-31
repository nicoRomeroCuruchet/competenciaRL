#!/usr/bin/env python3
"""
Arma LEADERBOARD.md leyendo todos los results/<equipo>.json.
Ordena por la métrica de ranking definida en config.yaml (iqm por defecto).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
METRIC = CFG.get("rank_metric", "iqm")

HEADER = "| # | Equipo | IQM | IC 95% | Media | Desvío | Min | Max | Episodios |"
SEP = "|---|--------|-----|--------|-------|--------|-----|-----|-----------|"


def row(rank: int, r: dict) -> str:
    ci = r.get("iqm_ci95", ["—", "—"])
    return (
        f"| {rank} | {r['team']} | {r['iqm']:.2f} | "
        f"[{ci[0]}, {ci[1]}] | {r['mean']:.2f} | {r['std']:.2f} | "
        f"{r['min']:.1f} | {r['max']:.1f} | {r['n_episodes']} |"
    )


def main():
    results = []
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue

    results.sort(key=lambda r: r.get(METRIC, float("-inf")), reverse=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 🏆 Leaderboard — Competencia RL (DQN)",
        "",
        f"Entorno: `{CFG['env_id']}` · Ranking por **{METRIC.upper()}** · "
        f"Actualizado: {now}",
        "",
        HEADER, SEP,
    ]
    if results:
        lines += [row(i, r) for i, r in enumerate(results, 1)]
    else:
        lines.append("| — | _sin submissions todavía_ | | | | | | | |")

    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print(f"Leaderboard regenerado con {len(results)} equipo(s).")


if __name__ == "__main__":
    main()
