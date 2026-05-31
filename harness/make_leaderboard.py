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
REFERENCE_TEAMS = set(CFG.get("reference_teams", []))

HEADER = "| # | Equipo | IQM | IC 95% | Media | Desvío | Min | Max | Episodios |"
SEP = "|---|--------|-----|--------|-------|--------|-----|-----|-----------|"


def row(tag: str, r: dict) -> str:
    ci = r.get("iqm_ci95", ["—", "—"])
    return (
        f"| {tag} | {r['team']} | {r['iqm']:.2f} | "
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

    # Separar la(s) linea(s) de referencia del ranking de alumnos.
    reference = [r for r in results if r.get("team") in REFERENCE_TEAMS]
    ranked = [r for r in results if r.get("team") not in REFERENCE_TEAMS]
    reference.sort(key=lambda r: r.get(METRIC, float("-inf")), reverse=True)
    ranked.sort(key=lambda r: r.get(METRIC, float("-inf")), reverse=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 🏆 Leaderboard — Competencia RL (DQN)",
        "",
        f"Entorno: `{CFG['env_id']}` · Ranking por **{METRIC.upper()}** · "
        f"Actualizado: {now}",
    ]

    if reference:
        lines += [
            "",
            "### 🏁 Referencia — óptimo a batir (fuera de ranking)",
            "",
            HEADER, SEP,
        ]
        lines += [row("🏁", r) for r in reference]

    lines += [
        "",
        "### Ranking alumnos",
        "",
        HEADER, SEP,
    ]
    if ranked:
        lines += [row(str(i), r) for i, r in enumerate(ranked, 1)]
    else:
        lines.append("| — | _sin submissions todavía_ | | | | | | | |")

    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print(f"Leaderboard regenerado: {len(reference)} referencia(s), "
          f"{len(ranked)} equipo(s) en ranking.")


if __name__ == "__main__":
    main()
