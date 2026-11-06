#!/usr/bin/env python3
"""
Arma LEADERBOARD.md leyendo todos los results/<equipo>.json.

Campeonato estilo F1: en cada circuito (config.yaml -> circuits) se ordena por
la métrica de ranking (mean por defecto) y se reparten puntos por posición
(config.yaml -> points). Gana quien sume más. Desempate: más primeros puestos,
después más segundos, etc.

Un auto TITULAR por escudería (cuenta de GitHub, ver escuderias.py): es el
único que suma y ocupa posiciones. Los demás son de RESERVA: se muestran, pero
no le sacan puestos a nadie. El titular es el auto que más puntos sumaría
corriendo contra los titulares de las otras escuderías (se itera hasta que la
elección no cambia).

Autos de REFERENCIA (results/<x>.json con "referencia": true, sin carpeta en
submissions/): fuera de concurso. Se muestran con los puestos que habrían
sacado contra los titulares, pero no suman ni le sacan puestos a nadie.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from escuderias import escuderias

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
METRIC = CFG.get("rank_metric", "mean")
METRIC_LABEL = {"mean": "MEDIA", "iqm": "IQM"}.get(METRIC, METRIC.upper())
CIRCUITS = CFG["circuits"]
POINTS = CFG["points"]


def load_results() -> list[dict]:
    results = []
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "circuits" not in r:
            print(f"Ignoro {f.name}: formato viejo (sin circuitos), se regenera en la próxima evaluación.")
            continue
        results.append(r)
    return results


def rank_circuit(results: list[dict], cid: str) -> list[tuple[int, dict]]:
    """[(posición, result)] ordenado. Empates exactos comparten posición."""
    rows = [r for r in results if cid in r["circuits"]]
    rows.sort(key=lambda r: r["circuits"][cid][METRIC], reverse=True)
    ranked, prev, pos = [], None, 0
    for i, r in enumerate(rows, 1):
        v = r["circuits"][cid][METRIC]
        if v != prev:
            pos, prev = i, v
        ranked.append((pos, r))
    return ranked


def points_for(pos: int) -> int:
    return POINTS[pos - 1] if pos <= len(POINTS) else 0


def positions_vs(car: dict, field: list[dict]) -> dict[str, int]:
    """Posición de `car` en cada circuito si corriera contra `field` (1 + los que le ganan)."""
    out = {}
    for c in CIRCUITS:
        cid = c["id"]
        if cid in car["circuits"]:
            v = car["circuits"][cid][METRIC]
            out[cid] = 1 + sum(1 for r in field if cid in r["circuits"] and r["circuits"][cid][METRIC] > v)
    return out


def champ_key(pos: dict[str, int], n: int) -> tuple:
    """Orden del campeonato: más puntos; desempate F1 por countback."""
    ps = list(pos.values())
    return (-sum(points_for(p) for p in ps), tuple(-ps.count(k) for k in range(1, n + 1)))


def pick_titulares(results: list[dict], owner: dict[str, str]) -> dict[str, dict]:
    """{escudería: result del titular}."""
    by_owner = defaultdict(list)
    for r in results:
        by_owner[owner[r["team"]]].append(r)
    n = len(results)

    def key_vs(r, tit):
        rivals = [t for o, t in tit.items() if o != owner[r["team"]]]
        return (champ_key(positions_vs(r, rivals), n), r["team"])

    # Arranque: el mejor de cada escudería con todos los autos en pista.
    tit = {o: min(cars, key=lambda r: (champ_key(positions_vs(r, results), n), r["team"]))
           for o, cars in by_owner.items()}
    for _ in range(20):
        new = {o: min(cars, key=lambda r: key_vs(r, tit)) for o, cars in by_owner.items()}
        if all(new[o]["team"] == tit[o]["team"] for o in tit):
            break
        tit = new
    return tit


def cells(pos: dict[str, int]) -> list[str]:
    out = []
    for c in CIRCUITS:
        p = pos.get(c["id"])
        out.append("—" if p is None else (f"{p}° ({points_for(p)})" if points_for(p) else f"{p}°"))
    return out


def main():
    results = load_results()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# 🏆 Leaderboard — Campeonato RL (DQN)",
        "",
        f"Entorno base: `{CFG['env_id']}` · {len(CIRCUITS)} circuitos · "
        f"Ranking por **{METRIC_LABEL}** en cada circuito · Actualizado: {now}",
        "",
        f"Puntos por circuito: {', '.join(f'{p}' for p in POINTS)} (del 1° al {len(POINTS)}°). "
        "La misma policy corre en todos los circuitos. Detalle de cada uno en el README.",
        "",
        "**Un auto titular por escudería** (cuenta de GitHub): es el único que suma puntos y "
        "ocupa posiciones. Es el auto de la escudería que más puntos suma contra los titulares "
        "de las demás. El resto son autos de reserva: aparecen abajo, pero no le sacan "
        "puestos a nadie.",
        "",
    ]
    if not results:
        lines.append("_Sin submissions todavía._")
        (ROOT / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
        print("Leaderboard regenerado con 0 equipo(s).")
        return

    refs = [r for r in results if r.get("referencia")]
    results = [r for r in results if not r.get("referencia")]
    owner = escuderias(ROOT, manual=CFG.get("escuderias"))
    for r in results:
        if r["team"] not in owner:
            print(f"Aviso: no encontré el PR que creó submissions/{r['team']}/; "
                  "cuenta como escudería propia (se corrige en config.yaml -> escuderias).")
            owner[r["team"]] = r["team"]

    tit = pick_titulares(results, owner)
    titulares = list(tit.values())
    is_tit = {r["team"] for r in titulares}
    reservas = [r for r in results if r["team"] not in is_tit]
    n = len(results)

    # Posiciones reales: solo corren los titulares.
    pos = {r["team"]: {} for r in titulares}
    ranked = {}
    for c in CIRCUITS:
        ranked[c["id"]] = rank_circuit(titulares, c["id"])
        for p, r in ranked[c["id"]]:
            pos[r["team"]][c["id"]] = p

    # Campeonato
    teams = sorted(pos, key=lambda t: (champ_key(pos[t], n), t))
    short = [c["name"].replace("GP ", "") for c in CIRCUITS]
    lines += [
        "## 🏁 Campeonato",
        "",
        "| # | Auto | Escudería | **Pts** | " + " | ".join(short) + " |",
        "|---|------|-----------|-----:|" + "|".join(":---:" for _ in CIRCUITS) + "|",
    ]
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    prev_key, rank = None, 0
    for i, t in enumerate(teams, 1):
        k = champ_key(pos[t], n)
        if k != prev_key:
            rank, prev_key = i, k
        lines.append(f"| {medal.get(rank, rank)} | {t} | {owner[t]} | **{-k[0]}** | " + " | ".join(cells(pos[t])) + " |")

    # Referencia: fuera de concurso, contra todos los titulares.
    if refs:
        lines += [
            "",
            "## 🚔 Auto de referencia (fuera de concurso)",
            "",
            "No compite: no suma ni le saca puestos a nadie. Las posiciones son las que habría "
            "sacado contra los titulares.",
            "",
            "| Auto | Pts que habría sumado | " + " | ".join(short) + " |",
            "|------|-----:|" + "|".join(":---:" for _ in CIRCUITS) + "|",
        ]
        for r in refs:
            hp = positions_vs(r, titulares)
            lines.append(f"| {r['team']} | {-champ_key(hp, n)[0]} | " + " | ".join(cells(hp)) + " |")

    # Reserva: qué haría cada auto si fuera el titular de su escudería.
    if reservas:
        hyp = {r["team"]: positions_vs(r, [t for o, t in tit.items() if o != owner[r["team"]]])
               for r in reservas}
        lines += [
            "",
            "## 🔧 Autos de reserva",
            "",
            "No suman ni ocupan posiciones. Las posiciones son las que tendría cada auto si fuera "
            "el titular de su escudería (contra los titulares de las demás).",
            "",
            "| Auto | Escudería | Pts como titular | " + " | ".join(short) + " |",
            "|------|-----------|-----:|" + "|".join(":---:" for _ in CIRCUITS) + "|",
        ]
        for r in sorted(reservas, key=lambda r: (owner[r["team"]].lower(), champ_key(hyp[r["team"]], n), r["team"])):
            t = r["team"]
            lines.append(f"| {t} | {owner[t]} | {-champ_key(hyp[t], n)[0]} | " + " | ".join(cells(hyp[t])) + " |")

    # Detalle por circuito: titulares con posición; reservas intercaladas con "R".
    for c in CIRCUITS:
        cid = c["id"]
        rows = ranked[cid]
        if not rows:
            continue
        leader = rows[0][1]["circuits"][cid]["mean"]
        all_rows = ([(p, r) for p, r in rows] + [(None, r) for r in reservas if cid in r["circuits"]]
                    + [("REF", r) for r in refs if cid in r["circuits"]])
        all_rows.sort(key=lambda pr: (-pr[1]["circuits"][cid][METRIC], not isinstance(pr[0], int)))
        lines += [
            "",
            f"## {c['name']}",
            "",
            f"_{c.get('desc', '')}_",
            "",
            "| # | Auto | Escudería | Pts | Media | IQM | Sin llegar | Min | Max | Episodios | Dif. al 1° |",
            "|---|------|-----------|----:|-------|-----|-----------:|-----|-----|-----------|-----------|",
        ]
        for p, r in all_rows:
            m = r["circuits"][cid]
            d = leader - m["mean"]
            gap = "—" if p == 1 else f"{'+' if d >= 0 else '−'}{abs(d):.3f}"
            lines.append(
                f"| {p if p else 'R'} | {r['team']} | {owner.get(r['team'], 'fuera de concurso')} | "
                f"{points_for(p) if isinstance(p, int) else '—'} | "
                f"{m['mean']:.3f} | {m['iqm']:.3f} | {100 * m['fail_rate']:.1f}% | {m['min']:.1f} | "
                f"{m['max']:.1f} | {r['n_episodes']} | {gap} |"
            )

    (ROOT / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print(f"Leaderboard regenerado con {len(titulares)} titular(es) y {len(reservas)} auto(s) de reserva.")


if __name__ == "__main__":
    main()
