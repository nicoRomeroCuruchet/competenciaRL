#!/usr/bin/env python3
"""
Evalúa un policy.onnx sobre un conjunto de seeds.

- Política: accion = argmax(Q(obs))  (greedy, determinista).
- Una corrida (episodio) por seed -> con política y entorno deterministas,
  más seeds = más muestras. La varianza entre seeds es la señal real en RL.
- Métrica de ranking: media de los retornos, con IC 95% por bootstrap.
  (También se reporta el IQM —media del 50% central— como dato informativo.)
- IMPORTANTE: nunca imprime los valores de las seeds (son secretas).

Las seeds se pasan por la variable de entorno EVAL_SEEDS (lista separada por
comas), que en el workflow viene de un GitHub Secret. Si no está, usa
seeds/public_seeds.txt (las de práctica).

Uso:
    EVAL_SEEDS="11,22,33" python evaluate.py --onnx policy.onnx --team mi_equipo
"""
import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def load_seeds() -> list[int]:
    env_seeds = os.environ.get("EVAL_SEEDS", "").strip()
    if env_seeds:
        return [int(s) for s in env_seeds.split(",") if s.strip()]
    public = ROOT / "seeds" / "public_seeds.txt"
    if public.exists():
        return [int(s) for s in public.read_text().split() if s.strip()]
    raise SystemExit("No hay seeds: definí EVAL_SEEDS o seeds/public_seeds.txt")


def run_episode(sess, input_name, output_name, n_actions, seed: int) -> float:
    env = gym.make(CFG["env_id"])
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(CFG["max_steps_per_episode"]):
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        q = sess.run([output_name], {input_name: x})[0]
        action = int(np.argmax(q[0]))
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    env.close()
    return total


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: media de los valores entre el percentil 25 y 75."""
    if len(x) == 0:
        return float("nan")
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    mid = x[(x >= lo) & (x <= hi)]
    return float(mid.mean()) if len(mid) else float(np.median(x))


def bootstrap_ci(x: np.ndarray, stat, n_resamples: int, alpha=0.05):
    rng = np.random.default_rng(0)  # IC reproducible
    boot = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_resamples)]
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--team", required=True)
    p.add_argument("--out", default=None, help="ruta del result.json de salida")
    args = p.parse_args()

    seeds = load_seeds()
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    returns = np.array(
        [run_episode(sess, CFG["input_name"], CFG["output_name"],
                     CFG["n_actions"], s) for s in seeds],
        dtype=np.float64,
    )

    score_iqm = iqm(returns)
    ci_iqm = bootstrap_ci(returns, iqm, CFG["bootstrap_resamples"])
    ci_mean = bootstrap_ci(returns, np.mean, CFG["bootstrap_resamples"])
    result = {
        "team": args.team,
        "n_episodes": len(returns),
        "mean": float(returns.mean()),
        "mean_ci95": [round(ci_mean[0], 3), round(ci_mean[1], 3)],
        "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "iqm": score_iqm,
        "iqm_ci95": [round(ci_iqm[0], 3), round(ci_iqm[1], 3)],
        "min": float(returns.min()),
        "max": float(returns.max()),
    }
    # No imprimimos seeds ni el detalle por-seed (info de las seeds ocultas).
    print(json.dumps(result, indent=2))

    out = args.out or str(ROOT / "results" / f"{args.team}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
