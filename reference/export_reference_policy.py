#!/usr/bin/env python3
"""
export_reference_policy.py — convierte la solucion EXACTA del profe (Policy
Iteration por DP en CUDA, guardada como .npz) en un policy.onnx que cumple el
contrato de la competencia, para que figure como LINEA DE REFERENCIA ("el optimo
a batir") en el leaderboard.

NO es una submission de alumno: vive fuera de submissions/ y por eso el validador
("solo pesos") nunca lo ve. Se corre UNA vez, offline, en la maquina del profe.
Solo necesita numpy + onnx (ni torch, ni numba, ni CUDA).

Que hace
--------
1. Lee el .npz del DP: value_function V, policy, grilla, bounds, action_space.
2. Reconstruye una Q-table [n_states, 3] por lookahead de 1 paso con la dinamica
   ANALITICA exacta de MountainCar:
       Q[s,a] = -1 + gamma * V_interp(step(s, a))      (= -1 si step cae en terminal)
   donde V_interp es interpolacion baricentrica de V (la misma que usa el DP).
   Asi argmax_a Q reproduce la politica optima y ademas da Q-values calibrados.
3. Hornea esa Q-table como constante dentro de un grafo ONNX minimo cuyo unico
   trabajo en inferencia es: obs -> celda de grilla mas cercana -> Gather -> q_values.
   Solo operadores estandar (Sub, Mul, Round, Clip, Cast, ReduceSum, Gather), opset 17.

Uso
---
    python competenciaRL/reference/export_reference_policy.py \
        --npz runners/results/mountain_car_cuda_policy.npz \
        --out competenciaRL/submissions/profe/policy.onnx \
        --gamma 0.99
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ── Dinamica analitica exacta de MountainCar-v0 (igual al kernel CUDA del runner) ──
def step_dynamics(pos, vel, force):
    """Un paso de MountainCar, vectorizado. force in {-1, 0, 1}."""
    vel = vel + force * 0.001 - 0.0025 * np.cos(3.0 * pos)
    vel = np.clip(vel, -0.07, 0.07)
    pos = pos + vel
    pos = np.clip(pos, -1.2, 0.6)
    # rebote pared izquierda: si toca el limite, velocidad a 0
    vel = np.where(pos <= -1.2, 0.0, vel)
    terminated = (pos >= 0.5) & (vel >= 0.0)
    return pos.astype(np.float32), vel.astype(np.float32), terminated


# ── Interpolacion baricentrica de V (vectorizada, equivalente a utils/barycentric) ──
def interp_value(points, V, low, high, grid_shape, strides, corner_bits):
    """V interpolado en `points` [M,2] por baricentrica sobre la grilla regular."""
    M = points.shape[0]
    n_dims = points.shape[1]
    step = (high - low) / (grid_shape - 1)              # [2]

    p = np.clip(points, low, high)                       # [M,2]
    cell = (p - low) / step                              # [M,2]
    base = np.floor(cell).astype(np.int64)               # [M,2]
    base = np.minimum(base, grid_shape - 2)
    base = np.maximum(base, 0)
    t = (p - (low + base * step)) / step                 # [M,2] en [0,1]

    out = np.zeros(M, dtype=np.float64)
    for c in range(corner_bits.shape[0]):                # 4 esquinas del hipercubo
        bits = corner_bits[c]                            # [2] en {0,1}
        w = np.ones(M, dtype=np.float64)
        flat = np.zeros(M, dtype=np.int64)
        for d in range(n_dims):
            w *= t[:, d] if bits[d] else (1.0 - t[:, d])
            flat += (base[:, d] + bits[d]) * strides[d]
        out += w * V[flat]
    return out


def build_q_table(npz_path: Path, gamma: float):
    d = np.load(npz_path, allow_pickle=True)
    V = d["value_function"].astype(np.float64)           # [N]
    policy = d["policy"].astype(np.int64)                # [N]
    states = d["states_space"].astype(np.float32)        # [N,2]
    low = d["bounds_low"].astype(np.float64)             # [2]
    high = d["bounds_high"].astype(np.float64)           # [2]
    grid_shape = d["grid_shape"].astype(np.int64)        # [2]
    strides = d["strides"].astype(np.int64)              # [2]
    corner_bits = d["corner_bits"].astype(np.int64)      # [4,2]
    action_space = d["action_space"].astype(np.float32)  # [3] = [-1,0,1]
    N = states.shape[0]
    n_actions = action_space.shape[0]

    # Sanity: el layout de la grilla coincide con la indexacion por strides.
    # idx(states[k]) debe dar k para toda celda.
    rec = np.round((states.astype(np.float64) - low) / ((high - low) / (grid_shape - 1)))
    rec = np.clip(rec, 0, grid_shape - 1).astype(np.int64)
    flat_rec = (rec * strides).sum(axis=1)
    assert np.array_equal(flat_rec, np.arange(N)), \
        "El mapeo obs->celda no coincide con states_space; revisar strides/bounds."

    # Q[s,a] = -1 + gamma * V_interp(step(s,a)),  o -1 si step es terminal.
    Q = np.full((N, n_actions), -1.0, dtype=np.float64)
    pos0, vel0 = states[:, 0], states[:, 1]
    for a in range(n_actions):
        force = float(action_space[a])
        npos, nvel, term = step_dynamics(pos0.copy(), vel0.copy(), force)
        nxt = np.column_stack([npos, nvel]).astype(np.float64)
        Vn = interp_value(nxt, V, low, high, grid_shape, strides, corner_bits)
        Q[:, a] = -1.0 + gamma * np.where(term, 0.0, Vn)

    agree = float((Q.argmax(axis=1) == policy).mean())
    print(f"[sanity] argmax(Q) coincide con la policy del DP en {agree*100:.2f}% de las celdas")
    if agree < 0.95:
        print("[warn] coincidencia baja: revisar gamma/dinamica/interpolacion.")

    meta = dict(low=low.astype(np.float32), high=high.astype(np.float32),
                grid_shape=grid_shape.astype(np.int64),
                strides=strides.astype(np.int64))
    return Q.astype(np.float32), meta


# ── Construccion del grafo ONNX: obs -> celda mas cercana -> Gather(Q) ──────────────
def build_onnx(Q, meta, out_path: Path, opset: int = 17):
    N, n_actions = Q.shape
    low = meta["low"]
    high = meta["high"]
    grid_shape = meta["grid_shape"]            # [2] int64
    strides = meta["strides"]                  # [2] int64
    inv_step = ((grid_shape - 1).astype(np.float32) / (high - low)).astype(np.float32)
    nbins_m1 = float(grid_shape[0] - 1)        # ambas dims = 200 -> 199; clip escalar sirve

    def const(name, arr):
        return numpy_helper.from_array(np.asarray(arr), name=name)

    initializers = [
        const("Qtable", Q),                                  # [N,3] float32
        const("low", low),                                   # [2]
        const("inv_step", inv_step),                         # [2]
        const("strides", strides),                           # [2] int64
        const("clip_min", np.array(0.0, dtype=np.float32)),
        const("clip_max", np.array(nbins_m1, dtype=np.float32)),
        const("sum_axis", np.array([1], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("Sub", ["observation", "low"], ["shifted"]),
        helper.make_node("Mul", ["shifted", "inv_step"], ["scaled"]),
        helper.make_node("Round", ["scaled"], ["rounded"]),
        helper.make_node("Clip", ["rounded", "clip_min", "clip_max"], ["clipped"]),
        helper.make_node("Cast", ["clipped"], ["idx"], to=TensorProto.INT64),
        helper.make_node("Mul", ["idx", "strides"], ["weighted"]),
        helper.make_node("ReduceSum", ["weighted", "sum_axis"], ["flat"], keepdims=0),
        helper.make_node("Gather", ["Qtable", "flat"], ["q_values"], axis=0),
    ]

    inp = helper.make_tensor_value_info(
        "observation", TensorProto.FLOAT, ["batch", 2])
    out = helper.make_tensor_value_info(
        "q_values", TensorProto.FLOAT, ["batch", n_actions])

    graph = helper.make_graph(nodes, "mountaincar_reference_policy",
                              [inp], [out], initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", opset)],
        producer_name="export_reference_policy")
    model.ir_version = 10  # compatible con onnxruntime / opset 17

    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"[ok] ONNX guardado en {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path,
                   default=Path("runners/results/mountain_car_cuda_policy.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("competenciaRL/submissions/profe/policy.onnx"))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    Q, meta = build_q_table(args.npz, args.gamma)
    build_onnx(Q, meta, args.out, args.opset)


if __name__ == "__main__":
    main()
