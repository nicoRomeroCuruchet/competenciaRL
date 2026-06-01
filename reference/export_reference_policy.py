#!/usr/bin/env python3
"""
export_reference_policy.py — convierte la solucion del profe (Policy Iteration por
DP en CUDA, guardada como .npz) en un policy.onnx que cumple el contrato de la
competencia y que REPRODUCE EXACTAMENTE su evaluacion baricentrica
(utils/barycentric.get_optimal_action + round(a+1)).

NO es una submission de alumno: vive fuera de submissions/ y por eso el validador
("solo pesos") nunca lo ve. Se corre UNA vez, offline. Solo numpy + onnx
(+ onnxruntime para el self-check). Ni torch, ni numba, ni CUDA.

Que hace el ONNX en inferencia (igual que get_optimal_action)
-------------------------------------------------------------
Para cada obs = [pos, vel]:
  1. Interpolacion baricentrica sobre la celda de grilla que la contiene:
       - base[d] = clip(floor((obs[d]-low[d])/step[d]), 0, n[d]-2)
       - t[d]    = (obs[d]-low[d])/step[d] - base[d]            (en [0,1])
       - para las 4 esquinas c: w_c = prod_d (t[d] si bit else 1-t[d]),
                                 flat_c = sum_d (base[d]+bit_d)*stride[d]
  2. accion continua  a = sum_c w_c * accion_de_la_policy(flat_c)   (en [-1,1])
  3. indice discreto  k = clip(round(a+1), 0, 2)
  4. q_values = one_hot(k)  ->  argmax(q_values) == k  (= action_to_gym del runner)

Todo con operadores estandar de ONNX (Max/Min/Floor/Sub/Mul/Cast/Gather/Add/
Round/Unsqueeze/Equal), opset 17.

Uso
---
    python competenciaRL/reference/export_reference_policy.py \
        --npz runners/results/mountain_car_cuda_policy.npz \
        --out competenciaRL/submissions/profe/policy.onnx
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# ── Referencia en numpy (replica get_optimal_action + round) para el self-check ──
def ref_actions(obs, pol_val, low, high, grid_shape, strides, corners):
    p = np.clip(obs.astype(np.float64), low, high)
    step = (high - low) / (grid_shape - 1)
    cell = (p - low) / step
    base = np.floor(cell).astype(np.int64)
    base = np.clip(base, 0, grid_shape - 2)
    t = cell - base
    a = np.zeros(len(obs), dtype=np.float64)
    for bits in corners:
        w = np.ones(len(obs), dtype=np.float64)
        flat = np.zeros(len(obs), dtype=np.int64)
        for d, bit in enumerate(bits):
            w *= t[:, d] if bit else (1.0 - t[:, d])
            flat += (base[:, d] + bit) * strides[d]
        a += w * pol_val[flat]
    return np.clip(np.round(a + 1.0), 0, 2).astype(np.int64)


# ── Construccion del grafo ONNX (baricentrica exacta) ──────────────────────────
def build_onnx(npz_path: Path, out_path: Path, opset: int = 17):
    d = np.load(npz_path, allow_pickle=True)
    policy = d["policy"].astype(np.int64)                  # [N] indice de accion por celda
    action_space = d["action_space"].astype(np.float32)   # [3] = [-1,0,1]
    low = d["bounds_low"].astype(np.float64)               # [2]
    high = d["bounds_high"].astype(np.float64)             # [2]
    grid_shape = d["grid_shape"].astype(np.int64)          # [2]
    strides = d["strides"].astype(np.int64)                # [2]

    # Valor de accion (-1/0/1) por celda: un solo Gather en vez de policy + action_space.
    pol_val = action_space[policy].astype(np.float32)      # [N]
    inv_step = ((grid_shape - 1).astype(np.float32) / (high - low).astype(np.float32))
    grid_minus2 = (grid_shape - 2).astype(np.float32)
    low_f = low.astype(np.float32)
    high_f = high.astype(np.float32)
    s0, s1 = int(strides[0]), int(strides[1])
    corners = list(product([0, 1], repeat=2))              # (0,0),(0,1),(1,0),(1,1)

    def c(name, arr):
        return numpy_helper.from_array(np.asarray(arr), name)

    inits = [
        c("pol_val", pol_val),
        c("low", low_f), c("high", high_f),
        c("inv_step", inv_step), c("grid_minus2", grid_minus2),
        c("zero_f", np.array(0.0, np.float32)),
        c("one_f", np.array(1.0, np.float32)),
        c("two_f", np.array(2.0, np.float32)),
        c("col0", np.array(0, np.int64)), c("col1", np.array(1, np.int64)),
        c("stride0", np.array(s0, np.int64)), c("stride1", np.array(s1, np.int64)),
        c("arange3", np.array([0.0, 1.0, 2.0], np.float32)),
        c("ax1", np.array([1], np.int64)),
    ]

    n = []  # nodos
    # 1) clip a los bounds (per-dim => Max/Min con broadcast, no Clip que pide escalares)
    n += [helper.make_node("Max", ["observation", "low"], ["p_lo"]),
          helper.make_node("Min", ["p_lo", "high"], ["p"])]
    # 2) cell = (p - low) * inv_step ;  base = clip(floor(cell), 0, n-2) ;  t = cell - base
    n += [helper.make_node("Sub", ["p", "low"], ["shift"]),
          helper.make_node("Mul", ["shift", "inv_step"], ["cell"]),
          helper.make_node("Floor", ["cell"], ["base_fl"]),
          helper.make_node("Max", ["base_fl", "zero_f"], ["base_lo"]),
          helper.make_node("Min", ["base_lo", "grid_minus2"], ["base_f"]),
          helper.make_node("Sub", ["cell", "base_f"], ["t"]),
          helper.make_node("Cast", ["base_f"], ["base_i"], to=TensorProto.INT64)]
    # columnas
    n += [helper.make_node("Gather", ["t", "col0"], ["t0"], axis=1),
          helper.make_node("Gather", ["t", "col1"], ["t1"], axis=1),
          helper.make_node("Sub", ["one_f", "t0"], ["mt0"]),   # 1 - t0
          helper.make_node("Sub", ["one_f", "t1"], ["mt1"]),   # 1 - t1
          helper.make_node("Gather", ["base_i", "col0"], ["b0"], axis=1),
          helper.make_node("Gather", ["base_i", "col1"], ["b1"], axis=1),
          helper.make_node("Mul", ["b0", "stride0"], ["b0s"]),
          helper.make_node("Mul", ["b1", "stride1"], ["b1s"]),
          helper.make_node("Add", ["b0s", "b1s"], ["base_flat"])]

    # 3) 4 esquinas: w_c * pol_val(flat_c), acumulando en interp_a
    term_names = []
    for ci, (bit0, bit1) in enumerate(corners):
        f0 = "t0" if bit0 else "mt0"
        f1 = "t1" if bit1 else "mt1"
        const_c = bit0 * s0 + bit1 * s1
        inits.append(c(f"off{ci}", np.array(const_c, np.int64)))
        n += [helper.make_node("Mul", [f0, f1], [f"w{ci}"]),
              helper.make_node("Add", ["base_flat", f"off{ci}"], [f"flat{ci}"]),
              helper.make_node("Gather", ["pol_val", f"flat{ci}"], [f"pv{ci}"], axis=0),
              helper.make_node("Mul", [f"w{ci}", f"pv{ci}"], [f"term{ci}"])]
        term_names.append(f"term{ci}")
    n += [helper.make_node("Add", [term_names[0], term_names[1]], ["s01"]),
          helper.make_node("Add", [term_names[2], term_names[3]], ["s23"]),
          helper.make_node("Add", ["s01", "s23"], ["interp_a"])]

    # 4) k = clip(round(a+1), 0, 2) ;  q_values = one_hot(k)
    n += [helper.make_node("Add", ["interp_a", "one_f"], ["a1"]),
          helper.make_node("Round", ["a1"], ["k0"]),
          helper.make_node("Max", ["k0", "zero_f"], ["k_lo"]),
          helper.make_node("Min", ["k_lo", "two_f"], ["k"]),
          helper.make_node("Unsqueeze", ["k", "ax1"], ["k_col"]),
          helper.make_node("Equal", ["k_col", "arange3"], ["oh"]),
          helper.make_node("Cast", ["oh"], ["q_values"], to=TensorProto.FLOAT)]

    inp = helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch", 2])
    out = helper.make_tensor_value_info("q_values", TensorProto.FLOAT, ["batch", 3])
    graph = helper.make_graph(n, "mountaincar_barycentric_policy", [inp], [out], inits)
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid("", opset)],
                              producer_name="export_reference_policy")
    model.ir_version = 10
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"[ok] ONNX guardado en {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")

    # ── self-check end-to-end: ONNX argmax vs referencia numpy de get_optimal_action ──
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    obs = np.column_stack([
        rng.uniform(low[0], high[0], 20000),
        rng.uniform(low[1], high[1], 20000),
    ]).astype(np.float32)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    q = sess.run(["q_values"], {"observation": obs})[0]
    onnx_k = q.argmax(axis=1)
    ref_k = ref_actions(obs, pol_val, low, high, grid_shape, strides, corners)
    agree = float((onnx_k == ref_k).mean())
    print(f"[self-check] argmax(ONNX) == get_optimal_action+round en {agree*100:.2f}% "
          f"de 20k puntos aleatorios")
    if agree < 1.0:
        bad = int((onnx_k != ref_k).sum())
        raise SystemExit(f"[ERROR] el ONNX NO replica la baricentrica ({bad} puntos difieren)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path,
                   default=Path("runners/results/mountain_car_cuda_policy.npz"))
    p.add_argument("--out", type=Path,
                   default=Path("competenciaRL/submissions/profe/policy.onnx"))
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    build_onnx(args.npz, args.out, args.opset)


if __name__ == "__main__":
    main()
