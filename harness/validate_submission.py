#!/usr/bin/env python3
"""
Valida una submission. Falla (exit != 0) si:
  1) El PR toca cualquier archivo que no sea submissions/<equipo>/policy.onnx
     o submissions/<equipo>/metadata.json  -> "código = descalificado".
  2) El .onnx no cumple el contrato (nombres, shapes, dtype, opset, ops estándar).

Uso:
    python validate_submission.py --changed-files "a.py b.onnx ..." --onnx ruta/policy.onnx

En el workflow, --changed-files se arma con el diff del PR contra main.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

# Solo se permiten estos dos archivos, y SOLO dentro de submissions/<equipo>/
ALLOWED_RE = re.compile(r"^submissions/[A-Za-z0-9_\-]+/(policy\.onnx|metadata\.json)$")

# Extensiones que delatan código -> descalificación inmediata.
CODE_EXTS = {
    ".py", ".ipynb", ".sh", ".bash", ".js", ".ts", ".rb", ".pl", ".c", ".cpp",
    ".java", ".go", ".rs", ".pyc", ".so", ".dll", ".exe", ".bat", ".ps1",
    ".yml", ".yaml", ".cfg", ".toml", ".mk", ".dockerfile",
}


def fail(msg: str) -> None:
    print(f"::error::DESCALIFICADO / SUBMISSION INVÁLIDA\n{msg}")
    sys.exit(1)


def check_changed_files(changed: list[str]) -> None:
    """Regla dura: el PR solo puede agregar/modificar el .onnx y el metadata."""
    bad_code, bad_other = [], []
    for f in changed:
        f = f.strip()
        if not f:
            continue
        if ALLOWED_RE.match(f):
            continue
        if Path(f).suffix.lower() in CODE_EXTS:
            bad_code.append(f)
        else:
            bad_other.append(f)

    if bad_code:
        fail(
            "Se detectó CÓDIGO en la submission. La competencia es solo de pesos.\n"
            "Archivos prohibidos:\n  - " + "\n  - ".join(bad_code)
        )
    if bad_other:
        fail(
            "El PR toca archivos fuera de submissions/<equipo>/.\n"
            "Solo se permite policy.onnx (y opcional metadata.json).\n"
            "Archivos no permitidos:\n  - " + "\n  - ".join(bad_other)
        )


def check_onnx(path: str) -> None:
    """El .onnx debe ser un grafo puro estándar que cumpla el contrato."""
    model = onnx.load(path)

    # 1) Estructura válida.
    try:
        onnx.checker.check_model(model)
    except Exception as e:  # noqa: BLE001
        fail(f"El ONNX no pasa el checker: {e}")

    # 2) Solo dominios estándar (sin custom ops que ejecuten cosas raras).
    for imp in model.opset_import:
        if imp.domain not in ("", "ai.onnx"):
            fail(f"Dominio de operadores no permitido: '{imp.domain}'. "
                 "Solo se aceptan operadores estándar de ONNX.")

    # 3) Una entrada y una salida con los nombres del contrato.
    inputs = model.graph.input
    outputs = model.graph.output
    if len(inputs) != 1 or len(outputs) != 1:
        fail(f"Se esperaba 1 entrada y 1 salida; hay {len(inputs)} y {len(outputs)}.")
    if inputs[0].name != CFG["input_name"]:
        fail(f"La entrada debe llamarse '{CFG['input_name']}', no '{inputs[0].name}'.")
    if outputs[0].name != CFG["output_name"]:
        fail(f"La salida debe llamarse '{CFG['output_name']}', no '{outputs[0].name}'.")

    # 4) Prueba de inferencia: shape correcta y dtype float32.
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    obs = np.zeros((1, CFG["obs_dim"]), dtype=np.float32)
    try:
        out = sess.run([CFG["output_name"]], {CFG["input_name"]: obs})[0]
    except Exception as e:  # noqa: BLE001
        fail(f"No se pudo correr inferencia con obs float32 [1,{CFG['obs_dim']}]: {e}")
    if out.shape[-1] != CFG["n_actions"]:
        fail(f"La salida debe tener {CFG['n_actions']} Q-values por acción; "
             f"tiene shape {out.shape}.")
    if out.dtype != np.float32:
        fail(f"La salida debe ser float32; es {out.dtype}.")

    print(f"OK: ONNX válido y conforme al contrato "
          f"(in '{CFG['input_name']}' [*, {CFG['obs_dim']}] -> "
          f"out '{CFG['output_name']}' [*, {CFG['n_actions']}]).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--changed-files", default="", help="lista separada por espacios")
    p.add_argument("--onnx", required=True, help="ruta al policy.onnx a validar")
    args = p.parse_args()

    if args.changed_files:
        check_changed_files(args.changed_files.split())
    check_onnx(args.onnx)
    print("Submission VÁLIDA.")
