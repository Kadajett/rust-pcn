#!/usr/bin/env python3
"""How much have the fresh run's byte-output columns moved since a saved reference, and are they collapsing?

Reads only the 257 byte columns of each expert's top weight matrix (about 9.5 MB each); cheap on CPU.
usage: river_bytecol_drift.py REFERENCE.npz [CHECKPOINT_ROOT]
"""

import json
import pathlib
import sys

import numpy as np

reference = np.load(sys.argv[1])
root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
                    "/bulk-storage/connectome-merc/river-universal-checkpoints/river-v7-fresh-seed20261002")
manifest = json.loads((root / "experts.json").read_text())
report = {"batch": manifest["cumulative_batches"], "reference": sys.argv[1]}
for expert in manifest["experts"]:
    snapshot = root / expert["checkpoint"]
    dims = json.loads((snapshot / "checkpoint.json").read_text())["dimensions"]
    offset = 16 + 4 * sum(a * b for a, b in list(zip(dims, dims[1:]))[:-1])
    top = np.memmap(snapshot / "pcn-weights.bin", dtype="<f4", mode="r", offset=offset, shape=(dims[-2], dims[-1]))
    now = np.asarray(top[:, 259:516], dtype=np.float64)
    before = reference[expert["role"]].astype(np.float64)
    delta = now - before
    eigen = np.linalg.eigvalsh(now.T @ now)
    delta_eigen = np.linalg.eigvalsh(delta.T @ delta)
    report[expert["role"]] = {
        "relative_change": float(np.linalg.norm(delta) / np.linalg.norm(before)),
        "column_norm2_median": float(np.median(np.einsum("ij,ij->j", now, now))),
        "rank1_fraction": float(eigen[-1] / eigen.sum()),
        "change_rank1_fraction": float(delta_eigen[-1] / delta_eigen.sum()) if delta_eigen.sum() > 0 else None,
    }
print(json.dumps(report, indent=2))
