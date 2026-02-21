#!/usr/bin/env python3
"""Live terminal dashboard for PCN training: baseline vs SEAL."""

import json
import os
import time

import plotext as plt

BASELINE = "data/output/metrics-baseline.jsonl"
SEAL = "data/output/metrics-seal.jsonl"
REFRESH = 5  # seconds


def load_epochs(path):
    epochs = []
    if not os.path.exists(path):
        return epochs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "epoch":
                epochs.append(rec)
    return epochs


def make_x(epochs):
    return [(e.get("round", 1) - 1) * 20 + e.get("epoch", 1) for e in epochs]


def draw(term_w, term_h):
    baseline = load_epochs(BASELINE)
    seal = load_epochs(SEAL)

    if not baseline and not seal:
        plt.clear_terminal()
        print("Waiting for metrics...")
        return

    bx = make_x(baseline)
    sx = make_x(seal)

    b_energy = [e["avg_energy"] for e in baseline]
    s_energy = [e["avg_energy"] for e in seal]

    # Accuracy and layer errors only exist on eval epochs (--eval-every)
    b_eval = [e for e in baseline if "accuracy" in e]
    s_eval = [e for e in seal if "accuracy" in e]
    bx_eval = make_x(b_eval)
    sx_eval = make_x(s_eval)

    b_acc = [e["accuracy"] * 100 for e in b_eval]
    s_acc = [e["accuracy"] * 100 for e in s_eval]

    b_err0 = [e["layer_errors"][0] for e in b_eval if e.get("layer_errors")]
    s_err0 = [e["layer_errors"][0] for e in s_eval if e.get("layer_errors")]
    b_err1 = [e["layer_errors"][1] for e in b_eval if e.get("layer_errors")]
    s_err1 = [e["layer_errors"][1] for e in s_eval if e.get("layer_errors")]
    bx_err = make_x([e for e in b_eval if e.get("layer_errors")])
    sx_err = make_x([e for e in s_eval if e.get("layer_errors")])

    # SEAL-specific
    s_mod0 = [e.get("seal", {}).get("modulation", [1, 1, 1])[0] for e in seal]
    s_mod1 = [e.get("seal", {}).get("modulation", [1, 1, 1])[1] for e in seal]
    s_var0 = [e.get("seal", {}).get("error_variance", [0, 0, 0])[0] for e in seal]
    s_var1 = [e.get("seal", {}).get("error_variance", [0, 0, 0])[1] for e in seal]

    # Latest stats for footer
    bl = baseline[-1] if baseline else {}
    sl = seal[-1] if seal else {}

    # Plot area: leave 4 lines for footer
    plot_h = term_h - 4

    plt.clear_figure()
    plt.theme("dark")
    plt.plot_size(term_w, max(plot_h, 20))
    plt.subplots(2, 3)

    # ── Row 1, Col 1: Energy ──
    plt.subplot(1, 1)
    if bx:
        plt.plot(bx, b_energy, label="baseline", color="blue")
    if sx:
        plt.plot(sx, s_energy, label="SEAL", color="red")
    plt.title("Avg Energy")
    plt.xlabel("epoch")

    # ── Row 1, Col 2: Accuracy ──
    plt.subplot(1, 2)
    if bx_eval:
        plt.plot(bx_eval, b_acc, label="baseline", color="blue")
    if sx_eval:
        plt.plot(sx_eval, s_acc, label="SEAL", color="red")
    plt.title("Accuracy (%)")
    plt.xlabel("epoch")

    # ── Row 1, Col 3: Layer Errors ──
    plt.subplot(1, 3)
    if bx_err:
        plt.plot(bx_err, b_err0, label="base L0", color="blue")
    if sx_err:
        plt.plot(sx_err, s_err0, label="SEAL L0", color="red")
    if bx_err:
        plt.plot(bx_err, b_err1, label="base L1", color="cyan")
    if sx_err:
        plt.plot(sx_err, s_err1, label="SEAL L1", color="magenta")
    plt.title("Layer Errors")
    plt.xlabel("epoch")

    # ── Row 2, Col 1: SEAL Modulation ──
    plt.subplot(2, 1)
    if sx:
        plt.plot(sx, s_mod0, label="L0 mod", color="yellow")
        plt.plot(sx, s_mod1, label="L1 mod", color="green")
        plt.hline(1.0, color="gray")
    plt.title("SEAL Modulation")
    plt.xlabel("epoch")

    # ── Row 2, Col 2: SEAL Error Variance ──
    plt.subplot(2, 2)
    if sx:
        plt.plot(sx, s_var0, label="L0 var", color="yellow")
        plt.plot(sx, s_var1, label="L1 var", color="green")
    plt.title("SEAL Error Variance")
    plt.xlabel("epoch")

    # ── Row 2, Col 3: Energy Delta (SEAL - Baseline) ──
    plt.subplot(2, 3)
    # Align by epoch index (min length)
    n = min(len(b_energy), len(s_energy))
    if n > 0:
        delta = [s_energy[i] - b_energy[i] for i in range(n)]
        dx = bx[:n]
        plt.plot(dx, delta, label="SEAL-base", color="red")
        plt.hline(0.0, color="gray")
    plt.title("Energy Delta (SEAL-Base)")
    plt.xlabel("epoch")

    plt.show()

    # ── Footer: live stats ──
    ts = time.strftime("%H:%M:%S")
    br = bl.get("round", "?")
    be = bl.get("epoch", "?")
    sr = sl.get("round", "?")
    se = sl.get("epoch", "?")

    seal_data = sl.get("seal", {})
    mod_str = ""
    if seal_data:
        mod = seal_data.get("modulation", [])
        mod_str = f"  mod=[{', '.join(f'{m:.3f}' for m in mod)}]"

    line1 = (
        f" [{ts}]  "
        f"BASELINE r{br}e{be}  energy={bl.get('avg_energy', 0):.4f}  acc={bl.get('accuracy', 0)*100:.2f}%  best={bl.get('best_accuracy', 0)*100:.2f}%  "
        f"|  "
        f"SEAL r{sr}e{se}  energy={sl.get('avg_energy', 0):.4f}  acc={sl.get('accuracy', 0)*100:.2f}%  best={sl.get('best_accuracy', 0)*100:.2f}%{mod_str}"
    )
    line2 = (
        f" epochs logged: baseline={len(baseline)} seal={len(seal)}  |  "
        f"batch time: ~{bl.get('elapsed_secs', 0)*1000:.0f}ms  |  "
        f"refreshing every {REFRESH}s  |  Ctrl-C to exit"
    )
    print(line1)
    print(line2, end="", flush=True)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

    while True:
        try:
            term_w = int(os.environ.get("COLUMNS", os.get_terminal_size().columns))
            term_h = int(os.environ.get("LINES", os.get_terminal_size().lines))
        except (ValueError, OSError):
            term_w, term_h = 200, 50

        try:
            draw(term_w, term_h)
        except Exception as e:
            plt.clear_terminal()
            print(f"Error: {e}")

        time.sleep(REFRESH)


if __name__ == "__main__":
    main()
