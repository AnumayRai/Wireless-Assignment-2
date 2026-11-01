import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from common.constellation import qam4
from common.channel import random_symbols, simo_channel
from common.detector import ml_detect, ser_curve
from common.theory import ser_qam4_rayleigh
from common.plotting import plot_simulated, finish, save, save_table

SEED = 3
N = 10**6
SNR_DB = np.arange(0, 25, 2)
L_VALUES = [1, 2, 4]
COLORS = {1: "tab:blue", 2: "tab:red", 4: "tab:green"}
CONST = qam4()


def mrc_combine(h, y):
    norm_h = np.linalg.norm(h, axis=1)
    z = np.sum(np.conj(h) * y, axis=1) / norm_h
    return z, norm_h


def sc_select(h, y):
    best = np.argmax(np.abs(h) ** 2, axis=1)
    rows = np.arange(h.shape[0])
    return y[rows, best], h[rows, best]


def make_link(L, combiner):
    def link(rng, snr, n):
        i, x = random_symbols(rng, CONST, n)
        h, y = simo_channel(rng, x, snr, L)
        z, g = combiner(h, y)
        return i, ml_detect(z, g, CONST)
    return link


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for L in L_VALUES:
        results[("MRC", L)] = ser_curve(rng, SNR_DB, N, make_link(L, mrc_combine))
        results[("SC", L)] = ser_curve(rng, SNR_DB, N, make_link(L, sc_select))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    snr = 10 ** (SNR_DB / 10)
    ax.semilogy(SNR_DB, ser_qam4_rayleigh(snr), "k:", label="4-QAM fading theory (Problem 1)")
    for L in L_VALUES:
        plot_simulated(ax, SNR_DB, results[("MRC", L)], N, linestyle="-", marker="o", color=COLORS[L], label=f"MRC, L={L}")
        plot_simulated(ax, SNR_DB, results[("SC", L)], N, linestyle="--", marker="x", color=COLORS[L], label=f"SC, L={L}")
    finish(ax, "Receive diversity, 4-QAM", loc="upper right")
    save(fig, "p2_receive_diversity")

    names = [f"{s}_L{L}" for L in L_VALUES for s in ("MRC", "SC")]
    cols = [SNR_DB] + [results[(s, L)] / N for L in L_VALUES for s in ("MRC", "SC")]
    save_table("p2_receive_diversity", "snr_db," + ",".join(names), cols)

    print("SNR_dB " + " ".join(f"{n:>11s}" for n in names))
    for k, s in enumerate(SNR_DB):
        print(f"{s:6d} " + " ".join(f"{results[(sc, L)][k]/N:11.3e}" for L in L_VALUES for sc in ("MRC", "SC")))


if __name__ == "__main__":
    main()
