import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from common.constellation import qam4
from common.channel import random_symbols, rayleigh_channel, rayleigh_coefficients, awgn_noise
from common.detector import ml_detect, ser_curve
from common.plotting import plot_simulated, finish, save, save_table
from problem2.receive_diversity import make_link as simo_link, mrc_combine

SEED = 4
N = 10**6
SNR_DB = np.arange(0, 25, 2)
CONST = qam4()
energy_log = {}


def single_antenna_link(rng, snr, n):
    i, x = random_symbols(rng, CONST, n)
    h, y = rayleigh_channel(rng, x, snr)
    energy_log.setdefault("Single antenna", []).append(np.mean(np.abs(x) ** 2))
    return i, ml_detect(y, h, CONST)


def beamforming_weights(h):
    return np.conj(h) / np.linalg.norm(h, axis=1, keepdims=True)


def beamforming_link(rng, snr, n):
    i, x = random_symbols(rng, CONST, n)
    h = rayleigh_coefficients(rng, (n, 2))
    s = beamforming_weights(h) * x[:, None]
    y = np.sum(h * s, axis=1) + awgn_noise(rng, n, snr)
    energy_log.setdefault("Beamforming", []).append(np.mean(np.sum(np.abs(s) ** 2, axis=1)))
    gain = np.sum(h * beamforming_weights(h), axis=1)
    return i, ml_detect(y, gain, CONST)


def alamouti_encode(x1, x2):
    s_t1 = np.column_stack([x1, x2]) / np.sqrt(2)
    s_t2 = np.column_stack([-np.conj(x2), np.conj(x1)]) / np.sqrt(2)
    return s_t1, s_t2


def alamouti_combine(h, y1, y2):
    h1, h2 = h[:, 0], h[:, 1]
    norm_h = np.linalg.norm(h, axis=1)
    z1 = (np.conj(h1) * y1 + h2 * np.conj(y2)) / norm_h
    z2 = (np.conj(h2) * y1 - h1 * np.conj(y2)) / norm_h
    return z1, z2, norm_h / np.sqrt(2)


def alamouti_link(rng, snr, n):
    i, x = random_symbols(rng, CONST, n)
    x1, x2 = x[0::2], x[1::2]
    blocks = x1.size
    h = rayleigh_coefficients(rng, (blocks, 2))
    s_t1, s_t2 = alamouti_encode(x1, x2)
    y1 = np.sum(h * s_t1, axis=1) + awgn_noise(rng, blocks, snr)
    y2 = np.sum(h * s_t2, axis=1) + awgn_noise(rng, blocks, snr)
    energy = np.concatenate([np.sum(np.abs(s_t1) ** 2, axis=1), np.sum(np.abs(s_t2) ** 2, axis=1)])
    energy_log.setdefault("Alamouti", []).append(np.mean(energy))
    z1, z2, gain = alamouti_combine(h, y1, y2)
    i_hat = np.empty(n, dtype=int)
    i_hat[0::2] = ml_detect(z1, gain, CONST)
    i_hat[1::2] = ml_detect(z2, gain, CONST)
    return i, i_hat


def main():
    rng = np.random.default_rng(SEED)
    schemes = {
        "Single antenna": single_antenna_link,
        "Beamforming": beamforming_link,
        "Alamouti": alamouti_link,
        "MRC, L=2": simo_link(2, mrc_combine),
    }
    results = {name: ser_curve(rng, SNR_DB, N, link) for name, link in schemes.items()}

    styles = {
        "Single antenna": dict(color="k", marker="o", linestyle="-"),
        "Beamforming": dict(color="tab:red", marker="s", linestyle="-"),
        "Alamouti": dict(color="tab:blue", marker="^", linestyle="-"),
        "MRC, L=2": dict(color="tab:green", marker="x", linestyle="--"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, err in results.items():
        plot_simulated(ax, SNR_DB, err, N, label=name, **styles[name])
    finish(ax, "Transmit diversity, 4-QAM", loc="upper right")
    save(fig, "p3_transmit_diversity")
    save_table("p3_transmit_diversity", "snr_db," + ",".join(n.replace(", ", "_").replace(" ", "_") for n in results),
               [SNR_DB] + [e / N for e in results.values()])

    print("Average transmitted energy per channel use (mean over all SNR points):")
    for name, values in energy_log.items():
        print(f"  {name:15s} {np.mean(values):.6f}")
    print("SNR_dB " + " ".join(f"{n:>15s}" for n in results))
    for k, s in enumerate(SNR_DB):
        print(f"{s:6d} " + " ".join(f"{results[n][k]/N:15.3e}" for n in results))


if __name__ == "__main__":
    main()
