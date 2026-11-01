import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from common.constellation import bpsk, qam4
from common.channel import random_symbols, awgn_channel
from common.detector import ml_detect, ser_curve
from common.theory import ser_bpsk_awgn, ser_qam4_awgn
from common.plotting import plot_simulated, finish, save, save_table

SEED = 1
N = 10**6
SNR_DB = np.arange(0, 13, 1)


def make_link(const):
    def link(rng, snr, n):
        i, x = random_symbols(rng, const, n)
        h, y = awgn_channel(rng, x, snr)
        return i, ml_detect(y, h, const)
    return link


def main():
    rng = np.random.default_rng(SEED)
    err_bpsk = ser_curve(rng, SNR_DB, N, make_link(bpsk()))
    err_qam = ser_curve(rng, SNR_DB, N, make_link(qam4()))
    snr = 10 ** (SNR_DB / 10)
    th_bpsk = ser_bpsk_awgn(snr)
    th_qam = ser_qam4_awgn(snr)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogy(SNR_DB, th_bpsk, "b-", label="BPSK theory")
    ax.semilogy(SNR_DB, th_qam, "r-", label="4-QAM theory")
    plot_simulated(ax, SNR_DB, err_bpsk, N, linestyle="none", marker="o", color="b", label="BPSK simulation")
    plot_simulated(ax, SNR_DB, err_qam, N, linestyle="none", marker="s", color="r", label="4-QAM simulation")
    finish(ax, "AWGN channel")
    save(fig, "p1_awgn")
    save_table("p1_awgn", "snr_db,err_bpsk,ser_bpsk,theory_bpsk,err_qam4,ser_qam4,theory_qam4",
               [SNR_DB, err_bpsk, err_bpsk / N, th_bpsk, err_qam, err_qam / N, th_qam])

    print("SNR_dB  BPSK_sim     BPSK_th      4QAM_sim     4QAM_th")
    for k, s in enumerate(SNR_DB):
        print(f"{s:6d}  {err_bpsk[k]/N:.4e}  {th_bpsk[k]:.4e}  {err_qam[k]/N:.4e}  {th_qam[k]:.4e}")


if __name__ == "__main__":
    main()
