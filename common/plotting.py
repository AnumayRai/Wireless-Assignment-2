import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def plot_simulated(ax, snr_db, errors, n, **style):
    keep = errors >= 10
    ax.semilogy(np.asarray(snr_db)[keep], errors[keep] / n, **style)


def finish(ax, title, loc="best"):
    ax.set_xlabel(r"SNR$_\mathrm{dB}$ (dB)")
    ax.set_ylabel(r"Symbol error rate $P_e$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=loc, fontsize=8)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150)
    plt.close(fig)


def save_table(name, header, columns):
    os.makedirs(FIG_DIR, exist_ok=True)
    data = np.column_stack(columns)
    np.savetxt(os.path.join(FIG_DIR, name + ".csv"), data, delimiter=",", header=header, comments="", fmt="%.6g")
