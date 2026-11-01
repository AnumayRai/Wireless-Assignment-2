import numpy as np
from scipy.special import erfc


def Q(a):
    return 0.5 * erfc(a / np.sqrt(2))


def ser_bpsk_awgn(snr):
    return Q(np.sqrt(2 * snr))


def ser_qam4_awgn(snr):
    q = Q(np.sqrt(snr))
    return 2 * q - q ** 2


def ser_bpsk_rayleigh(snr):
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def ser_qam4_rayleigh(snr):
    return 2 * ser_bpsk_rayleigh(snr / 2)
