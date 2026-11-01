import numpy as np


def normalize(points):
    points = np.asarray(points, dtype=complex)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def bpsk():
    return normalize([1, -1])


def qam4():
    return normalize([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])


def average_energy(const):
    return np.mean(np.abs(const) ** 2)
