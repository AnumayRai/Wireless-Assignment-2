import numpy as np


def db_to_linear(snr_db):
    return 10 ** (snr_db / 10)


def random_symbols(rng, const, n):
    i = rng.integers(0, len(const), n)
    return i, const[i]


def complex_gaussian(rng, shape, variance):
    scale = np.sqrt(variance / 2)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def awgn_noise(rng, shape, snr):
    return complex_gaussian(rng, shape, 1 / snr)


def rayleigh_coefficients(rng, shape):
    return complex_gaussian(rng, shape, 1.0)


def awgn_channel(rng, x, snr):
    h = np.ones(x.shape, dtype=complex)
    y = h * x + awgn_noise(rng, x.shape, snr)
    return h, y


def rayleigh_channel(rng, x, snr):
    h = rayleigh_coefficients(rng, x.shape)
    y = h * x + awgn_noise(rng, x.shape, snr)
    return h, y


def simo_channel(rng, x, snr, L):
    h = rayleigh_coefficients(rng, (x.size, L))
    y = h * x[:, None] + awgn_noise(rng, (x.size, L), snr)
    return h, y
