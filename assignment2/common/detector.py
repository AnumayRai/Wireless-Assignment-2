import numpy as np


def ml_detect(y, h, const):
    distances = np.abs(y[:, None] - h[:, None] * const[None, :]) ** 2
    return np.argmin(distances, axis=1)


def count_errors(i_hat, i):
    return int(np.count_nonzero(i_hat != i))


def ser_curve(rng, snr_db, n, link):
    errors = np.zeros(len(snr_db), dtype=int)
    for k, s in enumerate(snr_db):
        snr = 10 ** (s / 10)
        i, i_hat = link(rng, snr, n)
        errors[k] = count_errors(i_hat, i)
    return errors
