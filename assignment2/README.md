# EC5.407 Wireless Communications - Assignment 2

Anumay Rai (2025122013)

## Language and packages

- Python 3.13 (any Python 3.9+ should work)
- numpy 2.x, scipy 1.x, matplotlib 3.x

Install with:

    pip install numpy scipy matplotlib

No built-in modulation, demodulation, detection or error-rate functions are used. Only `numpy` random numbers, `scipy.special.erfc` and basic array functions.

## Layout

    assignment2/
    |-- common/
    |     constellation.py   BPSK and 4-QAM constellations (unit average energy)
    |     channel.py         symbols, AWGN noise, Rayleigh / SIMO channels
    |     detector.py        vectorized ML (minimum distance) detector, error counting, SER loop
    |     theory.py          Q(.) and the theoretical SER expressions
    |     plotting.py        semilogy plotting (only points with n_err >= 10), saving figures/CSV
    |-- problem1/  awgn_ser.py, rayleigh_ser.py
    |-- problem2/  receive_diversity.py
    |-- problem3/  transmit_diversity.py
    |-- figures/   output figures (.pdf/.png) and SER tables (.csv)
    |-- report/    report.pdf and its LaTeX source
    |-- README.md

## How to reproduce the figures

Run every command from inside the `assignment2` folder:

    python problem1/awgn_ser.py              -> figures/p1_awgn.pdf
    python problem1/rayleigh_ser.py          -> figures/p1_rayleigh.pdf
    python problem2/receive_diversity.py     -> figures/p2_receive_diversity.pdf
    python problem3/transmit_diversity.py    -> figures/p3_transmit_diversity.pdf

Each script also prints its SER table, and problem 3 prints the measured average
transmitted energy per channel use.

## Random seeds

The seed is set once at the start of each script with `np.random.default_rng(SEED)`:

| Script | Seed |
|---|---|
| problem1/awgn_ser.py | 1 |
| problem1/rayleigh_ser.py | 2 |
| problem2/receive_diversity.py | 3 |
| problem3/transmit_diversity.py | 4 |

N = 10^6 symbols per SNR point (per scheme, per L). Run times on a laptop: roughly 5 s, 7 s, 30 s and 17 s.
