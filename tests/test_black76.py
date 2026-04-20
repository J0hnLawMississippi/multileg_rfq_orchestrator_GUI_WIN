import numpy as np

from db_option_pricer_win import Black76


def test_call_put_parity_approximation():
    F = np.array([70000.0])
    K = np.array([68000.0])
    sigma = np.array([0.6])
    T = np.array([0.25])
    c = Black76.price(F, K, sigma, T, np.array([True]))[0]
    p = Black76.price(F, K, sigma, T, np.array([False]))[0]
    assert np.isclose(c - p, F[0] - K[0], atol=1e-6)


def test_expired_option_prices_to_intrinsic():
    F = np.array([70000.0, 70000.0])
    K = np.array([65000.0, 75000.0])
    sigma = np.array([0.6, 0.6])
    T = np.array([0.0, -1.0])
    prices = Black76.price(F, K, sigma, T, np.array([True, False]))
    assert prices[0] == 5000.0
    assert prices[1] == 5000.0
