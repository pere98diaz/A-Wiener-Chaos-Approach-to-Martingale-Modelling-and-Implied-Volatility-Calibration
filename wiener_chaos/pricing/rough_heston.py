"""Rough Heston reference prices and deltas from Fourier integration.

Approximates the fractional Riccati equation with rational formulas to evaluate
the characteristic function, using Mittag–Leffler functions in the variance
terms. The resulting prices generate the synthetic calibration and held-out
surfaces in the rough Heston experiments.
"""

from math import exp, pi

import numpy as np
from pymittagleffler import mittag_leffler
from scipy.integrate import quad, trapezoid
from scipy.special import gamma


def rat_approx(u, t, pars, N):
    """
    Python version of RatApprox: Pade approximations for rough Heston log-characteristic function.

    Parameters:
        u : array_like
            Points to evaluate the log-price transform.
        t : float
            Maturity.
        pars : list or array of float
            Model parameters [lambda, nu, rho, alpha].
        N : int
            Approximation order (2, 3, 4, or 5).

    Returns:
        psi : ndarray
            Approximated transform.
        Dalpha_psi : ndarray
            Second component as defined in the original function.
    """
    lam, nu, rho, al = pars
    #u = np.array(u, dtype=complex)
    lamp = lam / nu
    lamTilde = lamp - 1j * rho * u
    aa = np.sqrt(u * (u + 1j) + lamTilde**2)
    rm = lamTilde - aa
    rp = lamTilde + aa
    g0 = rm / nu
    y = t ** al
    g1 = -g0 / (aa * nu * gamma(1-al)) if al != 1 else 0.0

    if N == 2:
        b1 = -u * (u + 1j) / (2 * gamma(1 + al))
        b2 = -b1 * lamTilde * nu * gamma(1 + al) / gamma(1 + 2 * al)
        den = g0**2 + b1 * g1
        q1 = (b1 * g0 - b2 * g1) / den
        q2 = (b1**2 + b2 * g0) / den
        p1 = b1
        p2 = b2 + b1 * q1
        psi = (p1 * y + p2 * y**2) / (1 + q1 * y + q2 * y**2)

    elif N == 3:
        b1 = -u * (u + 1j) / (2 * gamma(1 + al))
        b2 = -b1 * lamTilde * nu * gamma(1 + al) / gamma(1 + 2 * al)
        b3 = (-b2 * lamTilde * nu + 0.5 * nu**2 * b1**2) * gamma(1 + 2 * al) / gamma(1 + 3 * al)
        g2 = -1 / (aa * nu) * (gamma(1 - al) / gamma(1 - 2 * al) * g1 - 0.5 * nu**2 * g1**2) if al != 1 else 0.0
        g3 = -1 / (aa * nu) * (gamma(1 - 2 * al) / gamma(1 - 3 * al) * g2 - nu**2 * g1 * g2) if al != 1 else 0.0
        den = g0**3 + 2 * b1 * g0 * g1 - b2 * g1**2 + b1**2 * g2 + b2 * g0 * g2
        p1 = b1
        p2 = (b1**2 * g0**2 + b2 * g0**3 + b1**3 * g1 + b1 * b2 * g0 * g1 -
              b2**2 * g1**2 + b1 * b3 * g1**2 + b2**2 * g0 * g2 - b1 * b3 * g0 * g2) / den
        q1 = (b1 * g0**2 + b1**2 * g1 - b2 * g0 * g1 + b3 * g1**2 -
              b1 * b2 * g2 - b3 * g0 * g2) / den
        q2 = (b1**2 * g0 + b2 * g0**2 - b1 * b2 * g1 - b3 * g0 * g1 +
              b2**2 * g2 - b1 * b3 * g2) / den
        q3 = (b1**3 + 2 * b1 * b2 * g0 + b3 * g0**2 - b2**2 * g1 + b1 * b3 * g1) / den
        p3 = g0 * q3
        psi = (p1 * y + p2 * y**2 + p3 * y**3) / (1 + q1 * y + q2 * y**2 + q3 * y**3)

    elif N == 5:
        lamp = lam / nu
        lamTilde = lamp - 1j * rho * u
        aa = np.sqrt(u * (u + 1j) + lamTilde**2)
        rm = lamTilde - aa
        rp = lamTilde + aa

        b1 = -u * (u + 1j) / (2 * gamma(1 + al))
        b2 = -b1 * lamTilde * nu * gamma(1 + al) / gamma(1 + 2 * al)
        b3 = (-b2 * lamTilde * nu + nu**2 * b1**2 / 2) * gamma(1 + 2 * al) / gamma(1 + 3 * al)
        b4 = (-b3 * lamTilde * nu + nu**2 * b1 * b2) * gamma(1 + 3 * al) / gamma(1 + 4 * al)
        b5 = (-b4 * lamTilde * nu + nu**2 * (0.5 * b2**2 + b1 * b3)) * gamma(1 + 4 * al) / gamma(1 + 5 * al)

        g0 = rm / nu
        g1 = (al != 1) * (-1 / (aa * nu) / gamma(1 - al) * g0)
        g2 = (al != 1) * (-1 / (aa * nu) * (gamma(1 - al) / gamma(1 - 2 * al) * g1 - 0.5 * nu**2 * g1**2))
        g3 = (al != 1) * (-1 / (aa * nu) * (gamma(1 - 2 * al) / gamma(1 - 3 * al) * g2 - nu**2 * g1 * g2))
        g4 = (al != 1) * (-1 / (aa * nu) * (gamma(1 - 3 * al) / gamma(1 - 4 * al) * g3 - nu**2 * (0.5 * g2**2 + g1 * g3)))

        den = (-g0**5 - 4*b1*g0**3*g1 - 3*b1**2*g0*g1**2 + 3*b2*g0**2*g1**2 + 2*b1*b2*g1**3 -
        2*b3*g0*g1**3 + b4*g1**4 - 3*b1**2*g0**2*g2 - 3*b2*g0**3*g2 - 2*b1**3*g1*g2 +
        2*b1*b2*g0*g1*g2 + 4*b3*g0**2*g1*g2 - b2**2*g1**2*g2 - 2*b1*b3*g1**2*g2 -
        3*b4*g0*g1**2*g2 + b1**2*b2*g2**2 - 2*b2**2*g0*g2**2 + 4*b1*b3*g0*g2**2 +
        b4*g0**2*g2**2 + 2*b2*b3*g1*g2**2 - 2*b1*b4*g1*g2**2 - b3**2*g2**3 +
        b2*b4*g2**3 - 2*b1**3*g0*g3 - 4*b1*b2*g0**2*g3 - 2*b3*g0**3*g3 +
        2*b1**2*b2*g1*g3 + 4*b2**2*g0*g1*g3 + 2*b4*g0**2*g1*g3 - 2*b2*b3*g1**2*g3 +
        2*b1*b4*g1**2*g3 - 2*b1*b2**2*g2*g3 + 2*b1**2*b3*g2*g3 - 2*b2*b3*g0*g2*g3 +
        2*b1*b4*g0*g2*g3 + 2*b3**2*g1*g2*g3 - 2*b2*b4*g1*g2*g3 + b2**3*g3**2 -
        2*b1*b2*b3*g3**2 + b1**2*b4*g3**2 - b3**2*g0*g3**2 + b2*b4*g0*g3**2 -
        b1**4*g4 - 3*b1**2*b2*g0*g4 - b2**2*g0**2*g4 - 2*b1*b3*g0**2*g4 - b4*g0**3*g4 +
        2*b1*b2**2*g1*g4 - 2*b1**2*b3*g1*g4 + 2*b2*b3*g0*g1*g4 - 2*b1*b4*g0*g1*g4 -
        b3**2*g1**2*g4 + b2*b4*g1**2*g4 - b2**3*g2*g4 + 2*b1*b2*b3*g2*g4 -
        b1**2*b4*g2*g4 + b3**2*g0*g2*g4 - b2*b4*g0*g2*g4)

        q1 = (-(b1*g0**4) - 3*b1**2*g0**2*g1 + b2*g0**3*g1 - b1**3*g1**2 + 4*b1*b2*g0*g1**2 -
        b3*g0**2*g1**2 - b2**2*g1**3 - 2*b1*b3*g1**3 + b4*g0*g1**3 - b5*g1**4 -
        2*b1**3*g0*g2 - b1*b2*g0**2*g2 + b3*g0**3*g2 + 4*b1**2*b2*g1*g2 +
        2*b1*b3*g0*g1*g2 - 2*b4*g0**2*g1*g2 + 2*b2*b3*g1**2*g2 + b1*b4*g1**2*g2 +
        3*b5*g0*g1**2*g2 - 2*b1*b2**2*g2**2 + b1**2*b3*g2**2 - 2*b1*b4*g0*g2**2 -
        b5*g0**2*g2**2 - b3**2*g1*g2**2 - b2*b4*g1*g2**2 + 2*b1*b5*g1*g2**2 +
        b3*b4*g2**3 - b2*b5*g2**3 - b1**4*g3 - b1**2*b2*g0*g3 + b2**2*g0**2*g3 +
        b4*g0**3*g3 - 2*b1**2*b3*g1*g3 - 4*b2*b3*g0*g1*g3 - 2*b5*g0**2*g1*g3 +
        b3**2*g1**2*g3 + b2*b4*g1**2*g3 - 2*b1*b5*g1**2*g3 + b2**3*g2*g3 -
        b1**2*b4*g2*g3 + b3**2*g0*g2*g3 + b2*b4*g0*g2*g3 - 2*b1*b5*g0*g2*g3 -
        2*b3*b4*g1*g2*g3 + 2*b2*b5*g1*g2*g3 - b2**2*b3*g3**2 + b1*b3**2*g3**2 +
        b1*b2*b4*g3**2 - b1**2*b5*g3**2 + b3*b4*g0*g3**2 - b2*b5*g0*g3**2 +
        b1**3*b2*g4 + 2*b1*b2**2*g0*g4 + b1**2*b3*g0*g4 + 2*b2*b3*g0**2*g4 +
        b1*b4*g0**2*g4 + b5*g0**3*g4 - b2**3*g1*g4 + b1**2*b4*g1*g4 - b3**2*g0*g1*g4 -
        b2*b4*g0*g1*g4 + 2*b1*b5*g0*g1*g4 + b3*b4*g1**2*g4 - b2*b5*g1**2*g4 +
        b2**2*b3*g2*g4 - b1*b3**2*g2*g4 - b1*b2*b4*g2*g4 + b1**2*b5*g2*g4 -
        b3*b4*g0*g2*g4 + b2*b5*g0*g2*g4) / den

        q2 = (-(b1**2 * g0**3) - b2 * g0**4 - 2 * b1**3 * g0 * g1 - b1 * b2 * g0**2 * g1 + b3 * g0**3 * g1 +
        2 * b1**2 * b2 * g1**2 + b2**2 * g0 * g1**2 - b4 * g0**2 * g1**2 + b1 * b4 * g1**3 +
        b5 * g0 * g1**3 - b1**4 * g2 - b1**2 * b2 * g0 * g2 - 2 * b2**2 * g0**2 * g2 + 3 * b1 * b3 * g0**2 * g2 +
        b4 * g0**3 * g2 - 2 * b1 * b2**2 * g1 * g2 - 4 * b1 * b4 * g0 * g1 * g2 - 2 * b5 * g0**2 * g1 * g2 -
        b2 * b4 * g1**2 * g2 + b1 * b5 * g1**2 * g2 + 2 * b1 * b2 * b3 * g2**2 - 2 * b1**2 * b4 * g2**2 -
        b3**2 * g0 * g2**2 + 3 * b2 * b4 * g0 * g2**2 - 2 * b1 * b5 * g0 * g2**2 + b3 * b4 * g1 * g2**2 -
        b2 * b5 * g1 * g2**2 - b4**2 * g2**3 + b3 * b5 * g2**3 + b1**3 * b2 * g3 + 3 * b1**2 * b3 * g0 * g3 +
        3 * b1 * b4 * g0**2 * g3 + b5 * g0**3 * g3 + b2**3 * g1 * g3 - 2 * b1 * b2 * b3 * g1 * g3 +
        b1**2 * b4 * g1 * g3 + b3**2 * g0 * g1 * g3 - b2 * b4 * g0 * g1 * g3 - b3 * b4 * g1**2 * g3 +
        b2 * b5 * g1**2 * g3 - b2**2 * b3 * g2 * g3 - b1 * b3**2 * g2 * g3 + 3 * b1 * b2 * b4 * g2 * g3 -
        b1**2 * b5 * g2 * g3 - b3 * b4 * g0 * g2 * g3 + b2 * b5 * g0 * g2 * g3 + 2 * b4**2 * g1 * g2 * g3 -
        2 * b3 * b5 * g1 * g2 * g3 + b2 * b3**2 * g3**2 - b2**2 * b4 * g3**2 - b1 * b3 * b4 * g3**2 +
        b1 * b2 * b5 * g3**2 - b4**2 * g0 * g3**2 + b3 * b5 * g0 * g3**2 - b1**2 * b2**2 * g4 +
        b1**3 * b3 * g4 - b2**3 * g0 * g4 + b1**2 * b4 * g0 * g4 - b2 * b4 * g0**2 * g4 + b1 * b5 * g0**2 * g4 +
        b2**2 * b3 * g1 * g4 + b1 * b3**2 * g1 * g4 - 3 * b1 * b2 * b4 * g1 * g4 + b1**2 * b5 * g1 * g4 +
        b3 * b4 * g0 * g1 * g4 - b2 * b5 * g0 * g1 * g4 - b4**2 * g1**2 * g4 + b3 * b5 * g1**2 * g4 -
        b2 * b3**2 * g2 * g4 + b2**2 * b4 * g2 * g4 + b1 * b3 * b4 * g2 * g4 - b1 * b2 * b5 * g2 * g4 +
        b4**2 * g0 * g2 * g4 - b3 * b5 * g0 * g2 * g4) / den

        q3 = (
            -(b1**3 * g0**2) - 2*b1*b2*g0**3 - b3*g0**4 - b1**4 * g1 - b1**2 * b2 * g0 * g1
            + 2*b2**2 * g0**2 * g1 - b1*b3*g0**2 * g1 + b4*g0**3 * g1 + b1*b2**2 * g1**2
            - 2*b1**2 * b3 * g1**2 - 2*b2*b3*g0*g1**2 - b5*g0**2 * g1**2 + b2*b4*g1**3
            - b1*b5*g1**3 + b1**3 * b2 * g2 + 3*b1**2 * b3 * g0 * g2 + 3*b1*b4*g0**2 * g2
            + b5*g0**3 * g2 + 2*b3**2 * g0 * g1 * g2 - 2*b2*b4*g0 * g1 * g2
            - b3*b4*g1**2 * g2 + b2*b5*g1**2 * g2 - b1*b3**2 * g2**2 + b1*b2*b4 * g2**2
            - b3*b4*g0 * g2**2 + b2*b5*g0 * g2**2 + b4**2 * g1 * g2**2 - b3*b5 * g1 * g2**2
            - b1**2 * b2**2 * g3 + b1**3 * b3 * g3 + b2**3 * g0 * g3 - 4*b1*b2*b3 * g0 * g3
            + 3*b1**2 * b4 * g0 * g3 - 2*b3**2 * g0**2 * g3 + b2*b4 * g0**2 * g3
            + b1*b5 * g0**2 * g3 - b2**2 * b3 * g1 * g3 + 3*b1*b3**2 * g1 * g3
            - b1*b2*b4 * g1 * g3 - b1**2 * b5 * g1 * g3 + 3*b3*b4 * g0 * g1 * g3
            - 3*b2*b5 * g0 * g1 * g3 - b4**2 * g1**2 * g3 + b3*b5 * g1**2 * g3
            + b2*b3**2 * g2 * g3 - b2**2 * b4 * g2 * g3 - b1*b3*b4 * g2 * g3
            + b1*b2*b5 * g2 * g3 - b4**2 * g0 * g2 * g3 + b3*b5 * g0 * g2 * g3
            - b3**3 * g3**2 + 2*b2*b3*b4 * g3**2 - b1*b4**2 * g3**2 - b2**2 * b5 * g3**2
            + b1*b3*b5 * g3**2 + b1*b2**3 * g4 - 2*b1**2 * b2 * b3 * g4 + b1**3 * b4 * g4
            + b2**2 * b3 * g0 * g4 - 2*b1*b3**2 * g0 * g4 + b1**2 * b5 * g0 * g4
            - b3*b4 * g0**2 * g4 + b2*b5 * g0**2 * g4 - b2*b3**2 * g1 * g4
            + b2**2 * b4 * g1 * g4 + b1*b3*b4 * g1 * g4 - b1*b2*b5 * g1 * g4
            + b4**2 * g0 * g1 * g4 - b3*b5 * g0 * g1 * g4 + b3**3 * g2 * g4
            - 2*b2*b3*b4 * g2 * g4 + b1*b4**2 * g2 * g4 + b2**2 * b5 * g2 * g4
            - b1*b3*b5 * g2 * g4
            ) / den

        q4 = (-(b1**4 * g0)- 3 * b1**2 * b2 * g0**2- b2**2 * g0**3- 2 * b1 * b3 * g0**3
            - b4 * g0**4+ b1**3 * b2 * g1+ 4 * b1 * b2**2 * g0 * g1- b1**2 * b3 * g0 * g1+ 4 * b2 * b3 * g0**2 * g1
            - b1 * b4 * g0**2 * g1+ b5 * g0**3 * g1- b2**3 * g1**2+ b1**2 * b4 * g1**2
            - 2 * b3**2 * g0 * g1**2+ 2 * b1 * b5 * g0 * g1**2+ b3 * b4 * g1**3- b2 * b5 * g1**3
            - b1**2 * b2**2 * g2+ b1**3 * b3 * g2- 2 * b2**3 * g0 * g2+ 2 * b1 * b2 * b3 * g0 * g2
            + b3**2 * g0**2 * g2- 2 * b2 * b4 * g0**2 * g2+ b1 * b5 * g0**2 * g2+ 2 * b2**2 * b3 * g1 * g2
            - 4 * b1 * b2 * b4 * g1 * g2+ 2 * b1**2 * b5 * g1 * g2 - b4**2 * g1**2 * g2
            + b3 * b5 * g1**2 * g2- b2 * b3**2 * g2**2+ b2**2 * b4 * g2**2+ b1 * b3 * b4 * g2**2
            - b1 * b2 * b5 * g2**2+ b4**2 * g0 * g2**2- b3 * b5 * g0 * g2**2+ b1 * b2**3 * g3
            - 2 * b1**2 * b2 * b3 * g3+ b1**3 * b4 * g3 + b2**2 * b3 * g0 * g3
            - 2 * b1 * b3**2 * g0 * g3+ b1**2 * b5 * g0 * g3- b3 * b4 * g0**2 * g3
            + b2 * b5 * g0**2 * g3- b2 * b3**2 * g1 * g3+ b2**2 * b4 * g1 * g3+ b1 * b3 * b4 * g1 * g3
            - b1 * b2 * b5 * g1 * g3+ b4**2 * g0 * g1 * g3- b3 * b5 * g0 * g1 * g3+ b3**3 * g2 * g3
            - 2 * b2 * b3 * b4 * g2 * g3+ b1 * b4**2 * g2 * g3+ b2**2 * b5 * g2 * g3
            - b1 * b3 * b5 * g2 * g3- b2**4 * g4+ 3 * b1 * b2**2 * b3 * g4- b1**2 * b3**2 * g4
            - 2 * b1**2 * b2 * b4 * g4+ b1**3 * b5 * g4+ 2 * b2 * b3**2 * g0 * g4- 2 * b2**2 * b4 * g0 * g4
            - 2 * b1 * b3 * b4 * g0 * g4+ 2 * b1 * b2 * b5 * g0 * g4- b4**2 * g0**2 * g4
            + b3 * b5 * g0**2 * g4- b3**3 * g1 * g4 + 2 * b2 * b3 * b4 * g1 * g4- b1 * b4**2 * g1 * g4
            - b2**2 * b5 * g1 * g4+ b1 * b3 * b5 * g1 * g4
        ) / den

        q5 = (-b1**5 - 4*b1**3*b2*g0 - 3*b1*b2**2*g0**2 - 3*b1**2*b3*g0**2 - 2*b2*b3*g0**3 -
        2*b1*b4*g0**3 - b5*g0**4 + 3*b1**2*b2**2*g1 - 3*b1**3*b3*g1 + 2*b2**3*g0*g1 +
        2*b1*b2*b3*g0*g1 - 4*b1**2*b4*g0*g1 + b3**2*g0**2*g1 + 2*b2*b4*g0**2*g1 -
        3*b1*b5*g0**2*g1 - b2**2*b3*g1**2 - 2*b1*b3**2*g1**2 + 4*b1*b2*b4*g1**2 -
        b1**2*b5*g1**2 - 2*b3*b4*g0*g1**2 + 2*b2*b5*g0*g1**2 + b4**2*g1**3 -
        b3*b5*g1**3 - 2*b1*b2**3*g2 + 4*b1**2*b2*b3*g2 - 2*b1**3*b4*g2 -
        2*b2**2*b3*g0*g2 + 4*b1*b3**2*g0*g2 - 2*b1**2*b5*g0*g2 + 2*b3*b4*g0**2*g2 -
        2*b2*b5*g0**2*g2 + 2*b2*b3**2*g1*g2 - 2*b2**2*b4*g1*g2 - 2*b1*b3*b4*g1*g2 +
        2*b1*b2*b5*g1*g2 - 2*b4**2*g0*g1*g2 + 2*b3*b5*g0*g1*g2 - b3**3*g2**2 +
        2*b2*b3*b4*g2**2 - b1*b4**2*g2**2 - b2**2*b5*g2**2 + b1*b3*b5*g2**2 +
        b2**4*g3 - 3*b1*b2**2*b3*g3 + b1**2*b3**2*g3 + 2*b1**2*b2*b4*g3 - b1**3*b5*g3 -
        2*b2*b3**2*g0*g3 + 2*b2**2*b4*g0*g3 + 2*b1*b3*b4*g0*g3 - 2*b1*b2*b5*g0*g3 +
        b4**2*g0**2*g3 - b3*b5*g0**2*g3 + b3**3*g1*g3 - 2*b2*b3*b4*g1*g3 +
        b1*b4**2*g1*g3 + b2**2*b5*g1*g3 - b1*b3*b5*g1*g3) / den

        p1 = b1
        p2 = b2 + b1 * q1
        p3 = b3 + b1 * q2 + b2 * q1
        p4 = b4 + b3 * q1 + b2 * q2 + b1 * q3
        p5 = g0 * q5

        y = t ** al
        psi = (p1 * y + p2 * y**2 + p3 * y**3 + p4 * y**4 + p5 * y**5) / (
            1 + q1 * y + q2 * y**2 + q3 * y**3 + q4 * y**4 + q5 * y**5)

    Dalpha_psi = 0.5 * (nu * psi - rm) * (nu * psi - rp)
    return psi, Dalpha_psi


# -------------------- ROUGH HESTON PRICER --------------------
def e_fun(t,a,b,kappa):
    return (t**(b-1))*mittag_leffler(-kappa*(t**a),a,b)

def payoff_transform(u, k):
    return -(k**(1+u))/(u+u**2)

def _phi_rh(xi, params, r, q, T, N=5, M=200):
    """
    Rough Heston characteristic function using rational approximation (Padé).

    """
    ixi = 1j*xi
    kappa, theta, sigma, rho, alpha, v0 = params
    interval = np.linspace(0, T, M + 1)
    gam = v0 + kappa * (theta - v0) * e_fun(interval, alpha, alpha + 1, kappa)

    # Ensure u is scalar for rational approximation
    psi, Dalpha_psi = rat_approx(ixi/1j, interval, (kappa,sigma,rho,alpha), 3)  # shape: (M+1,)

    g = kappa * psi + Dalpha_psi
    integral_term = trapezoid(np.flip(gam) * g, interval)
    return np.exp(ixi * (r - q) * T + integral_term)

def rh_price(params, r, q, T, S0, K, flag="call"):
    def integrand(xi):
        xi = xi + 1j*0.5
        return (_phi_rh(-xi, params, r, q, T) * payoff_transform(1j*xi,K/S0)).real

    call = np.exp(-q*T) - (1/pi) * quad(integrand, 0.0, 500.0, limit=200)[0]
    call = S0*call
    if flag.lower() == "call":
        return call
    elif flag.lower() == "put":
        return call - S0 * exp(-q * T) + K * exp(-r * T) # put–call parity with dividends: C - P = S0 e^{-qT} - K e^{-rT}
    else:
        raise ValueError("flag must be 'call' or 'put'")

def rh_delta(params, r, q, T, S0, K, flag="call"):
    """
    Delta of European call/put under Rough Heston model.

    Derived from the Lewis call price formula:
    C = S₀ e^{-qT} - (S₀/π) ∫₀^∞ Re[φ(-ξ) · payoff_transform(iξ, K/S₀)] dξ

    where ξ = u + i/2 and payoff_transform(v, k) = -k^{1+v} / (v + v²).

    Taking dC/dS₀:
    - d/dS₀[S₀ e^{-qT}] = e^{-qT}
    - d/dS₀[S₀ · payoff_transform(v, K/S₀)] requires product rule

    Let k = K/S₀, so dk/dS₀ = -K/S₀² = -k/S₀
    payoff(v,k) = -k^{1+v}/(v+v²)
    d/dk[payoff] = -(1+v)k^v/(v+v²)
    d/dS₀[payoff(v,K/S₀)] = d/dk[payoff] · dk/dS₀ = (1+v)k^{1+v}/(S₀(v+v²))

    Full derivative of S₀·payoff:
    d/dS₀[S₀·payoff] = payoff + S₀·d/dS₀[payoff]
                     = -k^{1+v}/(v+v²) + (1+v)k^{1+v}/(v+v²)
                     = k^{1+v}/(v+v²) · (1+v-1) = k^{1+v}·v/(v+v²)
                     = k^{1+v}/(1+v)

    So: Δ = e^{-qT} - (1/π) ∫₀^∞ Re[φ(-ξ) · (K/S₀)^{1+iξ} / (1+iξ)] dξ
    """
    k = K / S0

    def integrand(u):
        xi = u + 0.5j
        ixi = 1j * xi
        phi = _phi_rh(-xi, params, r, q, T)
        # (K/S0)^{1+i*xi} / (1 + i*xi)
        factor = phi * (k ** (1.0 + ixi)) / (1.0 + ixi)
        return factor.real

    integral, _ = quad(integrand, 0.0, 500.0, limit=200)
    delta_call = exp(-q * T) - (1.0 / pi) * integral

    if flag.lower() == "call":
        return delta_call
    elif flag.lower() == "put":
        return delta_call - exp(-q * T)
    else:
        raise ValueError("flag must be 'call' or 'put'")

if __name__ == "__main__":
    # Example usage
    params = (1.5,0.04,0.5,-0.65,0.55,0.06)  # kappa, theta, sigma, rho, alpha, v0
    r = 0
    q = 0
    T = 1.0
    S0 = 100.0
    K = 100.0

    price = rh_price(params, r, q, T, S0, K, flag="call")
    print(f"RH Call Price: {price:.4f}")

    delta = rh_delta(params, r, q, T, S0, K, flag="call")
    print(f"RH Call Delta: {delta:.4f}")
