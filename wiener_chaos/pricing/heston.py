"""Heston reference-model pricing, simulation, and parameter fitting.

Provides scalar and vectorized Fourier prices, calibration to a market surface,
and Euler and quadratic-exponential simulation schemes. Also prices selected
exotic payoffs. The notebooks use these routines to generate reference data and
fit the Heston model used to initialize chaos coefficients.
"""

import cmath
import itertools
import math
from math import exp, log, pi, sqrt

import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import minimize

from .options import (
    BlackScholesVegaCore,
    evaluate_payoff,
    implied_vol_bisection,
    option_parameter_grid,
)
from ..core.utils import simulate_bm


# -------------------- HESTON PRICER --------------------
def _phi_heston(xi, params, r, q, T, S0, typ):
    """
    Heston characteristic function φ(i*xi) with continuous dividend yield q.
    typ ∈ {1,2} corresponds to P1, P2.
    params = (kappa, theta, sigma, rho, v0)
    """
    kappa, theta, sigma, rho, v0 = params
    ixi = 1j * xi
    if typ == 1:
        w = 1.0
        b = kappa - rho * sigma
    else:
        w = -1.0
        b = kappa

    d = cmath.sqrt((rho*sigma*ixi - b)**2 - sigma**2 * (w*ixi - xi**2))
    g = (b - rho*sigma*ixi - d) / (b - rho*sigma*ixi + d)
    e_dt = cmath.exp(-d*T)

    # only change vs your version: r -> (r - q)
    C = (r - q) * ixi * T + (kappa*theta/sigma**2) * (
        (b - rho*sigma*ixi - d)*T - 2.0*cmath.log((1.0 - g*e_dt)/(1.0 - g))
    )
    D = ((b - rho*sigma*ixi - d)/sigma**2) * ((1.0 - e_dt)/(1.0 - g*e_dt))

    return cmath.exp(C + D*v0 + ixi*log(S0))


# 2) P1/P2 integral (Carr–Madan / Lewis style)
def _P_typ(params, r, q, T, S0, K, typ, xi_max=5000.0):
    """
    Computes P1 or P2:
        1/2 + (1/pi) ∫_0^∞ Re{ e^{-i xi ln K} φ(i xi) / (i xi) } dxi
    """
    def integrand(xi):
        ixi = 1j * xi
        return (cmath.exp(-ixi * log(K)) *
                _phi_heston(xi, params, r, q, T, S0, typ) / (ixi)).real

    from scipy.integrate import quad
    val, _ = quad(integrand, 0.0, xi_max, limit=1000)
    return 0.5 + (1.0 / pi) * val

def heston_price(params, r, q, T, S0, K, flag="call"):
    """
    European call/put under Heston with dividend yield q:
    C = S0 e^{-qT} P1 - K e^{-rT} P2; P via parity.
    """
    P1 = _P_typ(params, r, q, T, S0, K, typ=1)
    P2 = _P_typ(params, r, q, T, S0, K, typ=2)
    call = S0 * exp(-q * T) * P1 - K * exp(-r * T) * P2
    if flag.lower() == "call":
        return call
    elif flag.lower() == "put":
        return call - S0 * exp(-q * T) + K * exp(-r * T) # put–call parity with dividends: C - P = S0 e^{-qT} - K e^{-rT}
    else:
        raise ValueError("flag must be 'call' or 'put'")


def sim_qe_heston(S0, r, q, params, T, pi, n_paths, gamma1=0.5, gamma2=0.5):
    """Andersen QE-M simulation on the supplied grid (including 0 and T).

    params = (kappa, theta, sigma, rho, v0). Returns float64 spot, variance,
    and log-spot arrays of shape (n_paths, len(pi)). Uses NumPy's global RNG.
    The variance transition matches CIR conditional mean/variance; the stock
    update preserves its conditional forward mean. This is a time-stepping
    approximation, not exact Heston simulation or continuous extrema sampling.
    """
    kappa, theta, sigma, rho, V0 = params
    grid = np.asarray(pi, dtype=float)
    if not np.isfinite([S0, r, q, T, kappa, theta, sigma, rho, V0, gamma1, gamma2]).all():
        raise ValueError("Heston simulation inputs must be finite")
    if S0 <= 0 or T < 0 or min(kappa, theta, sigma, V0) < 0 or abs(rho) > 1:
        raise ValueError("Require positive spot, nonnegative times/variance parameters, and |rho| <= 1")
    if (min(gamma1, gamma2) < 0
            or not np.isclose(gamma1 + gamma2, 1., rtol=0., atol=1e-12)):
        raise ValueError("QE weights must be nonnegative and sum to one")
    if (isinstance(n_paths, (bool, np.bool_))
            or not isinstance(n_paths, (int, np.integer)) or n_paths < 1):
        raise ValueError("n_paths must be a positive integer")
    if (grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all()
            or grid[0] != 0 or not np.isclose(grid[-1], T, rtol=0., atol=1e-12)
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("pi must increase strictly from 0 to T")

    N = len(grid) - 1
    X = np.empty((n_paths, N + 1), dtype=float)
    V = np.empty_like(X)
    X[:, 0], V[:, 0] = np.log(S0), V0

    for t, h in enumerate(np.diff(grid)):
        vt = V[:, t]
        one_minus_decay = -np.expm1(-kappa * h)
        decay = np.exp(-kappa * h)
        decay_integral = one_minus_decay / kappa if kappa > 0 else h
        m = vt * decay + theta * one_minus_decay

        if sigma == 0:
            # Deterministic CIR variance: integrate it exactly over this step.
            integrated_v = np.maximum(vt * decay_integral + theta * (h - decay_integral), 0.)
            V[:, t + 1] = m
            X[:, t + 1] = (X[:, t] + (r - q) * h - 0.5 * integrated_v
                           + np.sqrt(integrated_v) * np.random.randn(n_paths))
            continue

        s2 = sigma**2 * decay_integral * (vt * decay + 0.5 * theta * one_minus_decay)
        psi = np.divide(s2, m*m, out=np.zeros_like(m), where=m > 0)
        V_next = np.zeros_like(vt)
        variance_deviation = np.zeros_like(vt)
        centered_ln_mgf = np.zeros_like(vt)

        K2 = gamma2 * h * (kappa * rho / sigma - 0.5) + rho / sigma
        K3 = gamma1 * h * (1 - rho**2)
        K4 = gamma2 * h * (1 - rho**2)
        A = K2 + 0.5 * K4

        quadratic = psi <= 1.5
        if np.any(quadratic):
            psi1, m1 = psi[quadratic], m[quadratic]
            # Equivalent to a*(sqrt(b2)+Z)^2, without division by psi.
            # The same expression handles the absorbing m=s2=0 state.
            a = m1 * psi1 / (2 * (1 + np.sqrt(1 - 0.5 * psi1)))
            Z = np.random.randn(np.count_nonzero(quadratic))
            V_next[quadratic] = (np.sqrt(np.maximum(m1 - a, 0.)) + np.sqrt(a) * Z)**2
            variance_deviation[quadratic] = 2*np.sqrt(a)*np.sqrt(np.maximum(m1-a, 0.))*Z + a*(Z*Z-1)
            Aa = A * a
            denom = 1 - 2 * Aa
            if np.any(denom <= 0):
                raise ValueError("QE martingale moment does not exist; use a smaller time step")
            # log E[exp(A*V_next)] - A*m, without cancellation as sigma -> 0.
            centered_ln_mgf[quadratic] = -0.5*np.log1p(-2*Aa) - Aa + 2*A*Aa*(m1-a)/denom

        exponential = ~quadratic
        if np.any(exponential):
            psi2, m2 = psi[exponential], m[exponential]
            one_minus_p = 2 / (psi2 + 1)
            p = 1 - one_minus_p
            beta = one_minus_p / m2
            if np.any(beta <= A):
                raise ValueError("QE martingale moment does not exist; use a smaller time step")
            U = np.random.rand(np.count_nonzero(exponential))
            V_next[exponential] = np.where(
                U <= p, 0., (np.log(one_minus_p) - np.log1p(-U)) / beta,
            )
            variance_deviation[exponential] = V_next[exponential] - m2
            centered_ln_mgf[exponential] = np.log1p(one_minus_p*A/(beta-A)) - A*m2

        # E[exp(delta X - (r-q)*h) | V_t] = 1, integrating over both draws.
        # Centering V_next and its log-MGF is algebraically the QE-M K0 update.
        var_term = K3 * vt + K4 * V_next
        X[:, t + 1] = (X[:, t] + (r - q) * h + K2 * variance_deviation
                       - 0.5 * (K3 * vt + K4 * m) - centered_ln_mgf
                       + np.sqrt(var_term) * np.random.randn(n_paths))
        V[:, t + 1] = V_next

    S = np.exp(X)
    S[:, 0] = S0  # Preserve exact initial barrier touches (exp(log(S0)) can round up).
    return S, V, X

@torch.no_grad()
def sim_qe_heston_torch(S0, r, q, params, pi, n_paths, device="cpu"):
    """QE-M spot paths on CPU/CUDA, using float64 and Torch's RNG.

    Same transition as sim_qe_heston; retain only spot paths and the current
    variance/log spot. Both QE branches are vectorized, without CPU transfers
    or device synchronization at every time step.
    """
    kappa, theta, sigma, rho, v0 = params
    grid = np.asarray(pi, dtype=float)
    if (not np.isfinite([S0, r, q, *params]).all() or S0 <= 0
            or min(kappa, theta, sigma, v0) < 0 or abs(rho) > 1):
        raise ValueError("Invalid Heston parameters")
    if (grid.ndim != 1 or not len(grid) or grid[0] != 0
            or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0)):
        raise ValueError("pi must increase strictly from zero")
    if isinstance(n_paths, (bool, np.bool_)) or not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    paths = torch.empty((n_paths, len(grid)), dtype=torch.float64, device=device)
    paths[:, 0] = S0
    v, x = paths.new_full((n_paths,), v0), paths.new_full((n_paths,), math.log(S0))
    invalid_moment = torch.zeros((), dtype=torch.bool, device=device)
    tiny, eps = torch.finfo(paths.dtype).tiny, torch.finfo(paths.dtype).eps
    for j, h in enumerate(np.diff(grid)):
        decay, omd = math.exp(-kappa*h), -math.expm1(-kappa*h)
        integral = omd/kappa if kappa else h
        m = v*decay + theta*omd
        if sigma == 0:
            iv = (v*integral + theta*(h-integral)).clamp_min(0)
            x = x + (r-q)*h - .5*iv + iv.sqrt()*torch.randn_like(v)
            v = m
        else:
            s2 = sigma*sigma*integral*(v*decay + .5*theta*omd)
            psi = torch.where(m > 0, s2/(m*m).clamp_min(tiny), 0.)
            quadratic = psi <= 1.5
            z, zs = torch.randn((2, n_paths), dtype=paths.dtype, device=device)
            a = m*psi.clamp_max(1.5)/(2*(1+(1-.5*psi.clamp_max(1.5)).sqrt()))
            b = (m-a).clamp_min(0)
            vq = (b.sqrt()+a.sqrt()*z).square()
            dq = 2*a.sqrt()*b.sqrt()*z + a*(z*z-1)
            k2 = .5*h*(kappa*rho/sigma-.5) + rho/sigma
            k3 = k4 = .5*h*(1-rho*rho)
            A = k2 + .5*k4
            Aa = A*a
            denom = 1-2*Aa
            mgf_q = -.5*torch.log1p((-2*Aa).clamp_min(-1+eps)) - Aa + 2*A*Aa*b/denom.clamp_min(tiny)
            tail = 2/(psi.clamp_min(1.5)+1)
            beta = tail/m.clamp_min(tiny)
            u = torch.rand_like(v)
            ve = torch.where(u <= 1-tail, 0., (tail.log()-torch.log1p(-u))/beta)
            mgf_e = torch.log1p(tail*A/(beta-A).clamp_min(tiny)) - A*m
            invalid_moment |= ((quadratic & (denom <= 0)) | (~quadratic & (beta <= A))).any()
            vn = torch.where(quadratic, vq, ve)
            deviation = torch.where(quadratic, dq, ve-m)
            mgf = torch.where(quadratic, mgf_q, mgf_e)
            x = x + (r-q)*h + k2*deviation - .5*(k3*v+k4*m) - mgf + (k3*v+k4*vn).sqrt()*zs
            v = vn
        paths[:, j+1] = x.exp()
    if invalid_moment.item():
        raise ValueError("QE martingale moment does not exist; use a smaller time step")
    return paths


def sim_euler_heston(bm, S0, r, q, params, T, pi, device, return_last = False):
    """Projected Euler variance with a log-Euler stock update.

    bm contains two independent Brownian paths, including their initial zero.
    Negative variance proposals are projected to zero. This Euler scheme can
    have substantial boundary bias and is used by Heston projection.
    """
    # bm -> tensor shape (n_paths, 2, len(pi))
    n_samples = bm.shape[0]
    bm_diff = torch.diff(bm, dim=-1)
    N = len(pi)-1
    kappa, theta, sigma, rho, V0 = params
    dt = np.diff(pi)
    X = torch.zeros((n_samples, N + 1), device=device, dtype=bm.dtype)  # log-price paths
    V = torch.zeros((n_samples, N + 1), device=device, dtype=bm.dtype)  # variance paths
    X[:, 0] = log(S0)
    V[:, 0] = V0

    for t in range(N):
        #dW_V = np.random.randn(n_paths) * np.sqrt(dt[t])
        dW_V = bm_diff[:, 0, t]
        dW_S = rho * dW_V + sqrt(1-rho**2) * bm_diff[:, 1, t]
        #dW_S = rho * dW_V + np.sqrt(dt[t]) * np.sqrt(1 - rho**2) * np.random.randn(n_paths)

        V[:, t + 1] = V[:, t] + kappa * (theta - V[:, t]) * dt[t] + sigma * torch.sqrt(V[:, t]) * dW_V
        V[:, t + 1] = torch.clamp(V[:, t + 1], min=0.)

        X[:, t + 1] = X[:, t] + (r - q - 0.5 * V[:, t]) * dt[t] + torch.sqrt(V[:, t]) * dW_S

    S = torch.exp(X)
    S[:, 0] = S0
    if return_last:
        S = S[:, -1]
    return S

_DEFAULT_HESTON_BOUNDS = (
    (1e-3, 50.0), (1e-6, 2.0), (1e-3, 5.0),
    (-0.999, 0.999), (1e-6, 2.0),
)


def heston_surface_initial_guess(
    iv_surface, Ts, strikes, S0, r=0.0, q=0.0, *, bounds=None,
    smile_window=0.05,
):
    """Deterministic Heston start from the supplied implied-volatility surface.

    A quadratic fit to the shortest near-ATM smile gives v0, rho and sigma
    via the short-time Heston expansion of Forde, Jacquier and Lee. The
    long-maturity ATM variance seeds theta; Clark's maturity rule, as reviewed
    by Cui et al., seeds kappa as 2.75 / longest maturity. These approximate
    parameters can initialize the full option-price fit or be used directly
    by the optional heuristic-only chaos initializer.

    Returns (kappa, theta, sigma, rho, v0), diagnostics.
    """
    Ts = np.asarray(Ts, dtype=float)
    if Ts.ndim != 1 or len(Ts) < 1 or not np.isfinite(Ts).all() or np.any(Ts <= 0):
        raise ValueError("At least one positive finite maturity is needed.")
    if len(iv_surface) != len(Ts) or len(strikes) != len(Ts):
        raise ValueError("IV, strike and maturity rows must match.")
    if not np.isfinite(S0) or S0 <= 0 or not np.isfinite(smile_window) or smile_window <= 0:
        raise ValueError("S0 and smile_window must be positive and finite.")

    def term_structure(values, name):
        if np.isscalar(values):
            result = np.full(len(Ts), values, dtype=float)
        else:
            result = np.asarray(values, dtype=float)
        if result.shape != Ts.shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite and aligned with maturities.")
        return result

    rates, dividends = term_structure(r, "r"), term_structure(q, "q")
    bound_array = np.asarray(_DEFAULT_HESTON_BOUNDS if bounds is None else bounds, dtype=float)
    if (bound_array.shape != (5, 2) or not np.isfinite(bound_array).all()
            or np.any(bound_array[:, 0] >= bound_array[:, 1])):
        raise ValueError("bounds must have five finite increasing intervals.")

    atm_ivs, moneyness_rows, smile_rows = [], [], []
    for T, row_strikes, row_ivs, rr, qq in zip(Ts, strikes, iv_surface, rates, dividends):
        K = np.asarray(row_strikes, dtype=float)
        iv = np.asarray(row_ivs, dtype=float)
        if K.ndim != 1 or K.shape != iv.shape or len(K) < 3 or np.any(K <= 0):
            raise ValueError("Each maturity needs at least three positive strikes and IVs.")
        forward = S0 * np.exp((rr - qq) * T)
        k = np.log(K / forward)
        valid = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
        k, iv = k[valid], iv[valid]
        if len(k) < 3 or not (k.min() <= 0 <= k.max()):
            raise ValueError("Each smile needs finite IV quotes around forward ATM.")
        order = np.argsort(k)
        atm_ivs.append(float(np.interp(0.0, k[order], iv[order])))
        moneyness_rows.append(k)
        smile_rows.append(iv)

    shortest = int(np.argmin(Ts))
    k, iv = moneyness_rows[shortest], smile_rows[shortest]
    near = np.abs(k) <= smile_window
    if near.sum() < 3 or not (k[near].min() < 0 < k[near].max()):
        nearest = np.argsort(np.abs(k))[:min(9, len(k))]
        near = np.zeros(len(k), dtype=bool)
        near[nearest] = True
    if near.sum() < 3 or not (k[near].min() < 0 < k[near].max()):
        raise ValueError("The shortest smile lacks quotes on both sides of ATM.")
    a, b, c = map(float, np.polynomial.polynomial.polyfit(k[near], iv[near], 2))
    if a <= 0:
        raise ValueError("Fitted short-maturity ATM volatility must be positive.")
    v0 = a * a
    rho_sigma = 4.0 * a * b
    raw_sigma_squared = 24.0 * a**3 * c + 2.5 * rho_sigma**2
    # Noisy finite-maturity curvature may contradict the asymptotic formula.
    # Keep a valid deterministic start while retaining the observed skew sign.
    sigma = max(np.sqrt(max(raw_sigma_squared, 0.0)), abs(rho_sigma) / 0.9, a)
    rho = rho_sigma / sigma

    atm_variances = np.square(np.asarray(atm_ivs))
    kappa = 2.75 / float(Ts.max())
    theta = float(atm_variances[int(np.argmax(Ts))])
    raw = np.asarray([kappa, theta, sigma, rho, v0], dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Surface-based Heston start is nonfinite.")
    guess = np.clip(raw, bound_array[:, 0] + 1e-8, bound_array[:, 1] - 1e-8)
    diagnostics = {
        "short_maturity": float(Ts[shortest]),
        "short_quotes_used": int(near.sum()),
        "short_smile_atm_iv": a,
        "short_smile_slope": b,
        "short_smile_quadratic_coefficient": c,
        "atm_implied_variances": atm_variances.tolist(),
        "clipped_to_optimizer_bounds": bool(np.any(guess != raw)),
        "smile_curvature_safeguard_used": bool(sigma**2 > raw_sigma_squared + 1e-12),
    }
    return tuple(guess.tolist()), diagnostics


def calibrate_heston_to_iv_market(
    prices_market,      # list[list[float]]  (market CALL prices)
    Ts,                 # list[float]
    strikes,            # list[list[float]] aligned with prices_market
    S0,                 # float
    r=0.0,              # float or list[float] aligned with Ts
    q=0.0,              # float or list[float] aligned with Ts
    *,
    params_limits=((1.0, 1.5), (0.01, 0.4), (0.3, 0.7), (-0.9, -0.6), (0.01, 0.06)),
    seed=None,          # used only by explicit initial_guess="random"
    x0=None,            # explicit starting point overrides the surface heuristic
    initial_guess="surface",  # "surface" (deterministic) or explicit "random"
    bounds=None,
    method="L-BFGS-B",
    options=None,
    feller_penalty=0.0, # e.g. 1e2 to softly encourage 2*kappa*theta >= sigma^2
    vega_weighted=True, # weights = 1/vega^2 computed from IV implied by prices_market
    timing_check=None,  # optional clock-only study deadline
):
    """
    Calibrate Heston parameters (kappa, theta, sigma, rho, v0) to market CALL prices.

    Inputs are prices_market (NOT iv_market). If vega_weighted=True, we compute
    implied vols from prices_market and use BS vegas to weight the price errors.
    If x0 is omitted, the default start is derived deterministically from
    those market-implied vols. Use initial_guess="random" to opt into the
    former seeded uniform start.

    Returns
    -------
    best_params : tuple[float, float, float, float, float]
    res         : scipy.optimize.OptimizeResult
    """
    # Requires these imports in your environment:
    #   import numpy as np
    #   from scipy.optimize import minimize
    #   from .heston import heston_price
    #   from .options import implied_vol_bisection, BlackScholesVegaCore

    # -------------------- checks --------------------
    if len(Ts) != len(prices_market) or len(Ts) != len(strikes):
        raise ValueError("Ts, prices_market, strikes must have the same outer length.")
    for i in range(len(Ts)):
        if len(prices_market[i]) != len(strikes[i]):
            raise ValueError(f"Row {i}: prices_market[i] and strikes[i] must have the same length.")

    # term structures
    if isinstance(r, (int, float)):
        r_list = [float(r)] * len(Ts)
    else:
        if len(r) != len(Ts):
            raise ValueError("If r is a list, it must have the same length as Ts.")
        r_list = [float(x) for x in r]

    if isinstance(q, (int, float)):
        q_list = [float(q)] * len(Ts)
    else:
        if len(q) != len(Ts):
            raise ValueError("If q is a list, it must have the same length as Ts.")
        q_list = [float(x) for x in q]

    # bounds default (used by optimizer)
    if bounds is None:
        bounds = _DEFAULT_HESTON_BOUNDS

    provided_x0 = x0 is not None

    # -------------------- flatten surface --------------------
    T_flat, K_flat, r_flat, q_flat, C_mkt_flat, w_flat = [], [], [], [], [], []
    iv_surface = []
    eps_vega = 1e-10

    for i, T in enumerate(Ts):
        T = float(T)
        rr, qq = r_list[i], q_list[i]
        iv_row = []
        for K, C in zip(strikes[i], prices_market[i]):
            K = float(K)
            C = float(C)

            T_flat.append(T)
            K_flat.append(K)
            r_flat.append(rr)
            q_flat.append(qq)
            C_mkt_flat.append(C)

            if vega_weighted or (x0 is None and initial_guess == "surface"):
                iv = implied_vol_bisection(C, S0, K, T, rr, qq, flag="call")
                iv_row.append(iv)
            if vega_weighted:
                vega = float(BlackScholesVegaCore(max(iv, 1e-12), K, T, S0, rr, qq))
                vega = max(vega, eps_vega)
                w_flat.append(1.0 / (vega * vega))
            else:
                w_flat.append(1.0)
        iv_surface.append(iv_row)

    if x0 is None:
        if initial_guess == "surface":
            x0, _ = heston_surface_initial_guess(
                iv_surface, Ts, strikes, S0, r_list, q_list, bounds=bounds,
            )
        elif initial_guess == "random":
            if params_limits is None or len(params_limits) != 5:
                raise ValueError("params_limits must be five (low, high) intervals.")
            rng = np.random.default_rng(seed)
            x0 = tuple(rng.uniform(low=a, high=b) for a, b in params_limits)
        else:
            raise ValueError("initial_guess must be 'surface' or 'random'.")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (5,) or not np.isfinite(x0).all():
        raise ValueError("x0 must be a finite five-parameter Heston vector.")

    T_flat = np.asarray(T_flat, dtype=float)
    K_flat = np.asarray(K_flat, dtype=float)
    r_flat = np.asarray(r_flat, dtype=float)
    q_flat = np.asarray(q_flat, dtype=float)
    C_mkt_flat = np.asarray(C_mkt_flat, dtype=float)
    w_flat = np.asarray(w_flat, dtype=float)

    n_quotes = C_mkt_flat.size
    if n_quotes == 0:
        raise ValueError("Provide at least one market call price.")

    # The Fourier grid, strike phases and discount factors do not depend on
    # the fitted parameters. Prepare them once per maturity, not on every
    # objective/finite-difference evaluation. Keep this cache local to the fit.
    pricing_groups = []
    for T in np.unique(T_flat):
        idx = np.flatnonzero(T_flat == T)
        pricer = _HestonCalibrationPricer(
            float(T), K_flat[idx], S0, r_flat[idx][0], q_flat[idx][0],
        )
        pricing_groups.append((pricer, C_mkt_flat[idx], w_flat[idx]))

    def objective(p):
        if timing_check is not None:
            timing_check()
        kappa, theta, sigma, rho, v0 = map(float, p)
        params = (kappa, theta, sigma, rho, v0)

        err2 = 0.0
        for pricer, market, weights in pricing_groups:
            diff = pricer.call_prices(params) - market
            err2 += np.sum(weights * diff * diff)

        if feller_penalty and feller_penalty > 0.0:
            viol = max(0.0, sigma * sigma - 2.0 * kappa * theta)
            err2 += float(feller_penalty) * viol * viol

        # Match the chaos calibration's mean squared vega-scaled residual.
        # Dividing by sum(weights) instead can shrink the objective enough
        # for L-BFGS-B's absolute gradient tolerance to accept a poor fit.
        return err2 / n_quotes


    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method=method,
        options={} if options is None else options,
    )
    res.initial_guess = tuple(x0.tolist())
    res.initial_guess_method = "explicit" if provided_x0 else initial_guess
    return tuple(res.x.tolist()), res


def price_exotic_option_Heston(
    params,
    pi,
    functional,
    n_samples_x_batch,
    n_batch,
    r,
    q,
    S0,
    device,
    scheme = "QE",
    **kwargs
):
    if scheme not in ("QE", "Euler"):
        raise ValueError("scheme must be 'QE' or 'Euler'")
    for value in (n_batch, n_samples_x_batch):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Monte Carlo batch sizes and counts must be positive integers")
    param_names, param_lists = option_parameter_grid(kwargs)
    param_sizes = [len(v) for v in param_lists]
    prices = torch.zeros((n_batch, *param_sizes), device=device, dtype=torch.float64)
    T = float(pi[-1])
    discount = math.exp(-float(r) * T)
    index_ranges = [range(s) for s in param_sizes]
    for n in range(n_batch):
        if scheme == "QE":
            stock, _, _ = sim_qe_heston(S0, r, q, params, T, pi, n_samples_x_batch)
            S = torch.from_numpy(stock).to(device)
        else:
            bm = simulate_bm(n_samples_x_batch, 2, pi, device, type_float=torch.float64)
            S = sim_euler_heston(bm, S0, r, q, params, T, pi, device)
        if not torch.isfinite(S).all() or (S < 0).any():
            raise FloatingPointError("Invalid simulated spot paths")
        for idx_tuple in itertools.product(*index_ranges):
            current = {name: param_lists[i][idx_tuple[i]] for i, name in enumerate(param_names)}
            payoff = evaluate_payoff(functional, S, current)
            prices[(n, *idx_tuple)] = payoff.mean(dtype=torch.float64) * discount
    return prices.mean(dim=0)


def _phi_heston_vec(xi, params, r, q, T, S0, typ):
    """
    Vectorized Heston characteristic function φ(i*xi) for xi array.
    Returns complex array of shape (n_xi,).
    """
    kappa, theta, sigma, rho, v0 = params
    ixi = 1j * xi  # (n_xi,)

    if typ == 1:
        w = 1.0
        b = kappa - rho * sigma
    else:
        w = -1.0
        b = kappa

    d = np.sqrt((rho * sigma * ixi - b) ** 2 - sigma ** 2 * (w * ixi - xi ** 2))
    g = (b - rho * sigma * ixi - d) / (b - rho * sigma * ixi + d)
    e_dt = np.exp(-d * T)

    C = (r - q) * ixi * T + (kappa * theta / sigma ** 2) * (
        (b - rho * sigma * ixi - d) * T - 2.0 * np.log((1.0 - g * e_dt) / (1.0 - g))
    )
    D = ((b - rho * sigma * ixi - d) / sigma ** 2) * ((1.0 - e_dt) / (1.0 - g * e_dt))

    return np.exp(C + D * v0 + ixi * np.log(S0))  # (n_xi,)


class _HestonCalibrationPricer:
    """Reusable fixed-grid Fourier pricing for one calibration maturity.

    Use the same grid and summation order as heston_price_vec so caching does
    not trade quadrature accuracy for speed or perturb numerical gradients.
    """

    def __init__(self, T, strikes, S0, r, q):
        self.T, self.S0, self.r, self.q = T, S0, r, q
        self.xi = np.linspace(1e-8, 200.0, 4096, dtype=float)
        self.ixi = 1j * self.xi[:, None]
        self.dxi = self.xi[1] - self.xi[0]
        strikes = np.asarray(strikes, dtype=float)
        self.phase = np.exp(-1j * self.xi[:, None] * np.log(strikes)[None, :])
        self.discounted_spot = S0 * np.exp(-q * T)
        self.discounted_strikes = strikes * np.exp(-r * T)

    def _probability(self, params, typ):
        phi = _phi_heston_vec(
            self.xi, params, self.r, self.q, self.T, self.S0, typ,
        )
        integrand = np.real(self.phase * (phi[:, None] / self.ixi))
        integral = self.dxi * (
            0.5 * integrand[0] + integrand[1:-1].sum(axis=0) + 0.5 * integrand[-1]
        )
        return 0.5 + (1.0 / np.pi) * integral

    def call_prices(self, params):
        p1 = self._probability(params, 1)
        p2 = self._probability(params, 2)
        return self.discounted_spot * p1 - self.discounted_strikes * p2


def _P_typ_vec(params, r, q, T, S0, K_vec, typ, xi_max=200.0, n_xi=4096):
    """
    Vectorized P1/P2 on a fixed xi-grid.

    Returns:
      P : (nK,) float
    """
    K_vec = np.asarray(K_vec, dtype=float)
    logK = np.log(K_vec)  # (nK,)

    # xi-grid (avoid 0 to prevent division by 0)
    eps = 1e-8
    xi = np.linspace(eps, xi_max, n_xi, dtype=float)  # (n_xi,)
    dxi = xi[1] - xi[0]

    phi = _phi_heston_vec(xi, params, r, q, T, S0, typ)  # (n_xi,) complex

    # exp(-i xi logK): (n_xi, nK)
    E = np.exp(-1j * xi[:, None] * logK[None, :])

    # integrand = Re{ exp(-i xi logK) * phi(xi) / (i xi) }
    integrand = np.real(E * (phi[:, None] / (1j * xi[:, None])))  # (n_xi, nK)

    # Trapezoidal rule over xi
    integ = dxi * (0.5 * integrand[0] + integrand[1:-1].sum(axis=0) + 0.5 * integrand[-1])  # (nK,)

    return 0.5 + (1.0 / np.pi) * integ  # (nK,)


def heston_price_vec(params, r, q, T, S0, K_vec, flag="call", xi_max=200.0, n_xi=4096):
    """
    Vectorized call/put prices for strikes K_vec.
    Returns np.ndarray shape (nK,).
    """
    K_vec = np.asarray(K_vec, dtype=float)

    P1 = _P_typ_vec(params, r, q, T, S0, K_vec, typ=1, xi_max=xi_max, n_xi=n_xi)  # (nK,)
    P2 = _P_typ_vec(params, r, q, T, S0, K_vec, typ=2, xi_max=xi_max, n_xi=n_xi)  # (nK,)

    call = S0 * np.exp(-q * T) * P1 - K_vec * np.exp(-r * T) * P2  # (nK,)

    if flag.lower() == "call":
        return call
    elif flag.lower() == "put":
        return call - S0 * np.exp(-q * T) + K_vec * np.exp(-r * T)
    else:
        raise ValueError("flag must be 'call' or 'put'")


def heston_forward_start_price(params, r, q, T_start, T_end, S0,
                                moneyness_levels, pi, npaths=100000, device="cpu", scheme="QE"):
    """
    Price forward-starting call options under Heston via Monte Carlo.

    A forward-starting option has strike set at T_start as K = m * S(T_start),
    where m is the moneyness level, and pays max(S(T_end) - K, 0) at T_end.

    Parameters:
    -----------
    params : tuple - (kappa, theta, sigma, rho, v0)
    r : float - Risk-free rate
    q : float - Dividend yield
    T_start : float - Time when strike is set
    T_end : float - Maturity
    S0 : float - Initial spot price
    moneyness_levels : array-like - Strike = m * S(T_start) for each m
    pi : array-like - Time grid for simulation [0, ..., T_end]
    npaths : int - Number of MC paths
    device : str - PyTorch device ("cpu" or "cuda")

    Returns:
    --------
    prices : ndarray - Forward-start call prices for each moneyness
    std_errors : ndarray - Standard errors for each price
    """
    if not np.isfinite([T_start, T_end]).all() or not 0 <= T_start <= T_end:
        raise ValueError("Require 0 <= T_start <= T_end")
    pi = np.asarray(pi, dtype=float)
    if (pi.ndim != 1 or not len(pi) or pi[0] != 0 or pi[-1] != T_end
            or np.any(np.diff(pi) <= 0) or not np.isfinite(pi).all()):
        raise ValueError("The simulation grid must increase from zero to T_end")
    if npaths < 2:
        raise ValueError("At least two paths are needed for a Monte Carlo standard error")
    moneyness_levels = np.asarray(moneyness_levels, dtype=float)
    if not np.isfinite(moneyness_levels).all() or np.any(moneyness_levels < 0):
        raise ValueError("Forward-start moneyness must be finite and nonnegative")
    pi = np.unique(np.r_[pi, T_start])
    if scheme == "QE":
        S, _, _ = sim_qe_heston(S0, r, q, params, T_end, pi, npaths)
    elif scheme == "Euler":
        bm = simulate_bm(npaths, 2, pi, device, type_float=torch.float64)
        S = sim_euler_heston(bm, S0, r, q, params, T_end, pi, device).cpu().numpy()
    else:
        raise ValueError("scheme must be 'QE' or 'Euler'")
    idx_start = int(np.searchsorted(pi, T_start))

    S_start = S[:, idx_start]  # Spot at T_start
    S_end = S[:, -1]           # Spot at T_end

    prices = np.zeros(len(moneyness_levels))
    std_errors = np.zeros(len(moneyness_levels))

    for k, m in enumerate(moneyness_levels):
        K = m * S_start  # Strike set at T_start
        payoff = np.maximum(S_end - K, 0)
        prices[k] = np.exp(-r * T_end) * np.mean(payoff)
        std_errors[k] = np.exp(-r * T_end) * np.std(payoff, ddof=1) / np.sqrt(npaths)

    return prices, std_errors
