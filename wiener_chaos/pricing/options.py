"""Common option payoffs, Black–Scholes formulas, and implied volatilities.

Includes vanilla, floating-strike lookback, down-and-out barrier, and
forward-start options. Provides price inversion, vega weights, payoff-parameter
grids, and evaluation on simulated paths. Chaos and reference-model pricing
share these routines to use the same payoff definitions.
"""

from collections.abc import Sequence
from inspect import signature
from math import exp, expm1, isfinite, log, pi, sqrt
from numbers import Integral
from typing import Union

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm


# -------------------- BS IMPLIED VOL (BISECTION) --------------------

def _norm_cdf_tensor(x: torch.Tensor) -> torch.Tensor:
    # erfc retains the small lower-tail probabilities lost by 1 + erf.
    return 0.5 * torch.erfc(-x / sqrt(2.0))

def _norm_pdf(x):  # standard normal PDF
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def _bs_price(flag, S0, K, T, r, q, vol):
    """Scalar Black–Scholes price with continuous dividend yield q."""
    flag = flag.lower()
    if flag not in ("call", "put"):
        raise ValueError("flag must be 'call' or 'put'")
    if not all(isfinite(x) for x in (S0, K, T, r, q, vol)):
        raise ValueError("Black–Scholes inputs must be finite")
    if S0 <= 0 or K < 0 or T < 0 or vol < 0:
        raise ValueError("Require S0 > 0, K >= 0, T >= 0, and vol >= 0")

    spot_pv, strike_pv = S0 * exp(-q * T), K * exp(-r * T)
    if vol == 0 or T == 0 or K == 0:
        return max(spot_pv - strike_pv, 0.) if flag == "call" else max(strike_pv - spot_pv, 0.)

    std_dev = vol * sqrt(T)
    d1 = (log(S0 / K) + (r - q) * T) / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    # Price the OTM side using a tail-safe CDF, then apply put–call parity.
    # This avoids subtracting nearly equal ITM prices to obtain time value.
    if spot_pv <= strike_pv:
        call = max(spot_pv * ndtr(d1) - strike_pv * ndtr(d2), 0.)
        return float(call if flag == "call" else call + (strike_pv - spot_pv))
    put = max(strike_pv * ndtr(-d2) - spot_pv * ndtr(-d1), 0.)
    return float(put if flag == "put" else put + (spot_pv - strike_pv))


def implied_vol_bisection(price, S0, K, T, r, q, flag="call",
                          v_min=1e-12, v_max=5.0, tol=1e-8, max_iter=100):
    """European IV; tol is an absolute volatility tolerance, not a price tolerance.

    At the zero-volatility price return 0; at the upper price bound return
    infinity. Out-of-bounds/nonfinite prices or expired/zero-strike contracts
    return NaN because no finite, identifiable IV exists. Invalid contract
    inputs or solver settings raise. The initial bracket can expand both ways.
    Roundoff at a positive intrinsic value is treated as the zero-vol boundary;
    strictly positive OTM prices, however small, are inverted without clamping.
    """
    lower = _bs_price(flag, S0, K, T, r, q, 0.)
    if not all(isfinite(x) for x in (v_min, v_max, tol)):
        raise ValueError("IV solver settings must be finite")
    if not 0 <= v_min < v_max or tol <= 0 or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("Invalid IV solver settings")
    if not isfinite(price) or T == 0 or K == 0:
        return float("nan")
    upper = S0 * exp(-q * T) if flag.lower() == "call" else K * exp(-r * T)
    if lower > 0 and abs(price-lower) <= 64*np.finfo(float).eps*max(upper, 1.):
        return 0.0
    if price < lower or price > upper:
        return float("nan")
    if price == lower:
        return 0.0
    if price == upper:
        return float("inf")

    a, b = v_min, v_max
    if _bs_price(flag, S0, K, T, r, q, a) > price:
        a = 0.0
    for _ in range(8):
        if _bs_price(flag, S0, K, T, r, q, b) >= price:
            break
        b *= 2.0
    if _bs_price(flag, S0, K, T, r, q, b) < price:
        raise RuntimeError("Could not bracket the European implied volatility")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        value = _bs_price(flag, S0, K, T, r, q, m)
        if b - a <= tol:
            return m
        if value < price:
            a = m
        else:
            b = m
    raise RuntimeError("European implied-volatility bisection did not converge")


def compute_ivs(prices, maturities, strikes, S0, r, q):
    """Call IV surface with scalar or per-maturity rates and dividend yields."""
    count = len(maturities)
    if len(prices) != count or len(strikes) != count:
        raise ValueError("Prices, strikes, and maturities must have the same number of rows")
    rates, dividends = np.asarray(r, dtype=float), np.asarray(q, dtype=float)
    rates = np.full(count, float(rates)) if rates.ndim == 0 else rates
    dividends = np.full(count, float(dividends)) if dividends.ndim == 0 else dividends
    if rates.shape != (count,) or dividends.shape != (count,):
        raise ValueError("r and q must be scalars or have one entry per maturity")
    ivs = []
    for t, T in enumerate(maturities):
        if len(prices[t]) != len(strikes[t]):
            raise ValueError(f"Different numbers of prices and strikes at maturity row {t}")
        ivs.append([
            implied_vol_bisection(price, S0, K, T, rates[t], dividends[t], flag="call")
            for price, K in zip(prices[t], strikes[t])
        ])
    return ivs


def bs_call_price_vec(S0, K_vec, T, r, q, sigma, device, dtype):
    """Torch call prices, with the same limits and tail handling as _bs_price."""
    S0 = torch.as_tensor(S0, device=device, dtype=dtype)
    K  = torch.as_tensor(K_vec, device=device, dtype=dtype)
    Tt = torch.as_tensor(T, device=device, dtype=dtype)
    r  = torch.as_tensor(r, device=device, dtype=dtype)
    q  = torch.as_tensor(q, device=device, dtype=dtype)
    sig= torch.as_tensor(sigma, device=device, dtype=dtype)

    if any(not torch.isfinite(x).all() for x in (S0, K, Tt, r, q, sig)):
        raise ValueError("Black–Scholes inputs must be finite")
    if (S0 <= 0).any() or (K < 0).any() or (Tt < 0).any() or (sig < 0).any():
        raise ValueError("Require S0 > 0, K >= 0, T >= 0, and sigma >= 0")
    spot_pv, strike_pv = S0 * torch.exp(-q*Tt), K * torch.exp(-r*Tt)
    intrinsic = (spot_pv - strike_pv).clamp_min(0.)
    std_dev = sig * torch.sqrt(Tt)
    safe_std = torch.where(std_dev > 0, std_dev, torch.ones_like(std_dev))
    safe_K = torch.where(K > 0, K, torch.ones_like(K))
    d1 = (torch.log(S0 / safe_K) + (r - q) * Tt) / safe_std + 0.5 * safe_std
    d2 = d1 - safe_std
    call = (spot_pv * _norm_cdf_tensor(d1) - strike_pv * _norm_cdf_tensor(d2)).clamp_min(0.)
    put = (strike_pv * _norm_cdf_tensor(-d2) - spot_pv * _norm_cdf_tensor(-d1)).clamp_min(0.)
    price = torch.where(spot_pv <= strike_pv, call, put + (spot_pv - strike_pv))
    return torch.where((std_dev > 0) & (K > 0), price, intrinsic)


def BlackScholesVegaCore(vol, K, T, S0, r, q):
    """Scalar BS vega per unit volatility, including the right limit at zero."""
    if not all(isfinite(x) for x in (S0, K, T, r, q, vol)):
        raise ValueError("Black–Scholes inputs must be finite")
    if S0 <= 0 or K < 0 or T < 0 or vol < 0:
        raise ValueError("Require S0 > 0, K >= 0, T >= 0, and vol >= 0")
    if T == 0 or K == 0:
        return 0.0
    sqrtT = sqrt(T)
    if vol == 0:
        return S0 * exp(-q*T) * _norm_pdf(0.) * sqrtT if S0 * exp(-q*T) == K * exp(-r*T) else 0.0
    d1 = (log(S0 / K) + (r - q + 0.5 * vol * vol) * T) / (vol * sqrtT)
    return S0 * exp(-q * T) * _norm_pdf(d1) * sqrtT

def bs_lookback_floating_strike(S0, T, r, q, sigma, flag="call"):
    """Continuous-monitoring floating-strike lookback price at inception.

    The initial minimum/maximum is S0. Call payoff: S_T - min S_t;
    put payoff: max S_t - S_T, including time zero in the extrema.
    Uses the standard formula and its analytic zero-carry limit; see
    QuantLib's AnalyticContinuousFloatingLookbackEngine::A.
    """
    flag = flag.lower()
    if flag not in ("call", "put"):
        raise ValueError("flag must be 'call' or 'put'")
    if not all(isfinite(x) for x in (S0, T, r, q, sigma)):
        raise ValueError("Lookback inputs must be finite")
    if S0 <= 0 or T < 0 or sigma < 0:
        raise ValueError("Require S0 > 0, T >= 0, and sigma >= 0")
    if T == 0:
        return 0.0

    carry = r - q
    disc_r, disc_q = exp(-r * T), exp(-q * T)
    eta = 1.0 if flag == "call" else -1.0
    if sigma == 0:
        # A deterministic increasing/decreasing path can still have a payoff.
        return max(eta * S0 * disc_r * expm1(carry * T), 0.0)

    std_dev = sigma * sqrt(T)
    carry_ratio = 2.0 * carry / (sigma * sigma)
    d1 = carry * T / std_dev + 0.5 * std_dev
    d2 = d1 - std_dev
    if abs(carry_ratio) < 1e-7 and abs(carry*T) < 1e-7:
        # Limit of [N(eta*(-d1+lambda*std_dev))
        #           - exp(carry*T)*N(-eta*d1)] / lambda as lambda -> 0.
        correction = std_dev * (
            eta * _norm_pdf(d1) - d1 * norm.cdf(-eta * d1)
        )
    else:
        d3 = -d1 + carry_ratio * std_dev
        n4 = norm.cdf(-eta*d1)
        correction = (
            norm.cdf(eta*d3) - n4 - expm1(carry*T)*n4
        ) / carry_ratio
    price = eta * S0 * (
        disc_q * norm.cdf(eta * d1) - disc_r * norm.cdf(eta * d2)
        + disc_r * correction
    )
    return max(float(price), 0.0)


def floating_strike_lookback_implied_vol(price, S0, T, r, q, flag="call",
                         v_min=1e-6, v_max=5.0, tol=1e-8, max_iter=100):
    """Invert the continuously monitored lookback price by bisection.

    v_min/v_max provide the initial bracket; it is extended towards zero
    or higher volatility when necessary. Impossible prices and failed
    brackets raise instead of returning a misleading endpoint volatility.
    Prices at the zero-volatility value return zero; tol is in volatility units.
    """
    if not all(isfinite(x) for x in (price, v_min, v_max, tol)):
        raise ValueError("Price and solver inputs must be finite")
    if not 0 <= v_min < v_max or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid lookback IV solver settings")
    lower = bs_lookback_floating_strike(S0, T, r, q, 0.0, flag)
    if lower > 0 and abs(price-lower) <= 64*np.finfo(float).eps*max(S0*exp(-q*T), 1.):
        return 0.0
    if price < lower:
        raise ValueError("Lookback price is below its zero-volatility value")
    if T == 0:
        if price == 0:
            return 0.0
        raise ValueError("A lookback option at inception with T=0 has zero payoff")
    if flag.lower() == "call" and price >= S0 * exp(-q * T):
        raise ValueError("Lookback call price must be below discounted spot")
    if price == lower:
        return 0.0

    a, b = v_min, v_max
    if bs_lookback_floating_strike(S0, T, r, q, a, flag) > price:
        a = 0.0
    fb = bs_lookback_floating_strike(S0, T, r, q, b, flag) - price
    for _ in range(8):
        if fb >= 0:
            break
        b *= 2.0
        fb = bs_lookback_floating_strike(S0, T, r, q, b, flag) - price
    if fb < 0:
        raise ValueError("Could not bracket the lookback implied volatility")

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = bs_lookback_floating_strike(S0, T, r, q, m, flag) - price
        if b - a < tol:
            return m
        if fm < 0:
            a = m
        else:
            b = m
    raise RuntimeError("Lookback implied-volatility bisection did not converge")


def bs_barrier_down_out_call(S0, K, T, r, q, sigma, H):
    """Continuously monitored down-and-out call, no rebate, touch knocks out.

    Reiner–Rubinstein formula: A-C when K >= H, B-D when K < H.
    Asset and strike image terms have different powers of H/S0.
    """
    vanilla = _bs_price("call", S0, K, T, r, q, sigma)
    if not isfinite(H) or H < 0 or K <= 0:
        raise ValueError("Require finite H >= 0 and K > 0")
    if S0 <= H:
        return 0.0
    if H == 0 or T == 0:
        return vanilla
    if sigma == 0:
        terminal = S0 * exp((r-q)*T)
        return vanilla if min(S0, terminal) > H else 0.0

    std_dev = sigma * sqrt(T)
    mu = (r-q)/(sigma*sigma) - 0.5
    log_ratio = log(H/S0)
    log_spot_pv, log_strike_pv = log(S0)-q*T, log(K)-r*T
    cutoff = max(K, H)
    x = log(S0/cutoff)/std_dev + (mu+1)*std_dev
    y = log(H*H/(S0*cutoff))/std_dev + (mu+1)*std_dev
    # Log-CDF products avoid overflow of the image powers at low volatility.
    direct = exp(log_spot_pv+log_ndtr(x)) - exp(log_strike_pv+log_ndtr(x-std_dev))
    image = (exp(log_spot_pv+2*(mu+1)*log_ratio+log_ndtr(y))
             - exp(log_strike_pv+2*mu*log_ratio+log_ndtr(y-std_dev)))
    return float(np.clip(direct-image, 0., vanilla))


def barrier_implied_vol(price, S0, K, T, r, q, H, barrier_type="down-and-out",
                        option_type="call", v_min=1e-6, v_max=5.0, tol=1e-8, max_iter=100):
    """A unique continuous-barrier IV within [v_min, v_max], otherwise NaN.

    Barrier prices need not be monotone in volatility. Search the requested
    range for sign-changing roots, and reject multiple roots rather than
    reporting an arbitrary branch or an unbracketed endpoint.
    """
    if barrier_type.lower() != "down-and-out" or option_type.lower() != "call":
        raise ValueError("Only down-and-out calls are supported")
    lower = bs_barrier_down_out_call(S0, K, T, r, q, 0., H)
    if (not all(isfinite(x) for x in (v_min, v_max, tol))
            or not 0 <= v_min < v_max or tol <= 0 or max_iter < 1):
        raise ValueError("Invalid barrier IV solver settings")
    if not isfinite(price) or price < 0 or T == 0 or S0 <= H:
        return float("nan")
    if H == 0:
        return implied_vol_bisection(price, S0, K, T, r, q, "call", v_min, v_max, tol, max_iter)
    if r == q and K == H:
        # Optional stopping: the price is exp(-r*T)*(S0-H) for every sigma.
        return float("nan")
    if price == 0:
        return 0.0 if lower == 0 else float("nan")
    if price >= S0*exp(-q*T):
        return float("nan")

    def residual(sigma):
        return bs_barrier_down_out_call(S0, K, T, r, q, sigma, H)-price

    if r == q:
        # With zero carry, K>H is increasing and K<H is decreasing in vol.
        if price == lower:
            return 0.0
        fa, fb = residual(v_min), residual(v_max)
        if (fa < 0 and fb < 0) or (fa > 0 and fb > 0):
            return float("nan")
        return brentq(residual, v_min, v_max, xtol=tol, maxiter=max_iter)

    # Both a logarithmic grid (small vol) and a linear grid (large vol)
    # resolve the nonmonotone branch over the explicitly requested range.
    grid = np.unique(np.r_[v_min, np.geomspace(max(v_min, 1e-10), v_max, 128),
                           np.linspace(v_min, v_max, 128)])
    values = [residual(v) for v in grid]
    roots = []
    for i, value in enumerate(values):
        if value == 0:
            roots.append(float(grid[i]))
        if i and ((values[i-1] < 0 < value) or (value < 0 < values[i-1])):
            roots.append(brentq(residual, grid[i-1], grid[i], xtol=tol, maxiter=max_iter))
    distinct = []
    for root in sorted(roots):
        if not distinct or root-distinct[-1] > 2*tol:
            distinct.append(root)
    return distinct[0] if len(distinct) == 1 else float("nan")


def _bs_forward_start_price(flag, S0, T_start, T_end, r, q, sigma, moneyness):
    if not all(isfinite(x) for x in (T_start, T_end, moneyness)):
        raise ValueError("Forward-start inputs must be finite")
    if T_start < 0 or T_end < T_start or moneyness < 0:
        raise ValueError("Require 0 <= T_start <= T_end and moneyness >= 0")
    vanilla = _bs_price(flag, S0, moneyness*S0, T_end-T_start, r, q, sigma)
    return exp(-q*T_start)*vanilla


def bs_forward_start_call(S0, T_start, T_end, r, q, sigma, moneyness):
    """Price max(S_end - moneyness*S_start, 0), paid at T_end.

    Homogeneity and independent BS returns give exp(-q*T_start) times
    the vanilla call with spot S0, strike moneyness*S0, tenor T_end-T_start.
    """
    return _bs_forward_start_price("call", S0, T_start, T_end, r, q, sigma, moneyness)


def bs_forward_start_put(S0, T_start, T_end, r, q, sigma, moneyness):
    """Price max(moneyness*S_start - S_end, 0), paid at T_end."""
    return _bs_forward_start_price("put", S0, T_start, T_end, r, q, sigma, moneyness)


def forward_start_implied_vol(price, S0, T_start, T_end, r, q, moneyness,
                               option_type="call", v_min=1e-6, v_max=5.0,
                               tol=1e-8, max_iter=100):
    """Forward-start BS IV using the audited vanilla inversion."""
    _bs_forward_start_price(option_type, S0, T_start, T_end, r, q, 0., moneyness)
    return implied_vol_bisection(
        price*exp(q*T_start), S0, moneyness*S0, T_end-T_start, r, q,
        option_type, v_min, v_max, tol, max_iter,
    )


def get_vegas(iv_market, maturities, strikes, S0, r_s, q_s, floor_vega = False, alpha = None, normalize = True):
    'Compute vega weights'
    vega_weights = []
    for t, T in enumerate(maturities):
        vega_weights_t = []
        for k, K in enumerate(strikes[t]):
            vega = BlackScholesVegaCore(iv_market[t][k], K, T, S0, r_s[t], q_s[t])
            if floor_vega:
                idx = np.abs(np.array(strikes[t]) - S0).argmin()
                vega_atm = BlackScholesVegaCore(iv_market[t][idx], S0, T, S0, r_s[t], q_s[t])
                vega = max(vega, alpha*vega_atm)

            vega_weights_t.append(1/vega**2)

        if normalize:
            vega_weights_t = [x / sum(vega_weights_t) for x in vega_weights_t]

        vega_weights.append(vega_weights_t)

    flat_weights = [x for row in vega_weights for x in row]

    return flat_weights, vega_weights


def option_parameter_grid(parameters):
    """Normalize scalar/one-dimensional option parameters for Cartesian pricing."""
    names, values = [], []
    for name, value in parameters.items():
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            if value.ndim > 1:
                raise ValueError(f"Parameter '{name}' must be scalar or one-dimensional")
            value = [value.item()] if value.ndim == 0 else value.tolist()
        elif isinstance(value, str) or not isinstance(value, Sequence):
            value = [value]
        if len(value) == 0:
            raise ValueError(f"Empty parameter grid for '{name}'")
        names.append(name)
        values.append(value)
    return names, values


def evaluate_payoff(functional, paths, parameters):
    """Support keyword/dictionary callbacks without masking errors in their body."""
    try:
        sig = signature(functional)
    except (TypeError, ValueError):
        value = functional(paths, **parameters)
    else:
        try:
            sig.bind(paths, **parameters)
        except TypeError:
            sig.bind(paths, parameters)
            value = functional(paths, parameters)
        else:
            value = functional(paths, **parameters)
    value = torch.as_tensor(value, device=paths.device)
    if value.ndim != 1 or value.shape[0] != paths.shape[0]:
        raise ValueError("The payoff must return one value per simulated path")
    if not torch.isfinite(value).all():
        raise FloatingPointError("The payoff produced nonfinite values")
    return value

def payoff_floating_strike_lookback(S: torch.Tensor, flag: str = "call") -> torch.Tensor:
    """
    Floating-strike lookback payoff from sampled paths.

    Parameters
    ----------
    S : torch.Tensor
        Shape (n_samples, n_times). Each row is one path on time grid pi.
        Assumes S[:, 0] is the initial value S0 and S[:, -1] is the terminal value S_T.
    flag : {"call","put"}
        "call": payoff = (S_T - min_t S_t)_+
        "put" : payoff = (max_t S_t - S_T)_+

    Returns
    -------
    torch.Tensor
        Shape (n_samples,), pathwise payoffs.
    """
    ST = S[:, -1]
    f = flag.lower()
    if f == "call":
        Smin = S.min(dim=1).values
        return torch.clamp(ST - Smin, min=0.0)
    elif f == "put":
        Smax = S.max(dim=1).values
        return torch.clamp(Smax - ST, min=0.0)
    else:
        raise ValueError("flag must be 'call' or 'put'")


def payoff_down_and_out_barrier(
    S: torch.Tensor,
    K: Union[float, torch.Tensor],
    H: Union[float, torch.Tensor],
    flag: str = "call",
    knock_on_touch: bool = True,
) -> torch.Tensor:
    """
    Down-and-out barrier option payoff from sampled paths.

    S: (n_samples, n_times)
    Call payoff (if not knocked out): (S_T - K)+
    Put  payoff (if not knocked out): (K - S_T)+
    Knock-out condition: min_t S_t <= H (or < H if knock_on_touch=False)
    """
    ST = S[:, -1]

    K_t = torch.as_tensor(K, device=S.device, dtype=S.dtype)
    H_t = torch.as_tensor(H, device=S.device, dtype=S.dtype)
    if (not torch.isfinite(K_t).all() or not torch.isfinite(H_t).all()
            or (K_t < 0).any() or (H_t < 0).any()):
        raise ValueError("Strike and barrier must be finite and nonnegative")

    Smin = S.min(dim=1).values
    if not torch.isfinite(Smin).all():
        raise FloatingPointError("Nonfinite path minimum")
    knocked_out = (Smin <= H_t) if knock_on_touch else (Smin < H_t)

    f = flag.lower()
    if f == "call":
        vanilla = torch.clamp(ST - K_t, min=0.0)
    elif f == "put":
        vanilla = torch.clamp(K_t - ST, min=0.0)
    else:
        raise ValueError("flag must be 'call' or 'put'")

    return torch.where(knocked_out, torch.zeros_like(vanilla), vanilla)


def payoff_forward_start(
    S: torch.Tensor,
    T_start_idx: int,
    moneyness: Union[float, torch.Tensor],
    flag: str = "call",
) -> torch.Tensor:
    """
    Forward-starting option payoff from sampled paths.

    At time T_start (given by index T_start_idx), the strike is set as K = moneyness * S(T_start).
    The payoff at maturity T_end is max(S(T_end) - K, 0) for calls.

    Parameters
    ----------
    S : torch.Tensor
        Shape (n_samples, n_times). Each row is one path on time grid pi.
        S[:, T_start_idx] is the spot at T_start.
        S[:, -1] is the terminal value S_T.
    T_start_idx : int
        Index in the path corresponding to T_start.
    moneyness : float or torch.Tensor
        Strike = moneyness * S(T_start). E.g., 1.0 = ATM, 0.9 = 10% ITM call.
    flag : {"call", "put"}
        "call": payoff = (S_T - K)+
        "put" : payoff = (K - S_T)+

    Returns
    -------
    torch.Tensor
        Shape (n_samples,), pathwise payoffs.
    """
    if (isinstance(T_start_idx, (bool, np.bool_)) or not isinstance(T_start_idx, Integral)
            or not 0 <= T_start_idx < S.shape[1]):
        raise ValueError("T_start_idx must identify a date in the simulated path")
    S_start = S[:, T_start_idx]  # Spot at T_start
    S_end = S[:, -1]             # Spot at T_end

    m = torch.as_tensor(moneyness, device=S.device, dtype=S.dtype)
    if not torch.isfinite(m).all() or (m < 0).any():
        raise ValueError("Forward-start moneyness must be finite and nonnegative")
    K = m * S_start              # Strike set at T_start

    f = flag.lower()
    if f == "call":
        return torch.clamp(S_end - K, min=0.0)
    elif f == "put":
        return torch.clamp(K - S_end, min=0.0)
    else:
        raise ValueError("flag must be 'call' or 'put'")
