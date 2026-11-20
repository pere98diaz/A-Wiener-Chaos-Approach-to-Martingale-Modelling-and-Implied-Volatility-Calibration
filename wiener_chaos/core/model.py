"""Stock-model interface used by the calibration and pricing notebooks.

ModelStock combines a chaos basis and evaluator with market data and calibrated
coefficients. It handles Heston-based or random initialization, Monte Carlo and
Gauss–Hermite pricing, control variates, coefficient optimization, and exotic
payoffs. Experiment notebooks supply the model settings and sampling budgets.
"""

import itertools
import math
import time
from collections import OrderedDict
from functools import partial
from math import exp, sqrt
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from .basis import WienerChaosDyson, WienerChaosPiecewiseConstant
from .chaos import ChaosGeneral, ChaosMartingale
from ..pricing.heston import calibrate_heston_to_iv_market, heston_surface_initial_guess, sim_euler_heston
from ..pricing.options import compute_ivs, evaluate_payoff, get_vegas, option_parameter_grid
from .utils import sample_tuple


Tensor = torch.Tensor


def _r6(x: float) -> float:
    return float(np.round(x, 6))


class ModelStock:
    def __init__(self, S0: float, device: torch.device, type_float: torch.dtype = torch.float32) -> None:
        """A stock with normalized conditional-square chaos dynamics."""
        self.S0 = float(S0)
        self.device = device
        self.type_float = type_float

        self.r_T, self.q_T = {}, {}
        self.rescaling_rq = {}
        # Quadrature features do not depend on coefficients or strikes. Bound the
        # retained tensors so exploratory node counts cannot fill device memory.
        self._quadrature_cache = OrderedDict()
        self._quadrature_cache_bytes = 0
        self._quadrature_cache_limit_bytes = 128 * 1024**2

    def initialize_chaos(
        self,
        type_chaos: str,
        T_horizon: float,
        d: int,
        P: int,
        M: int,
        pi_M: Optional[List[float]] = None,
        base_jitter: float = None,
        backend: str = "auto",
    ) -> None:
        """
        type_chaos:
          - "piecewise_constant"         (requires pi_M)
          - "piecewise_constant_dyson"   (Dyson with piecewise basis)
          - anything else gets passed as type_basis to WienerChaosDyson (e.g. "sine", "cosine", "fourier", ...)

        T_horizon: horizon
        d:         BM dimension
        P:         chaos order
        M:         number of 1D basis functions (so M*d total in L^2([0,T];R^d))
        pi_M:      only used for "piecewise_constant"
        base_jitter: numerical jitter for Dyson cholesky
        backend: "auto" (tensor for P=2/3, derivative above P=3),
                         "tensor", "derivative", or "grouped" for piecewise chaos;
                         general bases use the full-covariance tensor evaluator
        """
        self.clear_quadrature_cache()
        self.type_chaos = str(type_chaos)
        self.pi_M = pi_M
        self.d = int(d)
        self.P = int(P)
        self.T_horizon = T_horizon
        if backend == "auto":
            backend = (
                "derivative" if self.type_chaos == "piecewise_constant" and P > 3
                else "tensor"
            )
        if self.type_chaos == "piecewise_constant":
            if P < 2:
                raise ValueError("The piecewise chaos model requires P >= 2.")
            if P > 3 and backend not in ("derivative", "grouped"):
                raise ValueError(
                    "Orders above three require backend='derivative' or 'grouped'."
                )
        elif P not in (2, 3) or backend != "tensor":
            raise ValueError(
                "General-basis chaos supports P=2 or P=3 with backend='tensor'."
            )

        if self.type_chaos == "piecewise_constant":
            if pi_M is None:
                raise ValueError("You need to provide pi_M for type_chaos='piecewise_constant'.")

            if len(pi_M) - 1 != M:
                raise ValueError(
                    f"Expected pi_M to have {M+1} points (so M={M} intervals), got {len(pi_M)}. "
                    "Remember to include 0 at the beginning."
                )

            if pi_M[-1] != T_horizon:
                raise ValueError(f"pi_M[-1] should be {T_horizon}, got {pi_M[-1]}.")

            self.wienerchaos = WienerChaosPiecewiseConstant(d, P, pi_M, self.device, self.type_float)
            self.chaos = ChaosMartingale(self.wienerchaos, self.S0)
            self.chaos.set_backend(backend)
            return
        else:
            # Dyson variants
            if base_jitter is None:
                raise ValueError("You need to provide base_jitter for Dyson-based chaos.")

            self.wienerchaos = WienerChaosDyson(
                d, P, M, T_horizon, self.device,
                type_basis=self.type_chaos,
                type_float=self.type_float,
                base_jitter=base_jitter,
            )
            self.chaos = ChaosGeneral(self.wienerchaos, self.S0)

    def initialize_params(self, type="random", **kwargs):
        """
        type:
        - "random"
        - "random_heston"      : sample Heston params from params_limits
        - "calibrate_heston"   : fit Heston call prices, then project its terminal payoff
          (the square root of normalized terminal stock)
          Uses a deterministic surface-derived Heston starting guess unless
          x0 is supplied or initial_guess="random" is requested explicitly.
          Set fit_heston=False to project the surface heuristic directly,
          skipping Heston optimization (heston_fit_result is then None).
          projection_repetitions averages independent
          chaos projections after one Heston fit. projection_control selects
          an analytic Black--Scholes or linear Brownian control, and
          projection_antithetic pairs Brownian paths without adding paths.
        """
        if type == "random":
            nonconstant = kwargs["scale"] * torch.randn(
                len(self.wienerchaos.Lambda), device=self.device, dtype=self.type_float
            )
            # Include the trainable mean and start near the constant factor.
            nonconstant = torch.cat((nonconstant.new_ones(1), nonconstant))
            self._normalize_coefficients_(nonconstant, validate=True)
            d_a = torch.nn.Parameter(nonconstant)

        elif type == "random_heston":
            params = sample_tuple(kwargs["params_limits"])
            pi_f = np.linspace(0, self.T_horizon, kwargs.get("n_steps_heston", 100)).tolist()

            return self._initialize_from_heston(params, pi_f, kwargs)

        elif type == "calibrate_heston":
            maturities = kwargs.get("maturities", kwargs.get("Ts"))
            strikes = kwargs.get("strikes")
            prices_market = kwargs.get("prices_market", kwargs.get("price_market"))
            if maturities is None or strikes is None or prices_market is None:
                raise ValueError("Provide maturities (or Ts), strikes, and prices_market (or price_market).")

            if self.d != 2:
                raise ValueError("Heston projection requires two Brownian drivers (d=2).")

            fit_heston = kwargs.get("fit_heston", True)
            if not isinstance(fit_heston, bool):
                raise ValueError("fit_heston must be a bool")
            self.heston_heuristic_diagnostics = None
            if fit_heston:
                fit_options = {key: kwargs[key] for key in (
                    "params_limits", "seed", "x0", "initial_guess", "bounds", "method", "options",
                    "feller_penalty", "vega_weighted", "timing_check",
                ) if key in kwargs and kwargs[key] is not None}
                self.heston_params, self.heston_fit_result = calibrate_heston_to_iv_market(
                    prices_market=prices_market, Ts=maturities, strikes=strikes, S0=self.S0,
                    r=kwargs.get("r", 0.0), q=kwargs.get("q", 0.0), **fit_options,
                )
                self.heston_initial_guess = getattr(self.heston_fit_result, "initial_guess", None)
            else:
                if kwargs.get("x0") is not None or kwargs.get("initial_guess", "surface") != "surface":
                    raise ValueError("fit_heston=False requires initial_guess='surface' and no x0")
                ivs = compute_ivs(prices_market, maturities, strikes, self.S0,
                                  kwargs.get("r", 0.0), kwargs.get("q", 0.0))
                self.heston_params, self.heston_heuristic_diagnostics = heston_surface_initial_guess(
                    ivs, maturities, strikes, self.S0, kwargs.get("r", 0.0), kwargs.get("q", 0.0),
                    bounds=kwargs.get("bounds"),
                )
                self.heston_initial_guess = self.heston_params
                self.heston_fit_result = None

            pi_f = np.linspace(0, self.T_horizon, kwargs.get("n_steps_heston", 100)).tolist()
            return self._initialize_from_heston(self.heston_params, pi_f, kwargs)

        else:
            raise ValueError("type must be 'random', 'random_heston', or 'calibrate_heston'.")

        return d_a

    @torch.no_grad()
    def _normalize_coefficients_(self, theta, *, validate=False):
        norm_squared = self.chaos.normalizer(theta)
        if validate and (not torch.isfinite(norm_squared) or norm_squared <= 0):
            raise ValueError("The chaos factor must have a finite, nonzero norm.")
        theta.div_(norm_squared.sqrt())

    def _initialize_from_heston(self, params, pi_f, kwargs):
        """Project sqrt(S_T / forward); average independent estimates before normalization."""
        if self.d != 2:
            raise ValueError("Heston projection requires two Brownian drivers (d=2).")
        if len(pi_f) < 2:
            raise ValueError("n_steps_heston must be at least 2.")
        # S_T / forward has the same dynamics as zero-carry S_T / S0 for
        # deterministic rates, including maturity-dependent market curves.
        # Rates are used in the Heston fit and restored by sample_model_given_mat.
        forward = self.S0
        if not math.isfinite(forward) or forward <= 0:
            raise ValueError("The Heston projection requires a positive finite forward.")
        repetitions = kwargs.get("projection_repetitions", 1)
        if (isinstance(repetitions, bool) or not isinstance(repetitions, (int, np.integer))
                or repetitions < 1):
            raise ValueError("projection_repetitions must be a positive integer.")

        sim_terminal = partial(
            sim_euler_heston, S0=self.S0, r=0.0, q=0.0, params=params,
            T=self.T_horizon, pi=pi_f, device=self.device, return_last=True,
        )

        projection_antithetic = kwargs.get("projection_antithetic", False)
        if not isinstance(projection_antithetic, bool):
            raise ValueError("projection_antithetic must be a bool.")
        projection_control = kwargs.get("projection_control", "none")
        if projection_control not in ("none", "black_scholes", "linear_brownian"):
            raise ValueError(
                "projection_control must be 'none', 'black_scholes', or 'linear_brownian'."
            )
        if self.type_chaos != "piecewise_constant" and (
            projection_antithetic or projection_control != "none"
        ):
            raise ValueError(
                "Projection controls and antithetic paths require the piecewise-constant basis."
            )
        if projection_control != "none":
            kappa, theta_long, _, rho, v0 = map(float, params)
            T = float(self.T_horizon)
            mean_reversion_average = (-math.expm1(-kappa * T) / (kappa * T)
                                      if abs(kappa * T) > 1e-10 else 1.0)
            proxy_variance = theta_long + (v0 - theta_long) * mean_reversion_average
            if not math.isfinite(proxy_variance) or proxy_variance <= 0 or abs(rho) > 1:
                raise ValueError("Projection control requires positive variance and |rho| <= 1.")
            proxy_volatility = math.sqrt(proxy_variance)
            independent_weight = math.sqrt(1.0 - rho * rho)
            constant = math.exp(-proxy_variance * T / 8.0)
            intervals = torch.as_tensor(np.diff(self.wienerchaos.pi_M),
                                        device=self.device, dtype=self.type_float)
            loadings = (0.5 * proxy_volatility * intervals.sqrt()[:, None]
                        * torch.as_tensor([rho, independent_weight],
                                          device=self.device, dtype=self.type_float)[None, :])
            # The repository expands in He_alpha/alpha!. The exponential
            # proxy has known coefficients in every order; its linear part
            # has only constant and first-order coefficients.
            if projection_control == "black_scholes":
                powers = torch.pow(loadings.reshape(1, -1), self.wienerchaos.Lambda)
                nonconstant_coefficients = constant * powers.prod(dim=1)
            else:
                degrees = self.wienerchaos.Lambda.sum(dim=1)
                nonconstant_coefficients = (
                    constant
                    * (self.wienerchaos.Lambda * loadings.reshape(1, -1)).sum(dim=1)
                    * (degrees == 1)
                )
            proxy_coefficients = torch.cat((
                intervals.new_tensor([constant]), nonconstant_coefficients,
            ))

        def terminal_factor(bm):
            if kwargs.get("timing_check") is not None:
                kwargs["timing_check"]()
            factor = torch.sqrt(sim_terminal(bm) / forward)
            if projection_control != "none":
                stock_brownian = rho * bm[:, 0, -1] + independent_weight * bm[:, 1, -1]
                if projection_control == "black_scholes":
                    proxy = torch.exp(-0.25 * proxy_variance * T
                                      + 0.5 * proxy_volatility * stock_brownian)
                else:
                    proxy = constant * (1.0 + 0.5 * proxy_volatility * stock_brownian)
                return factor - proxy
            # Nonconstant chaos features have expectation zero. Removing this
            # known constant avoids projecting its Monte Carlo sampling noise.
            return factor - 1.0

        with torch.no_grad():
            total = None
            for _ in range(repetitions):
                # Each call advances the Brownian RNG and uses independent paths.
                projection_kwargs = {}
                if self.type_chaos == "piecewise_constant":
                    projection_kwargs["antithetic"] = projection_antithetic
                estimate = self.wienerchaos.compute_chaos_coeffs(
                    terminal_factor, pi_f, kwargs["N"], kwargs["Ns"],
                    **projection_kwargs,
                )
                total = estimate if total is None else total + estimate
            theta = total / repetitions
            if projection_control != "none":
                theta += proxy_coefficients
            else:
                # Restore the constant coefficient of sqrt(S_T / forward).
                theta[0] += 1.0
            normalizer = self.chaos.normalizer(theta)
            if not torch.isfinite(normalizer) or normalizer <= 0:
                raise ValueError("Heston projection produced an invalid chaos factor.")
            theta = theta / normalizer.sqrt()
        return torch.nn.Parameter(theta)


    def set_up_for_Ts(self, Ts, rs, qs):
        if isinstance(rs, (int, float)):
            rs = [rs] * len(Ts)
        if isinstance(qs, (int, float)):
            qs = [qs] * len(Ts)

        for T, r, q in zip(Ts, rs, qs):
            T = _r6(T)
            self.wienerchaos.set_step(T)
            self.chaos.set_step(T)
            self.rescaling_rq[T] = exp((r - q) * T)
            self.r_T[T], self.q_T[T] = float(r), float(q)

    def sample_model_given_mat(self, d_a, X, T):
        T = _r6(T)

        return self.chaos.spot(X, T, d_a, self.rescaling_rq[T]), d_a

    def sample_spot_price(self, d_a, pi, N, rs, qs, *, return_on_device=False):
        """Shared spot paths; optionally retain them on the model's device."""
        grid = np.round(np.asarray(pi, dtype=float), 6)
        if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
                or grid[0] != 0 or np.any(np.diff(grid) <= 0) or grid[-1] > self.T_horizon):
            raise ValueError("Path dates must increase from zero within the model horizon")
        if isinstance(N, (bool, np.bool_)) or not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("N must be a positive integer")
        norm = self.chaos.normalizer(d_a)
        if not torch.isfinite(norm) or abs(float(norm)-1.) > 1e-4:
            raise ValueError("Path simulation requires normalized chaos coefficients")
        path_device = self.device if return_on_device else torch.device("cpu")
        sample_S = [torch.full((N,), self.S0, dtype=self.type_float, device=path_device)]
        if len(grid) == 1:
            return sample_S[0][:, None]
        rs = np.asarray(rs, dtype=float)
        qs = np.asarray(qs, dtype=float)
        rs = np.full(len(grid)-1, float(rs)) if rs.ndim == 0 else rs
        qs = np.full(len(grid)-1, float(qs)) if qs.ndim == 0 else qs
        if (rs.shape != (len(grid)-1,) or qs.shape != (len(grid)-1,)
                or not np.isfinite(rs).all() or not np.isfinite(qs).all()):
            raise ValueError("Path rates and yields must be finite scalars or one per positive date")
        self.set_up_for_Ts(grid[1:].tolist(), rs.tolist(), qs.tolist())
        X = self.chaos.sample_past_same_omega(N, grid[1:].tolist())
        for t, T in enumerate(grid[1:]):
            ST, _ = self.sample_model_given_mat(d_a, X[t], T)
            sample_S.append(ST.to(path_device))
        paths = torch.stack(sample_S).T
        if not torch.isfinite(paths).all() or (paths < 0).any():
            raise FloatingPointError("Invalid simulated spot paths")
        return paths

    def compute_model_prices_MC(
        self,
        d_a,
        X,
        T,
        strikes,
        poly_cv_degree=None,
        n_samples_cv=None,   # int or None
        shared_moments=None,
    ):
        """
        Vanilla CALLS, MC.
        Control variates are applied to UNDISCOUNTED payoffs; discount at the end.

        n_samples_cv:
          - None: use all N samples both to fit lambdas and to estimate price
          - int:  use first n_samples_cv samples to fit lambdas; remaining samples for pricing MC
        """

        if poly_cv_degree is not None and not (1 <= poly_cv_degree <= 2):
            raise ValueError("poly_cv_degree must be None, 1, or 2 for this model.")

        T = _r6(T)

        split = n_samples_cv if (n_samples_cv is not None and poly_cv_degree is not None) else None
        X_cv, X_MC = X[:split], X[split:]


        # ------------------------------------------------------------
        # Build S_T
        # ------------------------------------------------------------
        ST, _ = self.sample_model_given_mat(d_a, X_MC, T)  # ST: (N_MC,)
        ST = ST.reshape(-1)  # ensure (N_MC,)

        # ------------------------------------------------------------
        # Payoff construction (CALLS): (S_T - K)+   [UNDISCOUNTED]
        # ------------------------------------------------------------
        K = torch.as_tensor(strikes, device=self.device, dtype=ST.dtype).view(-1, 1)  # (K,1)
        payoff = torch.relu(ST.unsqueeze(0) - K)                                      # (K, N_MC)

        if poly_cv_degree is not None:
            if n_samples_cv is not None:
                with torch.no_grad():
                    ST_cv, _ = self.sample_model_given_mat(d_a, X_cv, T)  # ST: (N_cv,)
                    ST_cv = ST_cv.reshape(-1)  # ensure (N_cv,)
                    payoff_cv = torch.relu(ST_cv.unsqueeze(0) - K)          # (K, N_cv)
            else:
                ST_cv = ST.detach()
                payoff_cv = payoff.detach()                                # (K, N_cv)

        # Discount factor
        r = self.r_T[T]
        disc = torch.exp(
            -torch.as_tensor(r, device=self.device, dtype=ST.dtype)
            * torch.as_tensor(T, device=self.device, dtype=ST.dtype)
        )

        # ============================================================
        # PREPARE CONTROL VARIATES
        # ============================================================

        # ---- Polynomial CV features (centered) ----
        P = None  # (N, p)
        if poly_cv_degree is not None:
            p = int(poly_cv_degree)
            ST_moments = self.chaos.moments(
                p, d_a, T, self.rescaling_rq[T], shared=shared_moments
            )

            ST_powers = ST.unsqueeze(1) ** torch.arange(1, p + 1, device=ST.device, dtype=ST.dtype)  # (N_MC, p)
            P = ST_powers - ST_moments  # (N_MC, p) centered powers

            if n_samples_cv is not None:
                ST_powers_cv = ST_cv.unsqueeze(1) ** torch.arange(1, p + 1, device=ST.device, dtype=ST.dtype)  # (N_cv, p)
                P_cv = ST_powers_cv - ST_moments  # (N_cv, p) centered powers
            else:
                P_cv = P.detach()

        # ============================================================
        # APPLY CONTROL VARIATES (UNDISCOUNTED convention)
        #   - use *_cv to compute lambdas
        #   - apply lambdas to *_MC and estimate price from MC part only
        # ============================================================

        # ---- Case 0: no CV ----
        if poly_cv_degree is None:
            payoff_mean = payoff.mean(dim=1)

        else:
            # payoff_cv: (K, N_cv), payoff: (K, N_MC)
            Y_cv = payoff_cv.T  # (N_cv, K)

            Yc_cv = Y_cv - Y_cv.mean(dim=0, keepdim=True)  # (N_cv, K)
            denom = float(Y_cv.shape[0] - 1)

            # P_cv: (N_cv, p), P: (N_MC, p)
            with torch.no_grad():
                Cov_PP = (P_cv.T @ P_cv) / denom                    # (p, p)
                Cov_PY = (P_cv.T @ Yc_cv) / denom                   # (p, K)

                ridge = torch.as_tensor(1e-12, device=P_cv.device, dtype=P_cv.dtype)
                I = torch.eye(P_cv.shape[1], device=P_cv.device, dtype=P_cv.dtype)
                Cov_PP = Cov_PP + ridge * I
                L = torch.linalg.cholesky(Cov_PP)
                lambdas_poly = torch.cholesky_solve(Cov_PY, L).T    # (K, p)

            # The fitted coefficients are fixed for the pricing sample, so the
            # correction can be applied after averaging. This avoids allocating
            # another paths-by-strikes tensor and keeps moment gradients intact.
            payoff_mean = payoff.mean(dim=1) - lambdas_poly @ P.mean(dim=0)

        # ------------------------------------------------------------
        # Discount at the end (PV of calls) using MC-set only
        # ------------------------------------------------------------
        price = disc * payoff_mean  # (K,)
        return price

    def clear_quadrature_cache(self):
        """Release cached quadrature tensors; also called when replacing the basis."""
        self._quadrature_cache.clear()
        self._quadrature_cache_bytes = 0

    def compute_model_prices_Gauss_Hermite(self, d_a, T, strikes, n=20, tol_w=0):
        """
        Vanilla calls, Gauss–Hermite over the Gaussian integrals.
        Only for type_chaos == "piecewise_constant". Fixed features are cached;
        maturity rescaling and coefficient-dependent prices are evaluated afresh.
        """
        if self.type_chaos != "piecewise_constant":
            raise ValueError("Gauss_Hermite is only available for type_chaos='piecewise_constant'.")

        T = float(T)
        M_T = sum(ts < T for ts in self.pi_M)
        X, W = self._gauss_hermite_features(T, M_T, n)
        ST, _ = self.sample_model_given_mat(d_a, X, T)

        K = torch.as_tensor(strikes, dtype = self.type_float, device=self.device)  # (S,)
        payoff = torch.relu(ST.unsqueeze(0) - K.unsqueeze(1))                  # (S,Nnodes)

        disc = exp(-self.r_T[_r6(T)] * T)
        return (W.unsqueeze(0) * payoff).sum(dim=1) * disc

    @torch.no_grad()
    def _gauss_hermite_features(self, T, M_T, n):
        """Cached physical past coordinates for the conditional-square model."""
        key = (_r6(T), int(n), str(self.device), self.type_float)
        if key in self._quadrature_cache:
            self._quadrature_cache.move_to_end(key)
            return self._quadrature_cache[key]
        nodes, weights = np.polynomial.hermite.hermgauss(int(n))
        z = torch.as_tensor(nodes, device=self.device, dtype=self.type_float)
        w = torch.as_tensor(weights, device=self.device, dtype=self.type_float)
        dim = self.d * M_T
        standard = sqrt(2.0) * torch.cartesian_prod(*([z] * dim)).reshape(-1, dim)
        X = torch.zeros((len(standard), self.d * self.wienerchaos.M),
                        device=self.device, dtype=self.type_float)
        v = self.chaos.past_variance[_r6(T)]
        X[:, :dim] = standard * v[:dim].sqrt()
        W = torch.cartesian_prod(*([w] * dim)).reshape(-1, dim).prod(dim=1) / (torch.pi ** (dim / 2))
        size = X.numel() * X.element_size() + W.numel() * W.element_size()
        if size <= self._quadrature_cache_limit_bytes:
            while self._quadrature_cache_bytes + size > self._quadrature_cache_limit_bytes:
                _, (old_X, old_W) = self._quadrature_cache.popitem(last=False)
                self._quadrature_cache_bytes -= (
                    old_X.numel() * old_X.element_size() + old_W.numel() * old_W.element_size()
                )
            self._quadrature_cache[key] = (X, W)
            self._quadrature_cache_bytes += size
        return X, W

    def compute_model_prices(self, d_a, T, strikes, method, **kwargs):
        if method == "MC":
            return self.compute_model_prices_MC(
                d_a, kwargs["X"], T, strikes,
                poly_cv_degree=kwargs.get("poly_cv_degree", None),
                n_samples_cv = kwargs.get("n_samples_cv", None),
            )
        elif method == "Gauss_Hermite":
            return self.compute_model_prices_Gauss_Hermite(
                d_a, T, strikes, kwargs.get("n", 20), kwargs.get("tol_w", 0)
            )
        raise ValueError("method must be 'MC' or 'Gauss_Hermite'.")


    def vega_weighted_mse(self, prices_market, prices_model, weights_flat):
        if torch.is_tensor(prices_market):
            prices_market_flat = prices_market.to(device=self.device).reshape(-1)
        else:
            prices_market_flat = torch.as_tensor([x for row in prices_market for x in row], device=self.device)
        prices_model_flat = torch.cat(prices_model, dim=0)
        diff = (prices_market_flat - prices_model_flat).reshape(-1)
        return torch.mean(weights_flat * diff ** 2)


    def price_exotic_option_MC(
        self,
        d_a,
        pi,
        functional,
        n_samples_x_batch,
        n_batch,
        r,
        q,
        **kwargs
    ):
        """Discount pathwise payoffs over a Cartesian parameter grid."""
        for value in (n_batch, n_samples_x_batch):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("Monte Carlo batch sizes and counts must be positive integers")
        param_names, param_lists = option_parameter_grid(kwargs)
        sizes = [len(v) for v in param_lists]
        prices = torch.zeros((n_batch, *sizes), device=self.device, dtype=torch.float64)
        T = float(pi[-1])
        discount = math.exp(-float(r)*T)
        index_ranges = [range(s) for s in sizes]
        for n in range(n_batch):
            S = self.sample_spot_price(d_a, pi, n_samples_x_batch, r, q)
            for idx_tuple in itertools.product(*index_ranges):
                current = {name: param_lists[i][idx_tuple[i]] for i, name in enumerate(param_names)}
                payoff = evaluate_payoff(functional, S, current)
                prices[(n, *idx_tuple)] = payoff.mean(dtype=torch.float64)*discount
        return prices.mean(dim=0)


    def _build_index_blocks(self, maturities_blocks):
        index_blocks = []
        counter = 0
        for block in maturities_blocks:
            L = len(block)
            index_blocks.append(list(range(counter, counter + L)))
            counter += L
        return index_blocks

    def _gather_block_lists(self, index_block, strikes, method_per_mat, N_per_mat, Ns_per_mat, nodes_per_mat, prices_market, iv_market):
        strikes_block = [strikes[i] for i in index_block]
        method_block  = [method_per_mat[i] for i in index_block]
        N_block       = [N_per_mat[i] for i in index_block]
        Ns_block       = [Ns_per_mat[i] for i in index_block]
        nodes_block   = [nodes_per_mat[i] for i in index_block]
        prices_mkt_b  = [prices_market[i] for i in index_block]
        iv_mkt_b      = [iv_market[i] for i in index_block]
        return strikes_block, method_block, N_block, Ns_block, nodes_block, prices_mkt_b, iv_mkt_b

    def _block_rates(self, maturities_block):
        q_block = [self.q_T[_r6(T)] for T in maturities_block]
        r_block = [self.r_T[_r6(T)] for T in maturities_block]
        return r_block, q_block

    def _block_vega_weights(self, iv_mkt_block, maturities_block, strikes_block):
        r_block, q_block = self._block_rates(maturities_block)
        flat_w, _ = get_vegas(
            iv_mkt_block, maturities_block, strikes_block,
            self.S0, r_block, q_block,
            False, alpha=1e-3, normalize=False
        )
        return torch.as_tensor(flat_w, device=self.device)

    def _simulate_X_block(
        self,
        maturities_block,
        method_block,
        N_block,
        Ns_block,
    ):
        """
        Returns
        -------
        if pi_bm == False:
            X : list, X[i] is Tensor (Ns*N, n_terms) = Phi(T) for maturity i (MC case), else None for GH case.
        if return_bm == True:
            (X, bm) where
            - X is as above
            - bm is a list with bm[i] Tensor (Ns*N, d, len(pi_bm)) for MC case, else None
        """
        X = []
        bm = []  # only used if return_bm=True

        for (T, method, N, Ns) in zip(maturities_block, method_block, N_block, Ns_block):
            if method != "MC":
                X.append(None)
            else:
                X.append(self.chaos.sample_past(N, Ns, T))

        return X

    def _simulate_X_block_with_seed(
        self, maturities_block, method_block, N_block, Ns_block, seed=None
    ):
        """Draw a fixed comparison sample without advancing the training RNG."""
        if seed is None:
            return self._simulate_X_block(maturities_block, method_block, N_block, Ns_block)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            return self._simulate_X_block(maturities_block, method_block, N_block, Ns_block)

    def _relative_anchor_penalty(self, theta, reference):
        """Squared weighted-L2 distance to a fixed factor, divided by its nonconstant norm.

        Both factors are first normalized in the chaos L2 norm, since a global
        scale does not change the squared model. The common starting sign fixes
        the otherwise equivalent plus/minus representation.
        """
        factorials = self.chaos.factorials.to(dtype=theta.dtype)
        unit = theta / self.chaos.normalizer(theta).sqrt()
        target = reference / self.chaos.normalizer(reference).sqrt()
        weights = factorials[1:].reciprocal()
        target_norm_sq = (target[1:].square() * weights).sum()
        if not torch.isfinite(target_norm_sq) or target_norm_sq <= 0:
            raise ValueError("The Heston anchor needs a finite nonconstant factor.")
        return ((unit[1:] - target[1:]).square() * weights).sum() / target_norm_sq


    def _prices_model_block(
        self,
        d_a,
        X_block,
        maturities_block,
        strikes_block,
        method_block,
        nodes_block,
        poly_cv_degree=None,
        n_samples_cv = None,
    ):
        prices_model = []
        shared_moments = None
        if poly_cv_degree == 2 and "MC" in method_block:
            shared_moments = self.chaos.shared_moment_kernels(d_a)
        for i in range(len(maturities_block)):
            T = maturities_block[i]
            method = method_block[i]

            if method == "MC":

                prices_model.append(
                    self.compute_model_prices_MC(
                        d_a,
                        X_block[i],
                        T,
                        strikes_block[i],
                        poly_cv_degree=poly_cv_degree,
                        n_samples_cv = n_samples_cv,
                        shared_moments=shared_moments,
                    )
                )
            else:
                prices_model.append(
                    self.compute_model_prices_Gauss_Hermite(
                        d_a, T, strikes_block[i], nodes_block[i], tol_w=0
                    )
                )

        return prices_model


    def _plot_block_fit(
        self,
        model_list,                 # list of (prices_model OR iv_model), each is list over maturities
        maturities_block,
        strikes_block,
        iv_mkt_block,
        title_suffix,
        labels,                     # list[str], same length as model_list
        save_folder=None,
        input_type: str = "price",  # "price" (default) or "iv"
    ):
        if len(model_list) != len(labels):
            raise ValueError("labels must have same length as model_list.")

        input_type = str(input_type).lower()
        if input_type not in {"price", "iv"}:
            raise ValueError("input_type must be 'price' or 'iv'.")

        # Optional: create folder if saving is requested
        if save_folder is not None:
            import os
            os.makedirs(save_folder, exist_ok=True)

        # Convert each model set into python lists
        model_list_py = []
        for model in model_list:
            model_list_py.append([
                m.detach().cpu().tolist() if torch.is_tensor(m) else m
                for m in model
            ])

        # Compute implied vols only if input is prices
        if input_type == "price":
            iv_models = []
            for prices_list in model_list_py:
                iv_models.append(
                    compute_ivs(
                        prices_list, maturities_block, strikes_block, self.S0,
                        [self.r_T[_r6(T)] for T in maturities_block],
                        [self.q_T[_r6(T)] for T in maturities_block],
                    )
                )
        else:
            # input is already IVs: model_list_py is iv_models
            iv_models = model_list_py

        markers = ["o", "s", "^", "D", "v", "P"]

        for t in range(len(maturities_block)):
            plt.figure()

            # Market data: x markers only
            plt.scatter(
                strikes_block[t], iv_mkt_block[t],
                marker="x", s=75, linewidths=2.0,
                label="market", zorder=10,
            )

            # Model data: markers only, no continuous lines
            for k, iv_model in enumerate(iv_models):
                mk = markers[k % len(markers)]

                plt.plot(
                    strikes_block[t], iv_model[t],
                    linestyle="None",      # no line
                    marker=mk,
                    markersize=6.5,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    label=labels[k],
                    zorder=2 + k,
                )

            plt.title(f"T = {maturities_block[t]} -- {title_suffix}")
            plt.xlabel("Strike")
            plt.ylabel("Implied Volatility")
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=False)

            if save_folder is not None:
                import os
                T_val = maturities_block[t]
                fname = f"iv_calibration_smile_{T_val}"
                plt.savefig(
                    os.path.join(save_folder, fname + ".png"),
                    dpi=200,
                    bbox_inches="tight",
                )

            plt.show()


    def _train_block(
        self,
        d_a,
        maturities_block,
        strikes_block,
        method_block,
        N_block,
        Ns_block,
        nodes_block,
        prices_market_block,
        flat_w_block,
        *,
        n_iter,
        lr,
        simulate_every,
        reg_lambda,
        poly_cv_degree = None,
        patience = 5000,
        tol_error = 1e-8,
        warmup_without_cv = 500,   # Number of steps without control variates
        limit_error_above = 1e-6, # If patience has been reached and the error is still above this value,
                                 #keep training until the value is reached
        plateau_kwargs = None,
        print_loss_every_it = 100,
        n_samples_cv = None,
        check_validation_set_MC = None,   # None: training loss; positive int: validation interval
        validation_seed = None,
        validation_metric = "price_mse",
        iv_market_block = None,
        anchor_lambda = 0.0,
        anchor_coefficients = None,
        timing_callback = None,
    ):
        validation_interval = check_validation_set_MC
        if validation_interval is not None:
            if (isinstance(validation_interval, bool)
                    or not isinstance(validation_interval, (int, np.integer))
                    or validation_interval < 1):
                raise ValueError("check_validation_set_MC must be None or a positive integer.")
        if validation_metric not in ("price_mse", "iv_mae"):
            raise ValueError("validation_metric must be 'price_mse' or 'iv_mae'.")
        if validation_metric == "iv_mae" and (validation_interval is None or iv_market_block is None):
            raise ValueError("IV validation requires a validation interval and market IVs.")

        # initial sims
        X = self._simulate_X_block(maturities_block, method_block, N_block, Ns_block)

        # Quotes are fixed throughout this block; transfer them to the device once.
        prices_market_block = torch.as_tensor(
            [x for row in prices_market_block for x in row], device=self.device
        )

        if validation_interval is not None:
            # Independent MC draws, held fixed for the entire block even when
            # training paths are refreshed. Use the same per-maturity budgets.
            X_val = self._simulate_X_block_with_seed(
                maturities_block, method_block, N_block, Ns_block, validation_seed
            )

        opt = torch.optim.AdamW([d_a], lr=lr, weight_decay=reg_lambda)
        if plateau_kwargs is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **plateau_kwargs)

        best_loss = float("inf")
        best_da = d_a.detach().clone()

        patience_accumulated = 0
        poly_cv_degree_ = None
        last_report_time = time.perf_counter()
        last_report_iteration = 0
        if timing_callback is not None:
            timing_callback(0, now=last_report_time)

        for it in range(int(n_iter)):

            if warmup_without_cv == it:
                if "MC" in method_block:
                    # print(f"Activating control variates")
                    poly_cv_degree_ = poly_cv_degree

            if simulate_every is not None and (it + 1) % int(simulate_every) == 0:
                if "MC" in method_block:
                    del X
                    torch.cuda.empty_cache()
                    X = self._simulate_X_block(maturities_block, method_block, N_block, Ns_block)

            validation_evaluated = validation_interval is not None and (
                it == 0 or (it + 1) % validation_interval == 0
            )
            if validation_evaluated:
                with torch.no_grad():
                    prices_model_val = self._prices_model_block(
                        d_a, X_val, maturities_block, strikes_block, method_block, nodes_block,
                        # Keep the validation estimator fixed, including during
                        # the training control-variate warmup.
                        poly_cv_degree=poly_cv_degree,
                        n_samples_cv = n_samples_cv,
                    )
                    loss_val = self.vega_weighted_mse(prices_market_block, prices_model_val, flat_w_block)
                    if validation_metric == "iv_mae":
                        ivs_val = compute_ivs(
                            [prices.detach().cpu().tolist() for prices in prices_model_val],
                            maturities_block, strikes_block, self.S0,
                            [self.r_T[_r6(t)] for t in maturities_block],
                            [self.q_T[_r6(t)] for t in maturities_block],
                        )
                        loss_val_value = float(np.mean([
                            abs(float(iv) - float(target))
                            for row, targets in zip(ivs_val, iv_market_block)
                            for iv, target in zip(row, targets)
                        ]))
                    else:
                        loss_val_value = loss_val.item()
                    if not math.isfinite(loss_val_value):
                        raise FloatingPointError("Nonfinite validation objective.")

            opt.zero_grad(set_to_none=True)

            prices_model = self._prices_model_block(
                d_a, X, maturities_block, strikes_block, method_block, nodes_block,
                poly_cv_degree=poly_cv_degree_,
                n_samples_cv = n_samples_cv,
            )

            loss = self.vega_weighted_mse(prices_market_block, prices_model, flat_w_block)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise FloatingPointError("Nonfinite calibration objective.")
            # Save the coefficients that produced this objective, before updating.
            # Selection, patience and scheduling must all use the same fresh
            # metric. Skipped validation iterations do not consume patience.
            monitor_evaluated = validation_interval is None or validation_evaluated
            candidate_loss = loss_val_value if validation_interval is not None else loss_value
            if monitor_evaluated:
                if candidate_loss + tol_error < best_loss:
                    best_loss = candidate_loss
                    best_da = d_a.detach().clone()
                    patience_accumulated = 0
                else:
                    patience_accumulated += 1
            objective = loss
            if anchor_lambda > 0:
                objective = objective + anchor_lambda * self._relative_anchor_penalty(
                    d_a, anchor_coefficients
                )
            objective.backward()

            # Gradient through theta / sqrt(nu) at nu=1, where
            # nu = sum(theta.square() / factorials). This recovers the
            # scale-invariant update without normalizing inside autograd.
            with torch.no_grad():
                d_a.grad.sub_(
                    (d_a.grad * d_a).sum() * d_a / self.chaos.factorials.to(d_a.dtype)
                )
            opt.step()
            self._normalize_coefficients_(d_a)
            # Reuse the scalar for scheduling, reporting and stopping; each item()
            # otherwise forces another host/device synchronization on CUDA.
            if plateau_kwargs is not None and monitor_evaluated:
                scheduler.step(candidate_loss)

            report_due = print_loss_every_it is not None and (it + 1) % print_loss_every_it == 0
            timing_due = timing_callback is not None and (it + 1) % 25 == 0
            if report_due or timing_due:
                # Reports and the optional study guard time the same completed steps.
                if d_a.is_cuda:
                    torch.cuda.synchronize(d_a.device)
                now = time.perf_counter()
                if timing_due:
                    timing_callback(it + 1, now=now)
            if report_due:
                iterations_per_second = (it + 1 - last_report_iteration) / (now - last_report_time)
                last_report_time = now
                last_report_iteration = it + 1
                message = f"[{it+1:04d}] loss = {loss_value:.6e}"
                if validation_interval is not None:
                    metric_name = "iv_val_mae" if validation_metric == "iv_mae" else "loss_val"
                    label = metric_name if validation_evaluated else f"{metric_name} (last check)"
                    message += f", {label} = {loss_val_value:.6e}"
                print(f"{message}, {iterations_per_second:.2f} iterations/s", flush=True)

            # We return best_da, so the quality gate must use that same saved
            # checkpoint. A later noisy/divergent value should not force
            # thousands of extra steps after a good solution has plateaued.
            if (monitor_evaluated and patience_accumulated >= patience
                    and best_loss < limit_error_above):
                break

        self.last_calibration_iterations = it + 1
        self.last_calibration_monitor_loss = best_loss
        with torch.no_grad():
            d_a.copy_(best_da)
            self._normalize_coefficients_(d_a)
            prices_model = self._prices_model_block(
                d_a, X, maturities_block, strikes_block, method_block, nodes_block,
                poly_cv_degree=poly_cv_degree_, n_samples_cv=n_samples_cv,
            )
        return d_a, X, prices_model

    @torch.enable_grad()
    def calibrate(
        self,
        iv_market,
        prices_market,
        maturities_blocks,
        strikes,
        method_per_mat,
        N_per_mat,
        Ns_per_mat,
        nodes_per_mat,
        d_a=None,
        n_iter=5000,
        lr=5e-4,
        scale_init=0.001,
        simulate_every=None,
        reg_lambda=0.0,
        poly_cv_degree=None,
        patience=100,
        tol_error=1e-8,
        warmup_without_cv=2000,   # Number of steps without control variates
        limit_error_above=1e-6,
        plateau_kwargs=None,
        print_loss_every_it=100,
        print_plots = True,
        n_samples_cv = None,
        check_validation_set_MC = None,
        validation_seed = None,
        validation_metric = "price_mse",
        evaluation_seed = None,
        anchor_lambda = 0.0,
        anchor_coefficients = None,
        timing_callback = None,
    ):
        """Fit the vega-weighted option-price objective.

        Enable gradient recording locally even when the caller disabled it for
        an earlier pricing or plotting operation.

        check_validation_set_MC=None monitors training loss (the default).
        A positive integer draws an independent validation set at the start of
        each block and checks it on the first iteration and every N iterations.
        Its loss controls checkpoint selection, early stopping and the plateau
        scheduler; patience values then count validation checks. The validation
        set uses the training pricing allocations and stays fixed while training
        paths may be refreshed with simulate_every. Only training loss supplies
        gradients. Optional validation_seed and evaluation_seed keep those
        comparison samples common across runs without advancing the training RNG.
        With validation_metric='iv_mae', checkpointing, early stopping, and the
        learning-rate scheduler use mean absolute implied-volatility error
        instead of vega-weighted price MSE. The training objective is unchanged.
        A positive anchor_lambda adds a scale-aware squared-chaos penalty toward
        anchor_coefficients (or the supplied starting vector) to the training
        gradient. Validation and checkpoint selection still use price error only.
        """
        import time
        t0 = time.perf_counter()

        loss_insample_by_blocks, loss_outofsample_by_blocks = [], []
        prices_model_outofsample_by_mat = [None] * len(iv_market)  # list[list[float]] aligned with maturities

        index_blocks = self._build_index_blocks(maturities_blocks)

        if d_a is None:
            d_a = self.initialize_params(type="random", scale=scale_init)
        else:
            if not isinstance(d_a, torch.nn.Parameter):
                d_a = torch.nn.Parameter(d_a)
        d_a.requires_grad_(True)

        if d_a.numel() != len(self.wienerchaos.Lambda) + 1:
            raise ValueError("Chaos requires the constant coefficient followed by all nonconstant coefficients.")
        self._normalize_coefficients_(d_a, validate=True)
        if not math.isfinite(float(anchor_lambda)) or anchor_lambda < 0:
            raise ValueError("anchor_lambda must be finite and nonnegative.")
        if anchor_lambda > 0:
            if anchor_coefficients is None:
                anchor_coefficients = d_a.detach().clone()
            else:
                anchor_coefficients = torch.as_tensor(
                    anchor_coefficients, dtype=d_a.dtype, device=d_a.device
                ).detach().clone()
            if (anchor_coefficients.shape != d_a.shape
                    or not torch.isfinite(anchor_coefficients).all()
                    or not anchor_coefficients.any()):
                raise ValueError("anchor_coefficients must match finite nonzero calibration coefficients.")

        for block_id, maturities_block in enumerate(maturities_blocks):
            index_block = index_blocks[block_id]

            strikes_block, method_block, N_block, Ns_block, nodes_block, prices_mkt_block, iv_mkt_block = self._gather_block_lists(
                index_block, strikes, method_per_mat, N_per_mat, Ns_per_mat, nodes_per_mat, prices_market, iv_market
            )

            flat_w_block = self._block_vega_weights(iv_mkt_block, maturities_block, strikes_block)

            if print_loss_every_it is not None:
                print("Calibrating...")

            d_a, X_insample, prices_model_insample = self._train_block(
                d_a,
                maturities_block,
                strikes_block,
                method_block,
                N_block,
                Ns_block,
                nodes_block,
                prices_mkt_block,
                flat_w_block,
                n_iter=n_iter,
                lr=lr,
                simulate_every=simulate_every,
                reg_lambda=reg_lambda,
                poly_cv_degree=poly_cv_degree,
                patience=patience,
                tol_error=tol_error,
                warmup_without_cv=warmup_without_cv,
                limit_error_above=limit_error_above,
                plateau_kwargs=plateau_kwargs,
                print_loss_every_it=print_loss_every_it,
                n_samples_cv = n_samples_cv,
                check_validation_set_MC = check_validation_set_MC,
                validation_seed = None if validation_seed is None else int(validation_seed) + block_id,
                validation_metric = validation_metric,
                iv_market_block = iv_mkt_block,
                anchor_lambda = anchor_lambda,
                anchor_coefficients = anchor_coefficients,
                timing_callback = timing_callback,
            )

            # resimulate (test set)
            del X_insample
            torch.cuda.empty_cache()
            X_outofsample = self._simulate_X_block_with_seed(
                maturities_block, method_block, N_block, Ns_block,
                None if evaluation_seed is None else int(evaluation_seed) + block_id,
            )
            with torch.no_grad():
                prices_model_outofsample = self._prices_model_block(
                    d_a, X_outofsample, maturities_block, strikes_block, method_block, nodes_block,
                    poly_cv_degree=poly_cv_degree,
                    n_samples_cv = n_samples_cv,
                )


            # plot BOTH on same figures (per maturity)
            idx_not_MC = [i for i, s in enumerate(method_block) if s != "MC"]
            idx_MC = [i for i, s in enumerate(method_block) if s == "MC"]

            if print_plots:

                self._plot_block_fit(
                    model_list=[[prices_model_insample[i] for i in idx_not_MC]],
                    labels=["Calibration"],
                    maturities_block=[maturities_block[i] for i in idx_not_MC],
                    strikes_block=[strikes_block[i] for i in idx_not_MC],
                    iv_mkt_block=[iv_mkt_block[i] for i in idx_not_MC],
                    title_suffix="Quadrature",
                )

                self._plot_block_fit(
                    model_list=[[prices_model_insample[i] for i in idx_MC],
                                    [prices_model_outofsample[i] for i in idx_MC]],
                    labels=["Calibration in-sample-set", "Calibration out-of-sample-set"],
                    maturities_block=[maturities_block[i] for i in idx_MC],
                    strikes_block=[strikes_block[i] for i in idx_MC],
                    iv_mkt_block=[iv_mkt_block[i] for i in idx_MC],
                    title_suffix="Monte Carlo",
                )

            del X_outofsample
            torch.cuda.empty_cache()

            loss_insample = self.vega_weighted_mse(prices_mkt_block, prices_model_insample, flat_w_block)

            loss_outofsample = self.vega_weighted_mse(prices_mkt_block, prices_model_outofsample, flat_w_block)
            loss_insample_by_blocks.append(loss_insample.detach().item())
            loss_outofsample_by_blocks.append(loss_outofsample.detach().item())
            for local_i, global_i in enumerate(index_block):
                prices_model_outofsample_by_mat[global_i] = prices_model_outofsample[local_i].detach().cpu().tolist()

            if print_loss_every_it is not None:
                print(f"Loss in-sample: {loss_insample:.2e}. Loss out-of-sample: {loss_outofsample:.2e}\n")

        loss_outofsample_mean = torch.mean(torch.tensor(loss_outofsample_by_blocks)).item()
        loss_insample_mean = torch.mean(torch.tensor(loss_insample_by_blocks)).item()

        elapsed_seconds = time.perf_counter() - t0

        return d_a.detach(), prices_model_outofsample_by_mat, loss_insample_mean, loss_outofsample_mean, elapsed_seconds

    def _plot_block_fit_final(
        self,
        model_list,                   # list of (prices_model OR iv_model), each is list over maturities
        maturities_block,
        strikes_block,
        iv_mkt_block=None,
        title_suffix="",
        labels=None,                  # list[str], same length as model_list
        save_folder=None,
        input_type: str = "price",    # "price" (default) or "iv"
        combined_plot: bool = False,  # if True, plot all maturities on same figure
        model_index: int = 0,         # which model to use for combined plot
        iv_bid_block=None,            # optional: same structure as iv_mkt_block
        iv_ask_block=None,            # optional: same structure as iv_mkt_block
        show_maturity_in_title: bool = False,
    ):
        if labels is None:
            raise ValueError("You need to provide labels.")

        if len(model_list) != len(labels):
            raise ValueError("labels must have same length as model_list.")

        input_type = str(input_type).lower()
        if input_type not in {"price", "iv"}:
            raise ValueError("input_type must be 'price' or 'iv'.")

        show_market = iv_mkt_block is not None
        show_bid_ask_band = (
            not show_market
            and iv_bid_block is not None
            and iv_ask_block is not None
        )

        if show_market:
            if len(iv_mkt_block) != len(maturities_block):
                raise ValueError(
                    "iv_mkt_block must have the same length as maturities_block."
                )

        if show_bid_ask_band:
            if (
                len(iv_bid_block) != len(maturities_block)
                or len(iv_ask_block) != len(maturities_block)
            ):
                raise ValueError(
                    "iv_bid_block and iv_ask_block must have the same length "
                    "as maturities_block."
                )

        # Optional: create folder if saving is requested
        if save_folder is not None:
            import os

            os.makedirs(save_folder, exist_ok=True)

        # Convert each model set into Python lists
        model_list_py = []
        for model in model_list:
            model_list_py.append(
                [
                    m.detach().cpu().tolist() if torch.is_tensor(m) else m
                    for m in model
                ]
            )

        # Compute implied vols only if input is prices
        if input_type == "price":
            iv_models = []

            for prices_list in model_list_py:
                iv_models.append(
                    compute_ivs(
                        prices_list,
                        maturities_block,
                        strikes_block,
                        self.S0,
                        [self.r_T[_r6(T)] for T in maturities_block],
                        [self.q_T[_r6(T)] for T in maturities_block],
                    )
                )
        else:
            # Input is already implied volatilities
            iv_models = model_list_py

        # Forward log-moneyness:
        # log(K / F_0T), where F_0T = S_0 exp((r_T - q_T) T)
        logstrikes_all = [
            np.log(
                np.asarray(strikes, dtype=float)
                / (
                    self.S0
                    * np.exp(
                        (
                            self.r_T[_r6(T)]
                            - self.q_T[_r6(T)]
                        )
                        * T
                    )
                )
            )
            for strikes, T in zip(strikes_block, maturities_block)
        ]

        # Combined plot: all maturities on the same figure
        # Colors = maturities, markers = models
        if combined_plot:
            import matplotlib.cm as cm
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            colors = cm.viridis(
                [
                    i / (len(maturities_block) - 1)
                    if len(maturities_block) > 1
                    else 0.5
                    for i in range(len(maturities_block))
                ]
            )

            markers_models = ["x", "D", "^", "s", "o", "P"]

            plt.figure(figsize=(12, 4))

            for t in range(len(maturities_block)):
                # Shaded bid-ask strip if market IVs are not passed
                if show_bid_ask_band:
                    bid_vals = np.asarray(iv_bid_block[t])
                    ask_vals = np.asarray(iv_ask_block[t])

                    plt.fill_between(
                        logstrikes_all[t],
                        bid_vals,
                        ask_vals,
                        color="royalblue",
                        alpha=0.55,
                        zorder=1,
                    )

                # Plot market data only if available
                if show_market:
                    plt.scatter(
                        logstrikes_all[t],
                        iv_mkt_block[t],
                        marker="s",
                        s=70,
                        facecolors="none",
                        edgecolors=colors[t],
                        linewidths=1.6,
                        zorder=10 + t,
                    )

                # Plot each model with a different marker
                for k, iv_model in enumerate(iv_models):
                    marker = markers_models[k % len(markers_models)]

                    # Keep the first model series orange and use crosses
                    if k == 0:
                        plt.scatter(
                            logstrikes_all[t],
                            iv_model[t],
                            marker="x",
                            s=75,
                            color="orange",
                            linewidths=1.8,
                            alpha=0.95,
                            zorder=20 + k,
                        )
                    else:
                        plt.scatter(
                            logstrikes_all[t],
                            iv_model[t],
                            marker=marker,
                            s=90,
                            facecolors="none",
                            edgecolors=colors[t],
                            linewidths=1.2,
                            alpha=0.85,
                            zorder=20 + k,
                        )

            # Maturity legend
            maturity_handles = [
                Line2D(
                    [0],
                    [0],
                    color=colors[t],
                    linewidth=3,
                    linestyle="-",
                    label=f"T = {maturities_block[t]}",
                )
                for t in range(len(maturities_block))
            ]

            # Model legend
            model_handles = []

            for k in range(len(iv_models)):
                if k == 0:
                    model_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="orange",
                            linestyle="none",
                            marker="x",
                            markersize=8,
                            markeredgewidth=1.8,
                            label=labels[k],
                        )
                    )
                else:
                    model_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="black",
                            linestyle="none",
                            marker=markers_models[k % len(markers_models)],
                            markerfacecolor="none",
                            markersize=8,
                            markeredgewidth=1.2,
                            label=labels[k],
                        )
                    )

            all_handles = maturity_handles.copy()

            # Add market marker handle if market IVs are shown
            if show_market:
                market_handle = Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="none",
                    marker="s",
                    markerfacecolor="none",
                    markersize=8,
                    markeredgewidth=1.6,
                    label="Market",
                )
                all_handles.append(market_handle)

            # Add bid-ask band handle if needed
            if show_bid_ask_band:
                band_handle = Patch(
                    facecolor="royalblue",
                    alpha=0.55,
                    edgecolor="none",
                    label="Market bid-ask",
                )
                all_handles.append(band_handle)

            all_handles += model_handles

            if title_suffix:
                plt.title(title_suffix, fontsize=16)

            plt.xlabel(
                r"Log moneyness",
                fontsize=14,
            )
            plt.ylabel("Implied Volatility", fontsize=14)
            plt.grid(True, alpha=0.25)

            plt.legend(
                handles=all_handles,
                frameon=False,
                loc="best",
                ncol=2,
                fontsize=12,
            )

            if save_folder is not None:
                import os

                fname = f"iv_smiles_combined_{title_suffix.replace(' ', '_')}"

                plt.savefig(
                    os.path.join(save_folder, fname + ".pdf"),
                    dpi=200,
                    bbox_inches="tight",
                )

            plt.show()
            return

        # Separate plot for each maturity
        for t in range(len(maturities_block)):
            plt.figure()

            # Shaded bid-ask strip if market IVs are not passed
            if show_bid_ask_band:
                bid_vals = np.asarray(iv_bid_block[t])
                ask_vals = np.asarray(iv_ask_block[t])

                plt.fill_between(
                    logstrikes_all[t],
                    bid_vals,
                    ask_vals,
                    color="royalblue",
                    alpha=0.55,
                    zorder=1,
                    label="Market bid-ask",
                )

            # Market: markers only, no line
            if show_market:
                plt.scatter(
                    logstrikes_all[t],
                    iv_mkt_block[t],
                    marker="s",
                    s=75,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.8,
                    label="Market",
                    zorder=10,
                )

            # Models: distinct markers, no line
            markers = ["x", "s", "^", "D", "o", "P"]

            for k, iv_model in enumerate(iv_models):
                marker = markers[k % len(markers)]

                if k == 0:
                    plt.scatter(
                        logstrikes_all[t],
                        iv_model[t],
                        marker="x",
                        s=75,
                        color="orange",
                        linewidths=1.8,
                        alpha=0.95,
                        label=labels[k],
                        zorder=20 + k,
                    )
                else:
                    plt.scatter(
                        logstrikes_all[t],
                        iv_model[t],
                        marker=marker,
                        s=90,
                        facecolors="none",
                        edgecolors=f"C{k + 1}",
                        linewidths=1.2,
                        alpha=0.9,
                        label=labels[k],
                        zorder=20 + k,
                    )

            if show_maturity_in_title:
                plot_title = f"T = {maturities_block[t]}"

                if title_suffix:
                    plot_title += f" -- {title_suffix}"

                plt.title(plot_title, fontsize=16)
            elif title_suffix:
                plt.title(title_suffix, fontsize=16)

            plt.xlabel(
                r"Log-moneyness",
                fontsize=16,
            )
            plt.ylabel("Implied Volatility", fontsize=16)

            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))

            plt.tick_params(labelsize=13)
            plt.grid(True, alpha=0.25)
            plt.legend(frameon=False, fontsize=13)

            if save_folder is not None:
                import os

                T_val = maturities_block[t]
                fname = f"iv_calibration_smile_{T_val}"

                plt.savefig(
                    os.path.join(save_folder, fname + ".pdf"),
                    dpi=200,
                    bbox_inches="tight",
                )

            plt.show()
