"""Construct Wiener-chaos bases and project terminal factors onto them.

Defines the piecewise-constant and general time-basis representations, including
the Legendre basis used in the experiments. These classes prepare multi-indices,
Brownian samples, and Dyson moment calculations for chaos evaluation and the
Monte Carlo projection used in Heston initialization.
"""

import math
from typing import Callable, List

import numpy as np
import torch

from .basis_integrals import (
    cholesky_spd,
    get_integrals,
    get_scalar_products,
    iterate_lists_with_trace,
)
from .utils import (
    create_Lambda,
    get_Gs,
    get_all_lambdas,
    merge_and_track_positions,
    simulate_bm,
    simulate_chaos,
)


def _round6(x: float) -> float:
    return float(np.round(x, 6))


class WienerChaosPiecewiseConstant:
    def __init__(self, d: int, P: int, pi_M: List[float], device: torch.device, type_float: torch.dtype = torch.float32) -> None:
        """
        d:      Brownian dimension
        P:      chaos truncation order
        pi_M:   grid [0=s0 < s1 < ... < sM = horizon] used for step functions
        device: torch device
        """
        self.d = int(d)
        self.P = int(P)
        self.pi_M = [float(x) for x in pi_M]
        self.device = device
        self.type_float = type_float

        self.horizon = float(self.pi_M[-1])
        self.M = len(self.pi_M) - 1

        # full Lambda on the full grid (M intervals)
        self.Lambda, self.Lambda_n01, self.Lambda_fact = get_all_lambdas(
            self.P, self.M, self.d, self.device, ignore_first=True
        )
        # Reused by every sample batch; keep gather indices on the target device.
        self.Lambda_n01 = [torch.as_tensor(i, device=self.device) for i in self.Lambda_n01]

        # Grids used to evaluate the conditional square at each maturity.
        self.pi_M_T = {}
        self.M_T = {}

        # interval lengths on the base grid
        self.deltas = [_round6(b - a) for a, b in zip(self.pi_M[:-1], self.pi_M[1:])]

    def set_step(self, T: float) -> None:
        """Store the basis grid truncated at maturity T."""
        T = _round6(T)
        pi_M_T = [_round6(x) for x in self.pi_M if x < T]
        pi_M_T.append(T)
        self.pi_M_T[T] = pi_M_T
        self.M_T[T] = len(pi_M_T) - 1

    def compute_chaos_coeffs(
        self, functional: Callable, pi_f: List[float], N: int, Ns: int,
        *, antithetic: bool = False,
    ) -> torch.Tensor:
        """
        MC projection of xi=functional(B evaluated on pi_f).

        Include the degree-zero coefficient E[xi]. Antithetic sampling pairs
        each Brownian path with its negative without changing N or Ns.
        """
        if int(N) < 1 or int(Ns) < 1:
            raise ValueError("N and Ns must be positive for chaos projection.")
        pi_union, pi_pos = merge_and_track_positions([self.pi_M, pi_f])
        d_a_list = []
        constants = []

        for _ in range(int(Ns)):
            bm = simulate_bm(int(N), self.d, pi_union, self.device,
                             type_float=self.type_float, antithetic=antithetic)
            bm_M = bm[:, :, pi_pos[0]]
            bm_f = bm[:, :, pi_pos[1]]

            Gs = get_Gs(bm_M, self.pi_M)
            X = simulate_chaos(self.P, self.M, self.d, Gs, self.Lambda_n01, self.device, self.type_float)  # (N, n_terms)

            xi = functional(bm_f).reshape(-1, 1)  # (N, 1)
            if xi.shape[0] != int(N) or not torch.isfinite(xi).all():
                raise ValueError("Projection functional must return one finite value per path.")
            d_a_list.append(torch.mean(X * xi, dim=0))
            constants.append(xi.mean())

        d_a = torch.stack(d_a_list, dim=0)
        coefficients = self.Lambda_fact * torch.mean(d_a, dim=0)
        return torch.cat((torch.stack(constants).mean().reshape(1), coefficients))


class WienerChaosDyson:
    def __init__(
        self,
        d: int,
        P: int,
        M: int,
        T_horizon: float,
        device: torch.device,
        type_basis: str = "sine",
        type_float: torch.dtype = torch.float32,
        base_jitter: float = 1e-13,
        cholesky_on_cpu: bool = False,
    ) -> None:
        self.d = int(d)
        self.P = int(P)
        self.M = int(M)
        self.T = float(T_horizon)
        self.device = device

        self.type_basis = str(type_basis)
        # if you want a hard check, uncomment:
        # if self.type_basis not in ("sine", "piecewise_constant", "cosine", "fourier", "kl", "haar", "legendre"):
        #     raise ValueError(f"Unknown type_basis={self.type_basis!r}")

        self.type_float = type_float
        self.base_jitter = float(base_jitter)
        self.cholesky_on_cpu = bool(cholesky_on_cpu)

        self.dyson = DysonFormulaFast(
            d=self.d,
            P=self.P,
            M=self.M,
            T=self.T,
            device=self.device,
            type_basis=self.type_basis,
            dtype=self.type_float,
            base_jitter=self.base_jitter,
            cholesky_on_cpu=self.cholesky_on_cpu,
        )

        self.Lambda = self.dyson.Lambda
        self.Lambda_fact = self.dyson.Lambda_fact
        self.n_terms = self.dyson.n_terms
        self._prepared_times: set[float] = set()


    def set_step(self, t: float) -> None:
        # warm up cache for ∫_t^T h_i h_j
        t = _round6(t)
        if not (0.0 <= t <= self.T):
            raise ValueError(f"t must be in [0, {self.T}], got {t}")
        _ = self.dyson._scalar_products_tT(t)
        self._prepared_times.add(t)

    def compute_chaos_coeffs(
        self, functional: Callable, pi_f: List[float], N: int, Ns: int,
    ) -> torch.Tensor:
        # MC projection onto the normalized Hermite basis, including its mean.
        d_a_list = []
        constant_list = []
        for _ in range(Ns):
            bm_f, Gs = self.dyson.simulate_B_and_integrals([0.0, self.T], pi_f, N)
            X = self.dyson.simulate_dyson(t_list=[0.0, self.T], G=Gs)[0]  # (N, n_terms)
            xi = functional(bm_f).unsqueeze(1)                            # (N, 1)
            d_a_list.append(torch.mean(X * xi, dim=0))                    # (n_terms,)
            constant_list.append(xi.mean())

        d_a = torch.stack(d_a_list, dim=0)  # (Ns, n_terms)
        fact = torch.as_tensor(self.Lambda_fact, device=self.device, dtype=d_a.dtype)
        coefficients = fact * torch.mean(d_a, dim=0)
        return torch.cat((torch.stack(constant_list).mean().reshape(1), coefficients))


class DysonFormulaFast:
    """
    Dyson conditional expectation features for truncated chaos.
    Works with whatever basis you plugged into:
      - get_scalar_products(s,t,...)  (∫_s^t h_i h_j)
      - get_integrals(n,s,t,...)      (∫_s^t h_n)
    """

    def __init__(
        self,
        d: int,
        P: int,
        M: int,
        T: float,
        device: torch.device,
        *,
        type_basis: str = "sine",
        dtype: torch.dtype = torch.float32,
        base_jitter: float = 1e-13,
        cholesky_on_cpu: bool = False,
    ):
        self.d = int(d)
        self.P = int(P)
        self.M = int(M)
        self.T = float(T)
        self.device = device

        self.type_basis = str(type_basis)
        self.dtype = dtype
        self.base_jitter = float(base_jitter)
        self.cholesky_on_cpu = bool(cholesky_on_cpu)

        # Lambda: flattened multi-indices of length M*d
        self.Lambda, _, self.Lambda_fact = create_Lambda(self.P, self.M, self.d, ignore_first=True)
        self.Lambda = np.asarray(self.Lambda, dtype=np.int64)  # (n_terms, M*d)
        self.n_terms = self.Lambda.shape[0]

        # Lambda_d[dim] is (n_terms, M)
        self.Lambda_d = [self.Lambda[:, dim::self.d] for dim in range(self.d)]

        # NEW: base triples per dimension (so we can compute a per-dim product)
        self._base_per_dim = self._build_base_per_dim()

        # Precompute Dyson decrement trees per dimension (even orders only)
        self._precomp = {N: self._build_precomp_for_N(N) for N in range(2, self.P + 1, 2)}

        self._scalar_cache: dict[float, torch.Tensor] = {}
        self._weight_cache: dict[tuple[float, int, int], torch.Tensor] = {}

        # device-specific tensors (moved lazily in simulate_dyson)
        # (kept as plain dicts to avoid extra dependencies)
        self._seg_chol_cache: dict[tuple, torch.Tensor] = {}
        self._int_vec_cache: dict[tuple, torch.Tensor] = {}
        self._scalar_st_cache: dict[tuple, torch.Tensor] = {}


    def _build_base_per_dim(self):
        """
        For each dim, build padded triples [deg, j, dim] of length P for every chaos term k.
        This lets us compute:  prod_{m} H_{a_{dim,m}}(G_{dim,m}(t))  via gather+prod.
        """
        out = []
        for dim in range(self.d):
            triples_dim = []
            for k in range(self.n_terms):
                a = self.Lambda_d[dim][k].tolist()  # length M
                tri = [[deg, j, dim] for j, deg in enumerate(a) if deg != 0]
                while len(tri) < self.P:
                    tri.append([0, 0, 0])  # H_0 = 1, safe padding
                triples_dim.append(tri[:self.P])

            arr = np.asarray(triples_dim, dtype=np.int64)  # (n_terms, P, 3)
            out.append({
                "deg": torch.as_tensor(arr[..., 0], dtype=torch.long),
                "j":   torch.as_tensor(arr[..., 1], dtype=torch.long),
                "dim": torch.as_tensor(arr[..., 2], dtype=torch.long),
            })
        return out

    def build_hermite_dict(self, G_t: torch.Tensor) -> torch.Tensor:
        # out: (n_paths, P+1, M, d)
        x = G_t
        n_paths = x.shape[0]
        out = torch.empty((n_paths, self.P + 1, self.M, self.d), device=x.device, dtype=x.dtype)
        out[:, 0] = 1.0
        if self.P >= 1:
            out[:, 1] = x
        for n in range(1, self.P):
            out[:, n + 1] = (x * out[:, n] - out[:, n - 1]) / float(n + 1)
        return out

    def _scalar_products_tT(self, t: float) -> torch.Tensor:
        # cache of ∫_t^T h_i h_j
        t = _round6(t)
        if t in self._scalar_cache:
            return self._scalar_cache[t]

        GtT = get_scalar_products(
            s=t,
            t=self.T,
            M=self.M,
            T=self.T,
            type_basis=self.type_basis,
            device=self.device,
            type_float=self.dtype,
        )
        self._scalar_cache[t] = GtT
        return GtT

    def _build_precomp_for_N(self, N: int):
        """
        Precompute, for each Brownian dim, all children obtained by N single decrements
        (N even), together with the trace indices used to build Gram-matrix weights.

        Output per dim:
          - keep_idx:   torch.LongTensor (n_kept,)
          - keep_pos:   dict[int,int] mapping global k -> row in the kept arrays
          - child_deg:  torch.LongTensor (n_kept, max_children, P)   padded triples (deg)
          - child_j:    torch.LongTensor (n_kept, max_children, P)   padded triples (j)
          - child_dim:  torch.LongTensor (n_kept, max_children, P)   padded triples (dim)
          - child_full: torch.LongTensor (n_kept, max_children, M)   full degree vectors per child (per-dim)
          - traces:     torch.LongTensor (n_kept, max_children, N)   decrement trace indices (slot indices in 0..M-1)
        """
        if N <= 0 or (N % 2) != 0:
            raise ValueError(f"N must be a positive even integer, got N={N}.")

        n = N // 2
        pre = {"n": n, "per_dim": []}

        for dim in range(self.d):
            keep: List[int] = []
            children_triples: List[List[List[List[int]]]] = []  # [i_kept][c][p][3]
            children_full: List[List[List[int]]] = []           # [i_kept][c][M]
            traces_all: List[List[List[int]]] = []              # [i_kept][c][N]

            for k in range(self.n_terms):
                initial = self.Lambda_d[dim][k].tolist()  # length M
                kids, traces = iterate_lists_with_trace(initial=initial, N=N)
                if not kids:
                    continue

                keep.append(k)

                # store full children degree vectors (length M)
                children_full.append([child[:] for child in kids])

                # store padded triples for fast Hermite gather/prod
                kids_padded: List[List[List[int]]] = []
                for child in kids:
                    tri = [[int(deg), int(j), int(dim)] for j, deg in enumerate(child) if deg != 0]
                    while len(tri) < self.P:
                        tri.append([0, 0, 0])
                    kids_padded.append(tri[: self.P])

                children_triples.append(kids_padded)
                traces_all.append([tr[:] for tr in traces])

            keep_idx = torch.as_tensor(keep, dtype=torch.long)

            if keep_idx.numel() == 0:
                pre["per_dim"].append(
                    {
                        "keep_idx": keep_idx,
                        "keep_pos": {},
                        "child_deg": None,
                        "child_j": None,
                        "child_dim": None,
                        "child_full": None,
                        "traces": None,
                    }
                )
                continue

            # Map global term index -> row position in the kept arrays (crucial for condexp_coeffs)
            keep_pos: Dict[int, int] = {int(k): i for i, k in enumerate(keep)}

            max_children = max(len(x) for x in children_triples)

            # Allocate arrays
            child_arr = np.zeros((len(children_triples), max_children, self.P, 3), dtype=np.int64)
            child_full_arr = np.zeros((len(children_full), max_children, self.M), dtype=np.int64)
            trace_arr = -np.ones((len(traces_all), max_children, N), dtype=np.int64)

            # Fill arrays
            for i_k, (kids_padded, kids_full, traces) in enumerate(
                zip(children_triples, children_full, traces_all)
            ):
                # kids_padded and traces are aligned (same number of children)
                for c, (triP, child_vec, trN) in enumerate(zip(kids_padded, kids_full, traces)):
                    child_arr[i_k, c, :, :] = np.asarray(triP, dtype=np.int64)
                    child_full_arr[i_k, c, :] = np.asarray(child_vec, dtype=np.int64)
                    trace_arr[i_k, c, :] = np.asarray(trN, dtype=np.int64)

            # Convert to tensors (kept on CPU by default; moved to device at call sites)
            child_deg = torch.as_tensor(child_arr[..., 0], dtype=torch.long)
            child_j = torch.as_tensor(child_arr[..., 1], dtype=torch.long)
            child_dim = torch.as_tensor(child_arr[..., 2], dtype=torch.long)
            child_full_t = torch.as_tensor(child_full_arr, dtype=torch.long)
            traces_t = torch.as_tensor(trace_arr, dtype=torch.long)

            pre["per_dim"].append(
                {
                    "keep_idx": keep_idx,
                    "keep_pos": keep_pos,
                    "child_deg": child_deg,
                    "child_j": child_j,
                    "child_dim": child_dim,
                    "child_full": child_full_t,
                    "traces": traces_t,
                }
            )

        return pre


    @staticmethod
    def _gather_prod(hermite_dict, deg, j, dim):
        return hermite_dict[:, deg, j, dim].prod(dim=-1)

    def _trace_weights(self, t, N, dim, traces):
        """
        Gram-trace weights for Dyson terms.

        traces: (n_kept, max_children, N) with N even
        """
        t = _round6(t)

        if N <= 0 or (N % 2) != 0:
            raise ValueError(f"N must be positive even, got N={N}.")

        device = traces.device

        GtT = self._scalar_products_tT(t)
        if GtT.device != device:
            GtT = GtT.to(device=device)

        key = (t, int(N), int(dim), str(device), str(GtT.dtype))
        if key in self._weight_cache:
            return self._weight_cache[key]

        n = N // 2

        # padding children have traces filled with -1
        valid = (traces[..., 0] >= 0)  # (n_kept, max_children)

        tp = traces.view(*traces.shape[:-1], n, 2).clamp_min(0)  # (n_kept, max_children, n, 2)
        a = tp[..., 0]
        b = tp[..., 1]

        w = GtT[a, b].prod(dim=-1)  # (n_kept, max_children)
        w = w * valid.to(w.dtype)

        self._weight_cache[key] = w
        return w


    def simulate_dyson(self, t_list, G):
        """Return [E[Phi | F_t]] evaluated at each t in t_list[1:]."""
        sims = []

        # Move index tensors to the right device once.
        dev = G.device

        base = []
        for dim in range(self.d):
            bd = self._base_per_dim[dim]
            base.append(
                {
                    "deg": bd["deg"].to(dev),
                    "j": bd["j"].to(dev),
                    "dim": bd["dim"].to(dev),
                }
            )

        precomp = {}
        for N in range(2, self.P + 1, 2):
            precomp[N] = []
            for dim in range(self.d):
                pd = self._precomp[N]["per_dim"][dim]
                keep_idx = pd["keep_idx"]
                if keep_idx is None or keep_idx.numel() == 0:
                    precomp[N].append(None)
                    continue

                precomp[N].append(
                    {
                        "keep_idx": keep_idx.to(dev),
                        "child_deg": pd["child_deg"].to(dev),
                        "child_j": pd["child_j"].to(dev),
                        "child_dim": pd["child_dim"].to(dev),
                        "traces": pd["traces"].to(dev),
                    }
                )

        for step, t in enumerate(t_list[1:]):
            hermite_dict = self.build_hermite_dict(G_t=G[step])

            dim_factors = []
            for dim in range(self.d):
                bd = base[dim]

                # n=0 term: just freeze omega^t in the Hermite arguments
                factor = self._gather_prod(hermite_dict, bd["deg"], bd["j"], bd["dim"])  # (N, n_terms)

                # Dyson corrections (even orders only)
                for N in range(2, self.P + 1, 2):
                    pd = precomp[N][dim]
                    if pd is None:
                        continue

                    n = self._precomp[N]["n"]
                    coeff = (0.5 ** n) / math.factorial(n)

                    child_prod = self._gather_prod(
                        hermite_dict, pd["child_deg"], pd["child_j"], pd["child_dim"]
                    )  # (N, n_kept, max_children)

                    w = self._trace_weights(t, N, dim, pd["traces"])  # (n_kept, max_children)

                    dyson_factor = (child_prod * w.unsqueeze(0)).sum(dim=-1)  # (N, n_kept)
                    factor[:, pd["keep_idx"]] += coeff * dyson_factor

                dim_factors.append(factor)

            # product across dimensions
            dyson = dim_factors[0]
            for f in dim_factors[1:]:
                dyson = dyson * f

            sims.append(dyson)

        return sims


    def simulate_B_and_integrals(self, t_list, pi_bm, n_samples):
        """
        Simulate (B on pi_bm, cumulative integrals G up to each t in t_list).

        We build, for each interval [t_i, t_{i+1}], the joint Gaussian vector
          (ΔB on the union BM grid inside the interval, ∫_{t_i}^{t_{i+1}} h_j dB)
        and then cum-sum across intervals.
        """
        t_list = [_round6(x) for x in t_list]
        pi_bm = [_round6(x) for x in pi_bm]

        if len(t_list) < 2:
            raise ValueError("t_list must have at least 2 points")
        if t_list[0] != 0.0:
            raise ValueError("t_list must start at 0.0")
        if pi_bm[0] != 0.0:
            raise ValueError("pi_bm must start at 0.0")
        if any(t_list[i] >= t_list[i + 1] for i in range(len(t_list) - 1)):
            raise ValueError("t_list must be strictly increasing")
        if any(pi_bm[i] >= pi_bm[i + 1] for i in range(len(pi_bm) - 1)):
            raise ValueError("pi_bm must be strictly increasing")
        if pi_bm[-1] > t_list[-1] + 1e-12:
            raise ValueError("require max(pi_bm) <= max(t_list)")
        if t_list[-1] > self.T + 1e-12:
            raise ValueError(f"require max(t_list) <= T={self.T}")

        pi_union, pos = merge_and_track_positions([t_list, pi_bm])
        pos_t, pos_b = pos[0], pos[1]

        n_union = len(pi_union)
        n_steps_t = len(t_list) - 1

        dB_union = torch.zeros((n_union - 1, n_samples, self.d), device=self.device, dtype=self.dtype)
        dG_steps = torch.zeros((n_steps_t, n_samples, self.M, self.d), device=self.device, dtype=self.dtype)

        # lazy caches (per instance)
        if not hasattr(self, "_seg_chol_cache"):
            self._seg_chol_cache = {}
        if not hasattr(self, "_int_vec_cache"):
            self._int_vec_cache = {}
        if not hasattr(self, "_scalar_st_cache"):
            self._scalar_st_cache = {}

        def _int_vec(a: float, b: float) -> torch.Tensor:
            key = (_round6(a), _round6(b), self.type_basis, str(self.device), str(self.dtype), int(self.M))
            out = self._int_vec_cache.get(key)
            if out is None:
                out = torch.stack(
                    [
                        get_integrals(
                            j + 1,
                            a,
                            b,
                            self.T,
                            type_basis=self.type_basis,
                            device=self.device,
                            type_float=self.dtype,
                            M=self.M,
                        )
                        for j in range(self.M)
                    ],
                    dim=0,
                )
                self._int_vec_cache[key] = out
            return out

        for i in range(n_steps_t):
            s = float(t_list[i])
            t = float(t_list[i + 1])

            u0 = pos_t[i]
            u1 = pos_t[i + 1]
            if u1 <= u0:
                raise RuntimeError("bad merge_and_track_positions output")

            K = u1 - u0
            times_seg = pi_union[u0 : u1 + 1]  # length K+1

            # Cache Cholesky per segment (depends on the local BM grid inside the interval).
            seg_key = (tuple(times_seg), self.type_basis, str(self.device), str(self.dtype), int(self.M))
            L = self._seg_chol_cache.get(seg_key)

            if L is None:
                dt = torch.as_tensor(np.diff(np.asarray(times_seg, dtype=np.float64)), device=self.device, dtype=self.dtype)

                dim_tot = K + self.M
                cov = torch.zeros((dim_tot, dim_tot), device=self.device, dtype=self.dtype)

                # Var(dB) block
                cov[:K, :K].diagonal().copy_(dt)

                # Cov(dB_k, ∫ h_j dB) block (fill by rows)
                V = torch.empty((K, self.M), device=self.device, dtype=self.dtype)
                for k in range(K):
                    a = float(times_seg[k])
                    b = float(times_seg[k + 1])
                    V[k] = _int_vec(a, b)

                cov[:K, K:] = V
                cov[K:, :K] = V.T

                # Cov(∫h dB, ∫h dB) block (depends only on the interval endpoints)
                st_key = (_round6(s), _round6(t), self.type_basis, str(self.device), str(self.dtype), int(self.M))
                cov_h = self._scalar_st_cache.get(st_key)
                if cov_h is None:
                    cov_h = get_scalar_products(
                        s=s,
                        t=t,
                        M=self.M,
                        T=self.T,
                        type_basis=self.type_basis,
                        device=self.device,
                        type_float=self.dtype,
                    )
                    self._scalar_st_cache[st_key] = cov_h
                cov[K:, K:] = cov_h

                L = cholesky_spd(cov, base_jitter=self.base_jitter, cholesky_on_cpu=self.cholesky_on_cpu)
                self._seg_chol_cache[seg_key] = L

            dim_tot = L.shape[0]
            Z = torch.randn((n_samples, dim_tot, self.d), device=self.device, dtype=self.dtype)

            # Y = L @ Z (broadcast over samples)
            Y = L @ Z  # (n_samples, dim_tot, d)

            dB_union[u0:u1] = Y[:, :K, :].transpose(0, 1)
            dG_steps[i] = Y[:, K:, :].reshape(n_samples, self.M, self.d)

        B_union = torch.zeros((n_union, n_samples, self.d), device=self.device, dtype=self.dtype)
        B_union[1:] = torch.cumsum(dB_union, dim=0)

        B_bm = B_union[pos_b]
        G_cum = torch.cumsum(dG_steps, dim=0)
        return B_bm.permute(1, 2, 0), G_cum
