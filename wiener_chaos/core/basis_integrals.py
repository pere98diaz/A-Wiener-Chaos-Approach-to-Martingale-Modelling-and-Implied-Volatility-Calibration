"""Integrals and scalar products of the time-basis functions.

These quantities determine the means and covariances of Brownian basis
integrals used by the chaos model. Includes piecewise-constant, Legendre,
trigonometric, Haar, and Karhunen–Loeve bases, with common dispatch functions
and a Cholesky factorization that adds jitter when needed.
"""

from typing import List, Literal, Optional, Tuple

import numpy as np
import torch


# torch.use_deterministic_algorithms(True)  # if you really want strict determinism


BasisType = Literal[
    "sine",
    "piecewise_constant",
    "cosine",
    "fourier",
    "kl",
    "haar",
    "legendre",
]


# ------------------------------------------------------------
# small helpers
# ------------------------------------------------------------
def _as(x, *, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype)


def _overlap_len(s: torch.Tensor, t: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # length of [s,t] ∩ [a,b]
    return torch.clamp(torch.minimum(t, b) - torch.maximum(s, a), min=0.0)


# ------------------------------------------------------------
# traces for Dyson precomputation
# ------------------------------------------------------------
def iterate_lists_with_trace(initial: List[int], N: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Like iterate_lists, but also keeps the indices where we decremented.
    """
    current_level = [(initial[:], [])]  # (list, trace)

    for _ in range(N):
        next_level = []
        for lst, trace in current_level:
            if all(x == 0 for x in lst):
                continue

            for i, x in enumerate(lst):
                if x > 0:
                    child = lst[:]
                    child[i] -= 1
                    next_level.append((child, trace + [i]))

        current_level = next_level
        if not current_level:
            break

    return [lst for lst, _ in current_level], [trace for _, trace in current_level]


# ============================================================
# 1) "SINE" BASIS WITH CONSTANT AS FIRST ELEMENT (ALWAYS ON)
#   h_1(t) = 1/sqrt(T)
#   h_n(t) = sqrt(2/T) * sin( (2*(n-1))π t/T ), n>=2   (even freqs only)
# ============================================================
def _freq_from_index(n: int) -> Optional[int]:
    if n == 1:
        return None
    return 2 * (n - 1)


def sine_basis_inner_product(n, m, s, t, T, *, device, type_float = torch.float32):
    """
    ∫_s^t h_n(r) h_m(r) dr with:
      h_1 = 1/sqrt(T),
      h_n = sqrt(2/T) sin( (2*(n-1))π r/T ) for n>=2
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)
    pi = torch.pi

    fn = _freq_from_index(int(n))
    fm = _freq_from_index(int(m))

    # constant-constant
    if fn is None and fm is None:
        return (t - s) / T

    # constant-sine
    if fn is None or fm is None:
        k = fm if fn is None else fn
        k_t = _as(k, device=device, dtype = type_float)
        return (torch.sqrt(_as(2.0, device=device, dtype = type_float)) / (k_t * pi)) * (
            torch.cos(k_t * pi * s / T) - torch.cos(k_t * pi * t / T)
        )

    # sine-sine
    n_t = _as(fn, device=device, dtype = type_float)
    m_t = _as(fm, device=device, dtype = type_float)

    if fn == fm:
        return (t - s) / T - (torch.sin(2 * n_t * pi * t / T) - torch.sin(2 * n_t * pi * s / T)) / (2 * n_t * pi)

    nm = n_t - m_t
    np_ = n_t + m_t
    return (1 / pi) * (
        (torch.sin(nm * pi * t / T) - torch.sin(nm * pi * s / T)) / nm
        - (torch.sin(np_ * pi * t / T) - torch.sin(np_ * pi * s / T)) / np_
    )


def sine_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    """
    ∫_s^t h_n(u) du with same "sine" basis (constant first).
    """
    if n <= 0:
        raise ValueError("n must be positive integer.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)
    pi = torch.pi

    fn = _freq_from_index(int(n))

    if fn is None:
        return (t - s) / torch.sqrt(T)

    k_t = _as(fn, device=device, dtype = type_float)
    return torch.sqrt(2 / T) * (T / (k_t * pi)) * (torch.cos(k_t * pi * s / T) - torch.cos(k_t * pi * t / T))


# ============================================================
# 2) PIECEWISE CONSTANT on uniform grid (n=1..M)
#   h_n = 1_{((n-1)Δ, nΔ]}/sqrt(Δ),  Δ=T/M
# ============================================================
def piecewise_constant_basis_integral(n, s, t, T, *, M: int, device, type_float = torch.float32):
    if n <= 0 or n > M:
        raise ValueError(f"n must be in 1..M (got n={n}, M={M}).")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s_t = _as(s, device=device, dtype = type_float)
    t_t = _as(t, device=device, dtype = type_float)
    T_t = _as(T, device=device, dtype = type_float)

    Delta = T_t / _as(M, device=device, dtype = type_float)
    a = _as(n - 1, device=device, dtype = type_float) * Delta
    b = _as(n, device=device, dtype = type_float) * Delta
    ol = _overlap_len(s_t, t_t, a, b)
    return ol / torch.sqrt(Delta)


def piecewise_constant_basis_inner_product(n, m, s, t, T, *, M: int, device, type_float = torch.float32):
    if n <= 0 or m <= 0 or n > M or m > M:
        raise ValueError(f"n,m must be in 1..M (got n={n}, m={m}, M={M}).")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")
    if n != m:
        return _as(0.0, device=device, dtype = type_float)

    s_t = _as(s, device=device, dtype = type_float)
    t_t = _as(t, device=device, dtype = type_float)
    T_t = _as(T, device=device, dtype = type_float)

    Delta = T_t / _as(M, device=device, dtype = type_float)
    a = _as(n - 1, device=device, dtype = type_float) * Delta
    b = _as(n, device=device, dtype = type_float) * Delta
    ol = _overlap_len(s_t, t_t, a, b)
    return ol / Delta


# ============================================================
# 3) COSINE with constant first:
#   h_1 = 1/sqrt(T)
#   h_n = sqrt(2/T) cos((n-1)π t/T), n>=2
# ============================================================
def cosine_basis_inner_product(n, m, s, t, T, *, device, type_float = torch.float32):
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers (starting at 1).")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)
    pi = torch.pi

    if n == 1 and m == 1:
        return (t - s) / T

    # constant-cosine
    if (n == 1) ^ (m == 1):
        other = m if n == 1 else n
        k = other - 1
        k_t = _as(k, device=device, dtype = type_float)
        return (torch.sqrt(_as(2.0, device=device, dtype = type_float)) / (k_t * pi)) * (
            torch.sin(k_t * pi * t / T) - torch.sin(k_t * pi * s / T)
        )

    # cosine-cosine
    k = n - 1
    ell = m - 1
    k_t = _as(k, device=device, dtype = type_float)
    ell_t = _as(ell, device=device, dtype = type_float)

    if k == ell:
        return (t - s) / T + (torch.sin(2 * k_t * pi * t / T) - torch.sin(2 * k_t * pi * s / T)) / (2 * k_t * pi)

    km = k_t - ell_t
    kp = k_t + ell_t
    return (1 / pi) * (
        (torch.sin(km * pi * t / T) - torch.sin(km * pi * s / T)) / km
        + (torch.sin(kp * pi * t / T) - torch.sin(kp * pi * s / T)) / kp
    )


def cosine_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    if n <= 0:
        raise ValueError("n must be a positive integer (starting at 1).")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)
    pi = torch.pi

    if n == 1:
        return (t - s) / torch.sqrt(T)

    k = n - 1
    k_t = _as(k, device=device, dtype = type_float)
    return torch.sqrt(2 / T) * (T / (k_t * pi)) * (torch.sin(k_t * pi * t / T) - torch.sin(k_t * pi * s / T))


# ============================================================
# 4) FOURIER with constant first:
#   h_1 = 1/sqrt(T)
#   then cos/sin with 2πk/T in the usual order
# ============================================================
def _fourier_mode(n: int):
    # n=1 -> const, n=2 -> cos(1), n=3 -> sin(1), n=4 -> cos(2), ...
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if n == 1:
        return ("const", 0)
    if n % 2 == 0:
        return ("cos", n // 2)
    return ("sin", (n - 1) // 2)


def _Ic(a, s, t):
    # ∫ cos(a u) du over [s,t], safe at a=0
    tiny = torch.finfo(a.dtype).tiny
    num = torch.sin(a * t) - torch.sin(a * s)
    return torch.where(a.abs() <= tiny, (t - s), num / a)


def _Is(a, s, t):
    # ∫ sin(a u) du over [s,t], safe at a=0
    tiny = torch.finfo(a.dtype).tiny
    num = torch.cos(a * s) - torch.cos(a * t)
    return torch.where(a.abs() <= tiny, torch.zeros_like(a), num / a)


def fourier_basis_inner_product(n, m, s, t, T, *, device, type_float = torch.float32):
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    kind_n, k_n = _fourier_mode(int(n))
    kind_m, k_m = _fourier_mode(int(m))

    an = (1.0 / torch.sqrt(T)) if kind_n == "const" else torch.sqrt(_as(2.0, device=device, dtype = type_float) / T)
    am = (1.0 / torch.sqrt(T)) if kind_m == "const" else torch.sqrt(_as(2.0, device=device, dtype = type_float) / T)

    if kind_n == "const" and kind_m == "const":
        raw = (t - s)
        return an * am * raw

    pi = torch.pi
    alpha = _as(2.0, device=device, dtype = type_float) * pi / T
    wn = alpha * _as(k_n, device=device, dtype = type_float)
    wm = alpha * _as(k_m, device=device, dtype = type_float)

    if kind_n == "const":
        raw = _Ic(wm, s, t) if kind_m == "cos" else _Is(wm, s, t)
        return an * am * raw

    if kind_m == "const":
        raw = _Ic(wn, s, t) if kind_n == "cos" else _Is(wn, s, t)
        return an * am * raw

    if kind_n == "cos" and kind_m == "cos":
        raw = 0.5 * (_Ic(wn - wm, s, t) + _Ic(wn + wm, s, t))
    elif kind_n == "sin" and kind_m == "sin":
        raw = 0.5 * (_Ic(wn - wm, s, t) - _Ic(wn + wm, s, t))
    else:
        raw = 0.5 * (_Is(wn + wm, s, t) + _Is(wn - wm, s, t))

    return an * am * raw


def fourier_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    if n <= 0:
        raise ValueError("n must be positive integer.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    kind, k = _fourier_mode(int(n))
    if kind == "const":
        return (t - s) / torch.sqrt(T)

    pi = torch.pi
    alpha = _as(2.0, device=device, dtype = type_float) * pi / T
    w = alpha * _as(k, device=device, dtype = type_float)
    a = torch.sqrt(_as(2.0, device=device, dtype = type_float) / T)

    return a * (_Ic(w, s, t) if kind == "cos" else _Is(w, s, t))


# ============================================================
# 5) KL(BM): h_n(t) = sqrt(2/T) sin((n-1/2)π t/T), n>=1
# ============================================================
def kl_basis_inner_product(n, m, s, t, T, *, device, type_float = torch.float32):
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)
    pi = torch.pi

    n_i = int(n)
    m_i = int(m)

    if n_i == m_i:
        k = 2 * n_i - 1
        k_t = _as(k, device=device, dtype = type_float)
        return (t - s) / T - (torch.sin(k_t * pi * t / T) - torch.sin(k_t * pi * s / T)) / (k_t * pi)

    nm = n_i - m_i
    np1 = n_i + m_i - 1

    nm_t = _as(nm, device=device, dtype = type_float)
    np1_t = _as(np1, device=device, dtype = type_float)

    term1 = (torch.sin(nm_t * pi * t / T) - torch.sin(nm_t * pi * s / T)) / nm_t
    term2 = (torch.sin(np1_t * pi * t / T) - torch.sin(np1_t * pi * s / T)) / np1_t
    return (1.0 / pi) * (term1 - term2)


def kl_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    n_i = int(n)
    pi = torch.pi
    a = ((_as(n_i, device=device, dtype = type_float) - _as(0.5, device=device, dtype = type_float)) * pi) / T

    return torch.sqrt(_as(2.0, device=device, dtype = type_float) / T) * (torch.cos(a * s) - torch.cos(a * t)) / a


# ============================================================
# 6) HAAR with constant first:
#   h_1 = 1/sqrt(T)
#   then wavelets in breadth-first order
# ============================================================
def _haar_index_to_jk(n: int):
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return None
    m = n - 2
    j = (m + 1).bit_length() - 1
    start = (1 << j) - 1
    k = m - start
    return j, k


def haar_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    if n == 1:
        return (t - s) / torch.sqrt(T)

    j, k = _haar_index_to_jk(int(n))
    two_j = 1 << j

    a = T * _as(k, device=device, dtype = type_float) / _as(two_j, device=device, dtype = type_float)
    b = T * _as(k + 1, device=device, dtype = type_float) / _as(two_j, device=device, dtype = type_float)
    mid = T * _as(2 * k + 1, device=device, dtype = type_float) / _as(2 * two_j, device=device, dtype = type_float)

    amp = torch.sqrt(_as(two_j, device=device, dtype = type_float) / T)
    L_left = _overlap_len(s, t, a, mid)
    L_right = _overlap_len(s, t, mid, b)
    return amp * (L_left - L_right)


def haar_basis_inner_product(n, m, s, t, T, *, device, type_float = torch.float32):
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    if n == 1 and m == 1:
        return (t - s) / T

    if n == 1 and m != 1:
        return (1.0 / torch.sqrt(T)) * haar_basis_integral(m, s, t, T, device=device, type_float=type_float)
    if m == 1 and n != 1:
        return (1.0 / torch.sqrt(T)) * haar_basis_integral(n, s, t, T, device=device, type_float=type_float)

    j1, k1 = _haar_index_to_jk(int(n))
    j2, k2 = _haar_index_to_jk(int(m))

    two_j1 = 1 << j1
    two_j2 = 1 << j2

    amp1 = torch.sqrt(_as(two_j1, device=device, dtype = type_float) / T)
    amp2 = torch.sqrt(_as(two_j2, device=device, dtype = type_float) / T)

    a1 = T * _as(k1, device=device, dtype = type_float) / _as(two_j1, device=device, dtype = type_float)
    b1 = T * _as(k1 + 1, device=device, dtype = type_float) / _as(two_j1, device=device, dtype = type_float)
    mid1 = T * _as(2 * k1 + 1, device=device, dtype = type_float) / _as(2 * two_j1, device=device, dtype = type_float)

    a2 = T * _as(k2, device=device, dtype = type_float) / _as(two_j2, device=device, dtype = type_float)
    b2 = T * _as(k2 + 1, device=device, dtype = type_float) / _as(two_j2, device=device, dtype = type_float)
    mid2 = T * _as(2 * k2 + 1, device=device, dtype = type_float) / _as(2 * two_j2, device=device, dtype = type_float)

    pieces1 = [(a1, mid1, +1.0), (mid1, b1, -1.0)]
    pieces2 = [(a2, mid2, +1.0), (mid2, b2, -1.0)]

    val = _as(0.0, device=device, dtype = type_float)
    for (u0, u1, sgn1) in pieces1:
        for (v0, v1, sgn2) in pieces2:
            inter = _overlap_len(s, t, torch.maximum(u0, v0), torch.minimum(u1, v1))
            val = val + (amp1 * amp2) * _as(sgn1 * sgn2, device=device, dtype = type_float) * inter

    return val


# ============================================================
# 7) SHIFTED LEGENDRE (constant is n=1)
# ============================================================
_GL_CACHE = {}  # (K, dtype_str) -> (z_cpu, w_cpu)


def _legendre_P_all(max_deg: int, x: torch.Tensor) -> torch.Tensor:
    if max_deg < 0:
        raise ValueError("max_deg must be >= 0")
    P = torch.empty((max_deg + 1, *x.shape), device=x.device, dtype=x.dtype)
    P[0] = 1.0
    if max_deg >= 1:
        P[1] = x
    for n in range(1, max_deg):
        P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
    return P


def _legendre_P(k: int, x: torch.Tensor) -> torch.Tensor:
    return _legendre_P_all(k, x)[k]


def _gauss_legendre_nodes_weights(K: int, *, type_float = torch.float32):
    key = (int(K), str(type_float))
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    z, w = np.polynomial.legendre.leggauss(int(K))
    z_t = torch.tensor(z, dtype = type_float)  # CPU
    w_t = torch.tensor(w, dtype = type_float)  # CPU
    _GL_CACHE[key] = (z_t, w_t)
    return z_t, w_t


def legendre_basis_integral(n, s, t, T, *, device, type_float = torch.float32):
    if n <= 0:
        raise ValueError("n must be positive integer.")
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    xs = 2.0 * s / T - 1.0
    xt = 2.0 * t / T - 1.0

    k = int(n) - 1
    if k == 0:
        integral_P = (xt - xs)
    else:
        denom = _as(2 * k + 1, device=device, dtype = type_float)
        Pkp1_t = _legendre_P(k + 1, xt)
        Pkm1_t = _legendre_P(k - 1, xt)
        Pkp1_s = _legendre_P(k + 1, xs)
        Pkm1_s = _legendre_P(k - 1, xs)
        integral_P = ((Pkp1_t - Pkm1_t) - (Pkp1_s - Pkm1_s)) / denom

    scale = 0.5 * torch.sqrt(_as(2 * n - 1, device=device, dtype = type_float) * T)
    return scale * integral_P


def get_scalar_products_legendre(s, t, M, T, device, *, type_float = torch.float32):
    if not (0 <= s <= t <= T):
        raise ValueError("require 0 <= s <= t <= T")

    s = _as(s, device=device, dtype = type_float)
    t = _as(t, device=device, dtype = type_float)
    T = _as(T, device=device, dtype = type_float)

    a = 2.0 * s / T - 1.0
    b = 2.0 * t / T - 1.0
    if torch.allclose(a, b):
        return torch.zeros((M, M), device=device, dtype = type_float)

    K = int(M)
    z_cpu, w_cpu = _gauss_legendre_nodes_weights(K, type_float = type_float)
    z = z_cpu.to(device=device, dtype = type_float)
    w = w_cpu.to(device=device, dtype = type_float)

    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * z
    dx = half

    Pvals = _legendre_P_all(M - 1, x)  # (M, K)
    I = dx * (Pvals * w.unsqueeze(0)) @ Pvals.T

    n = torch.arange(1, M + 1, device=device, dtype = type_float)
    svec = torch.sqrt(2.0 * n - 1.0)
    S = svec.unsqueeze(1) * svec.unsqueeze(0)
    return 0.5 * S * I


# ============================================================
# DISPATCH: keep these names
# ============================================================
def get_integrals(n, s, t, T, type_basis: BasisType, *, device, type_float = torch.float32, M: Optional[int] = None):
    if type_basis == "sine":
        return sine_basis_integral(n, s, t, T, device=device, type_float = type_float)

    if type_basis == "piecewise_constant":
        if M is None:
            raise ValueError("type_basis='piecewise_constant' requires M.")
        return piecewise_constant_basis_integral(n, s, t, T, M=M, device=device, type_float = type_float)

    if type_basis == "cosine":
        return cosine_basis_integral(n, s, t, T, device=device, type_float = type_float)

    if type_basis == "fourier":
        return fourier_basis_integral(n, s, t, T, device=device, type_float = type_float)

    if type_basis == "kl":
        return kl_basis_integral(n, s, t, T, device=device, type_float = type_float)

    if type_basis == "haar":
        return haar_basis_integral(n, s, t, T, device=device, type_float = type_float)

    if type_basis == "legendre":
        return legendre_basis_integral(n, s, t, T, device=device, type_float = type_float)

    raise ValueError(f"Unknown type_basis={type_basis!r}")


def get_scalar_products(s, t, M, T, type_basis: BasisType, device, *, type_float = torch.float32):
    """
    Matrix [i,j] = ∫_s^t h_{i+1}(r) h_{j+1}(r) dr
    """
    if type_basis == "legendre":
        return get_scalar_products_legendre(s, t, M, T, device, type_float = type_float)

    if type_basis == "sine":
        f = lambda i, j: sine_basis_inner_product(i, j, s, t, T, device=device, type_float = type_float)
    elif type_basis == "piecewise_constant":
        f = lambda i, j: piecewise_constant_basis_inner_product(i, j, s, t, T, M=M, device=device, type_float = type_float)
    elif type_basis == "cosine":
        f = lambda i, j: cosine_basis_inner_product(i, j, s, t, T, device=device, type_float = type_float)
    elif type_basis == "fourier":
        f = lambda i, j: fourier_basis_inner_product(i, j, s, t, T, device=device, type_float = type_float)
    elif type_basis == "kl":
        f = lambda i, j: kl_basis_inner_product(i, j, s, t, T, device=device, type_float = type_float)
    elif type_basis == "haar":
        f = lambda i, j: haar_basis_inner_product(i, j, s, t, T, device=device, type_float = type_float)
    else:
        raise ValueError(f"Unknown type_basis={type_basis!r}")

    mat = torch.empty((M, M), device=device, dtype = type_float)
    for i in range(M):
        mat[i, i] = f(i + 1, i + 1)
        for j in range(i + 1, M):
            v = f(i + 1, j + 1)
            mat[i, j] = v
            mat[j, i] = v
    return mat

def cholesky_spd(A: torch.Tensor, base_jitter: float = 1e-13, max_tries: int = 8, cholesky_on_cpu: bool = False):
    """
    Cholesky with a bit of jitter if needed.
    """
    if cholesky_on_cpu and A.is_cuda:
        L_cpu = cholesky_spd(A.detach().cpu(), base_jitter=base_jitter, max_tries=max_tries, cholesky_on_cpu=False)
        return L_cpu.to(device=A.device, dtype=A.dtype)

    A = (A + A.T) * 0.5
    n = A.shape[0]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)

    scale = A.diag().mean().clamp_min(torch.finfo(A.dtype).tiny)

    for k in range(max_tries):
        jitter = (10.0 ** k) * float(base_jitter) * scale
        L, info = torch.linalg.cholesky_ex(A + jitter * eye)
        if int(info.item()) == 0:
            return L

    # last fallback: PSD projection
    w, V = torch.linalg.eigh(A)
    w = w.clamp_min(0.0)
    A_psd = (V * w) @ V.T
    jitter = (10.0 ** (max_tries - 1)) * float(base_jitter) * scale
    return torch.linalg.cholesky(A_psd + jitter * eye)
