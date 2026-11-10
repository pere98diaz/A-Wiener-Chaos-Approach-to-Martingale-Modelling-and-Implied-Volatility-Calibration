"""Shared indexing, polynomial features, and Brownian sampling routines.

Builds the multi-indices and Hermite features used in truncated Wiener-chaos
expansions, simulates Brownian paths, and aligns simulation grids. Basis
coordinates are flattened as k = r*d + j, where r indexes time-basis functions
and j indexes Brownian drivers.
"""

from __future__ import annotations

import itertools
import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import scipy.special
import torch


Tensor = torch.Tensor


# ---------------------------------------------------------------------
# Lambda / multi-index construction
# ---------------------------------------------------------------------
def create_Lambda(
    P: int,
    M: int,
    d: int,
    ignore_first: bool = False,
) -> Tuple[
    List[List[int]],
    List[List[List[int]]],
    npt.NDArray[np.float32],
]:
    """
    Build the multi-index set used to truncate the Wiener chaos expansion.

    We consider multi-indices a = (a_1, ..., a_{Md}) in N^{Md} with total degree |a| <= P.
    (If ignore_first=True, we skip k=0 and start from k=1.)

    Returns
    -------
    Lambda : list[list[int]]
        List of integer vectors a of length Md.

    Lambda_n_compressed : list[list[list[int]]]
        For each multi-index a, a list of pairs [a_j, j] for the nonzero entries,
        padded with [0,0] to length P.

    Lambda_fact_prod : np.ndarray (float32), shape (len(Lambda),)
        a! = prod_j (a_j!) for each multi-index.
        NOTE: this can overflow float32 for large P/M/d; if that becomes an issue,
        store log-factorials instead (or keep float64).
    """
    Md = M * d
    start_k = 1 if ignore_first else 0

    Lambda_n: List[List[int]] = []

    # Stars-and-bars enumeration of compositions of k into Md nonnegative integers.
    # For each k, number of terms is binom(Md + k - 1, k).
    for k in range(start_k, P + 1):
        for c in itertools.combinations(range(k + Md - 1), Md - 1):
            parts = [b - a - 1 for a, b in zip((-1,) + c, c + (k + Md - 1,))]
            Lambda_n.append(parts[::-1])  # keep your existing slot convention

    # Keep a plain integer copy for downstream masking etc.
    Lambda = [row[:] for row in Lambda_n]

    # Vectorized factorial-product computation (much faster than nested Python loops).
    Lambda_arr = np.asarray(Lambda, dtype=np.int64)  # (n_terms, Md)
    # factorial returns float64; cast to float32 at the end to preserve old behavior
    fact = scipy.special.factorial(Lambda_arr, exact=False)  # float64
    Lambda_fact_prod = np.prod(fact, axis=1).astype(np.float32)

    # Enrich each coordinate a_j with index j -> [a_j, j] (mutates Lambda_n).
    for k in range(len(Lambda_n)):
        for j in range(Md):
            Lambda_n[k][j] = [int(Lambda_n[k][j]), j]

    # Compress out zeros and pad to length P.
    Lambda_n_compressed: List[List[List[int]]] = []
    for row in Lambda_n:
        comp = [pair for pair in row if pair[0] != 0]
        if len(comp) < P:
            comp.extend([[0, 0]] * (P - len(comp)))
        Lambda_n_compressed.append(comp[:P])

    return Lambda, Lambda_n_compressed, Lambda_fact_prod


def create_Lambda_n01(Lambda_n_compressed: List[List[List[int]]]) -> List[npt.NDArray[np.int64]]:
    """
    Flatten and separate the components of the compressed multi-index set.

    Input has shape (n_terms, P, 2) conceptually. Output is two flat arrays of length n_terms*P:
      - degrees (a_j),
      - positions (j).
    """
    arr = np.asarray(Lambda_n_compressed, dtype=np.int64).reshape(-1, 2)
    return [arr[:, 0], arr[:, 1]]


def get_all_lambdas(
    P: int,
    M: int,
    d: int,
    device: torch.device,
    ignore_first: bool,
):
    """
    Convenience wrapper used throughout the repo.

    Returns:
      Lambda:        torch.LongTensor on device, shape (n_terms, Md)
      Lambda_n01:    [deg_idx, pos_idx] as CPU int64 numpy arrays (cheap & reusable as indices)
      Lambda_fact:   torch.FloatTensor on device, shape (n_terms,)
    """
    Lambda, Lambda_n_compressed, Lambda_fact = create_Lambda(P, M, d, ignore_first)
    Lambda_t = torch.as_tensor(np.asarray(Lambda, dtype=np.int64), device=device)
    Lambda_n01 = create_Lambda_n01(Lambda_n_compressed)
    Lambda_fact_t = torch.from_numpy(Lambda_fact).to(device=device)
    return Lambda_t, Lambda_n01, Lambda_fact_t


# ---------------------------------------------------------------------
# Chaos simulation (piecewise-constant class relies on this)
# ---------------------------------------------------------------------
def simulate_chaos(
    P: int,
    M: int,
    d: int,
    G: torch.Tensor,
    Lambda_n01: Sequence[Union[np.ndarray, torch.Tensor]],
    device: torch.device,
    type_float: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Evaluate the truncated chaos basis along samples of the integrals G.

    Parameters
    ----------
    G : torch.Tensor, shape (N, d, M)
        G[:, j, r] = ∫ g_r^j dB^j.

    Lambda_n01
        Two 1D arrays/tensors (deg_idx, pos_idx), each length n_terms*P, where
        pos_idx is in {0,...,M*d-1} with ordering k=r*d+j.

    Returns
    -------
    chaos_sim : torch.Tensor, shape (N, n_terms)
    """
    # Flatten G with the repo convention k = r*d + j
    # (N, d, M) -> (N, M, d) -> (N, M*d)
    x = G.to(device=device, dtype=type_float).permute(0, 2, 1).reshape(G.shape[0], M * d)

    # Build normalized Hermite table H_n(x) = He_n(x)/n! via recurrence:
    #   H_0 = 1, H_1 = x, H_{n+1} = (x H_n - H_{n-1})/(n+1)
    Npaths = x.shape[0]
    dtype = type_float
    herm = torch.empty((Npaths, P + 1, M * d), device=device, dtype=dtype)
    herm[:, 0].fill_(1.0)
    if P >= 1:
        herm[:, 1] = x
    for n in range(1, P):
        herm[:, n + 1] = (x * herm[:, n] - herm[:, n - 1]) / float(n + 1)

    deg_idx, pos_idx = Lambda_n01

    # Indices are cheap to keep on CPU; PyTorch allows CPU indices for CUDA tensors,
    # but converting once per call avoids numpy->tensor conversion overhead.
    if not torch.is_tensor(deg_idx):
        deg_idx = torch.as_tensor(deg_idx, dtype=torch.long, device=device)
    else:
        deg_idx = deg_idx.to(device=device, dtype=torch.long)

    if not torch.is_tensor(pos_idx):
        pos_idx = torch.as_tensor(pos_idx, dtype=torch.long, device=device)
    else:
        pos_idx = pos_idx.to(device=device, dtype=torch.long)

    # Gather H_{deg}(x_pos) for each (term, slot) and multiply across the P slots.
    pre = herm[:, deg_idx, pos_idx].reshape(Npaths, -1, P)
    return pre.prod(dim=2)


# ---------------------------------------------------------------------
# Brownian utilities
# ---------------------------------------------------------------------
def simulate_bm(
    n_samples: int,
    d: int,
    pi: Sequence[float],
    device: torch.device,
    seed: Optional[int] = None,
    *,
    type_float: torch.dtype = torch.float32,
    antithetic: bool = False,
) -> torch.Tensor:
    """
    Simulate a d-dimensional Brownian motion on the grid pi.

    Returns bm of shape (n_samples, d, len(pi)). With antithetic=True,
    each sampled increment vector is paired with its negative; an odd final
    path remains unpaired.
    """
    pi_t = torch.as_tensor(pi, device=device, dtype=type_float)
    dt = torch.diff(pi_t)  # (len(pi)-1,)
    sqrt_dt = torch.sqrt(dt)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    if antithetic:
        # Every path remains marginally standard normal. Pairing z with -z
        # changes covariance between paths without changing the path count.
        half = torch.randn(((n_samples + 1) // 2, d, len(pi) - 1),
                           device=device, dtype=type_float, generator=gen)
        z = torch.cat((half, -half[:n_samples // 2]), dim=0)
    else:
        z = torch.randn((n_samples, d, len(pi) - 1),
                        device=device, dtype=type_float, generator=gen)
    dB = z * sqrt_dt  # broadcast over last axis

    bm = torch.empty((n_samples, d, len(pi)), device=device, dtype=type_float)
    bm[..., 0] = 0.0
    bm[..., 1:] = torch.cumsum(dB, dim=-1)
    return bm


def get_Gs(bm: Tensor, pi: Sequence[float]) -> Tensor:
    """
    Given BM values bm[:, :, k]=B_{pi[k]}, return normalized increments:

        G[:, :, k] = (B_{pi[k+1]} - B_{pi[k]}) / sqrt(pi[k+1]-pi[k])

    Output shape: (N, d, len(pi)-1)
    """
    if bm.shape[-1] != len(pi):
        raise ValueError("bm.shape[-1] must match len(pi).")

    DeltaBm = torch.diff(bm, dim=-1)  # (N,d,K)
    pi_t = torch.as_tensor(pi, device=bm.device, dtype=bm.dtype)
    dt = pi_t[1:] - pi_t[:-1]  # (K,)

    return DeltaBm / torch.sqrt(dt).view(1, 1, -1)


def merge_and_track_positions(lists: Sequence[Sequence[float]]):
    """
    Merge multiple sorted lists into one sorted unique list, and return positions
    mapping each original list element to its index in the merged list.

    NOTE: previous implementation used list.index in a loop (O(n^2)).
    This version is O(n) after sorting.
    """
    merged_list = sorted(set(el for lst in lists for el in lst))
    idx = {x: i for i, x in enumerate(merged_list)}
    list_positions = [[idx[el] for el in lst] for lst in lists]
    return merged_list, list_positions


def sample_tuple(bounds):
    """Sample uniformly from box constraints: bounds=((lo1,hi1),...,(lok,hik))."""
    return tuple(random.uniform(lo, hi) for lo, hi in bounds)
