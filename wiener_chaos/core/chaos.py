"""Evaluate stock models defined by conditional expectations of squared chaos.

Implements the piecewise-constant and general-basis representations, including
normalization, sampling at each maturity, and stock-price moments. Tensor,
derivative, and grouped evaluators compute the conditional quantities used by
ModelStock for option pricing and calibration.
"""

import math
from itertools import product
from math import factorial, prod

import torch

from .basis_integrals import get_scalar_products
from .utils import get_Gs, get_all_lambdas, merge_and_track_positions, simulate_bm, simulate_chaos


class ChaosMartingale:
    def __init__(self, basis, S0, chunk_size=None):
        if basis.P < 2:
            raise ValueError("The chaos model requires order P >= 2.")
        self.basis = basis
        self.S0 = float(S0)
        # Packed cubic contractions permit larger batches and fewer kernels.
        self.chunk_size = int(chunk_size if chunk_size is not None
                              else (16384 if basis.P == 3 else 4096))
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.n = basis.M * basis.d
        rows = basis.Lambda.cpu().tolist()
        self.factorials = torch.cat((
            torch.ones(1, device=basis.device, dtype=basis.type_float),
            basis.Lambda_fact.to(device=basis.device, dtype=basis.type_float),
        ))
        lookup = {tuple(row): k for k, row in enumerate(rows)}

        def coefficient_index(*coordinates):
            alpha = [0] * self.n
            for i in coordinates:
                alpha[i] += 1
            return 1 + lookup[tuple(alpha)]

        self.b_index = torch.tensor([coefficient_index(i) for i in range(self.n)], device=basis.device)
        self.A_index = torch.tensor(
            [coefficient_index(i, j) for i in range(self.n) for j in range(self.n)],
            device=basis.device,
        )
        self.C_index = None
        if basis.P == 3:
            self.C_index = torch.tensor(
                [coefficient_index(i, j, k) for i in range(self.n)
                 for j in range(self.n) for k in range(self.n)], device=basis.device,
            )
        self.past_variance = {}
        self._past_dimensions = {}
        self._symmetric_pairs = {}
        self.grouped = GroupedChaosEvaluator(basis, 4096 if chunk_size is None else chunk_size)
        self.derivative = DerivativeChaosEvaluator(basis, 4096 if chunk_size is None else chunk_size)
        self.backend = "tensor" if basis.P <= 3 else "derivative"

    def set_backend(self, backend):
        if backend not in ("tensor", "derivative", "grouped"):
            raise ValueError("Chaos backend must be 'tensor', 'derivative', or 'grouped'.")
        if self.basis.P > 3 and backend == "tensor":
            raise ValueError("Orders above three require the derivative or grouped backend.")
        self.backend = backend

    def set_step(self, T):
        """Past variance of each normalized Brownian increment at maturity T."""
        T = float(round(T, 6))
        u = self.basis.M_T[T]
        left = self.basis.pi_M_T[T][-2]
        rho = (T - left) / self.basis.deltas[u - 1]
        if not -1e-8 <= rho <= 1 + 1e-8:
            raise ValueError(f"Maturity {T} lies outside its basis interval.")
        rho = min(1.0, max(0.0, rho))
        v = torch.zeros(self.n, device=self.basis.device, dtype=self.basis.type_float)
        v[: (u - 1) * self.basis.d] = 1.0
        v[(u - 1) * self.basis.d : u * self.basis.d] = rho
        self.past_variance[T] = v
        # Grid geometry avoids synchronizing the device to inspect v during pricing.
        self._past_dimensions[T] = (u * self.basis.d,
                                    (u if rho == 1.0 else u - 1) * self.basis.d)

    def tensors(self, theta):
        if self.basis.P > 3:
            raise NotImplementedError("The tensor backend supports only P=2 or P=3.")
        if theta.numel() != len(self.factorials):
            raise ValueError(f"Expected {len(self.factorials)} coefficients, got {theta.numel()}.")
        c = theta[0]
        b = theta[self.b_index]
        A = (theta[self.A_index] / 2).reshape(self.n, self.n)
        if self.C_index is None:
            C = theta.new_zeros((self.n, self.n, self.n))
        else:
            C = (theta[self.C_index] / 6).reshape(self.n, self.n, self.n)
        return c, b, A, C

    def normalizer(self, theta):
        if theta.numel() != len(self.factorials):
            raise ValueError(f"Expected {len(self.factorials)} coefficients, got {theta.numel()}.")
        return (theta.square() / self.factorials.to(theta.dtype)).sum()

    def root_value(self, z, theta):
        if self.basis.P > 3:
            paths = z.shape[0]
            increments = z.reshape(paths, self.basis.M, self.basis.d).permute(0, 2, 1)
            features = simulate_chaos(
                self.basis.P, self.basis.M, self.basis.d, increments,
                self.basis.Lambda_n01, self.basis.device, z.dtype,
            )
            return theta[0] + features @ theta[1:]
        c, b, A, C = self.tensors(theta)
        linear = b - 3 * torch.einsum("ijj->i", C)
        return (c - torch.trace(A) + z @ linear
                + torch.einsum("bi,ij,bj->b", z, A, z)
                + torch.einsum("bi,ijk,bj,bk->b", z, C, z, z))

    def _cubic_square_coefficients(self, v, c, b, A, C, active, future_start):
        """Pack the mean, future gradient, and Hessian norm before visiting paths."""
        if active not in self._symmetric_pairs:
            self._symmetric_pairs[active] = torch.triu_indices(
                active, active, device=self.basis.device,
            )
        i, j = self._symmetric_pairs[active]
        multiplicity = (1 + (i != j)).to(A.dtype)
        bt = b - 3 * torch.einsum("ijj,j->i", C, v)
        g = 1 - v[future_start:]
        pair = g[:, None] * g[None, :]
        D = 2 * A[future_start:, future_start:]
        E = 6 * C[future_start:, future_start:, :active]
        h_const = 0.5 * (pair * D.square()).sum()
        h_linear = torch.einsum("ij,ij,ijk->k", pair, D, E)
        h_quadratic = 0.5 * torch.einsum("ij,ijk,ijl->kl", pair, E, E)
        future_C = C[future_start:, future_start:, future_start:]
        third = 6 * torch.einsum("ijk,ijk,i,j,k->", future_C, future_C, g, g, g)

        # C contracted against each distinct x_j*x_k is shared by the cubic
        # mean and its quadratic gradient. Off-diagonal pairs occur twice.
        quadratic = torch.cat((C[:, i, j].T, A[i, j, None],
                               h_quadratic[i, j, None]), dim=1)
        quadratic = quadratic * multiplicity[:, None]
        linear = torch.cat((bt[:active, None], 2*A[:active, future_start:],
                            h_linear[:, None]), dim=1)
        bias = torch.cat(((c - torch.diagonal(A).dot(v)).reshape(1),
                          bt[future_start:], (h_const + third).reshape(1)))
        return (i, j), linear, quadratic, bias, g

    def _conditional_square_chunk(self, x, active, future_start, pairs,
                                  linear, quadratic, bias, g):
        xa = x[:, :active]
        i, j = pairs
        pair_values = xa.index_select(1, i) * xa.index_select(1, j)
        contractions = pair_values @ quadratic
        affine = xa @ linear + bias
        # The last two packed columns hold the mean's quadratic term and the
        # squared-Hessian term; the preceding columns share the C contraction.
        mean = (affine[:, 0] + contractions[:, self.n]
                + (contractions[:, :active] * xa).sum(dim=1))
        gradient = affine[:, 1:-1] + 3*contractions[:, future_start:self.n]
        return (mean.square() + (gradient.square() * g).sum(dim=1)
                + affine[:, -1] + contractions[:, -1])

    def _quadratic_square_chunk(self, x, v, c, b, A, g, hessian_term, active):
        xa = x[:, :active]
        m = (c - torch.diagonal(A).dot(v) + xa @ b[:active]
             + torch.einsum("bi,ij,bj->b", xa, A[:active, :active], xa))
        dm = b + 2 * xa @ A[:active, :]
        return m.square() + (dm.square() * g).sum(dim=1) + hessian_term

    def conditional_square(self, x, v, theta, *, dimensions=None):
        """E[q(Z)^2 | X_t=x], for physical past coordinates x."""
        if self.basis.P > 3:
            raise NotImplementedError("Use the grouped evaluator above order three.")
        if x.shape[-1] != self.n:
            raise ValueError(f"Expected {self.n} Gaussian coordinates.")
        c, b, A, C = self.tensors(theta)
        if dimensions is None:
            active = int((v > 0).sum().item())
            future_start = int((v == 1).cumprod(dim=0).sum().item())
        else:
            active, future_start = dimensions
        if self.basis.P == 2:
            g = 1 - v
            hessian_term = 2 * torch.einsum("ij,ij,i,j->", A, A, g, g)
            return torch.cat([
                self._quadratic_square_chunk(block, v, c, b, A, g, hessian_term, active)
                for block in x.split(self.chunk_size)
            ])
        coefficients = self._cubic_square_coefficients(v, c, b, A, C, active, future_start)
        return torch.cat([
            self._conditional_square_chunk(block, active, future_start, *coefficients)
            for block in x.split(self.chunk_size)
        ])

    def spot(self, x, T, theta, forward_factor=1.0):
        T = float(round(T, 6))
        if self.backend == "grouped":
            numerator = self.grouped.conditional_square(x, T, theta)
        elif self.backend == "derivative":
            numerator = self.derivative.conditional_square(
                x, self.past_variance[T], theta, dimensions=self._past_dimensions[T],
            )
        else:
            numerator = self.conditional_square(
                x, self.past_variance[T], theta, dimensions=self._past_dimensions[T],
            )
        return self.S0 * forward_factor * numerator

    @torch.no_grad()
    def sample_past(self, N, Ns, T):
        """Independent maturity samples, in the same coordinates as quadrature."""
        T = float(round(T, 6))
        v = self.past_variance[T]
        active = self.basis.M_T[T] * self.basis.d
        chunks = []
        for _ in range(int(Ns)):
            x = torch.zeros((int(N), self.n), device=self.basis.device, dtype=self.basis.type_float)
            x[:, :active] = torch.randn(
                (int(N), active), device=self.basis.device, dtype=self.basis.type_float
            ) * v[:active].sqrt()
            chunks.append(x)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    @torch.no_grad()
    def sample_past_same_omega(self, N, times):
        """Observed coordinates at several times from a shared Brownian path."""
        times = [float(round(t, 6)) for t in times]
        grid, positions = merge_and_track_positions([self.basis.pi_M_T[t] for t in times])
        bm = simulate_bm(N, self.basis.d, grid, self.basis.device,
                         type_float=self.basis.type_float)
        result = []
        for T, pos in zip(times, positions):
            increments = get_Gs(bm[:, :, pos], self.basis.pi_M_T[T])
            active = self.basis.M_T[T] * self.basis.d
            x = torch.zeros((N, self.n), device=bm.device, dtype=bm.dtype)
            x[:, :active] = increments.permute(0, 2, 1).reshape(N, active) * self.past_variance[T][:active].sqrt()
            result.append(x)
        return result

    @staticmethod
    def _sym3_single(tensor):
        # The remaining pair is symmetric, leaving three distinct placements.
        return (tensor + tensor.permute(1, 0, 2) + tensor.permute(2, 1, 0)) / 3

    @staticmethod
    def _sym4_single(tensor):
        # The remaining triple is symmetric, leaving four placements.
        return (tensor + tensor.permute(1, 0, 2, 3)
                + tensor.permute(2, 1, 0, 3) + tensor.permute(3, 1, 2, 0)) / 4

    @staticmethod
    def _sym4_pairs(tensor):
        # Symmetric pairs and pair exchange leave three distinct pairings.
        return (tensor + tensor.permute(0, 2, 1, 3)
                + tensor.permute(0, 2, 3, 1)) / 3

    @staticmethod
    def _scale_axes(tensor, scale):
        out = tensor
        for axis in range(tensor.ndim):
            shape = [1] * tensor.ndim
            shape[axis] = len(scale)
            out = out * scale.reshape(shape)
        return out

    def shared_moment_kernels(self, theta):
        """Build parameter-dependent Wick kernels once for several maturities."""
        if self.basis.P > 3:
            raise NotImplementedError("The exact second control is available only for P=2 or P=3.")
        c, b, A, C = self.tensors(theta)
        nu = self.normalizer(theta)
        if self.basis.P == 2:
            h1 = 2*c*b + 4*A @ b
            h2 = 2*c*A + torch.outer(b, b) + 4*A @ A
            return {"b": b, "A": A, "nu": nu, "h": (h1, h2)}
        h1 = 2*c*b + 4*A @ b + 12*torch.einsum("ijk,jk->i", C, A)
        h2 = (2*c*A + torch.outer(b, b) + 4*A @ A
              + 6*torch.einsum("ijk,k->ij", C, b)
              + 18*torch.einsum("iab,jab->ij", C, C))
        h3 = (2*c*C + 2*self._sym3_single(torch.einsum("i,jk->ijk", b, A))
              + 12*self._sym3_single(torch.einsum("ia,jka->ijk", A, C)))
        h4 = (2*self._sym4_single(torch.einsum("i,jkl->ijkl", b, C))
              + self._sym4_pairs(torch.einsum("ij,kl->ijkl", A, A))
              + 9*self._sym4_pairs(torch.einsum("ija,kla->ijkl", C, C)))
        return {"A": A, "C": C, "nu": nu, "h": (h1, h2, h3, h4)}

    def exact_second_moment(self, v, theta, forward_factor=1.0, shared=None):
        """E[S_t^2] from Wick kernels, without rank-five/six arrays."""
        if shared is None:
            shared = self.shared_moment_kernels(theta)
        A, nu = shared["A"], shared["nu"]
        scale = v.sqrt()
        if self.basis.P == 2:
            b = shared["b"]
            h1, h2 = shared["h"]
            bv = b * scale
            Av = self._scale_axes(A, scale)
            bv2 = bv.square().sum()
            Av2 = Av.square().sum()
            variance = (self._scale_axes(h1, scale).square().sum()
                        + 2*self._scale_axes(h2, scale).square().sum()
                        + 8*bv2*Av2 + 16*(Av @ bv).square().sum()
                        + 8*Av2.square() + 16*(Av @ Av).square().sum())
            forward = self.S0 * forward_factor
            return forward**2 * (nu.square() + variance)
        C = shared["C"]
        h1, h2, h3, h4 = shared["h"]
        variance = sum(math.factorial(order) * self._scale_axes(h, scale).square().sum()
                       for order, h in enumerate((h1, h2, h3, h4), start=1))
        Av = self._scale_axes(A, scale)
        Cv = self._scale_axes(C, scale)
        AC1 = torch.einsum("ia,jka->ijk", Av, Cv)
        AC2 = torch.einsum("ab,iab->i", Av, Cv)
        CC2 = torch.einsum("iab,jab->ij", Cv, Cv)
        variance = (variance + 48*Av.square().sum()*Cv.square().sum()
                    + 288*AC1.square().sum() + 144*AC2.square().sum()
                    + 72*Cv.square().sum().square() + 648*CC2.square().sum())
        forward = self.S0 * forward_factor
        return forward**2 * (nu.square() + variance)

    def moments(self, degree, theta, T, forward_factor=1.0, shared=None):
        T = float(round(T, 6))
        nu = shared["nu"] if shared is not None else self.normalizer(theta)
        first = self.S0 * forward_factor * nu
        if degree == 1:
            return first[None]
        if degree == 2:
            second = self.exact_second_moment(self.past_variance[T], theta, forward_factor, shared)
            return torch.stack((first, second))
        raise ValueError("Only the first two moments are implemented.")


class ChaosGeneral:
    """Exact conditional square for quadratic or cubic terminal factors."""

    def __init__(self, basis, S0, chunk_size=8192):
        if basis.P not in (2, 3):
            raise ValueError("The general-basis chaos evaluator supports P=2 or P=3.")
        self.basis = basis
        self.S0 = float(S0)
        self.n = basis.M * basis.d
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")

        rows = basis.Lambda.tolist()
        lookup = {tuple(row): k + 1 for k, row in enumerate(rows)}

        def coefficient_index(*coordinates):
            alpha = [0] * self.n
            for coordinate in coordinates:
                alpha[coordinate] += 1
            return lookup[tuple(alpha)]

        device = basis.device
        self.b_index = torch.tensor(
            [coefficient_index(i) for i in range(self.n)], device=device
        )
        self.A_index = torch.tensor(
            [coefficient_index(i, j) for i in range(self.n) for j in range(self.n)],
            device=device,
        )
        self.C_index = None
        if basis.P == 3:
            self.C_index = torch.tensor(
                [coefficient_index(i, j, k)
                 for i in range(self.n) for j in range(self.n) for k in range(self.n)],
                device=device,
            )
        self.factorials = torch.cat((
            torch.ones(1, device=device, dtype=basis.type_float),
            torch.as_tensor(basis.Lambda_fact, device=device, dtype=basis.type_float),
        ))
        self.past_covariance = {}
        self.future_covariance = {}
        self.past_sqrt = {}
        self.past_full_sqrt = {}
        self.past_base_covariance = {}
        self.compact_past_cutoff = None
        self.compact_rotation = {}
        self.compact_columns = {}
        self.path_increment_sqrt = {}

    def enable_compact_past(self, relative_eigenvalue_cutoff=1e-10):
        """Use the numerically active past directions for sampled P=2 paths.

        Direct callers may still pass full observed coordinates to spot().
        The Gaussian draws keep their original (N, M, d) shape, so enabling
        this path preserves the random stream used by the full evaluator.
        """
        if self.basis.P != 2:
            raise ValueError("Compact past sampling is available only for P=2.")
        if not 0 < relative_eigenvalue_cutoff < 1:
            raise ValueError("relative_eigenvalue_cutoff must lie between 0 and 1.")
        self.compact_past_cutoff = float(relative_eigenvalue_cutoff)
        for t in self.past_covariance:
            self._prepare_compact_step(t)

    def _prepare_compact_step(self, t):
        if self.compact_past_cutoff is None or t in self.compact_rotation:
            return
        if t == 0 or t == self.basis.T:
            return
        # Build the tiny eigendecomposition in double precision. Float32
        # covariance roundoff would manufacture false small eigenvalues.
        covariance = get_scalar_products(
            0, t, self.basis.M, self.basis.T, self.basis.type_basis,
            torch.device("cpu"), type_float=torch.float64,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh((covariance + covariance.T) * 0.5)
        eigenvalues = eigenvalues.clamp_min(0)
        keep = eigenvalues > self.compact_past_cutoff * eigenvalues.max()
        if keep.sum().item() == self.basis.M:
            return
        rotation = eigenvectors[:, keep].to(
            device=self.basis.device, dtype=self.basis.type_float,
        )
        scales = eigenvalues[keep].sqrt().to(
            device=self.basis.device, dtype=self.basis.type_float,
        )
        eye_driver = torch.eye(
            self.basis.d, device=self.basis.device, dtype=self.basis.type_float,
        )
        self.compact_rotation[t] = rotation
        self.compact_columns[t] = torch.kron((rotation * scales).contiguous(), eye_driver)

    @staticmethod
    def _psd_sqrt(matrix):
        """A symmetric square root, also valid for a singular covariance."""
        eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
        return (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T

    def _basis_covariances(self, t):
        M, T = self.basis.M, self.basis.T
        eye = torch.eye(M, device=self.basis.device, dtype=self.basis.type_float)
        if t == 0:
            return torch.zeros_like(eye), eye
        if t == T:
            return eye, torch.zeros_like(eye)

        # Integrate the shorter side directly to avoid subtracting two nearly
        # equal matrices at very short maturities or near the horizon.
        if t <= T / 2:
            past = get_scalar_products(
                0, t, M, T, self.basis.type_basis, self.basis.device,
                type_float=self.basis.type_float,
            )
            values, vectors = torch.linalg.eigh((past + past.T) * 0.5)
            past = (vectors * values.clamp(0, 1)) @ vectors.T
            return past, eye - past
        future = self.basis.dyson._scalar_products_tT(t)
        values, vectors = torch.linalg.eigh((future + future.T) * 0.5)
        future = (vectors * values.clamp(0, 1)) @ vectors.T
        return eye - future, future

    def set_step(self, t):
        t = float(round(t, 6))
        if t in self.past_covariance:
            return
        if not 0 <= t <= self.basis.T:
            raise ValueError("Maturity must lie in the basis horizon.")
        past_base, future_base = self._basis_covariances(t)
        eye_driver = torch.eye(
            self.basis.d, device=self.basis.device, dtype=self.basis.type_float
        )
        self.past_base_covariance[t] = past_base
        self.past_covariance[t] = torch.kron(past_base.contiguous(), eye_driver)
        self.future_covariance[t] = torch.kron(future_base.contiguous(), eye_driver)
        self.past_sqrt[t] = self._psd_sqrt(past_base)
        self.past_full_sqrt[t] = torch.kron(self.past_sqrt[t].contiguous(), eye_driver)
        self._prepare_compact_step(t)

    def tensors(self, theta):
        if theta.numel() != len(self.factorials):
            raise ValueError(f"Expected {len(self.factorials)} coefficients, got {theta.numel()}.")
        c = theta[0]
        b = theta[self.b_index]
        A = (theta[self.A_index] / 2).reshape(self.n, self.n)
        C = None if self.C_index is None else (
            theta[self.C_index] / 6
        ).reshape(self.n, self.n, self.n)
        return c, b, A, C

    def normalizer(self, theta):
        if theta.numel() != len(self.factorials):
            raise ValueError(f"Expected {len(self.factorials)} coefficients, got {theta.numel()}.")
        return (theta.square() / self.factorials.to(theta.dtype)).sum()

    @staticmethod
    def _triple_covariance_contraction(C, G):
        """Contract all three axes without forming a six-index tensor."""
        contracted = torch.einsum("ia,ajk->ijk", G, C)
        contracted = torch.einsum("jb,ibk->ijk", G, contracted)
        contracted = torch.einsum("kc,ijc->ijk", G, contracted)
        return (C * contracted).sum()

    def _prepared_terms(self, theta, t, compact=False):
        c, b, A, C = self.tensors(theta)
        V = self.past_covariance[t]
        G = self.future_covariance[t]
        mean_constant = c - (A * V).sum()
        if C is None:
            # For P=2, E[F² | x] = m(x)² + q0 + q1·x + xᵀQx,
            # m(x) = c - <A,V> + b·x + xᵀAx. Contract the dense future
            # covariance once per coefficient vector, not once per path.
            AG = A @ G
            q0 = b @ (G @ b) + 2 * (AG * AG.T).sum()
            q1 = 4 * (AG @ b)
            Q = 4 * (AG @ A)
            if compact:
                columns = self.compact_columns[t]
                b = columns.T @ b
                A = columns.T @ A @ columns
                q1 = columns.T @ q1
                Q = columns.T @ Q @ columns
            return mean_constant, b, A, q0, q1, Q

        mean_linear = b - 3 * torch.einsum("ijk,jk->i", C, V)

        # Hessian contribution: 1/2 <H, G H G>. For P=3 the Hessian is
        # 2A + 6 C(., ., x), so this term is quadratic in the observed x.
        D = 2 * A
        GDG = G @ D @ G
        h_constant = 0.5 * (D * GDG).sum()
        E = 6 * C
        h_linear = torch.einsum("ij,ijk->k", GDG, E)
        GEG = torch.einsum("ia,abk,bj->ijk", G, E, G)
        h_quadratic = 0.5 * torch.einsum("ijk,ijl->kl", E, GEG)
        third = 6 * self._triple_covariance_contraction(C, G)
        h_constant = h_constant + third
        return (mean_constant, mean_linear, A, C, G,
                h_constant, h_linear, h_quadratic)

    @staticmethod
    def _quadratic_square_chunk(x, terms):
        mean_constant, b, A, q0, q1, Q = terms
        Ax = x @ A
        mean = mean_constant + x @ b + (x * Ax).sum(dim=1)
        return mean.square() + q0 + x @ q1 + (x * (x @ Q)).sum(dim=1)

    def _conditional_square_chunk(self, x, terms):
        mean_constant, mean_linear, A, C, G, h_constant, h_linear, h_quadratic = terms
        Ax = x @ A
        mean = mean_constant + x @ mean_linear + (x * Ax).sum(dim=1)
        gradient = mean_linear + 2 * Ax
        if C is not None:
            pair = (x[:, :, None] * x[:, None, :]).reshape(x.shape[0], -1)
            Cxx = pair @ C.reshape(self.n, -1).T
            mean = mean + (x * Cxx).sum(dim=1)
            gradient = gradient + 3 * Cxx
            h_term = (h_constant + x @ h_linear
                      + ((x @ h_quadratic) * x).sum(dim=1))
        else:
            h_term = h_constant
        gradient_term = ((gradient @ G) * gradient).sum(dim=1)
        return mean.square() + gradient_term + h_term

    def conditional_square(self, x, t, theta):
        t = float(round(t, 6))
        if x.ndim != 2:
            raise ValueError("Expected a matrix of observed basis integrals.")
        compact = x.shape[1] != self.n
        if compact:
            columns = self.compact_columns.get(t)
            if columns is None or x.shape[1] != columns.shape[1]:
                raise ValueError(f"Expected an (N, {self.n}) matrix of observed basis integrals.")
        terms = self._prepared_terms(theta, t, compact=compact)
        evaluate_chunk = (
            self._quadratic_square_chunk if self.basis.P == 2
            else self._conditional_square_chunk
        )
        return torch.cat([
            evaluate_chunk(block, terms)
            for block in x.split(self.chunk_size)
        ])

    def spot(self, x, t, theta, forward_factor=1.0):
        t = float(round(t, 6))
        numerator = self.conditional_square(x, t, theta)
        return self.S0 * forward_factor * numerator

    @torch.no_grad()
    def sample_past(self, N, Ns, t):
        t = float(round(t, 6))
        root = self.past_sqrt[t]
        rotation = self.compact_rotation.get(t)
        samples = []
        for _ in range(int(Ns)):
            eps = torch.randn(
                (int(N), self.basis.M, self.basis.d),
                device=self.basis.device, dtype=self.basis.type_float,
            )
            if rotation is None:
                samples.append(torch.einsum("ij,bjd->bid", root, eps).reshape(int(N), self.n))
            else:
                samples.append(torch.einsum("mk,bmd->bkd", rotation, eps).reshape(int(N), -1))
        return torch.cat(samples)

    @torch.no_grad()
    def sample_past_same_omega(self, N, times):
        """Share each Brownian basis integral across all requested times."""
        times = [float(round(t, 6)) for t in times]
        if any(left >= right for left, right in zip(times, times[1:])):
            raise ValueError("Path times must be strictly increasing.")
        if not times:
            return []
        previous_time = 0.0
        current = torch.zeros(
            (int(N), self.basis.M, self.basis.d),
            device=self.basis.device, dtype=self.basis.type_float,
        )
        paths = []
        for t in times:
            self.set_step(t)
            key = (previous_time, t)
            if key not in self.path_increment_sqrt:
                # Subtracting two float32 cumulative covariances and clipping
                # negative eigenvalues adds spurious variance on fine grids.
                # Integrate the increment directly and factor it in float64.
                covariance = get_scalar_products(
                    previous_time, t, self.basis.M, self.basis.T,
                    self.basis.type_basis, torch.device("cpu"),
                    type_float=torch.float64,
                )
                self.path_increment_sqrt[key] = self._psd_sqrt(covariance).to(
                    device=self.basis.device, dtype=self.basis.type_float,
                )
            increment_root = self.path_increment_sqrt[key]
            eps = torch.randn_like(current)
            current = current + torch.einsum("ij,bjd->bid", increment_root, eps)
            paths.append(current.reshape(int(N), self.n).clone())
            previous_time = t
        return paths

    @staticmethod
    def _sym3_single(tensor):
        return (tensor + tensor.permute(1, 0, 2) + tensor.permute(2, 1, 0)) / 3

    @staticmethod
    def _sym4_single(tensor):
        return (tensor + tensor.permute(1, 0, 2, 3)
                + tensor.permute(2, 1, 0, 3) + tensor.permute(3, 1, 2, 0)) / 4

    @staticmethod
    def _sym4_pairs(tensor):
        return (tensor + tensor.permute(0, 2, 1, 3)
                + tensor.permute(0, 2, 3, 1)) / 3

    @staticmethod
    def _transform_axes(tensor, root):
        """Apply the observed-covariance square root along every tensor axis."""
        transformed = tensor
        for axis in range(tensor.ndim):
            transformed = torch.tensordot(root, transformed, dims=([1], [axis]))
            transformed = transformed.movedim(0, axis)
        return transformed

    def shared_moment_kernels(self, theta):
        """Coefficient-dependent Wick contractions, shared across maturities."""
        c, b, A, C = self.tensors(theta)
        nu = self.normalizer(theta)
        if C is None:
            h1 = 2*c*b + 4*A @ b
            h2 = 2*c*A + torch.outer(b, b) + 4*A @ A
            return {"b": b, "A": A, "nu": nu, "h": (h1, h2)}

        h1 = 2*c*b + 4*A @ b + 12*torch.einsum("ijk,jk->i", C, A)
        h2 = (2*c*A + torch.outer(b, b) + 4*A @ A
              + 6*torch.einsum("ijk,k->ij", C, b)
              + 18*torch.einsum("iab,jab->ij", C, C))
        h3 = (2*c*C + 2*self._sym3_single(torch.einsum("i,jk->ijk", b, A))
              + 12*self._sym3_single(torch.einsum("ia,jka->ijk", A, C)))
        h4 = (2*self._sym4_single(torch.einsum("i,jkl->ijkl", b, C))
              + self._sym4_pairs(torch.einsum("ij,kl->ijkl", A, A))
              + 9*self._sym4_pairs(torch.einsum("ija,kla->ijkl", C, C)))
        return {"A": A, "C": C, "nu": nu, "h": (h1, h2, h3, h4)}

    def exact_second_moment(self, t, theta, forward_factor=1.0, shared=None):
        """E[S_t²] with dense past covariance; never form rank-five/six kernels."""
        t = float(round(t, 6))
        if shared is None:
            shared = self.shared_moment_kernels(theta)
        root = self.past_full_sqrt[t]
        A, nu = shared["A"], shared["nu"]
        Av = root @ A @ root
        Av2 = Av.square().sum()
        if self.basis.P == 2:
            b = shared["b"]
            h1, h2 = shared["h"]
            bv = root @ b
            h1v = root @ h1
            h2v = root @ h2 @ root
            variance = (h1v.square().sum() + 2*h2v.square().sum()
                        + 8*bv.square().sum()*Av2
                        + 16*(Av @ bv).square().sum()
                        + 8*Av2.square() + 16*(Av @ Av).square().sum())
        else:
            C = shared["C"]
            h1, h2, h3, h4 = shared["h"]
            variance = sum(
                math.factorial(order) * self._transform_axes(h, root).square().sum()
                for order, h in enumerate((h1, h2, h3, h4), start=1)
            )
            Cv = self._transform_axes(C, root)
            AC1 = torch.einsum("ia,jka->ijk", Av, Cv)
            AC2 = torch.einsum("ab,iab->i", Av, Cv)
            CC2 = torch.einsum("iab,jab->ij", Cv, Cv)
            variance = (variance + 48*Av2*Cv.square().sum()
                        + 288*AC1.square().sum() + 144*AC2.square().sum()
                        + 72*Cv.square().sum().square() + 648*CC2.square().sum())
        forward = self.S0 * forward_factor
        return forward**2 * (nu.square() + variance)

    def moments(self, degree, theta, t, forward_factor=1.0, shared=None):
        nu = shared["nu"] if shared is not None else self.normalizer(theta)
        first = self.S0 * forward_factor * nu
        if degree == 1:
            return first[None]
        if degree == 2:
            second = self.exact_second_moment(t, theta, forward_factor, shared)
            return torch.stack((first, second))
        raise ValueError("Only the first two moments are implemented.")


class DerivativeChaosEvaluator:
    def __init__(self, basis, chunk_size=4096):
        self.basis = basis
        self.n = basis.M * basis.d
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.rows = [(0,) * self.n] + [tuple(row) for row in basis.Lambda.cpu().tolist()]
        self.plans = {}

    def _plan(self, active, future_start):
        """Index the nonzero terms in every mixed derivative of m_t."""
        key = (active, future_start)
        if key in self.plans:
            return self.plans[key]

        observed_lookup = {}
        derivative_lookup = {}
        coefficient_ids, observed_ids, derivative_ids = [], [], []

        for coefficient_id, alpha in enumerate(self.rows):
            # Completed intervals cannot be differentiated (g=0); later
            # intervals have no observed part (v=0). Only the current
            # interval admits a split between observed and future degrees.
            current = alpha[future_start:active]
            tail = alpha[active:]
            for residual in product(*(range(degree + 1) for degree in current)):
                observed = alpha[:future_start] + tuple(
                    degree - k for degree, k in zip(current, residual)
                )
                beta = (0,) * future_start + residual + tail
                observed_id = observed_lookup.setdefault(observed, len(observed_lookup))
                derivative_id = derivative_lookup.setdefault(beta, len(derivative_lookup))
                coefficient_ids.append(coefficient_id)
                observed_ids.append(observed_id)
                derivative_ids.append(derivative_id)

        observed_indices = list(observed_lookup)
        device = self.basis.device
        betas = list(derivative_lookup)
        plan = {
            "observed_multiindices": torch.tensor(
                observed_indices, device=device, dtype=torch.long
            ).reshape(len(observed_indices), active),
            "observed_coordinates": tuple(
                coordinate for coordinate in range(active)
                if any(gamma[coordinate] for gamma in observed_indices)
            ),
            "derivative_multiindices": torch.tensor(
                betas, device=device, dtype=torch.long
            ).reshape(len(betas), self.n),
            "coefficient_ids": torch.tensor(coefficient_ids, device=device, dtype=torch.long),
            "observed_ids": torch.tensor(observed_ids, device=device, dtype=torch.long),
            "derivative_ids": torch.tensor(derivative_ids, device=device, dtype=torch.long),
            "future_degrees": torch.tensor(
                [sum(beta[future_start:active]) for beta in betas],
                device=device, dtype=torch.long,
            ),
            "inverse_factorials": torch.tensor(
                [1.0 / prod(factorial(degree) for degree in beta) for beta in betas],
                device=device, dtype=self.basis.type_float,
            ),
        }
        self.plans[key] = plan
        return plan

    def _observed_features(self, x, v, plan):
        """Evaluate H_gamma(y; v) for the distinct lowered indices gamma."""
        degrees = plan["observed_multiindices"]
        active = degrees.shape[1]
        paths = x.shape[0]
        features = x.new_ones((paths, degrees.shape[0]))
        if active == 0:
            return features

        observed = x[:, :active]
        hermites = [torch.ones_like(observed), observed]
        for degree in range(1, self.basis.P):
            hermites.append(
                (observed * hermites[-1] - v[:active] * hermites[-2]) / (degree + 1)
            )
        table = torch.stack(hermites, dim=1)
        for coordinate in plan["observed_coordinates"]:
            features = features * table[:, degrees[:, coordinate], coordinate]
        return features

    def _mixed_derivatives_chunk(self, x, v, coefficients, plan):
        """Columns are the actual values of d^beta m_t at observed paths."""
        features = self._observed_features(x, v, plan)
        values = features.index_select(1, plan["observed_ids"]) * coefficients
        return x.new_zeros((x.shape[0], plan["derivative_multiindices"].shape[0])).index_add(
            1, plan["derivative_ids"], values
        )

    def _prepared(self, v, theta, dimensions):
        if theta.numel() != len(self.rows):
            raise ValueError(f"Expected {len(self.rows)} coefficients, got {theta.numel()}.")
        active, future_start = dimensions
        if not 0 <= future_start <= active <= self.n:
            raise ValueError("Invalid past/future coordinate dimensions.")
        plan = self._plan(active, future_start)
        residual_variance = 1 - v[future_start] if future_start < active else v.new_tensor(1.)
        weights = plan["inverse_factorials"].to(theta.dtype) * residual_variance.pow(
            plan["future_degrees"]
        )
        coefficients = theta.index_select(0, plan["coefficient_ids"])
        return plan, coefficients, weights

    def mixed_derivatives(self, x, v, theta, *, dimensions):
        """Return (d^beta m_t(x), beta) for inspection or independent checks.

        Materializing all paths and derivative orders can be large; pricing
        instead processes them in bounded chunks below.
        """
        if x.shape[-1] != self.n:
            raise ValueError(f"Expected {self.n} Gaussian coordinates.")
        plan, coefficients, _ = self._prepared(v, theta, dimensions)
        result = self._mixed_derivatives_chunk(x, v, coefficients, plan)
        return result, plan["derivative_multiindices"]

    def conditional_square(self, x, v, theta, *, dimensions):
        """Evaluate the conditional derivative/Parseval sum for any finite P."""
        if x.shape[-1] != self.n:
            raise ValueError(f"Expected {self.n} Gaussian coordinates.")
        if x.shape[0] == 0:
            return x.new_empty((0,))
        plan, coefficients, weights = self._prepared(v, theta, dimensions)
        columns = max(
            len(plan["coefficient_ids"]),
            len(plan["observed_multiindices"]),
            len(plan["derivative_multiindices"]),
        )
        chunk_size = min(self.chunk_size, max(1, 2_000_000 // columns))
        values = []
        for block in x.split(chunk_size):
            derivatives = self._mixed_derivatives_chunk(block, v, coefficients, plan)
            values.append((derivatives.square() * weights).sum(dim=1))
        return torch.cat(values)


class GroupedChaosEvaluator:
    def __init__(self, basis, chunk_size=4096):
        self.basis = basis
        self.chunk_size = int(chunk_size)
        self.rows = [[0] * (basis.M * basis.d)] + basis.Lambda.cpu().tolist()
        self.plans = {}

    def _plan(self, interval):
        if interval in self.plans:
            return self.plans[interval]

        d, P = self.basis.d, self.basis.P
        prefix_len = (interval - 1) * d
        current_end = interval * d
        if prefix_len:
            past_rows, past_n01, _ = get_all_lambdas(
                P, interval - 1, d, self.basis.device, ignore_first=False
            )
            past_lookup = {tuple(row): i for i, row in enumerate(past_rows.cpu().tolist())}
            past_n01 = [torch.as_tensor(a, device=self.basis.device) for a in past_n01]
        else:
            past_lookup = {(): 0}
            past_n01 = None

        current_lookup = {}
        group_lookup = {}
        group_weights = []
        coefficient_ids, past_ids, current_ids, residual_degrees, group_ids = [], [], [], [], []
        for coeff_id, alpha in enumerate(self.rows):
            past = tuple(alpha[:prefix_len])
            current = alpha[prefix_len:current_end]
            tail = tuple(alpha[current_end:])
            for k in product(*(range(m + 1) for m in current)):
                observed = tuple(m - residual for m, residual in zip(current, k))
                current_id = current_lookup.setdefault(observed, len(current_lookup))
                group_key = (k, tail)
                if group_key not in group_lookup:
                    group_lookup[group_key] = len(group_lookup)
                    denom = prod(factorial(m) for m in (*k, *tail))
                    group_weights.append(1.0 / denom)
                coefficient_ids.append(coeff_id)
                past_ids.append(past_lookup[past])
                current_ids.append(current_id)
                residual_degrees.append(sum(k))
                group_ids.append(group_lookup[group_key])

        device = self.basis.device
        plan = {
            "prefix_len": prefix_len,
            "past_n01": past_n01,
            "current_degrees": list(current_lookup),
            "coefficient_ids": torch.tensor(coefficient_ids, device=device),
            "past_ids": torch.tensor(past_ids, device=device),
            "current_ids": torch.tensor(current_ids, device=device),
            "residual_degrees": torch.tensor(residual_degrees, device=device),
            "group_ids": torch.tensor(group_ids, device=device),
            "group_weights": torch.tensor(
                group_weights, device=device, dtype=self.basis.type_float
            ),
        }
        self.plans[interval] = plan
        return plan

    def _chunk(self, x, theta, rho, plan):
        paths = x.shape[0]
        prefix_len = plan["prefix_len"]
        if prefix_len:
            past = x[:, :prefix_len].reshape(paths, -1, self.basis.d).permute(0, 2, 1)
            prefix = simulate_chaos(
                self.basis.P, prefix_len // self.basis.d, self.basis.d,
                past, plan["past_n01"], self.basis.device, x.dtype,
            )
        else:
            prefix = x.new_ones((paths, 1))

        current_x = x[:, prefix_len:prefix_len + self.basis.d]
        hermites = [current_x.new_ones(current_x.shape), current_x]
        for degree in range(1, self.basis.P):
            hermites.append((current_x * hermites[-1] - rho * hermites[-2]) / (degree + 1))
        current = torch.stack([
            torch.prod(torch.stack([hermites[degree][:, driver]
                                    for driver, degree in enumerate(alpha)]), dim=0)
            for alpha in plan["current_degrees"]
        ], dim=1)

        q = 1 - rho
        residual_scale = torch.as_tensor(q, device=x.device, dtype=x.dtype).pow(
            plan["residual_degrees"].to(x.dtype) / 2
        )
        values = (theta[plan["coefficient_ids"]] * residual_scale
                  * prefix[:, plan["past_ids"]] * current[:, plan["current_ids"]])
        amplitudes = x.new_zeros((paths, len(plan["group_weights"]))).index_add(
            1, plan["group_ids"], values
        )
        return (amplitudes.square() * plan["group_weights"].to(x.dtype)).sum(dim=1)

    def conditional_square(self, x, T, theta):
        T = float(round(T, 6))
        if T == 0:
            nu = theta[0].square() + (
                theta[1:].square() / self.basis.Lambda_fact.to(theta.dtype)
            ).sum()
            return nu.expand(x.shape[0])
        interval = self.basis.M_T[T]
        left = self.basis.pi_M_T[T][-2]
        rho = (T - left) / self.basis.deltas[interval - 1]
        if not -1e-8 <= rho <= 1 + 1e-8:
            raise ValueError(f"Maturity {T} lies outside its basis interval.")
        rho = min(1.0, max(0.0, rho))
        plan = self._plan(interval)
        columns = max(len(plan["coefficient_ids"]), len(plan["group_weights"]))
        chunk_size = min(self.chunk_size, max(32, 2_000_000 // columns))
        return torch.cat([
            self._chunk(block, theta, rho, plan) for block in x.split(chunk_size)
        ])
