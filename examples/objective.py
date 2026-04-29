"""Objective for the constrained optimization problem.

Maximise:    L_final(F)
Subject to:  F = (F1, F2, F3, F4) ∈ [0, 100]^4         (box constraints)
             N1(0) = N2(0) = 0.05, Sn(0) = L(0) = 0    (fixed initial conditions)
             closed batch reactor for t_final hours    (no flow, no diffusion)

The model is the 8-state ODE in `src/CooperativeModel/kinetics.py` integrated
to t = t_final.  No spatial transport: a single grid cell with all transport
operators zero collapses the PDE to a pure ODE.

This module exposes:
  * `Objective.f(F)`           — scalar L_final(F)         (for maximisation)
  * `Objective.neg(F)`         — scalar -L_final(F)        (for scipy minimise)
  * `Objective.neg_with_grad(F)` — (-L, -∇L), gradient computed via batched
                                  9-point central FD in ONE Simulator call.
                                  Compatible with scipy `minimize(..., jac=True)`.
  * `Objective.evaluate_batch(F)` — vectorised over (B, 4) for landscape scans.
  * `Objective.bounds` — list of `(0, 100)` tuples, length 4.
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


class Objective:
    """Callable wrapper around the 72h closed-batch L_final model."""

    def __init__(self, t_final=72.0, N1=0.05, N2=0.05, device='cuda',
                 fd_h=2.0):
        self.t_final = t_final
        self.N1 = float(N1)
        self.N2 = float(N2)
        self.device = device
        self.fd_h = float(fd_h)
        self.bounds = [(0.0, 100.0)] * 4
        self.dim = 4
        self.n_evals = 0

    # ── Core simulator wrapper ──────────────────────────────────────────────

    def evaluate_batch(self, F_batch):
        """Evaluate L_final for a (B, 4) batch of initial sugar loadings.

        Returns: numpy array of shape (B,).
        """
        F_batch = np.atleast_2d(np.asarray(F_batch, dtype=float))
        B = len(F_batch)
        samples = np.zeros((B, 8))
        samples[:, 0] = self.N1
        samples[:, 1] = self.N2
        samples[:, 4:] = F_batch
        r = Simulator(
            samples=samples.tolist(),
            mode='batch', t_final=self.t_final, grid_size=1,
            omega=0.0, diffusion_scale=0.0, device=self.device,
        ).run()
        L = np.atleast_1d(r.L_final).astype(float)
        self.n_evals += B
        return L

    # ── scipy.optimize-compatible interfaces ────────────────────────────────

    def f(self, F):
        """Scalar L_final(F).  For maximisation."""
        return float(self.evaluate_batch(np.asarray(F).reshape(1, -1))[0])

    def neg(self, F):
        """Scalar -L_final(F).  For scipy.minimize."""
        return -self.f(F)

    def neg_with_grad(self, F):
        """Returns (-L, -∇L) computed via batched central FD in one call.

        scipy.optimize.minimize(jac=True) calls this once per outer step;
        the gradient costs the same as a single forward evaluation because
        all 9 perturbed points are batched together on the GPU.
        """
        F = np.asarray(F, dtype=float)
        h = self.fd_h
        # Adaptive step at the box: shrink h so we stay in [0, 100].
        h_eff = np.minimum(h, np.minimum(F, 100.0 - F))
        h_eff = np.where(h_eff > 1e-9, h_eff, h)

        pts = [F.copy()]
        for i in range(self.dim):
            sp = F.copy(); sp[i] = min(F[i] + h_eff[i], 100.0); pts.append(sp)
            sm = F.copy(); sm[i] = max(F[i] - h_eff[i], 0.0);   pts.append(sm)
        pts = np.array(pts)               # (9, 4)
        L_vals = self.evaluate_batch(pts)  # (9,)
        L0 = L_vals[0]
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            Lp = L_vals[1 + 2 * i]
            Lm = L_vals[2 + 2 * i]
            grad[i] = (Lp - Lm) / (2 * h_eff[i])
        return -float(L0), -grad
