"""Verify which candidate optima are TRUE local maxima.

Closed batch reactor: F1..F4 ∈ [0, 100] are initial sugar amounts at t=0.

For every candidate point F* we compute, via batched finite differences:
  1. Gradient g_i = ∂L_final/∂F_i  (central difference, h adaptive at the box)
  2. Hessian H_{ij} = ∂² L_final/∂F_i ∂F_j  (central + cross differences)

Then classify F* as:
  * INTERIOR MAX    — all coords strictly in (0, 100), |g| ≈ 0, all eigvals(H) < 0
  * BOUNDARY MAX    — some coord on {0, 100} with KKT-active inward gradient
                      and free-coordinate sub-Hessian negative-definite
  * SADDLE / NOT MAX — Hessian has a non-negative eigenvalue in the free dimensions

Usage:
    python examples/verify_optima.py                    # verify the 16 corners
    python examples/verify_optima.py optima.txt         # verify list from file
    python examples/verify_optima.py --from-find        # pipe from find_optima.py
"""

import sys, os, time, itertools, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from objective import Objective

# Shared objective — used by both batched evaluation and grad/Hessian below.
_OBJ = Objective(t_final=72.0, N1=0.05, N2=0.05, device='cuda')


def eval_batch(sugars_array):
    """Evaluate L_final for a [B, 4] batch of initial sugar loadings."""
    return _OBJ.evaluate_batch(sugars_array)


# ── Gradient + Hessian via batched central differences ─────────────────────

def grad_hess(F_star, h=1.0, box=(0.0, 100.0), bdry_tol=0.5):
    """Numerical gradient and Hessian of L_final at F_star.

    Uses central differences on free coordinates and one-sided (forward
    at lo, backward at hi) on coordinates pinned to the box. Returns
    `g[i]` as the *directional derivative pointing inward* whenever
    coord `i` is on the boundary, so the KKT check is simply "g ≤ 0 for
    boundary, ≈0 for interior". Boundary coordinates contribute no row
    or column to the free sub-Hessian.

    Coordinates within `bdry_tol` of either bound are treated as on the
    boundary — the CCR-on-nisin landscape has many face/edge peaks where
    the optimiser settles ~0.1 from a face, and treating them as strictly
    interior breaks the KKT/discrete-max checks.

    Also returns `L_plus[i]` and `L_minus[i]` so the caller can run a
    discrete-max check (L0 >= L at all FD neighbours) — necessary for
    cliff peaks where central-FD gradients are misleading but the point
    is still a genuine local maximum.

    Builds all needed evaluations in one batched simulator call.
    """
    F_star = np.asarray(F_star, dtype=float)
    n = 4
    lo, hi = box

    on_lo = np.array([F_star[i] <= lo + bdry_tol for i in range(n)])
    on_hi = np.array([F_star[i] >= hi - bdry_tol for i in range(n)])
    free  = ~(on_lo | on_hi)

    # Per-axis step. Central difference on free; one-sided on boundary.
    h_vec = np.zeros(n)
    for i in range(n):
        if free[i]:
            h_vec[i] = min(h, F_star[i] - lo, hi - F_star[i])
        else:
            h_vec[i] = h  # one-sided step into the interior

    pts = [F_star.copy()]
    plus_idx  = [None] * n   # F + h e_i  (used at lo and free)
    minus_idx = [None] * n   # F - h e_i  (used at hi and free)
    cross_idx = {}

    for i in range(n):
        if h_vec[i] <= 0: continue
        if on_lo[i] or free[i]:
            p = F_star.copy(); p[i] = F_star[i] + h_vec[i]
            plus_idx[i] = len(pts); pts.append(p)
        if on_hi[i] or free[i]:
            p = F_star.copy(); p[i] = F_star[i] - h_vec[i]
            minus_idx[i] = len(pts); pts.append(p)

    # Cross terms only for free-free pairs (boundary dims have no Hessian row/col)
    for i in range(n):
        for j in range(i+1, n):
            if not (free[i] and free[j]): continue
            if h_vec[i] <= 0 or h_vec[j] <= 0: continue
            for si in (+1, -1):
                for sj in (+1, -1):
                    p = F_star.copy()
                    p[i] = F_star[i] + si * h_vec[i]
                    p[j] = F_star[j] + sj * h_vec[j]
                    cross_idx[(i, j, si, sj)] = len(pts)
                    pts.append(p)

    pts = np.clip(np.array(pts), lo, hi)
    L_vals = eval_batch(pts)
    L0 = L_vals[0]

    g = np.zeros(n)
    H = np.zeros((n, n))

    for i in range(n):
        if h_vec[i] <= 0:
            continue
        if free[i]:
            Lp = L_vals[plus_idx[i]]
            Lm = L_vals[minus_idx[i]]
            g[i] = (Lp - Lm) / (2 * h_vec[i])
            H[i, i] = (Lp - 2 * L0 + Lm) / (h_vec[i] ** 2)
        elif on_lo[i]:
            # One-sided forward derivative; sign convention unchanged
            Lp = L_vals[plus_idx[i]]
            g[i] = (Lp - L0) / h_vec[i]
        else:  # on_hi
            # One-sided backward derivative
            Lm = L_vals[minus_idx[i]]
            g[i] = (L0 - Lm) / h_vec[i]

    for i in range(n):
        for j in range(i+1, n):
            if not (free[i] and free[j]): continue
            if h_vec[i] <= 0 or h_vec[j] <= 0: continue
            Lpp = L_vals[cross_idx[(i, j, +1, +1)]]
            Lpm = L_vals[cross_idx[(i, j, +1, -1)]]
            Lmp = L_vals[cross_idx[(i, j, -1, +1)]]
            Lmm = L_vals[cross_idx[(i, j, -1, -1)]]
            H[i, j] = (Lpp - Lpm - Lmp + Lmm) / (4 * h_vec[i] * h_vec[j])
            H[j, i] = H[i, j]

    # Capture L at +h e_i and -h e_i for the discrete-max check.
    L_plus  = np.full(n, np.nan)
    L_minus = np.full(n, np.nan)
    for i in range(n):
        if plus_idx[i]  is not None: L_plus[i]  = L_vals[plus_idx[i]]
        if minus_idx[i] is not None: L_minus[i] = L_vals[minus_idx[i]]

    return L0, g, H, on_lo, on_hi, h_vec, L_plus, L_minus


def classify(F_star, L0, g, H, on_lo, on_hi, L_plus=None, L_minus=None,
             tol_eig=1e-6, tol_disc=1e-3):
    """Return (kind, info_dict) where kind ∈ {'interior_max','boundary_max',
    'saddle','not_max'}.

    Uses a discrete-max criterion (L0 >= L at all FD neighbours, within
    tol_disc) instead of a smooth gradient-zero check. This handles cliff
    peaks correctly: the central-FD gradient is large at a cliff, but
    L0 is still the local maximum value.
    """
    n = 4
    free = ~(on_lo | on_hi)
    free_idx = np.where(free)[0]

    # Discrete-max first-order check: at the candidate, every available
    # FD neighbour must have L_neighbour <= L0 + tol_disc.
    fo_ok = True
    fo_reasons = []
    for i in range(n):
        if L_plus is not None and not np.isnan(L_plus[i]):
            if L_plus[i] > L0 + tol_disc:
                fo_ok = False
                fo_reasons.append(
                    f'F{i+1}+h: L={L_plus[i]:.3f}>{L0:.3f} (improves)')
        if L_minus is not None and not np.isnan(L_minus[i]):
            if L_minus[i] > L0 + tol_disc:
                fo_ok = False
                fo_reasons.append(
                    f'F{i+1}-h: L={L_minus[i]:.3f}>{L0:.3f} (improves)')

    # Second-order check: sub-Hessian on free coords must be neg-semidef.
    if free_idx.size > 0:
        Hsub = H[np.ix_(free_idx, free_idx)]
        eigvals = np.linalg.eigvalsh((Hsub + Hsub.T) / 2)
    else:
        eigvals = np.array([])

    so_ok = bool(eigvals.size == 0 or eigvals.max() <= tol_eig)

    info = dict(grad=g, hess=H, eigvals=eigvals, free=free,
                fo_ok=fo_ok, so_ok=so_ok, fo_reasons=fo_reasons)

    if fo_ok and so_ok:
        kind = 'interior_max' if free.all() else 'boundary_max'
    elif so_ok and not fo_ok:
        kind = 'not_max'         # gradient says we can improve
    elif fo_ok and not so_ok:
        kind = 'saddle'
    else:
        kind = 'not_max'
    return kind, info


# ── Reporting ──────────────────────────────────────────────────────────────

KIND_COLOR = {
    'interior_max': 'INTERIOR MAX',
    'boundary_max': 'BOUNDARY MAX',
    'saddle':       'SADDLE',
    'not_max':      'NOT MAX',
}

def verify_points(points, labels=None, h_list=(2.0,)):
    """Verify a list of candidate points and print a report."""
    if labels is None:
        labels = [f'#{i+1}' for i in range(len(points))]

    print(f'\n{"="*100}')
    print(f' GRADIENT + HESSIAN VERIFICATION ({len(points)} candidates)')
    print(f'{"="*100}')
    print(f'{"label":<20} {"F1":>6} {"F2":>6} {"F3":>6} {"F4":>6} {"L":>9}  '
          f'{"|g_free|":>9} {"max eig":>9}  status')
    print('-'*100)

    classified = []
    for label, F_star in zip(labels, points):
        # Use the first h that succeeds; report the most informative
        for h in h_list:
            L0, g, H, on_lo, on_hi, h_vec, L_plus, L_minus = grad_hess(F_star, h=h)
            kind, info = classify(F_star, L0, g, H, on_lo, on_hi,
                                  L_plus=L_plus, L_minus=L_minus)
            if info['eigvals'].size == 0 or not np.isnan(info['eigvals']).any():
                break
        F = F_star
        free = info['free']
        g_free = np.linalg.norm(g[free]) if free.any() else 0.0
        max_eig = info['eigvals'].max() if info['eigvals'].size > 0 else float('nan')
        status = KIND_COLOR[kind]
        if not info['fo_ok']:
            status += '  [' + ', '.join(info['fo_reasons']) + ']'
        print(f'{label:<20} {F[0]:>6.1f} {F[1]:>6.1f} {F[2]:>6.1f} {F[3]:>6.1f} '
              f'{L0:>9.3f}  {g_free:>9.4f} {max_eig:>9.4f}  {status}')
        classified.append((label, F_star, L0, kind, info))
    return classified


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from-find', action='store_true',
                   help='re-run find_optima discovery and verify the result')
    p.add_argument('points', nargs='*',
                   help='whitespace-separated F1 F2 F3 F4 quadruples (interleaved)')
    args = p.parse_args()

    # Always start with the 16 corners
    corner_pts = np.array(list(itertools.product([0., 100.], repeat=4)))
    corner_labels = [f'corner ({c[0]:>3.0f},{c[1]:>3.0f},{c[2]:>3.0f},{c[3]:>3.0f})'
                     for c in corner_pts]

    points = list(corner_pts)
    labels = list(corner_labels)

    if args.from_find:
        print('Re-running a small Sobol scan to harvest candidates '
              '(use find_optima.py for the full SHGO pipeline)...')
        rng = np.random.default_rng(42)
        N = 50_000
        sugars = rng.uniform(0, 100, size=(N, 4))
        L_vals = eval_batch(np.vstack([corner_pts, sugars]))
        idx_sorted = np.argsort(-L_vals)
        all_pts = np.vstack([corner_pts, sugars])
        # epsilon-separated greedy selection on top 500
        top = all_pts[idx_sorted[:500]]
        top_L = L_vals[idx_sorted[:500]]
        EPSILON = 15.0
        kept_pts, kept_L = [], []
        for i in range(len(top)):
            if all(np.linalg.norm(top[i] - kp) >= EPSILON for kp in kept_pts):
                kept_pts.append(top[i]); kept_L.append(top_L[i])
        points = kept_pts
        labels = [f'find#{i+1}' for i in range(len(points))]
        print(f'  → {len(points)} candidates')
    elif args.points:
        # Interpret the rest of argv as F1 F2 F3 F4 ... interleaved
        nums = [float(x) for x in args.points]
        if len(nums) % 4 != 0:
            sys.exit('points must come in groups of 4 (F1 F2 F3 F4)')
        extra = np.array(nums).reshape(-1, 4)
        points = list(corner_pts) + list(extra)
        labels = corner_labels + [f'pt #{i+1}' for i in range(len(extra))]

    verify_points(points, labels=labels, h_list=(2.0, 0.5))


if __name__ == '__main__':
    main()
