"""Verify which solutions are TRUE local maxima vs artifacts.

Closed batch reactor: F1..F4 are initial sugar amounts loaded at t=0.

1. Evaluate all 16 corners (each F_i ∈ {0, 100})
2. For each corner, check gradient signs (should all point inward)
3. Spot-check interior "optima" from find_optima.py by perturbing
"""

import sys, os, time
import numpy as np
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator

N1, N2, Sn, L = 0.05, 0.05, 0.0, 0.0
DEVICE = 'cuda'
DELTA = 0.5  # perturbation size for gradient check

def eval_batch(sugars_array):
    B = len(sugars_array)
    samples = np.zeros((B, 8))
    samples[:, 0] = N1
    samples[:, 1] = N2
    samples[:, 4:] = sugars_array
    r = Simulator(
        samples=samples.tolist(),
        mode='batch', t_final=72.0, grid_size=1,
        omega=0.0, diffusion_scale=0.0, device=DEVICE
    ).run()
    return r.L_final if B > 1 else np.array([r.L_final])

# ── 1. All 16 corners ──────────────────────────────────────────────────
print('='*60)
print(' ALL 16 CORNERS')
print('='*60)
corners = list(itertools.product([0, 100], repeat=4))
corner_sugars = np.array(corners, dtype=float)
corner_L = eval_batch(corner_sugars)

ranked = sorted(range(16), key=lambda i: -corner_L[i])
print(f'\n{"Rank":<5} {"F1":>5} {"F2":>5} {"F3":>5} {"F4":>5} {"Total":>6} {"L_final":>10}')
print('-'*48)
for rank, i in enumerate(ranked, 1):
    c = corners[i]
    print(f'{rank:<5} {c[0]:>5} {c[1]:>5} {c[2]:>5} {c[3]:>5} '
          f'{sum(c):>6} {corner_L[i]:>10.4f}')

# ── 2. Gradient check at each corner ───────────────────────────────────
print(f'\n{"="*60}')
print(f' GRADIENT CHECK AT CORNERS (delta={DELTA})')
print(f' Is each corner a LOCAL MAXIMUM?')
print(f'{"="*60}')

# For each corner, perturb each F_i inward by DELTA and check if L decreases
# At F_i=0: check F_i=DELTA (should decrease L if 0 is optimal)
# At F_i=100: check F_i=100-DELTA (should decrease L if 100 is optimal)
all_perturbed = []
corner_indices = []  # (corner_idx, dimension)
for ci, corner in enumerate(corners):
    for dim in range(4):
        perturbed = list(corner)
        if corner[dim] == 0:
            perturbed[dim] = DELTA
        else:
            perturbed[dim] = 100 - DELTA
        all_perturbed.append(perturbed)
        corner_indices.append((ci, dim))

perturbed_L = eval_batch(np.array(all_perturbed))

print(f'\n{"Corner":<22} {"L_corner":>9} {"is_max":>7}  gradient directions')
print('-'*72)
for ci in ranked:
    corner = corners[ci]
    L_c = corner_L[ci]
    is_max = True
    grads = []
    for dim in range(4):
        pidx = ci * 4 + dim
        L_p = perturbed_L[pidx]
        dL = L_p - L_c
        direction = 'in' if corner[dim] == 0 else 'out'
        # If perturbing inward INCREASES L, this corner is NOT a local max
        if dL > 0:
            is_max = False
            grads.append(f'F{dim+1}:+{dL:.3f}!')
        else:
            grads.append(f'F{dim+1}:{dL:.3f}')
    label = f'({corner[0]:>3},{corner[1]:>3},{corner[2]:>3},{corner[3]:>3})'
    status = 'YES' if is_max else 'no'
    print(f'{label:<22} {L_c:>9.4f} {status:>7}  {", ".join(grads)}')

# ── 3. Check interior "optima" from find_optima.py ─────────────────────
print(f'\n{"="*60}')
print(f' INTERIOR POINT VERIFICATION')
print(f' Perturb each coord ±{DELTA} — if any improves L, not a max')
print(f'{"="*60}')

interior_pts = [
    [37.64, 98.66, 71.78, 95.12],   # #5 from find_optima
    [96.98, 98.46, 28.82, 73.38],   # #6
    [43.89,  2.16, 82.63, 89.62],   # #7
    [15.43, 68.30, 74.48, 96.75],   # #8
    [50.0,  50.0,  50.0,  50.0],    # center of domain
    [0.0,   50.0,  50.0,  50.0],    # half-interior
]
labels = ['#5 (37,99,72,95)', '#6 (97,98,29,73)', '#7 (44,2,83,90)',
          '#8 (15,68,74,97)', 'center (50x4)', 'half (0,50,50,50)']

# Build all perturbations
all_pts = []
pt_meta = []  # (point_idx, 'base' or (dim, direction))
for pi, pt in enumerate(interior_pts):
    all_pts.append(pt)
    pt_meta.append((pi, 'base'))
    for dim in range(4):
        for sign in [-1, +1]:
            pert = list(pt)
            pert[dim] = np.clip(pert[dim] + sign * DELTA, 0, 100)
            all_pts.append(pert)
            pt_meta.append((pi, (dim, sign)))

all_pts_L = eval_batch(np.array(all_pts))

print(f'\n{"Point":<22} {"L_base":>9} {"is_max":>7}  improving perturbations')
print('-'*72)
for pi, (pt, label) in enumerate(zip(interior_pts, labels)):
    base_idx = pi * 9  # 1 base + 8 perturbations per point
    L_base = all_pts_L[base_idx]
    improving = []
    for j in range(8):
        idx = base_idx + 1 + j
        dL = all_pts_L[idx] - L_base
        dim = j // 2
        sign = [-1, +1][j % 2]
        if dL > 1e-6:
            improving.append(f'F{dim+1}{"+" if sign>0 else "-"}:{dL:+.4f}')
    is_max = len(improving) == 0
    status = 'YES' if is_max else 'no'
    print(f'{label:<22} {L_base:>9.4f} {status:>7}  {", ".join(improving) if improving else "(none)"}')
