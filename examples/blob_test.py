"""Passive-scalar blob test on the cached 32^3 flow.

Verifies that the stage-2 transport (pure Advection) is well-behaved on the
actual cached velocity field by tracking a single scalar Gaussian blob
under chaotic advection (no kinetics, no explicit diffusion).  Mixing
should be driven entirely by the impeller-induced flow.  Healthy outputs:

    - max(c) over time stays close to its initial value (decays monotonically
      due to diffusion and numerical dissipation, never blows up)
    - the blob translates and stretches along streamlines and gradually fills
      the cylinder via mixing
    - total mass drift remains small (a percent or two)

Writes:
    blob_dump.txt   – per-timestep min/max/mean/sum/argmax stats
    blob.gif        – mid-z animation of the blob with the wall contour overlaid
    blob.png        – snapshots at t = 0, t/4, t/2, 3t/4, t_final (mid-z)

Usage:
    python scripts/blob_test.py [flow_cache.h5]
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from CooperativeModel.flow_3d import load_flow
from CooperativeModel.spatial_operators import Advection

# ── Config ──────────────────────────────────────────────────────────────────
CACHE = sys.argv[1] if len(sys.argv) > 1 else 'flow_cache.h5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

T_FINAL = 48.0      # hours — ~40 eddy turnovers at |v|_max ≈ 1.6 cm/h on a 1 cm domain
N_OUTPUT = 49        # number of saved frames (~ every 30 min)
MIXING_SCALE = 1.0   # multiply cached velocity field (mass-conserving)
BLOB_CENTER = (0.5, 0.5, 0.25)   # (z, y, x) — off-center, near +z wall side
BLOB_WIDTH = 0.07                # gaussian sigma
BLOB_AMPL = 1.0

# CFL-derived dt (refined below once we know u_max)
DT_SAFETY = 0.4
# ────────────────────────────────────────────────────────────────────────────


print(f'Loading {CACHE}...')
u, v, w, fluid, meta = load_flow(CACHE)
fluid = fluid.to(device=DEVICE, dtype=DTYPE)
u = u.to(device=DEVICE, dtype=DTYPE) * MIXING_SCALE
v = v.to(device=DEVICE, dtype=DTYPE) * MIXING_SCALE
w = w.to(device=DEVICE, dtype=DTYPE) * MIXING_SCALE

Nz, Ny, Nx = fluid.shape
Lx = float(meta.get('Lx', 1.0))
Ly = float(meta.get('Ly', Lx))
Lz = float(meta.get('Lz', Lx))
dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
print(f'  grid {Nz}x{Ny}x{Nx}, extent {Lx}x{Ly}x{Lz} cm, '
      f'fluid cells = {int(fluid.sum().item())}/{Nz * Ny * Nx}')

mask_fluid = fluid                                                   # [Nz,Ny,Nx]
wall_mask = (1.0 - fluid).reshape(1, 1, Nz, Ny, Nx)                  # [1,1,Nz,Ny,Nx]
vel = torch.stack([u, v, w], dim=0).unsqueeze(0)                     # [1,3,Nz,Ny,Nx]
u_max = float(torch.maximum(torch.maximum(u.abs().max(), v.abs().max()),
                            w.abs().max()).item())
v_mean = float((torch.sqrt(u * u + v * v + w * w) * fluid).sum().item()
               / max(fluid.sum().item(), 1.0))
print(f'  |v|_max = {u_max:.3f} cm/h, |v|_mean = {v_mean:.3f} cm/h')

# CFL estimate (advection only — no explicit diffusion)
dt = DT_SAFETY * min(dx, dy, dz) / max(u_max, 1e-6)
n_steps = int(np.ceil(T_FINAL / dt))
dt = T_FINAL / n_steps
print(f'  dt = {dt:.4e} h  (n_steps = {n_steps}, '
      f'CFL_adv = {u_max * dt / min(dx, dy, dz):.3f})')

# Build operator (mask-aware open-face upwind)
adv = Advection(dx=dx, dy=dy, dz=dz, wall_mask=wall_mask)

# Initial Gaussian blob, zeroed inside walls
zg = (torch.arange(Nz, device=DEVICE, dtype=DTYPE) + 0.5) * dz
yg = (torch.arange(Ny, device=DEVICE, dtype=DTYPE) + 0.5) * dy
xg = (torch.arange(Nx, device=DEVICE, dtype=DTYPE) + 0.5) * dx
ZG, YG, XG = torch.meshgrid(zg, yg, xg, indexing='ij')
z0, y0, x0 = BLOB_CENTER
r2 = (ZG - z0) ** 2 + (YG - y0) ** 2 + (XG - x0) ** 2
c0 = BLOB_AMPL * torch.exp(-r2 / (2.0 * BLOB_WIDTH ** 2))
c0 = c0 * fluid
c = c0.clone().reshape(1, 1, Nz, Ny, Nx)

print(f'  blob IC: max = {c.max().item():.3f}, sum = {(c * fluid).sum().item():.3f}')

# ── Time stepping ──────────────────────────────────────────────────────────
save_every = max(1, n_steps // (N_OUTPUT - 1))
saved_t = []
saved_c = []
log_lines = []
log_lines.append(f'# Passive-scalar blob test on {CACHE}')
log_lines.append(f'# grid {Nz}x{Ny}x{Nx}, T_final={T_FINAL}, dt={dt:.4e}, '
                 f'blob_center={BLOB_CENTER}, blob_width={BLOB_WIDTH}')
log_lines.append('')
log_lines.append(f'{"step":>6} {"t":>8} {"min":>12} {"max":>12} {"mean":>12} '
                 f'{"sum":>12} {"max_loc(z,y,x)":>20}')

mass0 = float((c * fluid).sum().item())

def log(step, t):
    arr = c[0, 0]
    fv = arr[mask_fluid > 0.5]
    mn = float(fv.min().item()); mx = float(fv.max().item())
    mean = float(fv.mean().item()); s = float(fv.sum().item())
    flat_idx = int(torch.argmax(arr * mask_fluid).item())
    z = flat_idx // (Ny * Nx); rem = flat_idx % (Ny * Nx)
    y = rem // Nx; x = rem % Nx
    log_lines.append(f'{step:>6} {t:>8.4f} {mn:>12.4e} {mx:>12.4e} '
                     f'{mean:>12.4e} {s:>12.4e}  ({z:>2},{y:>2},{x:>2})')

log(0, 0.0)
saved_t.append(0.0); saved_c.append(c[0, 0].detach().cpu().numpy().copy())

for step in range(1, n_steps + 1):
    # Forward Euler step on ∂c/∂t = -∇·(v c)  (pure conservative advection)
    c = c + dt * adv(c, vel)
    if step % save_every == 0 or step == n_steps:
        t = step * dt
        log(step, t)
        saved_t.append(t)
        saved_c.append(c[0, 0].detach().cpu().numpy().copy())

mass1 = float((c * fluid).sum().item())
log_lines.append('')
log_lines.append(f'# total fluid mass: t=0  {mass0:.6e}')
log_lines.append(f'# total fluid mass: t=T  {mass1:.6e}')
log_lines.append(f'# drift fraction       : {(mass1 - mass0) / max(mass0, 1e-30):+.4%}')
log_lines.append(f'# c_max history (final / initial): '
                 f'{c.max().item() / max(c0.max().item(), 1e-30):.4f}')

with open('blob_dump.txt', 'w') as f:
    f.write('\n'.join(log_lines) + '\n')
print(f'Wrote blob_dump.txt')

# ── Visualisation ──────────────────────────────────────────────────────────
mask_np = mask_fluid.detach().cpu().numpy()
zmid = Nz // 2

xs = (np.arange(Nx) + 0.5) * dx
ys = (np.arange(Ny) + 0.5) * dy

# Per-frame normalisation: each frame's vmax = its own peak in the slice.
# Absolute concentration decays orders of magnitude as the blob mixes; without
# per-frame rescaling the structure becomes invisible after a fraction of the
# run.  Re-scaling shows *how* the blob mixes (its shape, location, stretch)
# rather than just how much it has decayed.
def _frame_vmax(arr2d):
    m = float(np.nanmax(arr2d))
    return max(m, 1e-12)

# 5-panel snapshot grid (per-frame normalised)
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
idxs = [0, len(saved_t) // 4, len(saved_t) // 2,
        3 * len(saved_t) // 4, len(saved_t) - 1]
for ax, i in zip(axes, idxs):
    arr = saved_c[i][zmid]
    arr_show = np.where(mask_np[zmid] > 0.5, arr, np.nan)
    vmax_i = _frame_vmax(arr_show)
    im = ax.pcolormesh(xs, ys, arr_show, cmap='magma',
                       vmin=0, vmax=vmax_i, shading='auto')
    ax.contour(xs, ys, mask_np[zmid], levels=[0.5],
               colors='cyan', linewidths=1.0)
    ax.set_aspect('equal')
    ax.set_title(f't = {saved_t[i]:.2f} h\nmax = {vmax_i:.3e}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.savefig('blob.png', dpi=120, bbox_inches='tight')
print(f'Wrote blob.png  ({os.path.getsize("blob.png") / 1024:.0f} KB)')
plt.close(fig)

# Animation (mid-z, per-frame normalised)
fig, ax = plt.subplots(figsize=(6, 6))
arr0 = np.where(mask_np[zmid] > 0.5, saved_c[0][zmid], np.nan)
vmax0 = _frame_vmax(arr0)
im = ax.pcolormesh(xs, ys, arr0, cmap='magma',
                   vmin=0, vmax=vmax0, shading='auto')
ax.contour(xs, ys, mask_np[zmid], levels=[0.5], colors='cyan', linewidths=1.0)
ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')
title = ax.set_title(f't = 0.00 h    max = {vmax0:.3e}  (per-frame norm)')
fig.colorbar(im, ax=ax, fraction=0.046, label='c / max(c)')

def update(i):
    arr = saved_c[i][zmid]
    arr_show = np.where(mask_np[zmid] > 0.5, arr, np.nan)
    vmax_i = _frame_vmax(arr_show)
    im.set_clim(vmin=0, vmax=vmax_i)
    im.set_array(arr_show.ravel() if im.get_array().ndim == 1 else arr_show)
    title.set_text(f't = {saved_t[i]:.2f} h    max = {vmax_i:.3e}  (per-frame norm)')
    return im, title

ani = animation.FuncAnimation(fig, update, frames=len(saved_t),
                              interval=80, blit=False)
ani.save('blob.gif', writer='pillow', dpi=80)
print(f'Wrote blob.gif  ({os.path.getsize("blob.gif") / 1024:.0f} KB)')
plt.close(fig)
