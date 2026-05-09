"""Steady incompressible Navier-Stokes flow solve and HDF5 cache I/O.

Stage-1 of the 3D pipeline.  Solves an unsteady NS in the cylindrical-tank
mask until ``||u^{n+1} - u^n||_inf / max(||u^n||_inf, eps) < tol``, holding the
non-axisymmetric impeller body force fixed.  The converged velocity field is
saved to HDF5 along with all parameters needed to reproduce it; Stage 2
loads the cache once per BO evaluation and never re-solves.

Numerics
--------
Chorin / fractional-step projection on a co-located uniform Cartesian grid:

    u*    = u^n + dt * (-(u^n . grad) u^n + nu * lap u^n + f_imp)
    apply no-slip on the wall mask (u* = 0 in wall cells)
    solve  lap p = (1/dt) div u*           with Neumann BCs (red-black GS)
    u^{n+1} = u* - dt * grad p
    re-zero u^{n+1} on the wall mask

Discretisation
--------------
* 7-point Laplacian for the viscous and pressure operators
* First-order upwind for (u . grad) u (L-stable on the explicit-Euler outer
  step; we only need the converged steady state, not a high-fidelity transient)
* Replicate / Neumann padding at all six box boundaries
* The pressure Poisson uses the **open-face MAC** divergence on the RHS and a
  matching **open-face FV Laplacian** on the LHS, so the corrector
  ``u_new = u* - dt * grad_open(p)`` drives ``div_open(u_new)`` to zero at
  every fluid cell — including those adjacent to walls.  This is the
  divergence operator the species advection actually uses, so a uniform field
  is preserved exactly under the converged flow (no spurious wall sources).

Citations
---------
* Pericleous & Patel (1987), "The modelling of tangential and axial agitators
  in chemical reactors" — impeller-as-body-force.
* Delafosse et al. (2014), Chemical Engineering Science 106, 76-85 — the
  Stage-2 compartment-model framing the cached velocity field feeds.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .velocity_fields import impeller_body_force


_PAD_3D = (1, 1, 1, 1, 1, 1)  # (left, right, top, bottom, front, back) — F.pad order


def _operator_fingerprint():
    """Hash of the operator-defining functions used by the NS solve.

    Stamped into the cache metadata as ``code_version`` so a cached flow
    can be detected as stale whenever any of the discretisation primitives
    (Laplacian, gradient, divergence, advection, pressure GS, CG cleanup)
    or the steady-state driver itself change source.  Cheaper than carrying
    a manual version string and impossible to forget to bump.
    """
    parts = []
    for fn in (_laplacian, _grad, _div, _udotgrad,
               _solve_pressure_rb_gs, _final_cg_projection,
               solve_steady_flow):
        parts.append(inspect.getsource(fn))
    digest = hashlib.sha256(''.join(parts).encode('utf-8')).hexdigest()
    return f'3d-openface-{digest[:10]}'


def _pad_replicate(t):
    """Replicate-pad a [..., Nz, Ny, Nx] tensor by 1 on every face.

    F.pad with mode='replicate' on 5D tensors expects [N, C, D, H, W];
    we always feed it that shape.
    """
    return F.pad(t, _PAD_3D, mode='replicate')


def _laplacian(u, dx, dy, dz):
    """7-point Laplacian on a co-located grid with Neumann BCs.

    Args:
        u: [N, C, Nz, Ny, Nx] tensor.
        dx, dy, dz: cell sizes.

    Returns:
        lap u, same shape as u.
    """
    p = _pad_replicate(u)
    cz, cy, cx = u.shape[-3], u.shape[-2], u.shape[-1]
    centre = p[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
    xp = p[..., 1:1 + cz, 1:1 + cy, 2:2 + cx]
    xm = p[..., 1:1 + cz, 1:1 + cy, 0:cx]
    yp = p[..., 1:1 + cz, 2:2 + cy, 1:1 + cx]
    ym = p[..., 1:1 + cz, 0:cy,     1:1 + cx]
    zp = p[..., 2:2 + cz, 1:1 + cy, 1:1 + cx]
    zm = p[..., 0:cz,     1:1 + cy, 1:1 + cx]
    return ((xp + xm - 2.0 * centre) / (dx * dx)
            + (yp + ym - 2.0 * centre) / (dy * dy)
            + (zp + zm - 2.0 * centre) / (dz * dz))


def _grad(p, dx, dy, dz, mask=None):
    """Backward-difference gradient with optional mask-aware face flux.

    Without ``mask`` this returns the plain backward-difference gradient.
    When ``mask`` is given (1=fluid, 0=wall), each face flux is killed if
    either of the two cells it joins is a wall — this enforces ∂p/∂n = 0
    at fluid-wall interfaces.  Paired with the forward-difference ``_div``
    and a mask-aware FV Laplacian in the pressure GS, the projection step
    drives ``div u_new`` to zero at every fluid cell, including those
    adjacent to walls.

    Args:
        p:    [N, 1, Nz, Ny, Nx].
        mask: optional [N, 1, Nz, Ny, Nx] fluid mask.

    Returns:
        gx, gy, gz, each [N, 1, Nz, Ny, Nx].
    """
    pp = _pad_replicate(p)
    cz, cy, cx = p.shape[-3], p.shape[-2], p.shape[-1]
    gx = (pp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
          - pp[..., 1:1 + cz, 1:1 + cy, 0:cx]) / dx
    gy = (pp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
          - pp[..., 1:1 + cz, 0:cy,     1:1 + cx]) / dy
    gz = (pp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
          - pp[..., 0:cz,     1:1 + cy, 1:1 + cx]) / dz
    if mask is not None:
        mp = _pad_replicate(mask)
        mxm = mp[..., 1:1 + cz, 1:1 + cy, 0:cx]
        mym = mp[..., 1:1 + cz, 0:cy,     1:1 + cx]
        mzm = mp[..., 0:cz,     1:1 + cy, 1:1 + cx]
        gx = gx * mxm * mask
        gy = gy * mym * mask
        gz = gz * mzm * mask
    return gx, gy, gz


def _div(u, v, w, dx, dy, dz, mask=None):
    """Forward-difference divergence of (u, v, w), each [N, 1, Nz, Ny, Nx].

    With ``mask`` (1=fluid, 0=wall), each face contribution is killed when
    either of the two cells the face joins is a wall — i.e. the operator
    becomes the open-face MAC divergence used by the species advection.
    Combined with the mask-aware backward gradient ``_grad`` and the open-face
    FV Laplacian solved in ``_solve_pressure_rb_gs``, this gives a consistent
    chain ``div_open ∘ grad_open = lap_open`` so the projection drives
    ``div_open(u_new)`` to zero at every fluid cell.
    """
    up = _pad_replicate(u)
    vp = _pad_replicate(v)
    wp = _pad_replicate(w)
    cz, cy, cx = u.shape[-3], u.shape[-2], u.shape[-1]

    u_xp = up[..., 1:1 + cz, 1:1 + cy, 2:2 + cx]
    u_xm = up[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
    v_yp = vp[..., 1:1 + cz, 2:2 + cy, 1:1 + cx]
    v_ym = vp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
    w_zp = wp[..., 2:2 + cz, 1:1 + cy, 1:1 + cx]
    w_zm = wp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]

    if mask is not None:
        mp = _pad_replicate(mask)
        xp_m = mp[..., 1:1 + cz, 1:1 + cy, 2:2 + cx]
        xm_m = mp[..., 1:1 + cz, 1:1 + cy, 0:cx]
        yp_m = mp[..., 1:1 + cz, 2:2 + cy, 1:1 + cx]
        ym_m = mp[..., 1:1 + cz, 0:cy,     1:1 + cx]
        zp_m = mp[..., 2:2 + cz, 1:1 + cy, 1:1 + cx]
        zm_m = mp[..., 0:cz,     1:1 + cy, 1:1 + cx]
        # Open-face indicators: face open iff both adjacent cells are fluid.
        op_xp = mask * xp_m
        op_xm = mask * xm_m
        op_yp = mask * yp_m
        op_ym = mask * ym_m
        op_zp = mask * zp_m
        op_zm = mask * zm_m
        u_xp = u_xp * op_xp; u_xm = u_xm * op_xm
        v_yp = v_yp * op_yp; v_ym = v_ym * op_ym
        w_zp = w_zp * op_zp; w_zm = w_zm * op_zm

    return (u_xp - u_xm) / dx + (v_yp - v_ym) / dy + (w_zp - w_zm) / dz


def _udotgrad(u, v, w, dx, dy, dz):
    """First-order upwind ``(u . grad)`` applied to each of u, v, w.

    Upwind is L-stable (CFL < 1 is sufficient), unlike central differences
    which are unconditionally unstable on the explicit-Euler outer step
    used here.  The added numerical dissipation is acceptable: we only need
    a converged steady state, not a high-fidelity transient.

    Returns convective derivatives ``((u.grad)u, (u.grad)v, (u.grad)w)``.
    """
    def _udg_scalar(s):
        sp = _pad_replicate(s)
        cz, cy, cx = s.shape[-3], s.shape[-2], s.shape[-1]
        c   = sp[..., 1:1 + cz, 1:1 + cy, 1:1 + cx]
        sxp = (sp[..., 1:1 + cz, 1:1 + cy, 2:2 + cx] - c) / dx
        sxm = (c - sp[..., 1:1 + cz, 1:1 + cy, 0:cx]) / dx
        syp = (sp[..., 1:1 + cz, 2:2 + cy, 1:1 + cx] - c) / dy
        sym = (c - sp[..., 1:1 + cz, 0:cy,     1:1 + cx]) / dy
        szp = (sp[..., 2:2 + cz, 1:1 + cy, 1:1 + cx] - c) / dz
        szm = (c - sp[..., 0:cz,     1:1 + cy, 1:1 + cx]) / dz
        ux = torch.where(u >= 0, sxm, sxp)
        vy = torch.where(v >= 0, sym, syp)
        wz = torch.where(w >= 0, szm, szp)
        return u * ux + v * vy + w * wz

    return _udg_scalar(u), _udg_scalar(v), _udg_scalar(w)


def _solve_pressure_rb_gs(p, rhs, mask, dx, dy, dz, n_sweeps, progress_bar=None):
    """Red-black Gauss-Seidel for the FV Laplacian with Neumann BCs at walls.

    Solves ``lap p = rhs`` at fluid cells (``mask = 1``), where ``lap`` is the
    finite-volume Laplacian summing only open (fluid-fluid) faces — equivalent
    to enforcing ∂p/∂n = 0 at every fluid-wall interface.  This stencil is the
    exact discrete adjoint of ``div_fwd ∘ grad_bwd_masked`` (see ``_grad``),
    so the corrector ``u_new = u* − dt · grad p`` produces a velocity whose
    forward divergence is zero at every fluid cell, including those adjacent
    to walls.  Walls are left at p = 0 (their values do not enter the stencil).

    The box boundary is replicate-padded — also Neumann.

    Args:
        p:    [N, 1, Nz, Ny, Nx] initial guess (modified in-place).
        rhs:  [N, 1, Nz, Ny, Nx] right-hand side.
        mask: [N, 1, Nz, Ny, Nx] fluid mask (1.0 inside fluid).
        dx, dy, dz: cell sizes.
        n_sweeps: number of red-black sweeps.
        progress_bar: optional tqdm bar to update once per sweep.

    Returns:
        Updated pressure tensor.
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    Nz, Ny, Nx = p.shape[-3], p.shape[-2], p.shape[-1]
    iz = torch.arange(Nz, device=p.device).reshape(-1, 1, 1)
    iy = torch.arange(Ny, device=p.device).reshape(1, -1, 1)
    ix = torch.arange(Nx, device=p.device).reshape(1, 1, -1)
    parity = ((ix + iy + iz) & 1).to(p.dtype)  # 0 = red, 1 = black

    # Open-face indicators (1 if the neighbour cell is fluid, 0 if wall).
    mp = _pad_replicate(mask)
    xp_m = mp[..., 1:1 + Nz, 1:1 + Ny, 2:2 + Nx]
    xm_m = mp[..., 1:1 + Nz, 1:1 + Ny, 0:Nx]
    yp_m = mp[..., 1:1 + Nz, 2:2 + Ny, 1:1 + Nx]
    ym_m = mp[..., 1:1 + Nz, 0:Ny,     1:1 + Nx]
    zp_m = mp[..., 2:2 + Nz, 1:1 + Ny, 1:1 + Nx]
    zm_m = mp[..., 0:Nz,     1:1 + Ny, 1:1 + Nx]

    diag = ((xp_m + xm_m) * inv_dx2
            + (yp_m + ym_m) * inv_dy2
            + (zp_m + zm_m) * inv_dz2)
    diag_safe = diag.clamp(min=1e-30)

    n_fluid = mask.sum().clamp(min=1.0)

    for _ in range(n_sweeps):
        for colour in (0.0, 1.0):
            pad = _pad_replicate(p)
            xp = pad[..., 1:1 + Nz, 1:1 + Ny, 2:2 + Nx]
            xm = pad[..., 1:1 + Nz, 1:1 + Ny, 0:Nx]
            yp = pad[..., 1:1 + Nz, 2:2 + Ny, 1:1 + Nx]
            ym = pad[..., 1:1 + Nz, 0:Ny,     1:1 + Nx]
            zp = pad[..., 2:2 + Nz, 1:1 + Ny, 1:1 + Nx]
            zm = pad[..., 0:Nz,     1:1 + Ny, 1:1 + Nx]
            num = ((xp * xp_m + xm * xm_m) * inv_dx2
                   + (yp * yp_m + ym * ym_m) * inv_dy2
                   + (zp * zp_m + zm * zm_m) * inv_dz2)
            new_p = (num - rhs) / diag_safe
            select = (parity == colour) & (mask > 0.5)
            p = torch.where(select, new_p, p)
        # Anchor mean-zero each sweep — Neumann Poisson is unique only up
        # to a constant, and warm-starting from previous outer iterations
        # can let GS push p far into the constants null space within a
        # single call.  The shift doesn't change grad p, only its bound.
        p = (p - (p * mask).sum() / n_fluid) * mask
        if progress_bar is not None:
            progress_bar.update(1)
    return p


def _final_cg_projection(u, v, w, mask, dx, dy, dz,
                         max_iters=2000, tol=1e-12):
    """Final tight projection: drive ``div_open(u, v, w)`` to roundoff.

    Solves ``-lap_open(p) = -div_open(u)`` to high accuracy with un-
    preconditioned CG, then applies the standard corrector
    ``u_new = u - grad_open(p)`` (and likewise for v, w).  ``A = -lap_open``
    is symmetric positive semi-definite on fluid cells under Neumann BCs;
    the chain ``-_div(_grad(p))`` matches the operator the corrector
    inverts, so the residual norm of CG equals ``|div_open(u_new)|``
    modulo roundoff.

    Scaling note: unpreconditioned CG on the Neumann Poisson problem has
    iteration count growing roughly as ``O(N^{1/3})`` on N^3 grids, since
    the condition number of the discrete Laplacian scales as h^{-2}.
    At 32^3 this converges in O(50-300) iterations and is fine.  For
    significantly larger grids a multigrid V-cycle preconditioner (or
    geometric MG as the solver outright) would be a drop-in win — it makes
    iteration count grid-independent at the cost of one extra restriction/
    prolongation pass per CG step.  Worth doing only if you scale up.

    Returns:
        u_new, v_new, w_new, n_iters, final_rel_res.
    """
    n_fluid = mask.sum().clamp(min=1.0)

    def matvec(p):
        gx, gy, gz = _grad(p, dx, dy, dz, mask=mask)
        return -_div(gx, gy, gz, dx, dy, dz, mask=mask) * mask

    rhs = _div(u, v, w, dx, dy, dz, mask=mask)
    b = -rhs * mask
    b_mean = (b * mask).sum() / n_fluid
    b = (b - b_mean) * mask

    p = torch.zeros_like(b)
    r = b - matvec(p)
    d = r.clone() * mask
    rr_old = (r * r * mask).sum()
    b_norm = rr_old.sqrt().clamp(min=1e-30)
    res_norm = b_norm

    for it in range(max_iters):
        Ad = matvec(d)
        dAd = (d * Ad * mask).sum().clamp(min=1e-60)
        alpha = rr_old / dAd
        p = p + alpha * d
        r = r - alpha * Ad
        # Anchor mean-zero so p doesn't drift in the constants null space.
        p = (p - (p * mask).sum() / n_fluid) * mask
        rr_new = (r * r * mask).sum()
        res_norm = rr_new.sqrt() / b_norm
        if res_norm < tol or torch.isnan(rr_new) or rr_new == 0:
            break
        beta = rr_new / rr_old.clamp(min=1e-60)
        d = (r + beta * d) * mask
        rr_old = rr_new

    gpx, gpy, gpz = _grad(p, dx, dy, dz, mask=mask)
    u_new = (u - gpx) * mask
    v_new = (v - gpy) * mask
    w_new = (w - gpz) * mask
    return u_new, v_new, w_new, it + 1, float(res_norm.item())


def solve_steady_flow(
    grid,
    mask=None,
    F0=10.0,
    r_imp=None,
    z_imp=None,
    sigma_r=None,
    sigma_z=None,
    theta_0=0.0,
    sigma_theta=math.pi / 6,
    nu=1e-3,
    dt=None,
    tol=1e-4,
    max_iters=20000,
    pressure_iters=200,
    device='cpu',
    dtype=torch.float64,
    progress=True,
):
    """Solve steady NS in the cylindrical mask via projection iteration.

    Args:
        grid: ``GridConfig`` instance (provides Nx, Ny, Nz, Lx, Ly, Lz).
        mask: ``[Nz, Ny, Nx]`` float fluid/wall mask (1=fluid).  If ``None``,
            ``cylinder_mask(grid)`` is used.
        F0, r_imp, z_imp, sigma_r, sigma_z, theta_0, sigma_theta:
            Forwarded to ``impeller_body_force``.
        nu: Kinematic viscosity [cm^2/h].
        dt: Pseudo-time step.  If ``None``, set to
            ``0.4 * min(dx, dy, dz) / max(0.5 * sqrt(F0 * Lx), 1e-3)``.
        tol: Convergence tolerance on ``||u^{n+1} - u^n||_inf / ||u^n||_inf``.
        max_iters: Maximum outer iterations.
        pressure_iters: Inner red-black Gauss-Seidel sweeps per outer step.
        device, dtype: Torch placement.
        progress: Show a tqdm bar over the outer iteration.

    Returns:
        u, v, w: Three ``[Nz, Ny, Nx]`` velocity components, all wall-zeroed.
        metadata: dict with reproducibility info.
    """
    from .velocity_fields import cylinder_mask  # local import avoids cycles

    if mask is None:
        mask = cylinder_mask(grid, device=device, dtype=dtype)
    else:
        mask = mask.to(device=device, dtype=dtype)

    dx, dy, dz = grid.dx, grid.dy, grid.dz

    if dt is None:
        # Conservative CFL on the upwind advection.  The rough impeller
        # force-balance estimate is ``u_ref = 0.5 sqrt(F0 Lx)``; the peak
        # velocity in the converged field is consistently 2-3x this
        # (boundary layers, recirculation), so realistic ``u_max`` is
        # ``~ (1-1.5) sqrt(F0 Lx)``.  Choosing ``dt = 0.15 h / u_ref``
        # gives  CFL = u_max dt / h = 0.3 (u_max / u_ref) ≈ 0.3-0.45 in
        # practice — safely below the upwind stability limit of 1.
        u_ref = max(0.5 * math.sqrt(max(F0, 0.0) * grid.Lx), 1e-3)
        dt = 0.15 * min(dx, dy, dz) / u_ref

    # Body force (shared across iterations).
    f = impeller_body_force(
        grid, F0=F0, r_imp=r_imp, z_imp=z_imp,
        sigma_r=sigma_r, sigma_z=sigma_z,
        theta_0=theta_0, sigma_theta=sigma_theta,
        device=device, dtype=dtype,
    )  # [3, Nz, Ny, Nx]
    fx = f[0:1].unsqueeze(0)  # [1, 1, Nz, Ny, Nx]
    fy = f[1:2].unsqueeze(0)
    fz = f[2:3].unsqueeze(0)

    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    u = torch.zeros(1, 1, Nz, Ny, Nx, device=device, dtype=dtype)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)
    p = torch.zeros_like(u)

    mask_5d = mask.reshape(1, 1, Nz, Ny, Nx)

    eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-12

    outer = tqdm(range(max_iters), desc='NS solve', disable=not progress)
    last_residual = float('nan')
    last_ns_residual = float('nan')
    last_iter = 0
    ns_residual_every = 50
    for it in outer:
        u_prev = u.clone()
        v_prev = v.clone()
        w_prev = w.clone()

        # --- Predictor: u* = u + dt*(-(u.grad)u + nu*lap u + f) ---
        cu, cv, cw = _udotgrad(u, v, w, dx, dy, dz)
        lu = _laplacian(u, dx, dy, dz)
        lv = _laplacian(v, dx, dy, dz)
        lw = _laplacian(w, dx, dy, dz)
        u_star = u + dt * (-cu + nu * lu + fx)
        v_star = v + dt * (-cv + nu * lv + fy)
        w_star = w + dt * (-cw + nu * lw + fz)
        u_star = u_star * mask_5d
        v_star = v_star * mask_5d
        w_star = w_star * mask_5d

        # --- Pressure Poisson: lap_open(p) = div_open(u*) / dt ---
        # Use the **open-face MAC** divergence on the RHS to match the
        # open-face FV Laplacian solved by the GS.  With this stencil pair
        # the discrete identity ``div_open(grad_open(p)) = lap_open(p)`` holds
        # exactly, so after the corrector  ``u_new = u* - dt · grad_open(p)``
        # we get  ``div_open(u_new) = div_open(u*) - lap_open(p) · dt = 0``
        # at every fluid cell — including those adjacent to walls.  This
        # eliminates the wall-localised divergence that the previous
        # forward-difference RHS left behind, which had to be removed by a
        # post-hoc CG cleanup pass.
        #
        # Compatibility of the Neumann Poisson problem: ``sum(div_open(u*))``
        # over fluid cells equals zero by construction (each fluid-fluid face
        # contributes once with each sign — Stokes), so the RHS is in the
        # range of ``lap_open`` to roundoff.  We still subtract the fluid-
        # mean to keep accumulated round-off bounded.
        rhs_raw = _div(u_star, v_star, w_star, dx, dy, dz, mask=mask_5d) / dt
        n_fluid = mask_5d.sum().clamp(min=1.0)
        rhs_mean = (rhs_raw * mask_5d).sum() / n_fluid
        rhs = (rhs_raw - rhs_mean) * mask_5d
        # Inner-bar only on the first outer step (avoid terminal flooding).
        inner_bar = None
        if progress and it == 0:
            inner_bar = tqdm(total=pressure_iters, desc='  pressure (it 0)',
                             leave=False)
        p = _solve_pressure_rb_gs(p, rhs, mask_5d, dx, dy, dz,
                                  pressure_iters, progress_bar=inner_bar)
        if inner_bar is not None:
            inner_bar.close()
        # Anchor pressure: Neumann Poisson is unique only up to a constant,
        # and warm-starting from the previous outer iteration lets that
        # constant drift unboundedly across iterations.  Subtracting the
        # fluid-mean keeps p bounded without changing grad p.
        p = (p - (p * mask_5d).sum() / n_fluid) * mask_5d

        # --- Corrector ---
        # Mask-aware backward gradient — kills face flux at fluid-wall faces
        # (Neumann), matching the FV Laplacian solved by the GS so the forward
        # divergence of u_new is identically zero at every fluid cell.
        gpx, gpy, gpz = _grad(p, dx, dy, dz, mask=mask_5d)
        u = (u_star - dt * gpx) * mask_5d
        v = (v_star - dt * gpy) * mask_5d
        w = (w_star - dt * gpz) * mask_5d

        # --- Convergence check ---
        diff = torch.maximum(
            (u - u_prev).abs().amax(),
            torch.maximum((v - v_prev).abs().amax(),
                          (w - w_prev).abs().amax()),
        )
        scale = torch.maximum(
            u.abs().amax(),
            torch.maximum(v.abs().amax(), w.abs().amax()),
        )
        residual = (diff / (scale + eps)).item()
        last_residual = residual
        last_iter = it + 1

        # Periodic NS residual: ||−(u·∇)u + ν∇²u + f − ∇p|| over fluid.
        # The step-size metric ``residual`` only confirms ``u`` has stopped
        # moving; the NS residual confirms ``u`` actually satisfies the
        # steady momentum equation.  Recompute the convective and
        # viscous terms on the *new* velocity and pressure for a faithful
        # check (the predictor used the old fields).
        if it % ns_residual_every == 0 or residual < tol:
            with torch.no_grad():
                cu_n, cv_n, cw_n = _udotgrad(u, v, w, dx, dy, dz)
                lu_n = _laplacian(u, dx, dy, dz)
                lv_n = _laplacian(v, dx, dy, dz)
                lw_n = _laplacian(w, dx, dy, dz)
                gpx_n, gpy_n, gpz_n = _grad(p, dx, dy, dz, mask=mask_5d)
                rx = (-cu_n + nu * lu_n + fx - gpx_n) * mask_5d
                ry = (-cv_n + nu * lv_n + fy - gpy_n) * mask_5d
                rz = (-cw_n + nu * lw_n + fz - gpz_n) * mask_5d
                ns_res = (rx * rx + ry * ry + rz * rz).sum().sqrt()
                last_ns_residual = float(ns_res.item())
            outer.set_postfix(res=f'{residual:.3e}',
                              ns_res=f'{last_ns_residual:.3e}',
                              umax=f'{scale.item():.3e}')
        if residual < tol:
            break
    outer.close()

    # ── Final tight projection (CG to machine epsilon) ─────────────────────
    # The per-step red-black GS only partially solves the pressure Poisson
    # (200 sweeps doesn't reach machine epsilon for Neumann Poisson on 32^3).
    # At convergence of the outer iteration the per-step velocity change is
    # already small but the cumulative leftover divergence is not.  A single
    # CG pass on the chain operator A = -lap_open = -div_open(grad_open(·))
    # drives ``div_open(u_new)`` to roundoff in O(50-300) iterations.  The
    # corrector is the same form used inside the loop.
    u, v, w, cg_iters, cg_res = _final_cg_projection(
        u, v, w, mask_5d, dx, dy, dz,
    )
    metadata_extra = {
        'final_cg_iters': int(cg_iters),
        'final_cg_relres': float(cg_res),
    }

    metadata = {
        'Nx': int(grid.Nx), 'Ny': int(grid.Ny), 'Nz': int(grid.Nz),
        'Lx': float(grid.Lx), 'Ly': float(grid.Ly), 'Lz': float(grid.Lz),
        'F0': float(F0),
        'r_imp': float(r_imp if r_imp is not None else 0.25 * grid.Lx),
        'z_imp': float(z_imp if z_imp is not None else 0.5 * grid.Lz),
        'sigma_r': float(sigma_r if sigma_r is not None else grid.Lx / 16.0),
        'sigma_z': float(sigma_z if sigma_z is not None else grid.Lz / 16.0),
        'theta_0': float(theta_0),
        'sigma_theta': float(sigma_theta),
        'nu': float(nu),
        'dt': float(dt),
        'tol': float(tol),
        'max_iters': int(max_iters),
        'pressure_iters': int(pressure_iters),
        'n_iters': int(last_iter),
        'converged_residual': float(last_residual),
        'dtype': str(dtype),
        # Derived from a SHA-256 of the operator-defining functions, so a
        # stencil change auto-invalidates cached flows on next compare.
        'code_version': _operator_fingerprint(),
        **metadata_extra,
    }

    return (u.squeeze(0).squeeze(0),
            v.squeeze(0).squeeze(0),
            w.squeeze(0).squeeze(0),
            metadata)


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def save_flow(path, u, v, w, mask, metadata):
    """Write the cached flow to HDF5.

    Datasets:  /u, /v, /w, /mask  — each [Nz, Ny, Nx], saved as float64.
    Attributes: every key in ``metadata`` written on the root group.
    """
    def _np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().to(dtype=torch.float64, device='cpu').numpy()
        return np.asarray(t, dtype=np.float64)

    u_np, v_np, w_np, m_np = _np(u), _np(v), _np(w), _np(mask)
    with h5py.File(path, 'w') as h:
        h.create_dataset('u', data=u_np, compression='gzip')
        h.create_dataset('v', data=v_np, compression='gzip')
        h.create_dataset('w', data=w_np, compression='gzip')
        h.create_dataset('mask', data=m_np, compression='gzip')
        for k, val in metadata.items():
            h.attrs[k] = val


def load_flow(path):
    """Load a cached flow from HDF5.

    Returns:
        u, v, w: ``[Nz, Ny, Nx]`` torch.float64 tensors on CPU.
        mask:    ``[Nz, Ny, Nx]`` torch.float64 tensor on CPU.
        metadata: dict of root-level attributes.
    """
    with h5py.File(path, 'r') as h:
        u = torch.from_numpy(np.asarray(h['u'][...], dtype=np.float64))
        v = torch.from_numpy(np.asarray(h['v'][...], dtype=np.float64))
        w = torch.from_numpy(np.asarray(h['w'][...], dtype=np.float64))
        mask = torch.from_numpy(np.asarray(h['mask'][...], dtype=np.float64))
        metadata = {k: (val.item() if hasattr(val, 'item') else val)
                    for k, val in h.attrs.items()}
    return u, v, w, mask, metadata
