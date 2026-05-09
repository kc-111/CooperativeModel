"""Spatial operators for 3D reaction-advection simulations.

Advection  : conservative first-order upwind for -nabla.(v c) on six cardinal
             faces, with an open-face MAC stencil that matches the divergence
             operator the flow-solver projection drives to zero.  This makes
             the operator mass-conserving by construction (every face flux
             enters two cells with opposite signs) and keeps an initially-
             uniform field exactly uniform under the converged flow.

Mixing is driven by chaotic advection from the non-axisymmetric impeller; no
explicit diffusion operator is provided.  The first-order upwind already
contributes the modest numerical diffusion needed to smooth steep fronts on
the 32^3 grid.

Shape conventions
-----------------
    c    : [B, C, Nz, Ny, Nx]
    vel  : either [B, 3, Nz, Ny, Nx] (shared across species; channels
           [vx, vy, vz]) or [B, 3*C, Nz, Ny, Nx] for per-species velocities.
    wall_mask : [1, 1, Nz, Ny, Nx], 1 = wall, 0 = fluid.
"""

import torch
import torch.nn.functional as F


_PAD_3D = (1, 1, 1, 1, 1, 1)  # F.pad expects (W_l, W_r, H_l, H_r, D_l, D_r)


class Advection:
    """3D ``-nabla . (v c)`` via conservative first-order upwind on the six
    cardinal faces.

    The face-velocity convention matches the MAC-style stencil that
    :func:`flow_3d._div` projects onto in the flow solver: each cell-centred
    component is reinterpreted as the velocity on one specific face.
    Concretely, ``u[k,j,i]`` is the velocity on the **-x face** of cell
    ``(k,j,i)`` (equivalently the +x face of cell ``(k,j,i-1)``); analogous
    for ``v`` (-y face) and ``w`` (-z face).  With this convention,

        div_flux[i] = (u[i+1] - u[i]) / dx + ...

    matches the operator the pressure projection drives to zero, so an
    initially-uniform field stays exactly uniform under a converged flow
    instead of being amplified by spurious sources at fluid-wall interfaces.

    The previous *centred face* averaging (``vx_xp = 0.5*(u[i]+u[i+1])``)
    produced a different divergence — central-difference, which the flow
    solver does *not* enforce — and could grow a uniform IC by 300x within
    a few hours at the cylinder wall.

    Args:
        dx, dy, dz: cell sizes in each direction.
        wall_mask:  optional ``[1, 1, Nz, Ny, Nx]`` binary mask
                    (1 = wall, 0 = fluid).  Output is zeroed inside walls.
    """

    def __init__(self, dx, dy=None, dz=None, wall_mask=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.dz = dz if dz is not None else dx
        self.wall_mask = wall_mask
        if wall_mask is not None:
            fluid = 1.0 - wall_mask
            Nz, Ny, Nx = fluid.shape[-3], fluid.shape[-2], fluid.shape[-1]
            fp = F.pad(fluid, _PAD_3D, mode='constant', value=0.0)
            # Open-face indicators (face is open iff both adjacent cells are
            # fluid).  With the MAC convention used below, the +x face of cell
            # (k,j,i) sits between cells (k,j,i) and (k,j,i+1) and carries
            # velocity u[k,j,i+1]; the -x face carries u[k,j,i].  The face is
            # open when both adjacent cells are fluid.
            self._open_xp = fluid * fp[..., 1:1 + Nz, 1:1 + Ny, 2:2 + Nx]
            self._open_xm = fluid * fp[..., 1:1 + Nz, 1:1 + Ny, 0:Nx]
            self._open_yp = fluid * fp[..., 1:1 + Nz, 2:2 + Ny, 1:1 + Nx]
            self._open_ym = fluid * fp[..., 1:1 + Nz, 0:Ny,     1:1 + Nx]
            self._open_zp = fluid * fp[..., 2:2 + Nz, 1:1 + Ny, 1:1 + Nx]
            self._open_zm = fluid * fp[..., 0:Nz,     1:1 + Ny, 1:1 + Nx]
        else:
            self._open_xp = self._open_xm = None
            self._open_yp = self._open_ym = None
            self._open_zp = self._open_zm = None

    def __call__(self, c, vel):
        """Return ``-nabla . (v c)`` with the same shape as ``c``.

        Args:
            c:   ``[B, C, Nz, Ny, Nx]``.
            vel: ``[B, 3, Nz, Ny, Nx]`` (shared across species; channels
                 [vx, vy, vz]) or ``[B, 3*C, Nz, Ny, Nx]`` (per-species).
        """
        Nz, Ny, Nx = c.shape[-3], c.shape[-2], c.shape[-1]

        if vel.shape[1] == 3:
            vx = vel[:, 0:1]
            vy = vel[:, 1:2]
            vz = vel[:, 2:3]
        else:
            vx = vel[:, 0::3]
            vy = vel[:, 1::3]
            vz = vel[:, 2::3]

        c_pad = F.pad(c, _PAD_3D, mode='replicate')
        c_c = c_pad[..., 1:1 + Nz, 1:1 + Ny, 1:1 + Nx]
        c_xp = c_pad[..., 1:1 + Nz, 1:1 + Ny, 2:2 + Nx]
        c_xm = c_pad[..., 1:1 + Nz, 1:1 + Ny, 0:Nx]
        c_yp = c_pad[..., 1:1 + Nz, 2:2 + Ny, 1:1 + Nx]
        c_ym = c_pad[..., 1:1 + Nz, 0:Ny,     1:1 + Nx]
        c_zp = c_pad[..., 2:2 + Nz, 1:1 + Ny, 1:1 + Nx]
        c_zm = c_pad[..., 0:Nz,     1:1 + Ny, 1:1 + Nx]

        # MAC-style face velocities (no averaging).  ``u[k,j,i]`` is the
        # -x face velocity of cell ``(k,j,i)``.  Therefore the +x face
        # velocity of cell ``(k,j,i)`` is ``u[k,j,i+1]``.
        vxp = F.pad(vx, (1, 1, 0, 0, 0, 0), mode='replicate')
        vx_xp = vxp[..., 2:2 + Nx]            # +x face = u[i+1]
        vx_xm = vxp[..., 1:1 + Nx]            # -x face = u[i]

        vyp = F.pad(vy, (0, 0, 1, 1, 0, 0), mode='replicate')
        vy_yp = vyp[..., 2:2 + Ny, :]         # +y face = v[j+1]
        vy_ym = vyp[..., 1:1 + Ny, :]         # -y face = v[j]

        vzp = F.pad(vz, (0, 0, 0, 0, 1, 1), mode='replicate')
        vz_zp = vzp[..., 2:2 + Nz, :, :]      # +z face = w[k+1]
        vz_zm = vzp[..., 1:1 + Nz, :, :]      # -z face = w[k]

        if self._open_xp is not None:
            vx_xp = vx_xp * self._open_xp
            vx_xm = vx_xm * self._open_xm
            vy_yp = vy_yp * self._open_yp
            vy_ym = vy_ym * self._open_ym
            vz_zp = vz_zp * self._open_zp
            vz_zm = vz_zm * self._open_zm

        # Upwind face values of c.
        flux_xp = torch.where(vx_xp > 0, vx_xp * c_c,  vx_xp * c_xp)
        flux_xm = torch.where(vx_xm > 0, vx_xm * c_xm, vx_xm * c_c)
        flux_yp = torch.where(vy_yp > 0, vy_yp * c_c,  vy_yp * c_yp)
        flux_ym = torch.where(vy_ym > 0, vy_ym * c_ym, vy_ym * c_c)
        flux_zp = torch.where(vz_zp > 0, vz_zp * c_c,  vz_zp * c_zp)
        flux_zm = torch.where(vz_zm > 0, vz_zm * c_zm, vz_zm * c_c)

        div_flux = ((flux_xp - flux_xm) / self.dx
                    + (flux_yp - flux_ym) / self.dy
                    + (flux_zp - flux_zm) / self.dz)

        result = -div_flux
        if self.wall_mask is not None:
            result = result * (1.0 - self.wall_mask)
        return result
