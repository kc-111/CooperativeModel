"""Conservative first-order upwind advection with zero wall flux."""

import torch
import torch.nn.functional as F


_PAD_3D = (1, 1, 1, 1, 1, 1)  # F.pad expects (W_l, W_r, H_l, H_r, D_l, D_r)


class Advection:
    """Compute -div(v*c) using conservative first-order upwind fluxes.

    Stored velocity components lie on the negative x, y, and z cell faces,
    matching the flow solver divergence. dx, dy, dz are cell sizes.
    The optional wall_mask is [1, 1, Nz, Ny, Nx], with 1=wall and 0=fluid.
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
            # Faces are open only when both adjacent cells are fluid.
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

        # Stored velocities lie on negative faces; positive faces use neighboring cells.
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
