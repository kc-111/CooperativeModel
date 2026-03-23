"""Spatial operators for 2D reaction-diffusion simulations.

Divergence: 8-direction flux-based stencil for nabla.(D nabla c) with no-flux BCs.
Advection:  Conservative upwind scheme for -nabla.(v c).
"""

import torch
import torch.nn.functional as F

INV_SQRT2 = 1.0 / (2.0 ** 0.5)


class Divergence:
    """
    Compute nabla.(D nabla c) using an 8-direction flux-based finite difference stencil.

    4 cardinal neighbours (weight 1.0) and 4 diagonal neighbours (weight 1/sqrt2).
    Face-averaged diffusion coefficients: D_face = (D_center + D_neighbour) / 2.
    Replicate padding enforces Neumann (no-flux) boundary conditions.

    Args:
        dx: Grid spacing in x-direction [cm].
        dy: Grid spacing in y-direction [cm]. Defaults to dx (square cells).
        wall_mask: Optional [1, 1, H, W] binary mask (1 = wall, 0 = fluid).
    """

    _OFFSETS = [
        # (row_offset, col_offset, weight)
        (-1,  0, 1.0),        # N
        ( 1,  0, 1.0),        # S
        ( 0, -1, 1.0),        # W
        ( 0,  1, 1.0),        # E
        (-1, -1, INV_SQRT2),  # NW
        (-1,  1, INV_SQRT2),  # NE
        ( 1, -1, INV_SQRT2),  # SW
        ( 1,  1, INV_SQRT2),  # SE
    ]

    def __init__(self, dx, dy=None, wall_mask=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.wall_mask = wall_mask

    def __call__(self, c, D):
        """
        Compute nabla.(D nabla c).

        Args:
            c: Concentration field, shape [B, C, H, W].
            D: Diffusion coefficients, broadcastable to [B, C, H, W].
               Typically [B, C, 1, 1] (uniform per species) or [B, C, H, W].

        Returns:
            Divergence field, shape [B, C, H, W].
        """
        H, W = c.shape[2], c.shape[3]
        c_pad = F.pad(c, (1, 1, 1, 1), mode='replicate')
        c_center = c_pad[:, :, 1:1+H, 1:1+W]

        # Detect spatially varying D
        spatially_varying = (D.dim() == 4 and D.shape[2] > 1 and D.shape[3] > 1)
        if spatially_varying:
            D_pad = F.pad(D, (1, 1, 1, 1), mode='replicate')
            D_center = D_pad[:, :, 1:1+H, 1:1+W]
        else:
            D_center = D  # broadcasts

        inv_dx2 = 1.0 / (self.dx * self.dy)
        result = torch.zeros_like(c)

        for dr, dc, w in self._OFFSETS:
            c_nbr = c_pad[:, :, 1+dr:1+dr+H, 1+dc:1+dc+W]
            if spatially_varying:
                D_nbr = D_pad[:, :, 1+dr:1+dr+H, 1+dc:1+dc+W]
                D_face = 0.5 * (D_center + D_nbr)
            else:
                D_face = D  # broadcasts
            result.add_(D_face * (c_nbr - c_center), alpha=w * inv_dx2)

        if self.wall_mask is not None:
            result.mul_(1.0 - self.wall_mask)

        return result


class Advection:
    """
    Compute -nabla.(v c) using a conservative first-order upwind scheme.

    Face velocities are averaged from cell-centred values. Upwind selection
    determines which cell value is used for the flux through each face.
    Replicate padding for boundary treatment.

    Args:
        dx: Grid spacing in x-direction [cm].
        dy: Grid spacing in y-direction [cm]. Defaults to dx.
        wall_mask: Optional [1, 1, H, W] binary mask (1 = wall, 0 = fluid).
    """

    def __init__(self, dx, dy=None, wall_mask=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx
        self.wall_mask = wall_mask

    def __call__(self, c, vel):
        """
        Compute -nabla.(v c).

        Args:
            c:   Concentration field, shape [B, C, H, W].
            vel: Velocity field. Either [B, 2, H, W] (shared across species,
                 vel[:,0]=vx, vel[:,1]=vy) or [B, 2*C, H, W] for per-species.

        Returns:
            Advection contribution, shape [B, C, H, W].
        """
        B, C, H, W = c.shape

        if vel.shape[1] == 2:
            vx = vel[:, 0:1, :, :]  # [B, 1, H, W], broadcasts over C
            vy = vel[:, 1:2, :, :]
        else:
            vx = vel[:, 0::2, :, :]  # [B, C, H, W]
            vy = vel[:, 1::2, :, :]

        # Pad concentration
        c_pad = F.pad(c, (1, 1, 1, 1), mode='replicate')
        c_center = c_pad[:, :, 1:1+H, 1:1+W]
        c_right  = c_pad[:, :, 1:1+H, 2:2+W]
        c_left   = c_pad[:, :, 1:1+H, 0:W]
        c_down   = c_pad[:, :, 2:2+H, 1:1+W]
        c_up     = c_pad[:, :, 0:H,   1:1+W]

        # --- X-direction (column axis) face velocities ---
        vx_pad = F.pad(vx, (1, 1, 0, 0), mode='replicate')
        vx_right = 0.5 * (vx_pad[..., 1:1+W] + vx_pad[..., 2:2+W])
        vx_left  = 0.5 * (vx_pad[..., 0:W]   + vx_pad[..., 1:1+W])

        # --- Y-direction (row axis) face velocities ---
        vy_pad = F.pad(vy, (0, 0, 1, 1), mode='replicate')
        vy_down = 0.5 * (vy_pad[:, :, 1:1+H, :] + vy_pad[:, :, 2:2+H, :])
        vy_up   = 0.5 * (vy_pad[:, :, 0:H,   :] + vy_pad[:, :, 1:1+H, :])

        # --- Upwind fluxes ---
        flux_x_right = torch.where(vx_right > 0, vx_right * c_center, vx_right * c_right)
        flux_x_left  = torch.where(vx_left  > 0, vx_left  * c_left,   vx_left  * c_center)
        flux_y_down  = torch.where(vy_down  > 0, vy_down  * c_center, vy_down  * c_down)
        flux_y_up    = torch.where(vy_up    > 0, vy_up    * c_up,     vy_up    * c_center)

        # Divergence of flux
        div_flux = ((flux_x_right - flux_x_left) / self.dx
                    + (flux_y_down - flux_y_up) / self.dy)

        result = -div_flux  # return -nabla.(v c)

        if self.wall_mask is not None:
            result.mul_(1.0 - self.wall_mask)

        return result
