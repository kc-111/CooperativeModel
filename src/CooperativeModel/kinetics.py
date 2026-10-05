"""Local growth, resource consumption, product formation, and toxin kinetics.

State channels: [N1..N4, L, R1..R4, T1..T4].
"""

import torch


def compute_reaction_rates(state, params):
    """Compute local reaction rates at every grid point.

    Args:
        state: ``[B, 13, ...]`` tensor with channel order
            ``[N1..N4, L, R1..R4, T1..T4]``.
        params: dict produced by ``ModelParameters.to_tensors()``.

    Returns:
        ``[B, 13, ...]`` tensor of d(state)/dt from reactions only.
    """
    # Clamp negative Runge-Kutta stages before evaluating rate laws.
    state = state.clamp(min=0.0)

    N = state[:, 0:4]      # [B, 4, ...]   biomass per species
    R = state[:, 5:9]      # [B, 4, ...]   primary resource concentrations
    T = state[:, 9:13]     # [B, 4, ...]   per-species toxin concentrations

    mu    = params['mu']     # [1, 4, 1, 1, 1]  per-species max growth rate
    K     = params['K']      # scalar           Monod half-saturation (also R-Hill K)
    h_R   = params['h_R']    # scalar           Hill exponent for R-poison inhibition
    K_T   = params['K_T']    # scalar           Hill K for toxin inhibition
    h_T   = params['h_T']    # scalar           Hill exponent for toxin inhibition
    c     = params['c']      # [1, 4, 1, 1, 1]  per-species stoichiometric coefficient
    Y     = params['Y']      # [1, 4, 1, 1, 1]  per-species lactate yield
    beta  = params['beta']   # scalar           toxin production per growth flux
    gamma = params['gamma']  # scalar           toxin first-order decay rate

    # Monod factor per resource: r_j / (K + r_j)
    R_mon = R / (K + R)        # [B, 4, ...]

    # Species i requires resources i and (i + 1) % 4.
    L0 = torch.minimum(R_mon[:, 0:1], R_mon[:, 1:2])
    L1 = torch.minimum(R_mon[:, 1:2], R_mon[:, 2:3])
    L2 = torch.minimum(R_mon[:, 2:3], R_mon[:, 3:4])
    L3 = torch.minimum(R_mon[:, 3:4], R_mon[:, 0:1])
    Lieb = torch.cat([L0, L1, L2, L3], dim=1)                      # [B, 4, ...]

    # Growth is inhibited by the sum of unused resources.
    R_poison_1 = torch.roll(R, shifts=-2, dims=1)                  # R_{i+2}
    R_poison_2 = torch.roll(R, shifts=-3, dims=1)                  # R_{i+3}
    R_poison_sum = R_poison_1 + R_poison_2                         # [B, 4, ...]
    KR_h = K ** h_R
    inh_R = KR_h / (KR_h + R_poison_sum ** h_R)                    # [B, 4, ...]

    # Exclude each species' own toxin from the inhibitory pool.
    T_tot = T.sum(dim=1, keepdim=True)                             # [B, 1, ...]
    T_other = T_tot - T                                            # [B, 4, ...]
    KT_h = K_T ** h_T
    inh_T = KT_h / (KT_h + T_other ** h_T)                         # [B, 4, ...]

    # Net specific growth rate
    g = mu * inh_R * inh_T * Lieb                                  # [B, 4, ...]

    # Species net growth (no cell-death; suppression is via inhibition)
    dN = g * N

    # Lactate production: sum of weighted growth fluxes
    dL = (Y * g * N).sum(dim=1, keepdim=True)      # [B, 1, ...]

    # Each species consumes both required resources.
    flux = c * g * N                                # [B, 4, ...]   per-resource uptake
    cons_R0 = flux[:, 0:1] + flux[:, 3:4]
    cons_R1 = flux[:, 0:1] + flux[:, 1:2]
    cons_R2 = flux[:, 1:2] + flux[:, 2:3]
    cons_R3 = flux[:, 2:3] + flux[:, 3:4]
    cons = torch.cat([cons_R0, cons_R1, cons_R2, cons_R3], dim=1)
    dR = -cons

    # Toxin dynamics: production tied to growth flux, decay first-order
    dT = beta * g * N - gamma * T

    # Assemble rates
    rates = torch.zeros_like(state)
    rates[:, 0:4] = dN
    rates[:, 4:5] = dL
    rates[:, 5:9] = dR
    rates[:, 9:13] = dT
    return rates
