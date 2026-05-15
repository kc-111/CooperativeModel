"""Local reaction kinetics for the 4-species Liebig consumer-resource model.

This replaces the prior Kong-2018-based consortium kinetics with a textbook
consumer-resource model (Tilman 1980; Marsland et al. 2019; Goyal & Maslov
2018) on a cyclic 4-cycle of species and essential resources.  The model is
run in **batch** mode — no chemostat dilution and no external supply terms
in the dynamics; the BO control vector enters only as the initial resource
concentrations.  A species-specific toxin term selects against the all-max
corner.

Variables (13 channels)::

    [N1..N4,  L,  R1..R4,    T1..T4]
     species  obj  resources  per-species toxins

Cyclic pairing (1-indexed in the comments, 0-indexed in code)::

    P_1 = {R_1, R_2},  P_2 = {R_2, R_3},  P_3 = {R_3, R_4},  P_4 = {R_4, R_1}

Each species i requires *both* resources in P_i and is Liebig-limited by
whichever is in shortest supply::

    g_i(R) = mu_i * min_{j in P_i}  R_j / (K + R_j)

Toxins (bacteriocin-style): each species secretes its own toxin at a rate
tied to its nutrient-uptake flux, the toxin decays first-order, and every
*other* species suffers a linear death term proportional to the toxins
not its own::

    dT_i/dt = beta * g_i * N_i  -  gamma * T_i
    death_i = delta * T_other * N_i
    T_other = sum_{j != i} T_j = T_tot - T_i

A species is immune to its own toxin (T_i is subtracted in T_other), so a
corner where only N_i grows produces only T_i and incurs zero kill on
itself.  At the all-max corner all four species grow simultaneously and
each one is suppressed by the other three toxins; this is what removes the
all-max corner as the global optimum independent of the resource budget.

Batch ODEs::

    dN_i/dt = g_i * N_i  -  delta * T_other * N_i
    dL/dt   = sum_i  Y_i * g_i * N_i
    dR_j/dt = - sum_{i : j in P_i}  c_i * g_i * N_i
    dT_i/dt = beta * g_i * N_i  -  gamma * T_i

Why four local optima of L(R_init) emerge intuitively from the equations:

  1. Liebig growth requires *both* of a species' paired resources.  Setting
     either paired resource to zero in the initial condition kills that
     species' growth term to zero.
  2. Each pair P_i defines a "corner" of supply space where only species i
     can grow — give R_{P_i} a high initial value and the two non-paired
     resources a low initial value.  There are four such pairs, hence four
     corners.
  3. At the all-max corner all four species coexist; each species is then
     poisoned by the three toxins it is *not* immune to, so net biomass
     never reaches the level achievable by a single uncontested species at
     a single-pair corner.  This removes the all-max corner as the global
     optimum and leaves the four pair-corners as the local maxima.

All operations are vectorised over the spatial grid [B, 13, Nz, Ny, Nx].
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
    # Intermediate RK stages can produce small negative concentrations that
    # would feed back into the rate laws; clamp to the physical orthant.
    state = state.clamp(min=0.0)

    N = state[:, 0:4]      # [B, 4, ...]   biomass per species
    R = state[:, 5:9]      # [B, 4, ...]   primary resource concentrations
    T = state[:, 9:13]     # [B, 4, ...]   per-species toxin concentrations

    mu    = params['mu']     # [1, 4, 1, 1, 1]  per-species max growth rate
    K     = params['K']      # scalar           Monod half-saturation
    c     = params['c']      # [1, 4, 1, 1, 1]  per-species stoichiometric coefficient
    Y     = params['Y']      # [1, 4, 1, 1, 1]  per-species lactate yield
    beta  = params['beta']   # scalar           toxin production / growth flux
    gamma = params['gamma']  # scalar           toxin first-order decay rate
    delta = params['delta']  # scalar           toxin kill coefficient (linear in T_other)

    # ---- Monod factor per resource: r_j / (K + r_j) -------------------------
    R_mon = R / (K + R)        # [B, 4, ...]

    # ---- Liebig growth on cyclic pairs P_i ---------------------------------
    # Species i (0-indexed) uses resources (i, (i+1) % 4).  Hardcoding the
    # cycle keeps the kinetics free of fancy tensor indexing while preserving
    # full vectorisation over the spatial grid.
    g0 = mu[:, 0:1] * torch.minimum(R_mon[:, 0:1], R_mon[:, 1:2])
    g1 = mu[:, 1:2] * torch.minimum(R_mon[:, 1:2], R_mon[:, 2:3])
    g2 = mu[:, 2:3] * torch.minimum(R_mon[:, 2:3], R_mon[:, 3:4])
    g3 = mu[:, 3:4] * torch.minimum(R_mon[:, 3:4], R_mon[:, 0:1])
    g = torch.cat([g0, g1, g2, g3], dim=1)        # [B, 4, ...]

    # ---- Toxin-induced death (linear in T_other) ----------------------------
    # Each species is immune to its own toxin. 
    T_tot = T.sum(dim=1, keepdim=True)             # [B, 1, ...]
    T_other = T_tot - T                            # [B, 4, ...]
    death = delta * T_other * N

    # ---- Species net growth: batch growth minus toxin death -----------------
    dN = g * N - death

    # ---- Lactate production: sum of weighted growth fluxes -----------------
    dL = (Y * g * N).sum(dim=1, keepdim=True)      # [B, 1, ...]

    # ---- Resource consumption (no back-secretion into R pool) ---------------
    # Each species i consumes c_i * g_i * N_i from *each* of its two paired
    # resources.  Under the cyclic pairing, resource R_j is consumed by the
    # two species whose pair contains j:
    #     R_0  <-  N_0, N_3
    #     R_1  <-  N_0, N_1
    #     R_2  <-  N_1, N_2
    #     R_3  <-  N_2, N_3
    flux = c * g * N                                # [B, 4, ...]   per-resource uptake
    cons_R0 = flux[:, 0:1] + flux[:, 3:4]
    cons_R1 = flux[:, 0:1] + flux[:, 1:2]
    cons_R2 = flux[:, 1:2] + flux[:, 2:3]
    cons_R3 = flux[:, 2:3] + flux[:, 3:4]
    cons = torch.cat([cons_R0, cons_R1, cons_R2, cons_R3], dim=1)
    dR = -cons

    # ---- Toxin dynamics: production tied to growth flux, decay first-order
    dT = beta * g * N - gamma * T

    # ---- Assemble rates -----------------------------------------------------
    rates = torch.zeros_like(state)
    rates[:, 0:4] = dN
    rates[:, 4:5] = dL
    rates[:, 5:9] = dR
    rates[:, 9:13] = dT
    return rates
