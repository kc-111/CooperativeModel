"""Local reaction kinetics for the 4-species Liebig consumer-resource model.

Textbook consumer-resource model (Tilman 1980; Marsland et al. 2019;
Goyal & Maslov 2018) on a cyclic 4-cycle of species and essential
resources, with **two** parallel inhibitory mechanisms acting
multiplicatively on each species' growth rate:

  1. **Resource-mediated inhibition** ("anti-nutrient" Hill term).
     Each species is repressed by the *sum* of its two non-paired
     resources on the cycle.  Biology: end-product / pH / undissociated
     organic-acid stress, allelopathic secondary metabolite, or
     differential ion susceptibility.  Summing both non-paired resources
     (rather than only the antipodal one) collapses the "extra
     resource" plateau in L(R_init): any single non-paired R-HI
     configuration suppresses the would-be grower, so only the four
     pure pair corners (both paired R's HI, both non-paired R's LO)
     leave a species fully un-poisoned.

  2. **Toxin-mediated inhibition** ("bacteriocin-style" Hill term).
     Each species secretes its own species-specific toxin (rate
     proportional to its growth flux) and is immune to its own toxin.
     Every *other* species' growth is repressed by the pool of toxins
     it is not immune to.  Toxin decays first-order, so the inhibitor
     persists after the producer's paired resources are gone — this is
     what prevents the late-phase "depletion cascade" that would let a
     suppressed species wake up once its poison resource is consumed.

Both mechanisms inhibit *growth* rather than killing cells.  A
poisoned / inhibited species simply does not grow, so it consumes no
resource and produces no L; this avoids the spurious "transient growth
then die" L-leak that a linear cell-death term would create.

Variables (13 channels)::

    [N1..N4,  L,  R1..R4,    T1..T4]
     species  obj  resources  per-species toxins

Cyclic pairing (1-indexed in the comments, 0-indexed in code)::

    P_1 = {R_1, R_2},  P_2 = {R_2, R_3},  P_3 = {R_3, R_4},  P_4 = {R_4, R_1}

Each species i requires *both* resources in P_i and is Liebig-limited
by whichever is in shortest supply::

    Liebig_i(R)  =  min_{j in P_i}  R_j / (K + R_j)

The poison-resource map sends species i to the two resources that are
*not* in P_i — ``R_{(i+2) mod 4}`` and ``R_{(i+3) mod 4}``::

    i = 0  ->  {R_2, R_3}     (P_0 = {R_0, R_1})
    i = 1  ->  {R_3, R_0}     (P_1 = {R_1, R_2})
    i = 2  ->  {R_0, R_1}     (P_2 = {R_2, R_3})
    i = 3  ->  {R_1, R_2}     (P_3 = {R_3, R_0})

Hill inhibition factors::

    inh_R_i = K^h_R   /  (K^h_R   +  (R_{(i+2) mod 4} + R_{(i+3) mod 4})^h_R)
    inh_T_i = K_T^h_T /  (K_T^h_T +  (T_total - T_i)^h_T)

Net specific growth rate::

    g_i = mu_i * inh_R_i * inh_T_i * Liebig_i(R)

Batch ODEs::

    dN_i/dt = g_i * N_i
    dL/dt   = sum_i  Y_i * g_i * N_i
    dR_j/dt = - sum_{i : j in P_i}  c_i * g_i * N_i
    dT_i/dt = beta * g_i * N_i  -  gamma * T_i

Why four local optima of L(R_init) emerge from these equations:

  1. Liebig growth requires *both* of a species' paired resources, so
     setting either paired resource low kills that species' growth.
  2. At a pair corner P_i (paired resources high, poison resource low,
     other non-paired resource arbitrary) only species i grows; the
     resource-Hill suppresses the species opposite the cycle, and the
     Liebig limitation kills the two cycle-neighbours.
  3. The toxin produced by species i persists (first-order decay) and
     suppresses any competitor that would otherwise wake up once
     species i has consumed its paired resources — closing the
     "depletion cascade" channel.
  4. The all-max corner has every species' poison resource high, so
     every species is fully R-repressed; total L is small and the
     all-max corner is not the global optimum.

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
    K     = params['K']      # scalar           Monod half-saturation (also R-Hill K)
    h_R   = params['h_R']    # scalar           Hill exponent for R-poison inhibition
    K_T   = params['K_T']    # scalar           Hill K for toxin inhibition
    h_T   = params['h_T']    # scalar           Hill exponent for toxin inhibition
    c     = params['c']      # [1, 4, 1, 1, 1]  per-species stoichiometric coefficient
    Y     = params['Y']      # [1, 4, 1, 1, 1]  per-species lactate yield
    beta  = params['beta']   # scalar           toxin production per growth flux
    gamma = params['gamma']  # scalar           toxin first-order decay rate

    # ---- Monod factor per resource: r_j / (K + r_j) -------------------------
    R_mon = R / (K + R)        # [B, 4, ...]

    # ---- Liebig growth on cyclic pairs P_i ---------------------------------
    # Species i (0-indexed) uses resources (i, (i+1) % 4).  Hardcoding the
    # cycle keeps the kinetics free of fancy tensor indexing while preserving
    # full vectorisation over the spatial grid.
    L0 = torch.minimum(R_mon[:, 0:1], R_mon[:, 1:2])
    L1 = torch.minimum(R_mon[:, 1:2], R_mon[:, 2:3])
    L2 = torch.minimum(R_mon[:, 2:3], R_mon[:, 3:4])
    L3 = torch.minimum(R_mon[:, 3:4], R_mon[:, 0:1])
    Lieb = torch.cat([L0, L1, L2, L3], dim=1)                      # [B, 4, ...]

    # ---- Hill inhibition by the "poison" (non-paired) resources -----------
    # Species i is inhibited by *both* non-paired resources
    # R_{(i+2) mod 4} and R_{(i+3) mod 4}.  Using both poisons (rather than
    # only the antipodal one) collapses the otherwise-flat "extra resource"
    # plateau: at a three-sugar configuration (e.g. R0=R1=R2 HI, R3 LO) the
    # single growing species would have one non-paired resource HI, which
    # now suppresses it.  Only the pure pair corners P_i — where both
    # non-paired resources are LO — leave a species free to grow.
    R_poison_1 = torch.roll(R, shifts=-2, dims=1)                  # R_{i+2}
    R_poison_2 = torch.roll(R, shifts=-3, dims=1)                  # R_{i+3}
    R_poison_sum = R_poison_1 + R_poison_2                         # [B, 4, ...]
    KR_h = K ** h_R
    inh_R = KR_h / (KR_h + R_poison_sum ** h_R)                    # [B, 4, ...]

    # ---- Hill inhibition by the toxins species i is NOT immune to ----------
    # Each species secretes its own toxin and is immune to it; every other
    # species' growth is repressed by the pool T_other = T_tot - T_i.  Toxin
    # decays first-order rather than being consumed by uptake, so this
    # inhibitor persists after the producer's paired resources have been
    # depleted — closing the "depletion cascade" loophole that pure
    # R-mediated inhibition leaves open.
    T_tot = T.sum(dim=1, keepdim=True)                             # [B, 1, ...]
    T_other = T_tot - T                                            # [B, 4, ...]
    KT_h = K_T ** h_T
    inh_T = KT_h / (KT_h + T_other ** h_T)                         # [B, 4, ...]

    # ---- Net specific growth rate ------------------------------------------
    g = mu * inh_R * inh_T * Lieb                                  # [B, 4, ...]

    # ---- Species net growth (no cell-death; suppression is via inhibition) -
    dN = g * N

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

    # ---- Toxin dynamics: production tied to growth flux, decay first-order -
    dT = beta * g * N - gamma * T

    # ---- Assemble rates -----------------------------------------------------
    rates = torch.zeros_like(state)
    rates[:, 0:4] = dN
    rates[:, 4:5] = dL
    rates[:, 5:9] = dR
    rates[:, 9:13] = dT
    return rates
