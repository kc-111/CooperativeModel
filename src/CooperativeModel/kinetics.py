"""Local reaction kinetics for the cooperative microbial consortium model.

Implements the full ODE system from the model description:
  - Monod growth kinetics on 4 nutrients for both strains
  - Diauxic shift mechanism (CoA only) with sharpness parameter n
  - Nisin-inhibited death rates
  - Cooperative nisin production depending on both strains and nutrients
  - Lactic acid production by CoA
  - Nutrient consumption by both strains

All operations are vectorized over the spatial grid [B, 8, H, W].
"""

import torch


def compute_reaction_rates(state, params):
    """
    Compute local reaction rates at every grid point.

    Args:
        state: [B, 8, H, W] tensor.
               Channels: [N1, N2, Sn, L, F1, F2, F3, F4].
        params: dict with parameter tensors (from ModelParameters.to_tensors()).

    Returns:
        [B, 8, H, W] tensor of d(state)/dt from reactions only.
    """
    # Clamp to prevent negatives from intermediate RK stages
    state = state.clamp(min=0.0)

    # Unpack state variables
    N1 = state[:, 0:1]   # [B, 1, H, W]
    N2 = state[:, 1:2]
    Sn = state[:, 2:3]
    L  = state[:, 3:4]
    F  = state[:, 4:8]   # [B, 4, H, W]  — F1, F2, F3, F4

    # Unpack parameters
    mu1    = params['mu1']      # [1, 4, 1, 1]
    mu2    = params['mu2']
    K1     = params['K1']
    K2     = params['K2']
    Ki_inh = params['Ki_inh']
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    dt1    = params['dt1']      # scalar
    dt2    = params['dt2']
    sigma  = params['sigma']
    alpha  = params['alpha']
    kp     = params['kp']
    rb     = params['rb']
    kn     = params['kn']
    ks     = params['ks']
    km     = params['km']
    YL     = params['YL']
    n      = params['n']
    Kp_L   = params['Kp_L']
    hL     = params['hL']
    c_osm  = params['c_osm']
    K_osm  = params['K_osm']
    h_osm  = params['h_osm']
    c_pH   = params['c_pH']
    K_pH   = params['K_pH']
    h_pH   = params['h_pH']
    K_coop = params['K_coop']
    K_ccr  = params['K_ccr']
    h_ccr  = params['h_ccr']
    c_tox  = params['c_tox']
    K_tox  = params['K_tox']
    h_tox  = params['h_tox']

    # ---- Individual Monod growth rates: g_{j,i} = mu_j * Fi / (K_{j,i} + Fi) ----
    g1 = mu1 * F / (K1 + F + F * F / Ki_inh)   # Haldane: peak at sqrt(K1*Ki_inh)
    g2 = mu2 * F / (K2 + F)    # [B, 4, H, W]  CoB on each nutrient

    # ---- Diauxic shift weights for CoA: beta_{1,i} = g_{1,i}^n / sum_j g_{1,j}^n ----
    g1_n = g1.pow(n)
    g1_n_sum = g1_n.sum(dim=1, keepdim=True).clamp(min=1e-30)
    beta1 = g1_n / g1_n_sum    # [B, 4, H, W]

    # ---- Total growth rates ----
    g1_total = (beta1 * g1).sum(dim=1, keepdim=True)   # [B, 1, H, W]
    g2_total = g2.mean(dim=1, keepdim=True)             # simple average over 4 nutrients

    # ---- Lactic-acid product inhibition of CoA growth (Hill form) ----
    # Slows CoA as L accumulates; couples all sugars through L and breaks the
    # yield ceiling, so L_final becomes non-monotonic in initial F.
    phi_L = 1.0 / (1.0 + (L / Kp_L).pow(hL))
    g1_total = g1_total * phi_L

    F_total = F.sum(dim=1, keepdim=True)

    # ---- Death rates: nisin protection × osmotic stress × (CoA: acid stress) ----
    # nisin protects: 1 / (1 + ks * Sn)
    # osmotic stress amplifies death of BOTH strains as F_total grows
    # acid stress amplifies death of CoA only as L grows
    nisin_inhibition = 1.0 / (1.0 + ks * Sn)
    osm_factor = 1.0 + c_osm * (F_total / K_osm).pow(h_osm)
    pH_factor  = 1.0 + c_pH  * (L      / K_pH ).pow(h_pH)
    # Per-sugar specific toxicity: sum of independent contributions, each peaks
    # at its own K_tox_i.  This breaks the F_total symmetry so the optimum can
    # land at non-uniform interior F_i values.
    tox_factor = 1.0 + c_tox * (F / K_tox).pow(h_tox).sum(dim=1, keepdim=True)
    It_Sn1 = dt1 * osm_factor * pH_factor * tox_factor * nisin_inhibition
    It_Sn2 = dt2 * osm_factor               * tox_factor * nisin_inhibition

    # ---- Cooperative nisin production ----
    # Carbon catabolite repression (CCR): high F_total downregulates nisin
    # biosynthesis (CcpA-mediated repression of secondary metabolites in LAB).
    # ccr_factor saturates at 1 for low F_total and decays at high F_total,
    # so corner4-style points produce LESS nisin than moderate-F interior
    # points and lose protection against the per-sugar toxicity death term.
    ccr_factor = 1.0 / (1.0 + (F_total / K_ccr).pow(h_ccr))
    # P_{S,Sn,F} = alpha * (Sn + rb) / (kp + Sn) * sum(Fi) * ccr_factor
    P_coeff = alpha * (Sn + rb) / (kp + Sn) * F_total * ccr_factor
    # Liebig minimum law: nisin production requires ALL 4 sugars present
    # (cross-feeding / complementary micronutrient hypothesis).  Each Hill
    # term saturates at 1, so this caps production rather than amplifying it.
    coop_factor = (F / (K_coop + F)).prod(dim=1, keepdim=True)   # [B, 1, H, W]
    # Pm = P_{S,Sn,F} * N1 * N2 / (km + N2) * coop_factor
    Pm = P_coeff * N1 * N2 / (km + N2) * coop_factor

    # ---- Assemble rates [B, 8, H, W] ----
    rates = torch.zeros_like(state)

    # dN1/dt = g1_total * N1 - It_Sn1 * N1
    rates[:, 0:1] = g1_total * N1 - It_Sn1 * N1

    # dN2/dt = (1/sigma) * g2_total * N2 - It_Sn2 * N2
    rates[:, 1:2] = (1.0 / sigma) * g2_total * N2 - It_Sn2 * N2

    # dSn/dt = Pm - kn * Sn
    rates[:, 2:3] = Pm - kn * Sn

    # dL/dt = YL * g1_total * N1
    rates[:, 3:4] = YL * g1_total * N1

    # dFi/dt = -(1/gamma1_i) * g1_i * beta1_i * N1 - (1/(gamma2_i * sigma)) * g2_i * N2
    rates[:, 4:8] = (-(1.0 / gamma1) * g1 * beta1 * N1
                     - (1.0 / (gamma2 * sigma)) * g2 * N2)

    return rates
