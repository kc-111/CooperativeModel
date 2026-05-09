"""Local reaction kinetics for the cooperative microbial consortium model.

Base model: the Monod-growth + diauxic-shift + nisin-cooperative-production +
nisin-protected-death structure follows

    Kong, W., Meldgin, D. R., Collins, J. J., and Lu, T. (2018).
    Designing microbial consortia with defined social interactions.
    Nature Chemical Biology, 14(8), 821-829.

Five mechanisms are added on top of the Kong et al. base so that L_final has
*interior* optima in the four-sugar input space rather than the trivial
corner-loaded optimum of the original (these are all opt-in via parameters
that recover the Kong et al. limit when set to zero / infinity):
  - Haldane substrate inhibition per sugar (Ki_inh -> infinity disables)
  - Lactic-acid product inhibition of CoA growth (Kp_L -> infinity disables)
  - Per-sugar saturating-Hill toxicity in death rate (c_tox = 0 disables)
  - Carbon catabolite repression of nisin biosynthesis (K_ccr -> infinity)
  - Multiplicative co-limitation of nisin biosynthesis on all 4 sugars
    (K_coop = 0 recovers single-sugar saturation)

Full system implemented:
  - Monod growth kinetics on 4 nutrients for both strains
  - Diauxic shift mechanism (CoA only) with sharpness parameter n
  - Death split into microbial (nisin-protectable) + chemical (osmotic /
    per-sugar toxic / acid-pH) components — chemical death is NOT suppressed
    by nisin self-immunity, since NisI/NisFEG protect against bacteriocin
    pore formation, not against generic physico-chemical stress
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

    # ---- Death rates: split into nisin-protectable (microbial) + chemical ----
    # Microbial death (bacteriocin-style): nisin self-protection 1/(1+ks*Sn).
    # Chemical death (osmotic / per-sugar toxicity / acid pH): NOT protected
    # by nisin — these are general physico-chemical stresses on cell envelope
    # and cytoplasm.  Without this split, once cells produce a tiny amount of
    # nisin (Sn ~ 0.005, ks=400), nisin protection collapses the entire death
    # term to ~0 and high-F configs simply consume all sugar with no penalty,
    # making L_final monotone in F_total.  Splitting keeps a chemical-death
    # floor that scales with F (and L), so high-F runs leave unconsumed sugar
    # and the L-vs-F surface acquires interior optima.
    nisin_inhibition = 1.0 / (1.0 + ks * Sn)
    # Chemical-stress excess (>= 0): each piece is the "factor - 1" so a
    # benign environment contributes 0 chemical death.
    osm_excess = c_osm * (F_total / K_osm).pow(h_osm)
    pH_excess  = c_pH  * (L       / K_pH ).pow(h_pH)
    # Per-sugar specific toxicity: each sugar contributes a *saturating* Hill
    # term (capped at 1), summed across the four sugars.  The sigmoidal Hill
    # form (h_tox > 1) keeps toxicity near zero below K_tox_i and saturates
    # cleanly above it, so death does not compound unboundedly when several
    # sugars are simultaneously high.  Per-sugar K_tox_i breaks F_total
    # symmetry so optima land at non-uniform interior F_i values.
    F_over_Ktox = (F / K_tox).pow(h_tox)
    tox_excess = c_tox * (F_over_Ktox / (1.0 + F_over_Ktox)).sum(
        dim=1, keepdim=True,
    )
    # CoA: microbial death + chemical (osmotic + toxicity + pH/acid)
    It_Sn1 = dt1 * (nisin_inhibition + osm_excess + tox_excess + pH_excess)
    # CoB: microbial death + chemical (osmotic + toxicity)
    It_Sn2 = dt2 * (nisin_inhibition + osm_excess + tox_excess)

    # ---- Cooperative nisin production ----
    # Carbon catabolite repression (CCR): high F_total downregulates nisin
    # biosynthesis (CcpA-mediated repression of secondary metabolites in LAB).
    # ccr_factor saturates at 1 for low F_total and decays at high F_total,
    # so corner4-style points produce LESS nisin than moderate-F interior
    # points and lose protection against the per-sugar toxicity death term.
    ccr_factor = 1.0 / (1.0 + (F_total / K_ccr).pow(h_ccr))
    # P_{S,Sn,F} = alpha * (Sn + rb) / (kp + Sn) * sum(Fi) * ccr_factor
    P_coeff = alpha * (Sn + rb) / (kp + Sn) * F_total * ccr_factor
    # Multiplicative co-limitation factor: nisin biosynthesis treats the four
    # sugars as complementary, non-substitutable inputs to a single secondary-
    # metabolite flux.  Justification: ribosomal synthesis of the 57-residue
    # precursor + NisB/C dehydration/cyclisation + NisT export are ATP- and
    # cofactor-intensive, so sustained nisin output requires balanced flux
    # through glycolysis / TCA / amino-acid pools (Stein et al. JBC 2003).
    # This is Saito et al. (2008) "Type I" co-limitation in the multiplicative
    # / Mankin form (Megee 1972; Bader 1978).  NB: this is NOT cross-feeding,
    # which would mean metabolic exchange between strains; this is single-
    # cell co-limitation of secondary-metabolite biosynthesis.
    # Each Hill term saturates at 1, so the product caps production rather
    # than amplifying it; any sugar going to zero kills the whole factor.
    # Strict Liebig would be (F / (K_coop + F)).min(dim=1) — keeping only the
    # single most-limiting sugar.  The product form penalises all four
    # limitations simultaneously and is what the 89 local-optima scan was
    # carried out under.
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
