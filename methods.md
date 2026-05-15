# Methods — 3D Cooperative Bioreactor

A two-stage simulator for a closed cylindrical stirred tank.  Stage 1 solves
the steady incompressible Navier–Stokes equations once and caches the
velocity field to disk; Stage 2 advects a 13-channel reacting state
(four species, four primary resources, four species-specific toxins, one
lactate-style objective) on top of that frozen flow.  Mixing is supplied
by chaotic advection from a non-axisymmetric impeller body force; **no
explicit (turbulent) diffusion operator is used**.

The code follows the framing of Delafosse et al. (2014) — every voxel in
the converged flow plays the role of a Delafosse-style compartment, and
the cached velocity field provides the inter-compartment fluxes.  The
impeller is modelled as a localised body force in the spirit of Pericleous
& Patel (1987).

---

## 1.  Geometry and grid

* Co-located uniform Cartesian grid, default 32 × 32 × 32 cells.
* Cube extent `Lx = Ly = Lz = 1` (arbitrary length units; the production
  runs use cm).
* The vessel is the cylinder inscribed in the (x, y) cross-section,
  axis aligned with z; aspect ratio H/D = 1.
* `cylinder_mask(grid)` returns a `[Nz, Ny, Nx]` indicator with 1 = fluid,
  0 = wall.  Walls are: outside the inscribed circle, the z-end caps
  (k = 0 and k = Nz − 1), and the four side faces of the bounding box
  (i = 0, i = Nx − 1, j = 0, j = Ny − 1).  The box-edge exclusion costs
  ≤ 1 % of fluid cells but is necessary so that the replicate padding in
  the projection chain does not silently collapse a face contribution and
  break the discrete identity `div_open(grad_open(p)) = lap_open(p)`.

---

## 2.  Stage 1 — Steady Navier–Stokes solve

Implemented in `src/CooperativeModel/flow_3d.py` (`solve_steady_flow`).

### 2.1  Governing equations

Incompressible NS with a stationary body force:

    ∂u/∂t + (u·∇)u  =  −∇p + ν ∇²u + f_imp(x)            in Ω_fluid
    ∇·u             =  0
    u               =  0                                  on ∂Ω_wall  (no-slip)
    ∂p/∂n           =  0                                  on ∂Ω_wall  (Neumann)

We integrate this in pseudo-time until the steady state, at which point
both `∂u/∂t` and the residual `‖uⁿ⁺¹ − uⁿ‖_∞ / ‖uⁿ‖_∞` vanish.

### 2.2  Impeller body force (non-axisymmetric)

`velocity_fields.impeller_body_force` returns a Cartesian field
`f ∈ ℝ^{3 × Nz × Ny × Nx}`:

    f(r, z, θ) = F0 · χ_rz(r, z) · χ_θ(θ) · θ̂

with

    χ_rz(r, z) = exp[ −((r − r_imp)² / 2σ_r²  +  (z − z_imp)² / 2σ_z²) ]
    χ_θ(θ)    = exp[ −d_circ(θ, θ_0)² / 2σ_θ² ]
    d_circ(a, b) = min(|a − b|,  2π − |a − b|)         (periodic distance)
    θ̂         = (−sin θ,  cos θ,  0)

The angular factor `χ_θ` localises the forcing to a **single blade** at
azimuth `θ_0` with width `σ_θ ≈ π/6`.  This deliberate breaking of
θ-symmetry is what makes the Lagrangian streamlines of the converged
flow chaotic — the canonical mixing regime of a real stirred tank.  An
axisymmetric (toroidal) forcing produces a purely circulating field whose
streamlines are closed circles, which does **not** mix.

Default geometric parameters: `r_imp = Lx / 4`, `z_imp = Lz / 2`,
`σ_r = Lx / 16`, `σ_z = Lz / 16`, `θ_0 = 0`, `σ_θ = π / 6`,
`F0 = 10`, `ν = 10⁻³`.

### 2.3  Numerics — Chorin / fractional-step projection

Each outer pseudo-time step:

1. **Predictor** (explicit Euler):

        u* = uⁿ + Δt · [ −(uⁿ·∇)uⁿ + ν Δuⁿ + f_imp ]

   * Convective term `(u·∇)u`: first-order upwind, applied componentwise.
     Upwind is L-stable on the explicit-Euler outer step (CFL < 1
     suffices), unlike central differences which would be unconditionally
     unstable here.  We only need the converged steady state, so the
     extra numerical dissipation is acceptable.
   * Viscous term: 7-point Laplacian.
   * Apply no-slip in walls: `u* ← u* · 𝟙_fluid`.

2. **Pressure Poisson** with Neumann BCs at walls and box boundaries:

        Δ_open p = (1 / Δt) · div_open(u*)

   solved by `pressure_iters = 200` red–black Gauss–Seidel sweeps on the
   open-face FV Laplacian (`_solve_pressure_rb_gs`).  The Neumann problem
   is unique only up to a constant, and warm-starting from the previous
   outer step lets that constant drift unboundedly across iterations, so
   `p` is shifted to fluid-mean-zero **after every red–black sweep**
   (inside the GS loop) and again at the end of each outer step.  The
   shift does not change `∇p`, only its bound — but it keeps round-off
   in the constants null space contained over `O(10⁴)` outer steps.

3. **Corrector**:

        uⁿ⁺¹ = u* − Δt · ∇_open p
        re-zero in walls.

Two metrics are tracked and a `tqdm` postfix is updated every 50 outer
steps:

* **Step-size residual** `‖uⁿ⁺¹ − uⁿ‖_∞ / max(‖uⁿ‖_∞, ε)` — the
  termination criterion (`< tol` ends the loop).  Confirms the velocity
  has stopped moving.
* **NS residual** `‖−(u·∇)u + ν Δu + f − ∇p‖₂` over fluid cells —
  recomputed periodically on the *new* velocity and pressure (the
  predictor used the pre-corrector fields, so a separate evaluation is
  needed).  Confirms the converged state actually satisfies the steady
  momentum equation.  Used as a sanity check; not a termination
  criterion.

Δt is set conservatively: with `u_ref = 0.5·√(F0·Lx)` from the impeller
force-balance estimate and `Δt = 0.15·h / u_ref`, the worst-case CFL
under the realistic peak `u_max ≈ (1–1.5)·√(F0·Lx)` is `0.3–0.45` —
safely below the upwind-advection stability limit of 1.

### 2.4  The open-face MAC stencil chain (and why it matters)

The species advection in Stage 2 uses a **MAC-style** divergence: the
cell-centred component `u[k,j,i]` is reinterpreted as the velocity on
the −x face of cell `(k,j,i)`, so

    div_open(u)[i] = (op_xp · u[i+1] − op_xm · u[i]) / Δx + …

with open-face indicators `op_xp[i] = mask[i] · mask[i+1]` (face open iff
both adjacent cells are fluid).  For the species transport to preserve a
spatially uniform field exactly under the converged flow, **the same
operator** must be the one driven to zero by the projection in Stage 1.

The Stage 1 components have therefore been chosen as a **consistent
discrete chain**:

    grad_open p   ←  backward difference, killed at any face touching a wall
    div_open u    ←  forward difference, killed at any face touching a wall
    lap_open p    =  div_open(grad_open p)         (FV 7-point at fluid cells)

These are the same three operators applied in `_grad`, `_div`, and the
red–black GS in `flow_3d.py`.  With this chain the corrector

    div_open(u_new) = div_open(u*) − Δt · lap_open(p) = 0

at every fluid cell, including those adjacent to walls.  An earlier
implementation used an unmasked forward divergence on the RHS; the
chain identity broke and the projection left a wall-localised residual
divergence on the order of `O(10)` in normalised units, which then
amplified an initially-uniform passive scalar by several hundred×
through the species advection.  The fix was to compute the RHS with
the **same** open-face MAC operator that the corrector inverts.

### 2.5  Compatibility projection

Neumann Poisson is solvable iff `∫_fluid rhs = 0`.  By Stokes on the
discrete chain, `Σ div_open(u*) = 0` over fluid cells exactly (each
fluid–fluid face contributes once with each sign).  The numerical
implementation still subtracts the fluid-mean of the RHS each step (and
of `p` each GS sweep, see § 2.3) so accumulated round-off in the
constant null-space stays bounded across `O(10⁴)` outer iterations.

### 2.6  Final tight projection (CG to machine epsilon)

200 red–black GS sweeps per step do **not** drive a Neumann Poisson on a
32³ grid to roundoff — they reduce the per-step error, but at outer
convergence the *cumulative* leftover divergence can still be
`O(10⁻³…10⁻¹)`.  After the outer loop terminates we run a single
unpreconditioned CG pass on the chain operator
`A p = −div_open(grad_open p)` (symmetric positive semi-definite on
fluid cells under Neumann BCs) until `‖rₖ‖ / ‖b‖ < 10⁻¹²`, then apply
the standard corrector.  This drives `|div_open(u_new)|_∞` to roundoff
in `O(50–300)` iterations.  The final divergence is recorded in the
HDF5 metadata as `final_cg_relres`.

For the production 32³ cache, `|div_open(u)|_∞ ≈ 7.5 × 10⁻¹⁵`.

**Scaling note.**  Unpreconditioned CG on the discrete Neumann Laplacian
has iteration count growing roughly as `O(N^{1/3})` on N³ grids (the
condition number scales as `h⁻²`), so the present `O(50–300)` number is
a 32³ figure.  At significantly larger resolutions a multigrid V-cycle
preconditioner — or geometric MG as the outer solver — would make
iteration count grid-independent at the cost of one restriction/
prolongation pass per step.  Worth doing only if Stage 1 ever scales up.

### 2.7  HDF5 cache

`save_flow` / `load_flow` round-trip `(u, v, w, mask)` plus a metadata
dict containing every parameter needed to reproduce the solve:
`Nx, Ny, Nz, Lx, Ly, Lz, F0, r_imp, z_imp, sigma_r, sigma_z, theta_0,
sigma_theta, nu, dt, tol, max_iters, pressure_iters, n_iters,
converged_residual, dtype, code_version, final_cg_iters,
final_cg_relres`.

`code_version` is **derived**, not a manual tag: it is a SHA-256 of the
source of the operator-defining functions (`_laplacian`, `_grad`, `_div`,
`_udotgrad`, `_solve_pressure_rb_gs`, `_final_cg_projection`,
`solve_steady_flow`), prefixed `3d-openface-` and truncated to 10 hex
chars.  Any stencil change therefore auto-bumps the version stamped into
the cache, so a stale cache can be detected by string comparison without
relying on a human to remember to bump a literal.

CLI: `python scripts/solve_flow.py --out flow_cache.h5`.

---

## 3.  Stage 2 — Reacting species transport

Implemented in `model.py` and `spatial_operators.py`.

### 3.1  State variables

The model carries 13 channels at every fluid voxel:

| index | symbol     | meaning                                                            |
|-------|------------|--------------------------------------------------------------------|
| 0–3   | `N₁..N₄`   | per-species biomass                                                |
| 4     | `L`        | scalar objective (lactate-style accumulator)                       |
| 5–8   | `R₁..R₄`   | primary nutrient resources                                         |
| 9–12  | `T₁..T₄`   | species-specific toxins (bacteriocin-style)                        |

### 3.2  Local kinetics (`compute_reaction_rates`)

A textbook 4-species Liebig consumer-resource model (Tilman 1980) on a
**cyclic 4-cycle** of species and essential resources.  Pairing is

    P₁ = {R₁, R₂},  P₂ = {R₂, R₃},  P₃ = {R₃, R₄},  P₄ = {R₄, R₁}

so each resource is essential to exactly two species and each species is
Liebig-limited by whichever of its two paired resources is in shortest
supply:

    g_i(R) = μ_i · min_{j ∈ P_i}  R_j / (K + R_j)

**Species-specific toxins** (bacteriocin-style; Riley & Wertz 2002;
Czárán et al. 2002; Kerr et al. 2002): each species secretes its own
toxin at a rate tied to its nutrient-uptake flux, the toxin decays
first-order, and every *other* species suffers a linear death term
proportional to the toxins not its own.  A species is **immune to its
own toxin** (T_i is excluded from the cross-toxin pool seen by N_i):

    T_other,i  =  Σ_{j ≠ i} T_j  =  T_tot − T_i
    death_i    =  δ · T_other,i · N_i
    dT_i/dt    =  β · g_i · N_i  −  γ · T_i

The toxin term **removes the all-max corner** of the resource initial
condition box as a viable global optimum.  At a single-pair corner
(e.g., R₁, R₂ high; R₃, R₄ low) only one species grows, produces only its
own toxin, and incurs zero kill on itself.  At the all-max corner all
four species grow and each is suppressed by the three cross-toxins,
collapsing total biomass and the L objective.  Linearity in T_other is
deliberate: a saturating (Hill) death term cannot overcome the carbon
stoichiometry advantage of the all-max corner under unconstrained
resource budgets, while a linear term scales with the cross-toxin pool
and suppresses all-max regardless of the budget.

Yield heterogeneity (`Y = [0.98, 1.02, 1.01, 0.99]`) breaks the
exact cyclic degeneracy across the four pair corners without changing
the qualitative topology.

The full batch ODE system at a voxel is

    dN_i/dt  =  g_i · N_i  −  δ · T_other,i · N_i
    dL/dt    =  Σ_i  Y_i · g_i · N_i
    dR_j/dt  =  − Σ_{i : j ∈ P_i}  c_i · g_i · N_i
    dT_i/dt  =  β · g_i · N_i  −  γ · T_i

All operations are vectorised over the spatial grid
`[B, 13, Nz, Ny, Nx]`.

### 3.3  Transport equations

For each channel `c_α`:

    ∂c_α/∂t  =  −∇·(v c_α) + R_α(c)

* `v = (vx, vy, vz)` is the cached steady velocity field, **frozen** for
  the entire BO run.  No two-way coupling; concentration changes do not
  feed back into the flow.
* `R_α(c)` is the reaction-rate vector from `compute_reaction_rates`.
* **No explicit diffusion term.**  Mixing comes entirely from chaotic
  advection by the non-axisymmetric impeller flow, plus the small
  numerical diffusion implicit in first-order upwind on a 32³ grid.

### 3.4  Spatial discretisation

`Advection` (in `spatial_operators.py`) implements
`−∇·(v c)` with conservative first-order upwind:

* The same MAC convention as Stage 1: `u[k,j,i]` is the −x face
  velocity of cell `(k,j,i)`; `u[k,j,i+1]` is the +x face velocity.
* Open-face indicators kill flux through any face that touches a wall.
* Upwind face values of `c` make the operator monotone (no spurious
  oscillations or negative concentrations on otherwise non-negative
  fields).
* Mass-conservative by construction: every internal face flux enters
  exactly two cells with opposite signs.  Walls have no flux.  At
  fluid–wall faces both sides are killed.

Because the divergence stencil here is identical to the one zeroed by
Stage 1 projection, a uniform field is preserved exactly under the
converged flow.

### 3.5  Time integration

`Tsit5SolverTorch` (an explicit RK45 with PI step controller) integrates
`∂y/∂t = R(y) − ∇·(v y)` on the flattened state `y ∈ ℝ^{B × 13 × Nz × Ny × Nx}`.
The CFL bound used to seed the initial step is

    Δt_adv = h_min / |v|_∞

Reaction stiffness is handled by the adaptive controller.

### 3.6  Conservation diagnostics

A passive-scalar blob test (`scripts/blob_test.py`) integrates a Gaussian
blob under pure advection on the production cache and reports:

* `|v|_∞ = 1.6` (cm/h on the production cache),
* fluid-volume-integrated mass drift over the run: **+0.0000 %**
  (machine precision),
* `c_max` decay: monotone, no overshoot.

A divergence diagnostic (`scripts/check_div.py`) computes three
divergence operators on the cached field and reports:

| operator                         | uses        | `|div|_∞`            |
|----------------------------------|-------------|----------------------|
| Open-face centred-velocity       | (legacy)    | small but non-zero   |
| **Open-face MAC**                | Stage 2 adv | **≈ 7.5 × 10⁻¹⁵**    |
| Forward difference (no mask)     | (legacy)    | small but non-zero   |

The MAC value is the operationally relevant one — it is the divergence
the species advection sees.

---

## 4.  Public API and reproducibility

The two stages communicate **only** through `flow_cache.h5`.  Stage 2
loads it once at the start of every `Simulator(...).run()` call and
never re-solves.  This makes a Bayesian-optimisation outer loop over
initial conditions cheap: each evaluation is one cached flow load plus
one species integration on the same fixed velocity field.

The `Simulator` constructor accepts `flow_cache_path=None` to force a
zero-velocity, all-fluid 1×1×1 grid — recovers the 0D well-mixed limit
used by `examples/example_ode.py` for ODE-only sanity checks against
the spatial code.

---

## 5.  Citations

### Flow (Stage 1)

* **Pericleous, K. A. & Patel, M. K.** (1987). *The modelling of
  tangential and axial agitators in chemical reactors.*
  PhysicoChemical Hydrodynamics **8**(2), 105–123.  Source for the
  body-force impeller model.

* **Delafosse, A. et al.** (2014). *CFD-based compartmental modelling
  of single-phase stirred-tank reactors: a multi-scale approach.*
  Chemical Engineering Science **106**, 76–85.  Source for the
  compartment-network framing of Stage 2 transport on the cached flow.

* **Chorin, A. J.** (1968). *Numerical solution of the Navier–Stokes
  equations.*  Mathematics of Computation **22**(104), 745–762.
  Fractional-step / projection method used in `solve_steady_flow`.

### Consumer-resource kinetics (Stage 2)

* **Tilman, D.** (1980). *Resources: a graphical-mechanistic approach
  to competition and predation.*  The American Naturalist **116**(3),
  362–393.  Original Liebig-style essential-resource competition
  framework adapted to the cyclic 4-cycle pairing here.

### Bacteriocin-style toxins

* **Riley, M. A. & Wertz, J. E.** (2002). *Bacteriocins: evolution,
  ecology, and application.*  Annual Review of Microbiology **56**,
  117–137.  Biological grounding for species-specific toxin secretion
  with self-immunity.

* **Czárán, T. L., Hoekstra, R. F. & Pagie, L.** (2002). *Chemical
  warfare between microbes promotes biodiversity.*  Proceedings of the
  National Academy of Sciences **99**(2), 786–790.  Mathematical model
  of mutually-antagonistic strains via bacteriocin-style toxins —
  source for the linear cross-killing term `δ · T_other · N`.

* **Kerr, B., Riley, M. A., Feldman, M. W. & Bohannan, B. J. M.**
  (2002). *Local dispersal promotes biodiversity in a real-life game of
  rock–paper–scissors.*  Nature **418**, 171–174.  Empirical and
  theoretical demonstration that toxin-mediated, non-transitive
  inhibition supports coexistence on cyclic-pairing graphs.
