# CooperativeModel

3D reaction-advection simulation of a two-strain cooperative microbial consortium (CoA + CoB) inside a cylindrical stirred tank, implemented in PyTorch.  Mixing is supplied entirely by chaotic advection from a non-axisymmetric impeller body force; there is no explicit (eddy / Fickian) diffusion operator.

The package is organised as a **two-stage pipeline**: a steady incompressible Navier–Stokes solve over the cylinder produces a velocity field once and caches it to HDF5; species transport then integrates the cooperative kinetics on top of that fixed flow for every Bayesian-optimisation evaluation. The flow is held constant through the BO loop — concentrations evolve, the velocity field does not.

The architecture follows the **CFD-based compartment-model framing** of Delafosse et al. (2014, *Chemical Engineering Science* **106**, 76–85): each finite-volume cell acts as a compartment whose inter-compartment fluxes come from the resolved velocity field. The impeller is represented as a localised body force in the spirit of Pericleous & Patel (1987) and is **non-axisymmetric** (a single angular Gaussian "blade") so that the resulting Lagrangian streamlines are chaotic rather than purely toroidal.

![Flow](blob.gif)
![Flow-through bioreactor simulation](sample0_topdown.gif)
![Flow-through bioreactor simulation2](sample0_vertical.gif)

## Quick Start

```bash
# Stage 1 — solve and cache the steady flow (slow, run once)
python scripts/solve_flow.py --out flow_cache.h5

# Stage 2 — species transport on the cached flow (fast, repeated)
python examples/example.py
```

```python
from CooperativeModel import Simulator

# Stage 2 in code: the BO objective wraps Simulator.run() and never re-solves Stage 1.
r = Simulator(
    samples=[[0.05, 0.05, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0]],
    t_final=48.0, grid_shape=(32, 32, 32),
    flow_cache_path='flow_cache.h5',
    ic_mode='uniform',
    device='cuda',
).run()
print(f'L={r.L_final:.2f}, Sn={r.Sn_final:.2f}')
r.gif('sample0.gif')              # mid-z slice, wall boundary contoured

# Well-mixed limit: 1x1x1 grid, no flow → the 8-ODE system.
r = Simulator(samples=[[0.05, 0.05, 0, 0, 25, 25, 25, 25]],
              grid_shape=(1, 1, 1), flow_cache_path=None,
              t_final=72.0).run()
```

See [`examples/example.py`](examples/example.py) and [`examples/find_optima.py`](examples/find_optima.py) for the full runnable scripts.

## Model

### State Variables

| Symbol | Channel | Description |
|--------|---------|-------------|
| $N_1$ | 0 | Population density of strain CoA |
| $N_2$ | 1 | Population density of strain CoB |
| $S_n$ | 2 | Nisin concentration |
| $L$   | 3 | Lactic acid concentration |
| $F_1$ | 4 | Glucose concentration |
| $F_2$ | 5 | Fructose concentration |
| $F_3$ | 6 | Sucrose concentration |
| $F_4$ | 7 | Maltose concentration |

### Well-Mixed ODE (Local Reaction Kinetics)

Each spatial cell evolves according to a cooperative-consortium ODE built on top of [Kong et al. (2018)]. The base structure (Monod growth + diauxic shift + nisin-cooperative production + nisin-protected death) is unchanged; five biological mechanisms have been added on top so that $L_{\text{final}}$ has *interior* optima in the four-sugar input space rather than the single trivial corner-loaded optimum of the original. These additions are listed together in [Mechanisms added on top of Kong et al.](#mechanisms-added-on-top-of-kong-et-al).

The presentation below builds up named intermediates (growth rates, then totals, then production/death rate factors) so that the final ODE system on the right-hand side reads cleanly. The convention is that anything *boxed* on the left of an `:=` is a definition; the final ODE block at the bottom uses only those symbols.

**Step 1 — Per-sugar growth rates** (Haldane for CoA, Monod for CoB):

$$g_{1,i} \;:=\; \mu_{1,i} \, \frac{F_i}{K_{1,i} + F_i + F_i^{2}/K_{\text{inh},i}} \qquad g_{2,i} \;:=\; \mu_{2,i} \, \frac{F_i}{K_{2,i} + F_i}$$

The CoA Haldane form has a maximum at $F_i^{*} = \sqrt{K_{1,i} K_{\text{inh},i}}$ — past this concentration the same sugar that fed CoA starts to inhibit it. With the values in Table 1 the per-sugar peaks sit around $(F_1^{*},F_2^{*},F_3^{*},F_4^{*}) \approx (75, 50, 30, 60)$ g/L, well inside the operating box, which is what makes the lactic-acid landscape multimodal in $(F_1,F_2,F_3,F_4)$.

**Step 2 — Diauxic weighting and totals.** CoA preferentially consumes whichever sugar gives it the fastest individual growth rate, with sharpness $n$:

$$\beta_{1,i} \;:=\; \frac{g_{1,i}^{\,n}}{\sum_{j} g_{1,j}^{\,n}} \qquad \tilde g_{1,\text{tot}} \;:=\; \sum_{i} \beta_{1,i} \, g_{1,i} \qquad g_{2,\text{tot}} \;:=\; \tfrac{1}{4} \sum_{i} g_{2,i}$$

**Step 3 — Lactate product inhibition on CoA total growth.** Accumulated lactate slows CoA but not CoB:

$$g_{1,\text{tot}} \;:=\; \frac{\tilde g_{1,\text{tot}}}{1 + (L/K_{p,L})^{h_L}}$$

The unhatted $g_{1,\text{tot}}$ is what the rest of the equations use. Setting $K_{p,L}\to\infty$ recovers the original uninhibited $\tilde g_{1,\text{tot}}$.

**Step 4 — Net death rate (split-death model).** Death is decomposed into a *microbial* component (bacteriocin-style, attenuated by nisin self-immunity) and a *chemical* component (per-sugar toxicity, NOT protected by nisin):

$$\text{Hill}(x) \;:=\; \frac{x^{h_{\text{tox}}}}{1 + x^{h_{\text{tox}}}} \qquad \tau \;:=\; c_{\text{tox}} \sum_{i} \text{Hill}\!\bigl(F_i / K_{\text{tox},i}\bigr)$$

$$\delta_1 \;:=\; d_{t_1} \cdot \!\left[\,\underbrace{\frac{1}{1 + k_s \, S_n}}_{\text{nisin-protected microbial death}} \;+\; \underbrace{\tau}_{\text{chemical (not nisin-protected)}}\right] \qquad \delta_2 \;:=\; d_{t_2} \cdot \!\left[\frac{1}{1 + k_s \, S_n} + \tau\right]$$

Two biological points:

1. **Per-sugar saturating Hill toxicity.** Each sugar contributes a separate Hill term `Hill(F_i / K_tox,i)` capped at 1, so the four sugars together contribute at most `4·c_tox` to the death-rate multiplier (vs. an unbounded sum in the linear form).  `h_tox > 1` makes the threshold sigmoidal — near-zero below `K_tox,i`, sharp rise above — which keeps moderate-F regions viable while still creating a clear "death zone" at high F.  LAB are sensitive to different sugars in different ways (Maillard / methylglyoxal for glucose, osmotic stress for sucrose, transporter overload for maltose), so the toxicity profile is *species-of-sugar* specific, not a function of $F_{\text{tot}}$.  This per-coordinate asymmetry is what places optima at non-uniform interior $F_i^{*}$ rather than along the diagonal.

2. **Why split-death** (microbial + chemical, rather than a single nisin-protected term). The producer-cell self-immunity machinery — the lipoprotein **NisI** (binds and sequesters nisin at the membrane, blocks pore formation) and the ABC exporter **NisFEG** (removes cell-associated nisin into the medium) — is *bacteriocin-specific*: it neutralises nisin, not generic physico-chemical stresses on the cell envelope and cytoplasm (Stein et al., *J. Biol. Chem.* 2003; AlKhatib et al., *PLoS ONE* 2014).  Putting the per-sugar toxicity inside the same `1/(1 + k_s S_n)` denominator as the microbial term would be saying "any amount of nisin protects against high glucose" — biologically wrong.  Operationally it also collapses the model: once nisin reaches `S_n ≈ 0.005` (with `k_s = 400` this gives `1/(1 + k_s S_n) ≈ 0.33`), the entire death term shrinks to near-zero and high-F runs simply consume all sugar with no penalty, making `L_final` monotone in `F_tot`.  Splitting keeps a chemical-death floor that scales with `F` and exposes the multimodal interior structure of the L-landscape.

**Step 5 — Cooperative nisin production rate** (Kong base × four-sugar co-limitation × CCR):

$$P_m \;:=\; \underbrace{\alpha \, \frac{S_n + r_b}{k_p + S_n} \, F_{\text{tot}} \, \frac{N_1 \, N_2}{k_m + N_2}}_{\text{Kong et al. base}} \cdot \underbrace{\prod_{i} \frac{F_i}{K_{\text{coop}} + F_i}}_{\text{co-limitation}} \cdot \underbrace{\frac{1}{1 + (F_{\text{tot}}/K_{\text{ccr}})^{h_{\text{ccr}}}}}_{\text{CCR}}$$

with $F_{\text{tot}} = \sum_i F_i$. The two multiplicative factors are biologically motivated:

- **Co-limitation** is the *interactive* (multiplicative / Mankin) form, used here for nisin biosynthetic flux rather than for biomass growth: the product is suppressed whenever *any* sugar goes to zero, pushing optima away from the lower bounds of every coordinate. This is the classical product-Monod form for **complementary, non-substitutable inputs** to a single biosynthetic pathway (Megee et al., *Biotechnol. Bioeng.* 1972; Bader, *Biotechnol. Bioeng.* 1978; reviewed in Kovárová-Kovar & Egli, *MMBR* 1998). Saito et al. (2008) categorise this as **Type I (independent) co-limitation** — distinct biochemical roles, all required. Justification for nisin: ribosomal synthesis of the 57-residue precursor, NisB/C-mediated dehydration and cyclisation of Ser/Thr/Cys, and NisT export are all ATP- and cofactor-intensive; sustained throughput requires balanced flux through glycolysis, TCA replenishment, and amino-acid biosynthesis. A diet of one sugar yields imbalanced flux and downregulates secondary metabolism in LAB. The strict Liebig minimum $\min_i F_i / (K_{\text{coop}} + F_i)$ would only track the single most-limiting sugar; the product instead lets all four limitations compound, which is the empirically better fit for complementary inputs (PNAS 2024 dynamic-colimitation framework).
- **CCR** (carbon-catabolite repression) implements CcpA-mediated repression of secondary-metabolite biosynthesis at high carbohydrate load in LAB. It penalises *high* $F_{\text{tot}}$ and pushes optima away from the upper bounds.

Combined with the per-sugar toxicity term in $\delta_j$, these break the "more sugar → more nisin → more bacteria → more $L$" monotonic chain; the result is a multimodal $L$ landscape with **30 distinct local maxima** under the well-mixed scan in `examples/find_optima.py` (eps = 2 separation, `t_final = 72 h`), the majority of which lie strictly inside the operating box.  The current top optimum is `F* ≈ (59.0, 84.9, 0.5, 85.3)` with `L_final ≈ 61.21`; #2 is at `(5.7, 80.1, 92.8, 77.2)` with `L_final ≈ 60.65`.  Each near-edge configuration drops one sugar near zero to escape that sugar's per-sugar toxicity term — `F_3 ≈ 0` is preferred because `K_tox,3 = 55` is the tightest threshold.

**Step 6 — Final ODE system.** Using the symbols defined above:

$$\frac{dN_1}{dt} = (g_{1,\text{tot}} - \delta_1) \, N_1 \qquad \frac{dN_2}{dt} = \tfrac{1}{\sigma} \, g_{2,\text{tot}} \, N_2 - \delta_2 \, N_2$$

$$\frac{dS_n}{dt} = P_m - k_n \, S_n \qquad \frac{dL}{dt} = Y_L \, g_{1,\text{tot}} \, N_1$$

$$\frac{dF_i}{dt} = -\frac{1}{\gamma_{1,i}} \, \beta_{1,i} \, g_{1,i} \, N_1 - \frac{1}{\sigma \, \gamma_{2,i}} \, g_{2,i} \, N_2 \qquad i \in \{1,2,3,4\}$$

### Parameters (Table 1)

| Symbol | Description | Value |
|--------|-------------|-------|
| $\mu_{1,i}$ | Max growth rate of CoA on $F_i$ | $[0.53,\; 0.5,\; 0.6,\; 0.55]$ h$^{-1}$ |
| $\mu_{2,i}$ | Max growth rate of CoB on $F_i$ | $[0.68,\; 0.64,\; 0.61,\; 0.7]$ h$^{-1}$ |
| $d_{t_1},\; d_{t_2}$ | Max death rates | $0.39,\; 0.34$ h$^{-1}$ |
| $\sigma$ | CoB growth scaling factor | $1.5$ |
| $\alpha$ | Nisin production constant | $0.33$ |
| $k_p$ | Nisin production saturation | $8.0$ |
| $r_b$ | Nisin basal production rate | $0.060$ |
| $k_n$ | Nisin degradation rate | $0.065$ h$^{-1}$ |
| $k_s$ | Nisin death inhibition | $400$ |
| $k_m$ | Nisin cooperative saturation | $0.014$ |
| $K_{1,i}$ | Monod const. CoA | $[0.19,\; 0.2,\; 0.18,\; 0.17]$ |
| $K_{2,i}$ | Monod const. CoB | $[0.72,\; 0.75,\; 0.65,\; 0.6]$ |
| $K_{\text{inh},i}$ | Haldane inhibition const. CoA | $[3.0\!\times\!10^4,\; 1.25\!\times\!10^4,\; 5.0\!\times\!10^3,\; 2.1\!\times\!10^4]$ |
| $\gamma_{1,i}$ | Yield const. CoA | $[0.6,\; 0.7,\; 0.72,\; 0.78]$ |
| $\gamma_{2,i}$ | Yield const. CoB | $[0.575,\; 0.625,\; 0.6,\; 0.5]$ |
| $Y_L$ | Lactic acid yield | $1.0$ |
| $n$ | Diauxic shift sharpness | $2.0$ |
| $K_{p,L},\; h_L$ | Lactate product inhibition (CoA growth) | $35,\; 2$ |
| $c_{\text{tox}},\; K_{\text{tox},i},\; h_{\text{tox}}$ | Per-sugar toxicity (saturating Hill) | $0.4,\; [85, 75, 55, 80],\; 4$ |
| $K_{\text{coop}}$ | Co-limitation Hill half-sat. (nisin) | $2.0$ |
| $K_{\text{ccr}},\; h_{\text{ccr}}$ | Catabolite repression (nisin) | $300,\; 2$ |

Note: $k_s$ has been reduced from the original $1.2\!\times\!10^3$ to $400$ in this implementation. The original value places the survival/death boundary at very low $S_n$ which makes the protected/unprotected behaviour effectively bistable across the operating box, collapsing all candidate $L$-trajectories to a single broad plateau. Softening it to 400 moves that boundary into the interior and — combined with the additional kinetic mechanisms above — exposes the multimodal structure of $L_{\text{final}}$ in $(F_1,F_2,F_3,F_4)$.

#### Mechanisms added on top of Kong et al.

The base Kong et al. ODE has a single trivial optimum: max all four sugars. To make the model a useful test bed for spatial / multi-modal optimisation, five biologically-grounded effects are layered on top. Each is gated by a parameter that recovers the original model in a limit, so the additions are opt-in.

| # | Mechanism | Where it acts | Recover original by | Why it's there |
|---|-----------|---------------|---------------------|----------------|
| 1 | Haldane substrate inhibition (CoA) | $g_{1,i}$ growth | $K_{\text{inh},i} \to \infty$ | High sugar inhibits the same uptake it feeds — places per-sugar growth peaks at $\sqrt{K_{1,i}K_{\text{inh},i}}$ inside $(0,100)$. |
| 2 | Lactic-acid product inhibition | $g_{1,\text{tot}}$ | $K_{p,L} \to \infty$ | Standard Luedeking-Piret-style end-product inhibition; couples all four sugars through a shared $L$ pool and breaks the uninhibited yield ceiling. |
| 3 | Per-sugar specific toxicity | death rate $\delta_j$ | $c_{\text{tox}} \to 0$ | Different sugars damage LAB through different routes (Maillard / methylglyoxal for glucose, osmotic stress for sucrose, transporter overload for maltose). Each $K_{\text{tox},i}$ is independent, which breaks the $F_{\text{tot}}$ symmetry and shifts optima away from the equal-mix diagonal. |
| 4 | Multiplicative co-limitation (nisin) | $P_m$ | $K_{\text{coop}} \to 0$ | Bacteriocin biosynthesis treats the four sugars as complementary, non-substitutable inputs to a single secondary-metabolite flux — the product $\prod_i F_i/(K_{\text{coop}}+F_i)$ of four Hill terms is bounded above by 1 and is suppressed whenever any sugar goes to zero. This is Saito et al. (2008) Type I co-limitation in the multiplicative / Mankin form (Megee 1972; Bader 1978), not metabolic cross-feeding between strains. Pushes optima away from the lower bounds. |
| 5 | Carbon catabolite repression (nisin) | $P_m$ | $K_{\text{ccr}} \to \infty$ | CcpA-mediated repression of secondary-metabolite biosynthesis at high carbohydrate load. Penalises high $F_{\text{tot}}$ and pushes optima away from the upper bounds. |

(1) and (3) make the death + growth balance asymmetric across sugars; (2) couples sugars through $L$; (4) and (5) make the nisin-protection signal a non-monotone function of $F_{\text{tot}}$, peaked at moderate values. Together these break the single-optimum structure of the base model: the well-mixed scan in `examples/find_optima.py` finds **30 distinct local maxima** of $L_{\text{final}}(F_1,F_2,F_3,F_4)$ at $\varepsilon = 2$ separation, most of which lie strictly inside $(0,100)^4$.

### 3D Spatial Extension (PDE)

Each species $y_k$ evolves on a $32\times32\times32$ Cartesian grid masked to a cylindrical fluid region (axis = $z$, $H/D = 1$, the cylinder fills the cube):

$$\frac{\partial y_k}{\partial t} = R_k(\mathbf{y}) - \nabla \cdot (\mathbf{v} \, y_k)$$

where:
- $R_k(\mathbf{y})$ — local reaction rate from the ODE above (kinetics is purely local, so the same `compute_reaction_rates` is reused across the well-mixed and 3D regimes).
- $-\nabla \cdot (\mathbf{v} \, y_k)$ — advection by the cached velocity field (conservative first-order upwind on the open-face MAC stencil; flux is killed at any face touching a wall).

**No explicit (Fickian / eddy) diffusion operator.**  Mixing is supplied entirely by chaotic advection from the non-axisymmetric impeller force, plus the small numerical diffusion implicit in first-order upwind on the $32^3$ grid.  This is a deliberate choice: the impeller's chaotic Lagrangian streamlines do the mixing the same way they would in a real stirred tank, and a sub-grid eddy-diffusivity term would just smear out the structure that the projection chain takes care to preserve.

**Open-face MAC stencil.** Both the Stage-1 projection and the Stage-2 advection use the same forward-difference divergence with face-flux indicators $\text{op}_{xp}[i] = \text{mask}[i]\cdot\text{mask}[i{+}1]$.  Because the projection drives this exact divergence to zero, a spatially uniform passive scalar is preserved *exactly* under the converged flow (verified in `tests/`).  Mass over the fluid region is conserved to ODE-solver tolerance.

**Time integration** — Tsitouras 5(4) adaptive Runge–Kutta with dense output via cubic Hermite interpolation and built-in non-negativity clamping. The advective CFL is automatic:

$$\Delta t_{\text{adv}} < \frac{\Delta x}{|\mathbf{v}|_{\max}}$$

Reaction stiffness is handled by the adaptive controller.

### Stage 1 — Steady NS solve and HDF5 cache

The cylindrical-tank flow is computed once by `solve_steady_flow` (in `flow_3d.py`) and cached to disk by `scripts/solve_flow.py`. The Bayesian-optimisation loop loads the cache and **never** re-solves the flow.

**Geometry.** Closed cylinder ($H/D = 1$) inscribed in a unit cube; the wall mask is built by `cylinder_mask`. No inlet, no outlet — the previous 2D `flow_through` mode is removed, since the spec is a closed vessel.

**Forcing — non-axisymmetric impeller body force.** Following the body-force impeller approach of Pericleous & Patel (1987), the impeller is represented as a localised tangential body force rather than a no-slip moving boundary. The new ingredient compared to the classical axisymmetric form is a single angular Gaussian "blade":

$$\mathbf{f}(r,\theta,z) \;=\; F_0 \,\chi_{rz}(r,z)\,\chi_\theta(\theta)\, \hat{\boldsymbol{\theta}}$$

$$\chi_{rz}(r,z) = \exp\!\left(-\frac{(r-r_{\text{imp}})^2}{2\sigma_r^2} - \frac{(z-z_{\text{imp}})^2}{2\sigma_z^2}\right) \qquad \chi_\theta(\theta) = \exp\!\left(-\frac{d_{\text{circ}}(\theta,\theta_0)^2}{2\sigma_\theta^2}\right)$$

with $d_{\text{circ}}(\theta,\theta_0) = \min(|\theta-\theta_0|,\,2\pi-|\theta-\theta_0|)$. The angular blade breaks the toroidal $\theta$-symmetry of the otherwise axisymmetric forcing. This azimuthal asymmetry is what unlocks **chaotic Lagrangian streamlines** characteristic of real stirred tanks; it is verified by `tests/test_asymmetry.py`.

**Solver.** Chorin fractional-step projection on a co-located grid:

1. **Predictor** — explicit Euler with first-order upwind $(\mathbf{u}\cdot\nabla)\mathbf{u}$, a 7-point Laplacian for $\nu\nabla^2\mathbf{u}$, and the impeller body force.
2. **Pressure Poisson** — 200 red–black Gauss–Seidel sweeps on the **open-face FV Laplacian** with embedded-boundary Neumann BCs.  The right-hand side uses the **open-face MAC divergence** of $\mathbf{u}^*$ so that the discrete chain `div_open ∘ grad_open = lap_open` holds exactly, making the corrector drive `div_open(u_new)` to zero at every fluid cell — including those adjacent to walls.  Pressure is anchored to fluid-mean-zero **after every red–black sweep** (inside the GS loop, not just at the end of each outer step) so warm-starting from previous iterations cannot let the unconstrained Neumann constant drift.
3. **Corrector** — `u_new = u* − Δt · grad_open(p)`, then re-zero in walls.

Two metrics are tracked: a step-size residual $\lVert\mathbf{u}^{n+1}-\mathbf{u}^n\rVert_\infty / \max(\lVert\mathbf{u}^n\rVert_\infty,\varepsilon)$ used as the termination criterion, and a periodic **NS residual** $\lVert -(\mathbf{u}\!\cdot\!\nabla)\mathbf{u} + \nu\nabla^2\mathbf{u} + \mathbf{f} - \nabla p\rVert_2$ used as a sanity check on the converged momentum balance.

After the outer loop converges, a single **unpreconditioned CG pass** on `−lap_open` drives $|\mathrm{div\_open}(\mathbf{u})|$ to roundoff (typical 50–300 iterations to relative residual $10^{-12}$ at $32^3$); on the production cache the final divergence is $\approx 7.5\times10^{-15}$.  This step is what guarantees a uniform passive scalar is preserved exactly under Stage-2 advection.

**HDF5 cache layout.** `save_flow` writes datasets `/u`, `/v`, `/w`, `/mask` along with attributes that fully reproduce the run (`Nx, Ny, Nz, Lx, Ly, Lz, F0, r_imp, z_imp, sigma_r, sigma_z, theta_0, sigma_theta, nu, dt, n_iters, converged_residual, dtype, code_version, final_cg_iters, final_cg_relres`). `code_version` is **derived from a SHA-256 of the operator-defining functions** (Laplacian, gradient, divergence, advection, pressure GS, CG cleanup, outer driver), so any stencil change auto-bumps the version stamped into the cache — no manual literal to forget. `load_flow` is bit-identical on round-trip (verified by `tests/test_flow_solve_smoke.py`).

### Stage 2 — Species transport on the cached flow

`Simulator.run()` calls `load_flow(path)` once per BO evaluation, builds a uniform IC over fluid cells, and integrates the 3D PDE above. Every voxel is a Delafosse-style compartment; inter-compartment fluxes come from the cached velocity field. With `flow_cache_path=None` the simulator uses zero velocity and an all-fluid mask, recovering the well-mixed (0D-equivalent) limit — this is what `examples/example_ode.py` runs.

## Simulator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N1, N2` | 0.05 | Initial population densities (CoA, CoB) |
| `Sn, L` | 0.0 | Initial nisin / lactic acid |
| `F1, F2, F3, F4` | 25, 25, 25, 25 | Initial sugars (glucose, fructose, sucrose, maltose) |
| `samples` | `None` | Optional `[B, 8]` IC tensor (overrides per-channel args) |
| `t_final` | 72.0 | Integration time [hours] |
| `n_output` | 145 | Number of output time points |
| `grid_shape` | `(32, 32, 32)` | `(Nz, Ny, Nx)`. Use `(1, 1, 1)` for the well-mixed limit. |
| `flow_cache_path` | `'flow_cache.h5'` | HDF5 cache from `scripts/solve_flow.py`. `None` ⇒ zero flow + all-fluid mask. |
| `mixing_scale` | 1.0 | Multiplier on the cached velocity field (e.g. `2.0` halves the eddy turnover time without re-solving Stage 1). |
| `ic_mode` | `'uniform'` | `'uniform'` distributes each species across all fluid cells; `'octant'` localises the IC to one octant of the cylinder (used to study advective dieback in `examples/example.py`). |
| `ic_octant` | `(1, 1, 1)` | Sign of `(x, y, z)` selecting the octant when `ic_mode='octant'`. |
| `device` | `'cpu'` | `'cpu'` or `'cuda'` |

## Results

`Simulator.run()` returns a `SimResults` object. Spatial averages are taken over **fluid cells only** (the wall mask does not dilute the reported means):

```python
r.L_final           # final lactic acid (fluid average)
r.Sn_final          # final nisin (fluid average)
r.elapsed           # wall-clock time [seconds]
r.final_values()    # dict of all 8 channels at final time
r.spatial_average() # numpy array [T, 8] (or [B, T, 8] for B>1)

r.gif('out.gif')        # mid-z slice animation, wall boundary contoured
r.snapshot('out.png')   # mid-z heatmap at final time
r.timeseries('out.png') # fluid-averaged time series
```

## Installation

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# CPU only
pip install -e .

# GPU (CUDA 12.6)
pip install -e .[gpu] --extra-index-url https://download.pytorch.org/whl/cu126
```

To use GPU, pass `device='cuda'` to the `Simulator`:

```python
r = Simulator(F4=100.0, device='cuda').run()
```

## References

1. Kong, W., Meldgin, D. R., Collins, J. J., and Lu, T. (2018). Designing microbial consortia with defined social interactions. *Nature Chemical Biology*, 14(8), 821-829.
2. Pericleous, K. A., and Patel, M. K. (1987). The modelling of tangential and axial agitators in chemical reactors. *PCH PhysicoChemical Hydrodynamics*, 8(2), 105-123. — body-force impeller model used in Stage 1.
3. Delafosse, A., Collignon, M.-L., Calvo, S., Delvigne, F., Crine, M., Thonart, P., and Toye, D. (2014). CFD-based compartment model for description of mixing in bioreactors. *Chemical Engineering Science*, **106**, 76-85. — compartment-network framing used in Stage 2.
4. Chorin, A. J. (1968). Numerical solution of the Navier–Stokes equations. *Mathematics of Computation*, 22(104), 745-762. — fractional-step projection method.
5. Oliveira, A. P., Nielsen, J., and Forster, J. (2005). Modeling *Lactococcus lactis* using a genome-scale flux model. *BMC Microbiology*, 5(1), 39.
6. Marsland, R., Cui, W., Goldford, J., and Mehta, P. (2020). The Community Simulator: A Python package for microbial ecology. *PLoS ONE*, 15(3), e0230430.
7. Tsitouras, C. (2011). Runge-Kutta pairs of order 5(4). *Computers & Mathematics with Applications*, 62(2), 770-775.
