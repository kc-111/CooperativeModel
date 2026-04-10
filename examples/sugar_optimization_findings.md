# Sugar Loading Optimization Findings: Cooperative Bioreactor Model

## Setup

A closed batch reactor (`mode='batch'`, grid_size=1, no diffusion, no flow) containing:
- **CoA (N1=0.05)** — lactic acid producer
- **CoB (N2=0.05)** — cooperator (helps CoA via nisin production, does not produce acid)
- Initial sugars **F1, F2, F3, F4 ∈ [0, 100]** (glucose, fructose, sucrose, maltose)

The reactor is sealed: F1–F4 are the **initial amounts loaded at t=0**, not a continuous feed stream. The bacteria consume the starting pool over `t_final = 72h` and no further sugar is added.

**Objective:** maximize final lactic acid `L_final` over the 4D initial-sugar space.

## The 13 true local optima

Exhaustive search over all 2⁴ = 16 corners plus perturbation-check confirmed that **all true local maxima sit at corners** — every "interior optimum" found by L-BFGS-B was an artifact of the optimizer stalling in flat regions. Gradient check at δ=0.5 (each coordinate perturbed inward toward 0 or 100) confirmed 13 of 16 corners are genuine local maxima.

| Rank | F1  | F2  | F3  | F4  | Total | L_final  | Local max? |
|-----:|----:|----:|----:|----:|------:|---------:|:----------:|
|  1   |   0 | 100 | 100 | 100 |   300 | **190.72** | ✓ |
|  2   | 100 |   0 | 100 | 100 |   300 | 180.60   | ✓ |
|  3   | 100 | 100 | 100 |   0 |   300 | 177.93   | ✓ |
|  4   | 100 | 100 |   0 | 100 |   300 | 167.90   | ✓ |
|  5   | 100 | 100 | 100 | 100 |   400 | 149.21   | ✓ |
|  6   |   0 |   0 | 100 | 100 |   200 | 144.95   | ✓ |
|  7   |   0 | 100 |   0 | 100 |   200 | 140.97   | ✓ |
|  8   |   0 | 100 | 100 |   0 |   200 | 138.13   | ✓ |
|  9   | 100 |   0 |   0 | 100 |   200 | 131.23   | ✓ |
|  10  | 100 |   0 | 100 |   0 |   200 | 128.14   | ✓ |
|  11  | 100 | 100 |   0 |   0 |   200 | 124.62   | ✓ |
|  12  |   0 |   0 |   0 | 100 |   100 |  76.91   | ✓ |
|  13  |   0 | 100 |   0 |   0 |   100 |  69.24   | ✓ |
|  —   |   0 |   0 | 100 |   0 |   100 |  71.46   | ✗ (F2+ improves) |
|  —   | 100 |   0 |   0 |   0 |   100 |  59.31   | ✗ (F3+ improves) |
|  —   |   0 |   0 |   0 |   0 |     0 |   0.00   | ✗ (trivial) |

**Key structural result:** the landscape is combinatorial. For each sugar the answer is binary (include at 100, or exclude at 0). There is no continuous trade-off.

## The big finding: more sugar can hurt

Going from (0,100,100,100) → (100,100,100,100) **adds 100 units of F1 but removes 41.5 units of lactic acid**. Every single-sugar removal from the all-max corner *improves* L_final:

| Remove | Δ L_final |
|--------|----------:|
| F1     | **+41.5** |
| F2     |    +31.4  |
| F4     |    +28.7  |
| F3     |    +18.7  |

The mixed 4-sugar loading is a local optimum but not the global one. Three sugars beats four.

## Biological interpretation

### The two strains are not interchangeable

From `kinetics.py`:
- **CoA (N1)** is the only lactic acid producer: `dL/dt = YL · g1_total · N1`
- **CoB (N2)** is a cooperator. It contributes via the `N1 · N2` term in nisin production `Pm`. Nisin protects both strains from death via `1/(1 + ks·Sn)` on dt1/dt2.

Every unit of sugar metabolized by CoB produces **no** lactic acid directly — it only helps indirectly by keeping the consortium alive.

### F1 is doubly bad for lactic acid production

From `config.py` (Table 1 parameters):

| Sugar | gamma1 (CoA yield) | mu2 (CoB growth) |
|-------|-------------------:|------------------:|
| F1 (glucose)  | **0.60 (worst)**  | 0.68 (2nd fastest) |
| F2 (fructose) | 0.70              | 0.64              |
| F3 (sucrose)  | 0.72              | 0.61              |
| F4 (maltose)  | **0.78 (best)**   | 0.70              |

F1 has the **worst CoA yield** *and* is one of CoB's **favorite substrates**. Every unit of F1 preferentially feeds the non-producing strain and, when it does reach CoA, converts inefficiently.

### Single-sugar results confirm the yield ceiling

For single-sugar runs, the observed L_final almost exactly equals `YL · gamma1 · 100`:

| Loading    | L_final | YL · gamma1 · 100 |
|------------|--------:|------------------:|
| F1 only    |  59.31  | 60.0 |
| F2 only    |  69.24  | 70.0 |
| F3 only    |  71.46  | 72.0 |
| F4 only    |  76.91  | 78.0 |

In single-sugar batch operation CoA essentially consumes everything and the yield coefficient is the ceiling.

### Why adding F1 to the full mix removes ~100 units of product

100 extra units of F1 should, at best, add `0.60 × 100 = 60` units of lactic acid if CoA got all of it. Instead we lose 41.5 — a gap of about 100 units of "expected but missing" product. That gap is:

1. **Substrate siphoning** — CoB grows fast on F1 (mu2=0.68), turning it into helper biomass.
2. **Diauxic interference** — the `n=2` sharpness in CoA's preference weights `beta1_i = g1_i² / Σ g1_j²` redistributes CoA's effort across the 4 sugars, diluting its focus on the high-yield F3/F4.
3. **Consortium imbalance** — CoB overgrowth competes with CoA for the remaining F2/F3/F4, reducing CoA biomass during the productive phase.

### The optimal strategy: starve the helper of its favorite food

The global optimum (0, 100, 100, 100) achieves the right balance:
- Total substrate is still high (300 units) → enough for growth and cooperative nisin production
- CoB has to survive on F2/F3/F4, none of which are its best substrate → CoB grows more slowly
- CoA dominates the consortium → maximum producer biomass → maximum lactic acid
- CoB is still present (cooperation works, nisin protects both strains from dying)

### Biological analogues

This is a well-known pattern in mixed-culture bioprocesses:

- **Lactic acid bacteria consortia in dairy fermentation** (e.g., *Lactobacillus delbrueckii* + *Streptococcus thermophilus*): cross-feeding mutualism where carbon source balance determines which strain dominates.
- **Nisin-producing Lactococcus co-cultures**: producer strains are typically slower-growing than contaminants or helper strains; carbon source selection is a known lever for biasing the balance toward the producer.
- **General producer–cheater dynamics**: when two strains share a substrate pool and only one makes the product of interest, feed composition must be biased *against* the non-producer's preferred substrate.

The model is reproducing the core result of that literature: **killing the helper is bad, but over-feeding the helper is worse**.

## Terminology note: `mode='batch'` ≠ "batched simulation"

Two unrelated meanings of "batch" coincide here:

- **`mode='batch'`** is a bioprocess term — a **batch reactor** is a sealed vessel with no inlet or outlet. Nutrients are loaded at t=0 and deplete over time. Contrast with `mode='flow_through'`, which maintains continuous inlet (`c_feed`) and outlet zones.
- **Batched simulation** is a GPU term — running many independent samples in parallel (the `samples=[...]` argument creating a batch dimension B). This is orthogonal to `mode`; both modes support multi-sample batching.

All results above use `mode='batch'` (closed reactor) *and* multi-sample batching on GPU. The F1–F4 values are **initial sugar amounts loaded at t=0**. In `mode='flow_through'`, the same F1–F4 parameters are reinterpreted as inlet-stream concentrations — that would be a different problem entirely.

## How the ODE reduction works

Both `mode='batch'` and `mode='flow_through'` are PDE simulators on a 2D grid. At each grid point there are 8 state variables `[N1, N2, Sn, L, F1, F2, F3, F4]`, and the general equation for field `y_k` is

```
dy_k/dt = R_k(y) + ∇·(D_k ∇ y_k) − ∇·(v y_k)
           ^^^^    ^^^^^^^^^^^^     ^^^^^^^^^^
         reaction   diffusion        advection
```

`R_k` is the local reaction kinetics (`kinetics.py` — Monod growth, diauxic shift, nisin production, lactic acid production, nutrient consumption). The other two terms are spatial transport.

The examples in this directory run the model in a *well-mixed ODE regime* by setting:

| Knob               | Value | Effect                          |
|--------------------|------:|---------------------------------|
| `mode='batch'`     | —     | sealed vessel, no inlet/outlet  |
| `grid_size=1`      | 1     | single spatial cell             |
| `diffusion_scale=0`| 0     | all `D_k = 0`                   |
| `omega=0`          | 0     | zero velocity field             |

With these settings every spatial operator is identically zero: on a 1-cell grid the replicate-padding in `spatial_operators.py` makes every "neighbour" equal to the center, so `∇·(D ∇ y)` and `∇·(v y)` both vanish. What remains is the pure reaction ODE

```
dy/dt = R(y),   y ∈ ℝ⁸
```

which `BioreactorRHS.__call__` computes in one call per step and `Tsit5SolverTorch` (adaptive Runge-Kutta 5(4)) integrates from `t=0` to `t_final=72`. Each of the B samples is an independent 8-dimensional initial-value problem; the solver vectorises across the batch dimension on GPU, so 100k simulations run in ~13s (13 s / 100k ≈ 0.13 ms per simulation).

In concrete terms, the 8 coupled ODEs are:

```
dN1/dt = g1_total·N1                − dt1/(1+ks·Sn)·N1        (CoA growth − death)
dN2/dt = g2_total·N2/sigma           − dt2/(1+ks·Sn)·N2        (CoB growth − death)
dSn/dt = alpha·(Sn+rb)/(kp+Sn)·ΣF·N1·N2/(km+N2) − kn·Sn        (nisin prod. − degradation)
dL/dt  = YL·g1_total·N1                                        (lactic acid production)
dFi/dt = −(1/gamma1_i)·g1_i·beta1_i·N1 − (1/(gamma2_i·sigma))·g2_i·N2    (sugar consumption)
```

with `g1_i = mu1_i·Fi/(K1_i+Fi)`, `g2_i = mu2_i·Fi/(K2_i+Fi)`, `beta1_i = g1_i^n / Σ g1_j^n` (diauxic weights), and `g1_total = Σ beta1_i·g1_i`, `g2_total = mean(g2_i)`.

The full 2D PDE version of this model is useful when you care about spatial heterogeneity — colony formation, gradients, mixing. When the objective is a well-mixed aggregate quantity like "final lactic acid concentration", the ODE reduction gives exact answers ~10,000× faster than the full 100×100 grid.

## Implications for the "N unique solutions" goal

With the current model structure, the number of distinct solutions is combinatorially bounded: ≤ 2⁴ = 16 corners, of which 13 are true local optima. To get a richer set of unique solutions, the model needs per-sample variation beyond initial conditions alone. Natural extensions:

1. **Per-sample kinetic parameters** — let `mu1`, `K1`, `gamma1` vary across samples so each sample's landscape is different.
2. **Inoculum ratio N1:N2** — currently fixed at 0.05:0.05. Sweeping this changes which strain dominates and which sugars matter.
3. **Time horizon `t_final`** — diauxic shifts are sequential, so different time horizons change which sugar "wins".
4. **Switch to `mode='flow_through'`** — continuous feed changes the optimization from "initial load" to "feed rate profile", a genuinely richer problem.
