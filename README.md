# CooperativeModel

2D reaction-diffusion simulation of a two-strain cooperative microbial consortium (CoA + CoB), implemented in PyTorch.

![Flow-through bioreactor simulation](flow_through.gif)

## Quick Start

```python
from CooperativeModel import Simulator

# Flow-through reactor (inlet upper-left, outlet lower-right)
r = Simulator(
    N1=0.05, N2=0.05, Sn=0.0, L=0.0, F1=0.0, F2=0.0, F3=0.0, F4=100.0,
    mode='flow_through', t_final=72.0, grid_size=100,
    omega=-0.25, diffusion_scale=0.1, flow_rate=5.0
).run()
print(f'L={r.L_final:.2f}, Sn={r.Sn_final:.2f}')
r.gif('flow_through.gif')
```

See [`examples/example.py`](examples/example.py) for the full runnable script.

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

Each spatial cell evolves according to the cooperative consortium ODE from [Kong et al. (2018)]:

**Population dynamics:**

$$\frac{dN_1}{dt} = g_{1,\text{total}} \, N_1 - \frac{d_{t_1}}{1 + k_s \, S_n} \, N_1$$

$$\frac{dN_2}{dt} = \frac{1}{\sigma} \, g_{2,\text{total}} \, N_2 - \frac{d_{t_2}}{1 + k_s \, S_n} \, N_2$$

**Nisin dynamics:**

$$\frac{dS_n}{dt} = P_m - k_n \, S_n$$

$$P_m = \alpha \, \frac{S_n + r_b}{k_p + S_n} \left(\sum_{i=1}^{4} F_i\right) \frac{N_1 \, N_2}{k_m + N_2}$$

**Lactic acid production:**

$$\frac{dL}{dt} = Y_L \cdot g_{1,\text{total}} \, N_1$$

**Nutrient consumption:**

$$\frac{dF_i}{dt} = -\frac{1}{\gamma_{1,i}} \, g_{1,i} \, \beta_{1,i} \, N_1 - \frac{1}{\gamma_{2,i} \, \sigma} \, g_{2,i} \, N_2 \qquad i \in \{1,2,3,4\}$$

**Growth rates** — Monod kinetics on each nutrient:

$$g_{j,i} = \mu_{j,i} \, \frac{F_i}{K_{j,i} + F_i}$$

**Diauxic shift** (CoA only) — preferential consumption with sharpness $n$:

$$\beta_{1,i} = \frac{g_{1,i}^{\,n}}{\sum_{j=1}^{4} g_{1,j}^{\,n}}$$

$$g_{1,\text{total}} = \sum_{i=1}^{4} \beta_{1,i} \, g_{1,i}$$

**CoB total growth** — simple average:

$$g_{2,\text{total}} = \frac{1}{4} \sum_{i=1}^{4} g_{2,i}$$

**Death rate inhibition** — nisin reduces death:

$$I_{t,S_n,j} = \frac{d_{t_j}}{1 + k_s \, S_n}$$

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
| $k_s$ | Nisin death inhibition | $1.2 \times 10^3$ |
| $k_m$ | Nisin cooperative saturation | $0.014$ |
| $K_{1,i}$ | Monod const. CoA | $[0.19,\; 0.2,\; 0.18,\; 0.17]$ |
| $K_{2,i}$ | Monod const. CoB | $[0.72,\; 0.75,\; 0.65,\; 0.6]$ |
| $\gamma_{1,i}$ | Yield const. CoA | $[0.6,\; 0.7,\; 0.72,\; 0.78]$ |
| $\gamma_{2,i}$ | Yield const. CoB | $[0.575,\; 0.625,\; 0.6,\; 0.5]$ |
| $Y_L$ | Lactic acid yield | $1.0$ |
| $n$ | Diauxic shift sharpness | $2.0$ |

### 2D Spatial Extension (PDE)

Each field $y_k$ evolves as a reaction-diffusion-advection PDE:

$$\frac{\partial y_k}{\partial t} = R_k(\mathbf{y}) + \nabla \cdot (D_k \, \nabla y_k) - \nabla \cdot (\mathbf{v} \, y_k)$$

where:
- $R_k(\mathbf{y})$ — local reaction rate from the ODE above
- $\nabla \cdot (D_k \, \nabla y_k)$ — Fickian diffusion (8-point stencil, no-flux BCs)
- $-\nabla \cdot (\mathbf{v} \, y_k)$ — advection by a divergence-free velocity field (upwind scheme)

**Diffusion operator** — 8-direction stencil with face-averaged coefficients:

$$\left[\nabla \cdot (D \nabla c)\right]_{i,j} \approx \frac{1}{\Delta x^2} \sum_{k=1}^{8} w_k \, \bar{D}_k \, (c_k - c_{i,j})$$

where $w_k = 1$ for cardinal neighbours and $w_k = 1/\sqrt{2}$ for diagonals, and $\bar{D}_k = (D_{i,j} + D_k)/2$.

**Advection operator** — conservative first-order upwind scheme. We discretise $-\nabla \cdot (\mathbf{v}\, c)$ in flux form. For each cell face, the face velocity is averaged from the two adjacent cell centres, and the upwind concentration is selected based on the flow direction:

$$\Phi^x_{i+\frac{1}{2},j} = \bar{v}^x_{i+\frac{1}{2}} \cdot \begin{cases} c_{i,j} & \text{if } \bar{v}^x_{i+\frac{1}{2}} > 0 \\ c_{i+1,j} & \text{otherwise} \end{cases}$$

where $\bar{v}^x_{i+\frac{1}{2}} = \tfrac{1}{2}(v^x_{i,j} + v^x_{i+1,j})$, and analogously for the $y$-direction. The advection contribution is then:

$$-\left[\nabla \cdot (\mathbf{v}\, c)\right]_{i,j} \approx -\frac{\Phi^x_{i+\frac{1}{2},j} - \Phi^x_{i-\frac{1}{2},j}}{\Delta x} - \frac{\Phi^y_{i,j+\frac{1}{2}} - \Phi^y_{i,j-\frac{1}{2}}}{\Delta y}$$

Upwind is chosen for stability: it introduces numerical diffusion that damps oscillations, and is monotone (preserves non-negativity). This is appropriate here because the physical diffusion operator already provides the dominant smoothing; the advection scheme only needs to be stable, not high-order.

**Diffusion coefficients:**

| Species | $D$ \[$cm^2/h$\] |
|---------|-----------|
| $N_1, N_2$ (bacteria) | $10^{-6}$ |
| $S_n, L$ (small molecules) | $5 \times 10^{-4}$ |
| $F_1$–$F_4$ (sugars) | $10^{-4}$ |

**Velocity field** — rigid-body vortex from a polynomial stream function:

$$\psi(x,y) = A \left(a^2 - u^2\right)\left(b^2 - v^2\right)$$

where $u = x - L_x/2$, $v = y - L_y/2$, $a = L_x/2$, $b = L_y/2$, and $A = \omega / (2ab)$. This vanishes on all four walls (no-penetration, divergence-free). The resulting velocity near the centre is:

$$v_x \approx -\omega\,(y - L_y/2), \qquad v_y \approx \omega\,(x - L_x/2)$$

i.e. rigid-body rotation with speed $|\mathbf{v}| \approx \omega \, r$.

**Time integration** — Tsitouras 5(4) adaptive Runge-Kutta with dense output via cubic Hermite interpolation and built-in non-negativity clamping.

**CFL condition** — the maximum time step is automatically limited to satisfy both diffusion and advection stability:

$$\Delta t_{\text{diff}} < \frac{\Delta x^2}{2 \, D_{\max} \cdot d} \qquad \Delta t_{\text{adv}} < \frac{\Delta x}{|\mathbf{v}|_{\max}}$$

where $d = 2$ is the spatial dimension. The solver uses $h_{\max} = 0.4 \cdot \min(\Delta t_{\text{diff}},\, \Delta t_{\text{adv}})$. This is computed automatically from the grid spacing, diffusion coefficients, and velocity field — no user tuning required.

### Flow-Through Mode

In `mode='flow_through'`, source/sink terms are added at corner zones:

$$\left.\frac{\partial y_k}{\partial t}\right|_{\text{inlet}} \mathrel{+}= \phi \, (c_{\text{feed},k} - y_k)$$

$$\left.\frac{\partial y_k}{\partial t}\right|_{\text{outlet}} \mathrel{-}= \phi \, y_k$$

where $\phi$ is the `flow_rate` parameter. The feed contains only nutrients (no microbes). Bacteria come from random initial inoculation.

## Simulator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N1, N2` | 0.05 | Initial population densities (CoA, CoB) |
| `Sn, L` | 0.0 | Initial nisin / lactic acid |
| `F1, F2, F3, F4` | 0, 0, 0, 100 | Initial nutrients (glucose, fructose, sucrose, maltose) |
| `mode` | `'flow_through'` | `'batch'` or `'flow_through'` |
| `t_final` | 72.0 | Integration time [hours] |
| `n_output` | 145 | Number of output time points |
| `grid_size` | 100 | Spatial grid points per side |
| `omega` | -0.25 | Vortex angular velocity [rad/h]. Negative = clockwise |
| `diffusion_scale` | 0.1 | Multiplier on diffusion coefficients |
| `flow_rate` | 5.0 | Inlet/outlet turnover rate [h$^{-1}$] (flow_through only) |
| `device` | `'cpu'` | `'cpu'` or `'cuda'` |

## Results

`Simulator.run()` returns a `SimResults` object:

```python
r.L_final           # final lactic acid (spatial average)
r.Sn_final          # final nisin (spatial average)
r.elapsed           # wall-clock time [seconds]
r.final_values()    # dict of all 8 channels at final time
r.spatial_average() # numpy array [T, 8]

r.gif('out.gif')           # all-channel animated GIF with Sn/L curves
r.snapshot('out.png')      # spatial heatmap at final time
r.timeseries('out.png')    # spatially-averaged time series
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
2. Oliveira, A. P., Nielsen, J., and Forster, J. (2005). Modeling *Lactococcus lactis* using a genome-scale flux model. *BMC Microbiology*, 5(1), 39.
3. Marsland, R., Cui, W., Goldford, J., and Mehta, P. (2020). The Community Simulator: A Python package for microbial ecology. *PLoS ONE*, 15(3), e0230430.
4. Tsitouras, C. (2011). Runge-Kutta pairs of order 5(4). *Computers & Mathematics with Applications*, 62(2), 770-775.
