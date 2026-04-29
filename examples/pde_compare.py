"""Run representative optima through the PDE simulator and compare to ODE.

Picks 5 candidates from the latest find_optima.py output (top interior, a far
interior cluster, a low-F4 interior, top boundary, F1=100 boundary) and runs:

  * mode='batch', grid_size=1            — pure ODE  (find_optima.py setting)
  * mode='batch', grid_size=50, D=0.1    — full PDE, uniform IC, no flow.
        With spatially uniform IC and no advection there are no gradients,
        so the answer must equal the ODE up to discretization noise.  This
        is a SANITY check on the PDE solver.  (diffusion_scale=1.0 makes
        the system numerically stiff for the explicit TSit5 solver; 0.1
        is the physical default and is stable.)
  * mode='flow_through', grid_size=50,
        omega=-0.25, D=0.1, flow_rate=5.0 — vortex-stirred CSTR with inlet
        and outlet zones.  Sugars are continuously fed at the inlet
        concentrations F* and continuously washed out, so L_final no longer
        means "yield from a fixed sugar pool" — it means "steady-state
        lactate accumulation against washout."  This re-ranks the optima
        because the constant-feed regime kills product inhibition (L caps
        at the wash-out balance) and removes substrate depletion as a
        differentiator.

Why we expect the rankings to change in flow_through:
  - In batch mode a corner like (100, 100, 100, 100) loses because of CCR,
    osmotic stress, and product inhibition once L builds up.
  - In flow_through mode the inlet keeps F1..F4 close to their setpoints
    forever, but the outlet also keeps L bounded.  The substrate-toxicity
    + nisin protection trade-off becomes the dominant axis, so interior
    points around F_i ~ K_tox_i should win even more decisively.
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from CooperativeModel import Simulator


# Representative optima from find_optima.py (latest run, see /tmp/find_optima_v3.log)
CANDIDATES = [
    # (label, F1, F2, F3, F4, L_ODE_reported, kind)
    ('#1  top INT ',     58.59, 91.77, 17.22, 68.32, 64.919, 'INT'),
    ('#2  far INT ',     92.41, 28.85,  7.77, 95.90, 64.689, 'INT'),
    ('#11 low-F4 INT',   87.41, 94.17, 28.40, 16.79, 63.545, 'INT'),
    ('#18 top BDY ',     66.34, 44.16, 32.92, 99.86, 63.046, 'BDY'),
    ('#20 F1=100 BDY',  100.00, 36.09,  6.25, 75.97, 62.938, 'BDY'),
]


def make_samples(F_list):
    """Build [B, 8] IC tensor: N1=N2=0.05, Sn=L=0, F=F_list."""
    samples = np.zeros((len(F_list), 8))
    samples[:, 0] = 0.05
    samples[:, 1] = 0.05
    samples[:, 4:] = F_list
    return samples.tolist()


F_arr = np.array([[c[1], c[2], c[3], c[4]] for c in CANDIDATES])
samples = make_samples(F_arr)


# ── Mode 1: pure ODE ────────────────────────────────────────────────────────
print('=' * 80)
print('Mode 1: ODE (mode=batch, grid_size=1, no diffusion, no flow)')
print('=' * 80)
t0 = time.time()
r_ode = Simulator(
    samples=samples, mode='batch', t_final=72.0,
    grid_size=1, omega=0.0, diffusion_scale=0.0, device='cuda',
).run()
print(f'  {time.time()-t0:.1f}s, L_final = {np.array(r_ode.L_final)}')


# ── Mode 2: PDE batch, uniform IC, no advection (sanity check) ──────────────
print('\n' + '=' * 80)
print('Mode 2: PDE batch, uniform IC (mode=batch, grid_size=50, D=0.1, omega=0)')
print('  → no gradients exist → must agree with ODE')
print('=' * 80)
t0 = time.time()
r_pde_uniform = Simulator(
    samples=samples, mode='batch', t_final=72.0,
    grid_size=50, omega=0.0, diffusion_scale=0.1, device='cuda',
).run()
print(f'  {time.time()-t0:.1f}s, L_final = {np.array(r_pde_uniform.L_final)}')


# ── Mode 3: PDE flow-through with vortex (full spatial + advection) ─────────
print('\n' + '=' * 80)
print('Mode 3: PDE flow-through (grid_size=50, omega=-0.25, D=0.1, flow_rate=5)')
print('  → inlet feeds F*, outlet washes out lactate; vortex stirs the field')
print('=' * 80)
t0 = time.time()
r_pde_flow = Simulator(
    samples=samples, mode='flow_through', t_final=72.0,
    grid_size=50, omega=-0.25, diffusion_scale=0.1, flow_rate=5.0,
    device='cuda',
).run()
print(f'  {time.time()-t0:.1f}s, L_final = {np.array(r_pde_flow.L_final)}')


# ── Comparison table ────────────────────────────────────────────────────────
print('\n' + '=' * 100)
print('  Comparison: L_final under three model fidelities')
print('=' * 100)
print(f'{"label":<18} {"F1":>6} {"F2":>6} {"F3":>6} {"F4":>6}   '
      f'{"L_ODE":>8} {"L_uniform":>10} {"L_flow":>8}   {"Δ_uniform":>10} {"Δ_flow":>9}')
print('-' * 100)

L_ode = np.atleast_1d(r_ode.L_final)
L_unf = np.atleast_1d(r_pde_uniform.L_final)
L_flw = np.atleast_1d(r_pde_flow.L_final)

# Re-rank by flow result to see if ordering survives
order_ode  = np.argsort(L_ode)[::-1]
order_flow = np.argsort(L_flw)[::-1]

for i, (lbl, F1, F2, F3, F4, _, kind) in enumerate(CANDIDATES):
    d_unf = L_unf[i] - L_ode[i]
    d_flw = L_flw[i] - L_ode[i]
    print(f'{lbl:<18} {F1:>6.1f} {F2:>6.1f} {F3:>6.1f} {F4:>6.1f}   '
          f'{L_ode[i]:>8.3f} {L_unf[i]:>10.3f} {L_flw[i]:>8.3f}   '
          f'{d_unf:>+10.3f} {d_flw:>+9.3f}')

print('\nRanking under each model:')
print(f'  ODE:           {[CANDIDATES[i][0].strip() for i in order_ode]}')
print(f'  PDE flow:      {[CANDIDATES[i][0].strip() for i in order_flow]}')
print(f'  Same top-1?    {order_ode[0] == order_flow[0]}')
print(f'  Spearman-like rank correlation: ' +
      f'{np.corrcoef(np.argsort(np.argsort(-L_ode)), np.argsort(np.argsort(-L_flw)))[0,1]:.3f}')
