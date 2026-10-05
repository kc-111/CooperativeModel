# CooperativeModel

CooperativeModel simulates four microbial populations consuming nutrients and producing a product and growth inhibitors. It supports a well-mixed model and a 3D cylindrical batch bioreactor. The 3D workflow computes fluid flow once, saves it, and reuses it to simulate different initial nutrient allocations.

## Installation

Requires Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

For the CUDA 12.6 GPU option:

```bash
python -m pip install -e '.[gpu]' --extra-index-url https://download.pytorch.org/whl/cu126
```

## Run the examples

Run a well-mixed comparison on CPU:

```bash
python examples/example_ode.py
```

For a 3D simulation, generate the flow cache and then run the reactor:

```bash
python examples/solve_flow.py --out flow_cache.h5
python examples/example.py
```

The flow solver runs on CPU by default; add `--device cuda` to use a GPU. The reactor example uses `device='cuda'`; change it to `'cpu'` for a CPU run. It simulates 24 hours with biomass and nutrients initially loaded into one octant and saves GIFs of the concentrations. The simulation grid must match the flow cache.

| Example | Purpose |
| --- | --- |
| [solve_flow.py](examples/solve_flow.py) | Generate a flow cache; use `--help` for options. |
| [example.py](examples/example.py) | Run and visualize the 3D bioreactor. |
| [example_ode.py](examples/example_ode.py) | Compare initial nutrient allocations without spatial transport. |
| [find_optima.py](examples/find_optima.py) | Search for initial nutrient allocations that maximize final product using L-BFGS-B. |
| [blob_test.py](examples/blob_test.py) | Visualize passive-scalar transport on the cached flow. |

## Python API

```python
from CooperativeModel import Simulator

result = Simulator(
    R1=2.0, R2=2.0, R3=0.05, R4=0.05,
    t_final=24.0,
    grid_shape=(1, 1, 1),
    flow_cache_path=None,
    device='cpu',
).run()

print(result.L_final)
```

For a 3D run, use `grid_shape=(32, 32, 32)` and `flow_cache_path='flow_cache.h5'`. Set `ic_mode='octant'` to match the spatial example, or `'uniform'` to fill all fluid cells. Multiple initial states can be passed through `samples`, with channel order `[N1..N4, L, R1..R4, T1..T4]`.

Results provide final product (`L_final`), final channel means (`final_values()`), and concentration trajectories (`spatial_average()`). Spatial runs can save visualizations:

```python
result.gif('reactor.gif', view='midz')
result.snapshot('reactor.png')
result.timeseries('concentrations.png')
```

## Source code

| Module | Responsibility |
| --- | --- |
| [kinetics.py](src/CooperativeModel/kinetics.py) | Microbial growth, nutrient consumption, product formation, and inhibition. |
| [flow_3d.py](src/CooperativeModel/flow_3d.py) | Flow calculation and cache storage. |
| [model.py](src/CooperativeModel/model.py) | Reaction and transport integration. |
| [simulate_ode.py](src/CooperativeModel/simulate_ode.py) | `Simulator` interface and results. |
| [config.py](src/CooperativeModel/config.py) | Kinetic, grid, and solver settings. |
| [visualization.py](src/CooperativeModel/visualization.py) | Plots and animations. |

To customize kinetic parameters or solver tolerances, use `SimulationConfig` with the lower-level `simulate` function.
