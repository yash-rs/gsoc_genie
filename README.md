# gsoc_genie
Test task for GSoC GENIE

Physics-Informed Neural Network (PINN) for the damped harmonic oscillator. This repo provides a configurable training pipeline (Adam + optional LBFGS refinement), inference against analytical and numerical solutions, and logging/plot outputs.

**Summary**
- Problem: damped harmonic oscillator with damping ratio `xi`
- Model: FCNN PINN with hard-coded initial conditions
- Inputs: normalized time $t \in [0, 1]$ and normalized damping $\xi_{\mathrm{norm}} \in [0, 1]$
- Outputs: predicted displacement $x(t)$
- Training: collocation-based residual minimization
- Inference: compares PINN to analytical and numerical solutions across `xi` values

**Equation**
The target ODE is (in normalized form inside the residual):

$$
x'' + 2 \xi x' + x = 0
$$

Time is normalized as $t = z / z_{\max}$. The residual uses derivatives with respect to normalized time and rescales by $z_{\max}$ internally.

**Project Layout**
- `harmonic_pinn.py` — training and inference pipeline
- `network.py` — FCNN and Fourier-embedding FCNN architectures
- `configs/pinn_configs.yaml` — default experiment configuration
- `damped_oscillator.ipynb` — quick-start notebook
- `outputs/` — run logs, plots, inference results

**Requirements**
- Python 3.12
- `numpy`, `matplotlib`, `scipy`, `torch`, `pyyaml`

**Quick Start (Notebook)**
Open and run `damped_oscillator.ipynb`. The notebook loads the YAML config and instantiates the trainer:

```python
from harmonic_pinn import HarmonicPINN
config_path = "./configs/pinn_configs.yaml"
config = load_config(config_path)
pinn = HarmonicPINN(config)
```


**Configuration**
The config file `configs/pinn_configs.yaml` controls everything. Key sections:

- Experiment:
  - `run_name`, `seed`, `device` (`auto`, `cpu`, `cuda`)
- Optimizer:
  - `optimizer.adam`: `use_adam`, `epochs`, `lr`
  - `optimizer.lbfgs`: `use_lbfgs`, `lr`, `max_iter`, `history_size`
- Model:
  - `network`: `fcnn` or `fourier`
  - `input_dim`, `hidden_dim`, `hidden_layers`, `activation`
  - `frequencies`, `scale` (for Fourier embeddings)
- ODE/Sampling:
  - `t_min`, `t_max`
  - `xi_min`, `xi_max`
  - `n_collocation`, `n_test`
  - `x_0`, `v_0` (initial conditions)

**Model Details**
- **Hard ICs**: $x(0) = x_0$ and $x'(0) = v_0$ are enforced by construction:
  - $x(t) = x_0 + (v_0 t_{\max}) t + t^2 N(t, \xi)$ (considers time rescaling)
- **Residual**: computed via autograd on normalized inputs.
- **FourierFCNN**: adds sinusoidal features of time and concatenates `xi`.

**Training**
- Collocation points are sampled uniformly in normalized `t` and `xi`.
- Training uses:
  - Adam for the configured epochs (default `100000`)
  - Optional LBFGS refinement (default `use_lbfgs: True`)
- The loss is MSE of the ODE residual.
- Loss history is stored and plotted.

**Inference**
For a small grid of `xi` values, inference function computes:
- Analytical solution
- Numerical solution (via `scipy.integrate.solve_ivp`)
- PINN predictions

Plots are saved to `outputs/<run_name>_<timestamp>/inference_results/` with L2 error in the title.

**Outputs and Logging**
Each run creates a timestamped directory:
```
outputs/<run_name>_<YYYYMMDD_HHMMSS>/
  train.log
  train_loss.png
  inference_results/
    inference_xi_*.png
```

