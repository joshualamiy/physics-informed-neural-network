# Physics-Informed Neural Network: Damped Harmonic Oscillator

A physics-informed neural network (PINN) built in TensorFlow that models a damped harmonic oscillator by training against its governing differential equation, not just observed data.

## Overview

A standard neural network learns purely from data. A PINN also penalizes the network for violating the physics it is supposed to obey. This model minimizes a weighted combination of four loss terms:

| Loss term | What it enforces |
|---|---|
| ODE residual | `m·x'' + c·x' + k·x = 0` across the domain |
| Initial displacement | `x(0) = x₀` |
| Initial velocity | `x'(0) = v₀` |
| Data fit | Predictions match observed training points |

Derivatives `x'` and `x''` are computed with nested `tf.GradientTape` automatic differentiation rather than finite differences, so the ODE residual is exact to machine precision.

The initial-condition terms are weighted 100× relative to the others. They are evaluated at a single point while the ODE and data terms average over hundreds, so without the weighting they would be swamped — and the ODE alone is satisfied by the trivial solution `x(t) = 0`, so something has to anchor the network to the correct one.

## Method

- **Architecture:** 3 hidden layers, 64 units each, `tanh` activation, single scalar output. `tanh` is used because it is smooth and infinitely differentiable — the loss depends on the network's second derivative, so an activation like ReLU would give a residual that is zero almost everywhere and carries no gradient signal.
- **Collocation points:** 1500 points sampled uniformly over `t ∈ [0, 30]`, roughly 4.8 oscillations. These require no labels; that is what makes the method physics-informed rather than data-driven.
- **Training:** Adam optimizer, learning rate `1e-3`, 20000 epochs.

## Verifying the dataset parameters

The physical constants were not taken on faith. Because `train.csv` includes an `acceleration` column, the governing equation can be rearranged as

```
x'' = -(c/m)·x' - (k/m)·x
```

which is linear in the unknowns, so `c/m` and `k/m` are recoverable by least squares across all 200 rows. This gives `k/m = 1.0` and `c/m = 0.1`, and the resulting closed-form solution reproduces the dataset to within `1.6 × 10⁻⁶`. The notebook performs this fit and asserts the constants in use match the data, so a mismatch fails loudly instead of silently training against the wrong physics.

## Validation

The underdamped oscillator has a closed-form solution:

```
x(t) = e^(-ζωₙt) · [A·cos(ω_d·t) + B·sin(ω_d·t)]
```

where `ωₙ = √(k/m)`, `ζ = c / (2√(mk))`, and `ω_d = ωₙ√(1 - ζ²)`.

The notebook plots the PINN prediction against this exact solution and reports max absolute error, mean absolute error, and RMSE — so accuracy is measured against ground truth rather than a held-out split.

## Parameters

| Parameter | Symbol | Value |
|---|---|---|
| Mass | `m` | 1.0 kg |
| Spring constant | `k` | 1.0 N/m |
| Damping coefficient | `c` | 0.1 N·s/m |
| Damping ratio | `ζ` | 0.05 (underdamped) |
| Period | `T` | 6.29 s |

Initial conditions `x₀ = 1.0 m` and `v₀ = 0.0 m/s` are read from the first row of the training data.

## Data

`train.csv` contains 200 rows with columns `time`, `displacement`, `velocity`, and `acceleration`, spanning `t = 0` to `t = 300`. The notebook filters this to the training domain so that the ODE residual and the data-fit term describe the same interval.

## Running it

Open `PhysicsModel.ipynb` in Google Colab or Jupyter with `train.csv` in the same directory, then run all cells top to bottom.

## Requirements

- TensorFlow
- NumPy
- pandas
- matplotlib
