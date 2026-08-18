# Physics-Informed Neural Network: Damped Harmonic Oscillator

A physics-informed neural network (PINN) built in TensorFlow that models a damped harmonic oscillator by training against its governing differential equation

## Overview

A normal neural net only knows what the data tells it. A PINN adds the physics as part of the loss, so the network has to satisfy m*x'' + c*x' + k*x = 0 too. The loss has four parts:
 - **ODE residual**: How badly the network violates the equation
- **Initial displacement**: `x(0) = x0`
- **Initial velocity**: `x'(0) = v0`
- **Data fit**: How far predictions are from the observed points
The derivatives come from nested tf.GradientTape calls, so x' and x'' are exact instead of finite-difference approximations. That was the part I thought was the most interesting.
The initial condition terms get weighted 100x because they're only evaluated at one point while the other two average over hundreds, so otherwise they get drowned out.

## Method

- **Architecture**: 3 hidden layers, 64 units each, `tanh` activation, single scalar output. `tanh` is used because it is smooth and infinitely differentiable. The loss depends on the network's second derivative, so an activation like ReLU would give a residual that is zero almost everywhere and carries no gradient signal.
- **Collocation points**: 1500 points sampled uniformly over `t ∈ [0, 30]`, roughly 4.8 oscillations.
- **Training**: Adam optimizer, learning rate `1e-3`, 50000 epochs.

## Verifying the dataset parameters

Because `train.csv` includes an `acceleration` column, the governing equation can be arranged as

```
x'' = -(c/m)·x' - (k/m)·x
```

which is linear in the unknowns, so `c/m` and `k/m` are recoverable by least squares across all 200 rows. This gives `k/m = 1.0` and `c/m = 0.1`, and the resulting closed-form solution reproduces the dataset to within `1.6 × 10⁻⁶`. The notebook runs this fit and asserts the values in use match the data, so if someone changes a parameter it clearly fails instead of training on the wrong physics.

## Validation

The underdamped oscillator has a closed form solution:

```
x(t) = e^(-ζωₙt) · [A·cos(ω_d·t) + B·sin(ω_d·t)]
```

where `ωₙ = √(k/m)`, `ζ = c / (2√(mk))`, and `ω_d = ωₙ√(1 - ζ²)`.

I compare against that directly rather than holding out part of the data. The notebook prints max absolute error, mean absolute error, and RMSE

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

## Results

Max absolute error:  1.408e-03
Mean absolute error: 1.141e-04
RMSE:                2.116e-04

Measured against the closed-form solution over t = 0 to 30.

## Running it

Open `PhysicsModel.ipynb` in Google Colab or Jupyter with `train.csv` in the same directory, then run all cells top to bottom.

## Requirements

- TensorFlow
- NumPy
- pandas
- matplotlib

## Problems Encountered
My first run looked great until about t = 12 and then the prediction just flattened out toward zero while the real solution kept oscillating. Max error was 0.389.
Took me a while to figure out why. The problem is that x(t) = 0 satisfies the ODE perfectly. And it's an easy one for the network to find, because sitting at zero makes the ODE residual tiny. So my ODE loss looked healthy at around 0.002 while the actual fit was obviously wrong. The initial conditions hold it down near t = 0 but that only propagates so far.

I fixed this by normalizing the input, as the network was getting raw t values up to 30 fed into tanh, and tanh(30) is 1.0 to fifteen decimal places, so the gradients basically vanish out there. Rescaling t to [-1, 1] inside the model's `call` fixes it, and GradientTape handles the chain rule on its own so the loss function didn't need to change.

I also weighed the data term more, as the observed points are the only thing in the loss that knows the answer isn't zero at t = 25. Bumping `w_data` from 1 to 20 keeps the ODE term from outvoting them.

Finally, I simply allowed the model to train longer. I increased the epoch amount to 50000 as well as adding a decaying learning rate.

## Things I'd like to try
  - Curriculum training: fit [0, 10] first, then extend to [0, 20] and [0, 30], so the correct solution propagates outward instead of the trivial one creeping in
  - Learning the damping coefficient from the data instead of assuming it (inverse problem)
  - Sampling collocation points randomly each epoch instead of on a fixed grid
