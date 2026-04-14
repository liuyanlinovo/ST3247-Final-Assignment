# ST3247-Assignment

Epidemic Parameter Inference on Adaptive Networks

This repository contains a Python implementation of **Simulation-Based Inference (SBI)** using the **Approximate Bayesian Computation (ABC)** method. The goal is to infer unknown parameters of a stochastic SIR epidemic model spreading on an adaptive network.

## Project Overview

The project focuses on a Susceptible-Infected-Recovered (SIR) model where individuals can "rewire" their social connections to avoid infection. We aim to estimate three key parameters:
- `beta` ($\beta$): Infection probability.
- `gamma` ($\gamma$): Recovery probability.
- `rho` ($\rho$): Rewiring (behavioral adaptation) probability.

The inference is based on observed macroscopic data (timeseries and degree histograms) across 40 independent replicates.

## Repository Structure

- `simulator.py`: The core stochastic SIR simulator.
- `run_inference.py`: The ABC inference script that calculates the posterior distribution.
- `data/`: Folder containing the observed CSV datasets.
  - `infected_timeseries.csv`
  - `rewiring_timeseries.csv`
  - `final_degree_histograms.csv`
- `inference_result.png`: The generated posterior distribution plot.

## Methodology: Approximate Bayesian Computation (ABC)

Since the likelihood function of this stochastic network model is intractable, we use **ABC Rejection Sampling**:
1. Sample candidate parameters $(\beta, \gamma, \rho)$ from uniform priors.
2. Simulate the epidemic using the provided `simulator.py`.
3. Calculate the Euclidean distance between simulated statistics and the mean of observed data.
4. Accept the top 5% of simulations with the smallest distances to form the posterior distribution.
