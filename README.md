# Energy Grid QAOA

**Resilience-Aware Controlled Islanding & MPES (Max-Cut) --- Rigetti
Hardware + Classical Scaling**

------------------------------------------------------------------------

## Overview

This repository implements a dual-track approach for solving energy grid
optimization problems using classical and quantum methods.

### 🔹 Track 1 --- Hackathon (MPES / Weighted Max-Cut)

-   Solve the Maximum Power Energy Section (MPES) problem
-   Implement classical baselines (Greedy, Simulated Annealing, GW-SDP)
-   Implement QAOA (p=1, p=2 optional)
-   Validate small instance on Rigetti hardware

### 🔹 Track 2 --- Research Extension (Resilience-Aware Islanding)

-   Multi-objective QUBO formulation
-   Penalize fragmentation and enforce balance
-   Perform λ parameter sweeps
-   Develop hybrid divide-and-conquer strategy for scaling

------------------------------------------------------------------------

## Repository Structure

    energy-grid-qaoa/
    │
    ├── README.md
    ├── environment.yml
    ├── docs/
    ├── data/
    ├── notebooks/
    │   ├── track_1_hackathon/
    │   └── track_2_research/
    ├── src/
    └── results/

------------------------------------------------------------------------

## Setup Instructions

### 1️⃣ Clone the repository

``` bash
git clone https://github.com/<your-username>/energy-grid-qaoa.git
cd energy-grid-qaoa
```

### 2️⃣ Create the conda environment

``` bash
conda env create -f environment.yml
conda activate hackathon
```

### 3️⃣ Launch Jupyter

``` bash
jupyter lab
```

------------------------------------------------------------------------

## Recommended Execution Order

### Track 1 (Hackathon)

1.  01_eda.ipynb\
2.  02_resource_estimation.ipynb\
3.  03_classical_baselines.ipynb\
4.  04_qaoa_simulator.ipynb\
5.  05_rigetti_hardware.ipynb

### Track 2 (Research)

6.  06_qubo_builder.ipynb\
7.  07_lambda_sweep.ipynb\
8.  08_fragmentation_validation.ipynb\
9.  09_islanding_results.ipynb

------------------------------------------------------------------------

## Hardware Notes

-   Tune parameters on simulator before hardware submission.
-   Keep circuits shallow (≤ 10 qubits, ≤ 100 two-qubit gates
    recommended).
-   Store raw Rigetti results under
    `results/track_1_hackathon/hardware_logs/`.

------------------------------------------------------------------------

## Reproducibility

-   All random seeds explicitly defined in notebooks.
-   Environment version controlled via `environment.yml`.
-   All generated plots and tables saved under `results/`.

------------------------------------------------------------------------

## Authors

Rushikesh Ubale; Yasar Mulani: Vashti Chowla and Aya R

------------------------------------------------------------------------




