# Quantum Pulse  
## Hardware-Aware Hybrid Quantum Optimization of Large-Scale Power Transmission Networks  

### Q-volution Hackathon 2026  
Team: Rushikesh Ubale · Yasar Mulani · Vashti Chowla · Aya Rhazi  

---

## 1. Project Overview

This project investigates how quantum optimization can be applied to a large-scale power-grid inspired Maximum Weighted Cut (MPES) problem under realistic hardware constraints.

The challenge problem (Problem B) contains 180 nodes and 226 edges. A naive one-qubit-per-node QAOA implementation exceeds the practical limits of current superconducting quantum hardware due to:

- Two-qubit gate overhead from routing
- Limited qubit connectivity
- Decoherence and noise accumulation
- Exponential classical simulation cost

Instead of executing a single large circuit, we developed a **hardware-aware hybrid workflow** that integrates:

- Resource estimation
- Native gate compilation auditing
- Graph decomposition under hardware constraints
- Cluster-level QAOA
- Classical refinement
- Physical hardware validation on Rigetti Ankaa-3

The central idea is that scalability at this scale comes from the workflow, not deeper circuits.

---

## 2. Repository Structure
.
├── data/
│ ├── problemA.parquet
│ └── problemB.parquet
│
├── notebooks/
│ ├── 01_resource_estimation.ipynb
│ ├── 02_partitioning.ipynb
│ ├── 03_cluster_qaoa.ipynb
│ ├── 04_hardware_validation.ipynb
│ └── 05_analysis_and_plots.ipynb
│
├── results/
│ ├── problemB_partition/
│ ├── problemB_qaoa_layer2/
│ ├── track_1_hackathon/
│   ├── hardware_logs/
│   ├── simulator_logs/
│   ├── tables/
│   └── plots/
│ 
│
├── src/
│ ├── qubo_builder.py
│ ├── qaoa_model.py
│ ├── graph_utils.py
│ ├── rigetti_runner.py
│ └── partitioning.py
│
└── README.md


---

## 3. Methodology

### 3.1 Resource Estimation

We first estimate the feasibility of a direct QAOA implementation on the full 180-node graph:

- One qubit per node → 180 qubits required
- Approximately 244 native two-qubit gates after routing (p = 1)
- Predicted survival in the noise-dominated regime
- Classical simulation requires 2^180 amplitudes (infeasible)

Conclusion: direct full-graph QAOA is not physically executable.

---

### 3.2 Hardware-Aware Graph Decomposition

We partition the graph into 16 clusters subject to hardware constraints:

- Maximum 12 nodes per cluster
- Maximum 15 internal edges
- Native iSWAP budget per cluster
- Predicted survival threshold S ≥ 0.20

Partitioning uses recursive splitting guided by connectivity structure.

Each cluster remains executable within the feasible regime of the Rigetti Ankaa-3 device.

---

### 3.3 Cluster-Level QAOA

For each cluster:

- Depth p = 1 QAOA
- Grid search + COBYLA optimization
- 1024 shots per cluster (simulator)
- Local bitstring solution extracted

---

### 3.4 Stitching and Classical Refinement

The 16 local cluster solutions are stitched into a global 180-node partition.

Since cluster boundaries ignore cross-cluster correlations, we apply:

- One-exchange greedy refinement
- Boundary repair

Final result is a hybrid quantum-classical solution.

---

### 3.5 Hardware Validation

Three pre-registered clusters (low, medium, high compiled two-qubit counts) were executed on the Rigetti Ankaa-3 QPU:

- Angles frozen from simulator optimization
- Compilation to native ISA
- Native iSWAP audit performed
- Embedding refinement applied to reduce routing overhead

Measured results confirm a clear relationship between compiled two-qubit gate count and performance degradation.

---

## 4. Key Results (Problem B)

Classical greedy baseline:6778.19

Raw stitched QAOA:5346.62

After classical refinement:6435.60 (~95% of greedy baseline)

Noisy simulation (device-calibrated noise model):3602.92


Hardware validation confirms compiled two-qubit count strongly impacts performance.

---

## 5. Reproducibility

To reproduce results:

1. Run `01_resource_estimation.ipynb`
2. Run `02_partitioning.ipynb`
3. Run `03_cluster_qaoa.ipynb`
4. Run `05_analysis_and_plots.ipynb`

For hardware execution:

- Requires QCS SDK and access to Rigetti Ankaa-3
- Execute `04_hardware_validation.ipynb`
- Hardware logs stored in `results/track_1_hackathon/hardware_logs/`

---

## 6. Technical Stack

- Python
- NumPy, SciPy, NetworkX
- Qiskit (simulation)
- pyQuil + QCS SDK (hardware)
- Rigetti Ankaa-3 (84 qubits)

---

## 7. Contribution

This project introduces a hardware-first methodology for executing large combinatorial optimization problems on NISQ devices by combining:

- Compile-time native gate accounting
- Feasibility envelope modeling
- Hardware-constrained decomposition
- Embedding refinement
- Hybrid quantum-classical stitching

Rather than scaling circuits, we scale the workflow.

---

## 8. Limitations and Future Work

- Noise remains the primary bottleneck
- Cluster boundary correlations are approximated
- Per-qubit calibration variability can impact performance

Future directions:

- Device-aware partitioning
- Advanced error mitigation
- Adaptive embedding search
- Higher-depth cluster circuits within bounded budgets

---

## Authors

Rushikesh Ubale; Yasar Mulani: Vashti Chowla and Aya Rhazi

------------------------------------------------------------------------




