# Priority-Aware Federated Reinforcement Learning for Adaptive Resource Allocation in Multi-Modal Medical Wearable IoT Networks

**PA-FedRL** — Our proposed work for the 7th International Conference on Frontiers in Computing and Systems (COMSYS-2026)

---

## Overview

Medical wearable devices face a hard constraint that generic IoT resource allocation methods ignore: not all sensor data carries equal clinical weight. Deferring a step-count reading is acceptable; deferring an ECG reading during a cardiac event is not.

PA-FedRL addresses this by combining Deep Q-Network (DQN) reinforcement learning with federated averaging, where the reward function is explicitly shaped by task priority. Devices learn locally and share only model weights, not raw sensor data, preserving patient privacy while allowing the policy to generalize across a heterogeneous device fleet.

The simulation covers seven sensor modalities (ECG, SpO2, EEG, Fall/IMU, Respiration, Skin Temperature, Step Count), three operating scenarios (normal operation, battery stress, cardiac event pre-emption), and is benchmarked against Round-Robin scheduling, Static Priority scheduling, and Vanilla FedAvg.

---

## Repository Structure

```
.
├── simulation.py          # Full simulation: environment, agent, training, baselines, figures
├── README.md
```

Output files generated on run:

```
figure1_battery_depletion.pdf / .png
figure2_cardiac_latency.pdf   / .png
```

---

## Method

### Environment

Each simulated device runs a `WearableEnv` with:

- **State space (10 dimensions):** battery level, per-task urgency scores (7), channel quality, last accuracy estimate
- **Action space (19 discrete actions):** run one task at full resolution, optionally throttle one high-priority task alongside it, or defer
- **Device heterogeneity:** processor speed (80-240 MHz), RAM (256-512 KB), and channel quality drawn at initialization to reflect real-world fleet diversity

Task priorities are fixed as:

| Task | Priority | Urgency Score |
|---|---|---|
| ECG | CRITICAL | 1.0 |
| SpO2 | CRITICAL | 1.0 |
| EEG | HIGH | 0.7 |
| Fall / IMU | HIGH | 0.7 |
| Respiration | MEDIUM | 0.4 |
| Skin Temperature | LOW | 0.1 |
| Step Count | LOW | 0.1 |

### Reward Function

The reward at each timestep is:

```
R = α·U + β·A − λ_e·E − δ·L
```

Where:
- `U` — weighted urgency of tasks executed
- `A` — mean accuracy across executed tasks
- `E` — normalized energy consumed (relative to E_max = 27.5 mJ)
- `L` — count of CRITICAL task deferrals
- Coefficients: α = 0.4, β = 0.3, λ_e = 0.2, δ = 0.1

This formulation penalizes both energy waste and critical deferrals, so the agent cannot exploit high reward by simply running every task at full resolution.

### DQN Agent

- **Architecture:** 10 → 64 → 64 → 19 fully connected network (~5,952 multiply-adds per inference, suitable for embedded deployment)
- **Discount factor:** γ = 0.95
- **Replay buffer:** 10,000 transitions
- **Epsilon-greedy exploration:** decays from 1.0 to 0.05 at rate 0.995

### Federated Aggregation

After every K = 10 episodes, each device uploads its local policy weights. The server performs uniform FedAvg:

```
θ_global = (1/N) · Σ θ_i
```

Uniform averaging is used deliberately to prevent the global policy from biasing toward devices with higher battery levels. Only model weights are transmitted — no sensor readings leave the device.

---

## Experimental Scenarios

**Scenario A — Normal Operation:** Battery initialized at 90-100%, standard channel conditions. Measures steady-state energy efficiency, task accuracy, and critical deferral rate.

**Scenario B — Battery Stress:** Battery initialized at 6-14% remaining. Tests whether the priority-aware policy conserves energy on critical tasks under severe resource constraints.

**Scenario C — Cardiac Event Pre-emption:** A cardiac anomaly is injected at a random step between t = 50 and t = 150. Latency is measured from event injection to the next ECG full-resolution execution (each step = 10 ms). The clinical threshold is 100 ms.

---

## Baselines

- **Round-Robin:** Cycles through all seven tasks in fixed order, regardless of urgency or battery level
- **Static Priority:** Always executes tasks in decreasing urgency order, no learning or adaptation
- **Vanilla FedAvg:** Federated DQN with the same architecture but no priority weighting in the reward (ablation, isolates the contribution of α and δ terms)

---

## Requirements

```
python >= 3.8
numpy
torch
matplotlib
```

Install dependencies:

```bash
pip install numpy torch matplotlib
```

---

## Running the Simulation

```bash
python simulation.py
```

Default configuration:

```python
NUM_DEVICES        = 5
NUM_ROUNDS         = 10
EPISODES_PER_ROUND = 10
NUM_SEEDS          = 5
```

The script runs all four conditions sequentially, prints a results table to stdout, and saves two figures to the working directory.

To adjust the scale of experiments, edit the constants at the top of the `if __name__ == "__main__"` block.

---

## Output

After a full run, the terminal prints a results table in the format below:

```
method               energy/task (mJ)    acc (%)      latency (ms)
-------------------------------------------------------------------
round-robin                   3.962       95.0      31.9 +/- 19.7
static priority               4.358       95.0      29.5 +/- 31.2
vanilla fedavg                    —          —     269.2 +/- 285.2
pa-fedrl (ours)               4.512       83.6      43.2 +/- 68.2
```

And generates:
- `figure1_battery_depletion.pdf` — battery depletion curves across all methods under Scenario B
- `figure2_cardiac_latency.pdf` — bar chart of cardiac event re-prioritization latency with the 100 ms clinical threshold marked

---

## Citation

This paper is currently under review at COMSYS-2026. If you use this code, please cite the corresponding paper once it is published through the COMSYS-2026 proceedings.

---

## License

This repository is released for research and reproducibility purposes. Contact the authors before using any part of this work in a commercial product.
