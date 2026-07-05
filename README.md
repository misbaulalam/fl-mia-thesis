# Robust Defense Against Membership Inference Attacks in Privacy-Preserving Federated CNNs Under Data Heterogeneity

**Undergraduate Thesis Project — B.Sc. in Computer Science and Engineering**
Varendra University, Department of Computer Science and Engineering

## Team

| Name | Student ID | Role |
|---|---|---|
| Md. Misbaul Alam | 231311013 | Project Lead & Core Architect |
| Israt Jahan Emu | 231311122 | Security & Data Engineer |
| Hasib al Nabil | 231311094 | QA & Analysis Specialist |

**Supervisor:** Rokaiya Tasnim, Lecturer, Department of Computer Science and Engineering

## Project Status

🔶 **In Progress — First Defense Checkpoint**

This repository reflects work completed for the first thesis defense (6-month checkpoint). Core framework components are implemented and verified. Final experiment tuning and full-scale results are in progress and will be completed for the final defense.

## Overview

Federated Learning (FL) allows multiple clients to collaboratively train a shared model without sharing raw data. However, FL remains vulnerable to Membership Inference Attacks (MIA), where an adversary can infer whether a specific sample was used during training — a risk that worsens under Non-IID (heterogeneous) client data and with CNN-based models that produce high-confidence predictions.

Most existing Differential Privacy (DP) defenses apply the same protection strength to every client regardless of their data heterogeneity. This project investigates a **heterogeneity-aware adaptive DP defense**, which scales privacy protection strength based on each client's measured data skew, and compares it against a standard flat/uniform DP baseline across varying levels of Non-IID heterogeneity.

## Research Objectives

1. Build a CNN-based Federated Learning framework using FedAvg, with controllable Non-IID data partitioning via Dirichlet distribution.
2. Implement a Membership Inference Attack model to measure privacy leakage using prediction confidence and loss values.
3. Design a heterogeneity-aware adaptive DP defense, where clipping and noise strength are scaled per client based on a KL-divergence-based skew score.
4. Compare adaptive DP against a flat/uniform DP baseline across multiple heterogeneity levels to evaluate the privacy-utility trade-off.

## Repository Structure

fl-mia-thesis/
├── data/               # CIFAR-10 dataset (downloaded, not tracked in git)
├── models/             # Saved trained model weights (.pt files)
├── results/            # Experiment output: CSVs, plots, metrics
├── notebooks/          # Jupyter notebooks for each development stage
│   ├── 00_setup_check.ipynb
│   ├── 01_partition_test.ipynb
│   ├── 02_model_test.ipynb
│   ├── 03_federated_test.ipynb
│   ├── 04_mia_test.ipynb
│   ├── 05_defense_test.ipynb
│   └── 06_full_experiment.ipynb
├── src/                # Core reusable source code
│   ├── partition.py    # Dirichlet-based Non-IID data partitioning
│   ├── model.py         # CNN architecture (SimpleCNN)
│   ├── federated.py     # FedAvg training, aggregation, evaluation
│   ├── attack.py        # Membership Inference Attack model
│   └── defense.py       # Flat DP and heterogeneity-aware adaptive DP
├── requirements.txt
└── README.md

## Methodology

The pipeline has five stages:

1. **Heterogeneity-Controlled Data Partitioning** — CIFAR-10 is split across clients using a Dirichlet distribution with tunable concentration parameter α (lower α = more skewed/Non-IID). A per-client skew score is computed using KL divergence between the client's local label distribution and the global distribution.

2. **CNN-Based Federated Learning** — A lightweight CNN is trained using a from-scratch FedAvg implementation: clients train locally on their partitioned data, and updates are aggregated into a global model over multiple communication rounds.

3. **Membership Inference Attack** — After training, an attack classifier (logistic regression) is trained on prediction confidence and loss values to distinguish training samples (members) from unseen samples (non-members), evaluated via accuracy, precision, recall, and F1-score.

4. **Heterogeneity-Aware Adaptive DP Defense** — Each client's gradient clipping threshold and noise scale are adjusted based on their skew score, rather than using one fixed setting for every client. A flat/uniform DP baseline is also implemented for comparison.

5. **Comparative Evaluation** — Three configurations (no defense, flat DP, adaptive DP) are compared across multiple α values, recording both model utility (test accuracy) and privacy leakage (attack accuracy) to evaluate the privacy-utility trade-off.

## Tools and Technologies

- **Language:** Python 3.12
- **Deep Learning:** PyTorch (CPU)
- **Data Handling:** NumPy, Torchvision
- **Evaluation:** Scikit-learn, Matplotlib
- **Dataset:** CIFAR-10
- **Environment:** VS Code, Jupyter Notebook

## Completed So Far

- [x] Dirichlet-based Non-IID partitioning with verified skew levels across α = 0.1, 0.5, 5.0, 100
- [x] KL-divergence-based client skew score computation
- [x] Lightweight CNN architecture for CIFAR-10
- [x] FedAvg implementation (local training + aggregation) — verified accuracy improves across rounds
- [x] Membership Inference Attack model (confidence + loss based) — verified functional
- [x] Flat/uniform DP defense implementation
- [x] Heterogeneity-aware adaptive DP defense implementation
- [x] Full experiment pipeline (multi-alpha × multi-defense comparison loop)

## In Progress

- [ ] Fine-tuning DP noise/clipping parameters for a clearer privacy-utility signal
- [ ] Running the full experiment across all planned α values with increased communication rounds
- [ ] Final results table, comparison plots, and privacy-utility trade-off analysis
- [ ] Thesis report writing

## How to Run

```bash
# Clone the repository
git clone <your-repo-url>
cd fl-mia-thesis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order from notebooks/, starting with 00_setup_check.ipynb
```

## References

1. R. Shokri, M. Stronati, C. Song, V. Shmatikov, "Membership Inference Attacks against Machine Learning Models," IEEE S&P, 2017.
2. M. Nasr, R. Shokri, A. Houmansadr, "Comprehensive Privacy Analysis of Deep Learning," IEEE S&P, 2019.
3. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017.
4. M. Abadi et al., "Deep Learning with Differential Privacy," ACM CCS, 2016.
5. T. Li et al., "Federated Optimization in Heterogeneous Networks," MLSys, 2020.
6. L. Bai et al., "Membership Inference Attacks and Defenses in Federated Learning: A Survey," ACM Computing Surveys, 2024.
7. Y. Gu, Y. Bai, S. Xu, "CS-MIA: Membership Inference Attack Based on Prediction Confidence Series in Federated Learning," 2022.
8. H. Chang et al., "United We Defend: Collaborative Membership Inference Defenses in Federated Learning," arXiv, 2026.
9. C. Dwork, "Differential Privacy," ICALP, 2006.
