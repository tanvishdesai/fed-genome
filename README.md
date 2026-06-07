# FedGenome · Federated Cancer Variant Classification

> **Part of the Bioinformatics AI Portfolio** — Project 3 of 4.
> See [PLANNING.md](../PLANNING.md) for the full roadmap.

Ports the **FedAlert precision-weighted consensus aggregation framework** to
multi-site cancer genomic somatic variant classification on ClinVar data.
Simulates 3 hospital sites (non-IID by chromosome) with federated learning.

---

## Architecture

```
Simulated Site 1        Simulated Site 2        Simulated Site 3
ClinVar chr 1-8         ClinVar chr 9-16        ClinVar chr 17-22,X,Y
(BRCA-like partition)   (lung-like partition)   (colon-like partition)
       │                       │                       │
 LocalGenomeNet          LocalGenomeNet          LocalGenomeNet
 (1D-CNN or              (same arch)             (same arch)
  Transformer)
       │ ΔW₁ + precision       │ ΔW₂ + precision       │ ΔW₃ + precision
       └───────────────────────┼───────────────────────┘
                               ▼
              FedGenome Aggregator
              (precision-weighted FedAvg)
              — higher-precision sites vote more —
                               │
                        Global Genome Model
                   (multi-site variant classifier)
```

**Novelty**: FedAlert's precision-weighted consensus (originally on histopathology images)
is ported to genomic variant pathogenicity classification. Same algorithm, different domain.

---

## Kaggle Setup (Recommended)

1. Create a Kaggle notebook, enable GPU
2. Add dataset: `kevinarvai/clinvar-conflicting`
   [https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting)
3. Run:

```python
!pip install torch scikit-learn matplotlib flwr -q

# Copy ClinVar from Kaggle input
!python scripts/download_data.py --kaggle

# Build site partitions
!python prepare_data.py

# Run federated experiment (all strategies, 10 rounds)
!python run_federation.py --rounds 10
```

---

## Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ClinVar (~400 MB, one-time)
python scripts/download_data.py

# 3. Build non-IID site partitions
python prepare_data.py

# 4. Run federated experiment
python run_federation.py --strategy all --rounds 20

# Single strategy
python run_federation.py --strategy fedgenome --rounds 20
```

---

## Run Order

```
1. python scripts/download_data.py        ← get ClinVar data (once)
2. python prepare_data.py                 ← partition into 3 sites (once)
3. python run_federation.py               ← run all 4 strategies + plots
```

Results in `results/ablation.json` and plots `results/ablation.png`, `results/convergence.png`.

---

## Strategies Compared

| Strategy | Description |
|----------|-------------|
| Centralized | All data pooled — upper bound |
| FedAvg | Standard weighted average by dataset size |
| FedProx | FedAvg + proximal regularisation (μ=0.01) |
| **FedGenome** | **Precision-weighted FedAvg** (FedAlert-style) |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Global AUC-ROC | Mean AUC across all 3 sites |
| Per-site AUC | Site-level classification performance |
| Equity Gap | std(site AUCs) — lower = more equitable |
| Convergence | AUC vs communication round |

---

## Project Structure

```
fedgenome/
  run_federation.py     Main experiment: all 4 strategies + plots
  prepare_data.py       ClinVar → 3 non-IID site partitions
  model.py              LocalGenomeNet (CNN-1D + Transformer variants)
  client.py             Flower client (+ FedProx proximal term)
  strategy.py           FedGenomeStrategy (precision-weighted FedAvg)
  requirements.txt
  scripts/
    download_data.py    ClinVar download (FTP or Kaggle)
  data/                 (ignored) variant_summary.txt + site JSON files
  results/              (ignored) ablation.json + plots
```

---

## Publication Path

- **Briefings in Bioinformatics** (Oxford, IF 9.5)
- **Nature Machine Intelligence**
- **RECOMB 2027** — premier computational genomics conference
- **ICLR 2027 workshop**: FL + healthcare track
