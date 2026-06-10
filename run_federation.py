"""
FedGenome Federated Learning Simulation
========================================
Simulates 3 hospital sites training a shared LocalGenomeNet on ClinVar
somatic variant data, partitioned by chromosome (non-IID).

Strategies:
  centralized  — all data pooled (upper bound)
  fedavg       — standard FedAvg (size-weighted)
  fedprox      — FedAvg + proximal term (client-side)
  fedgenome    — precision-weighted FedAvg (ours, ported from FedAlert)

Usage:
  python run_federation.py                          # run all strategies
  python run_federation.py --strategy fedgenome    # single strategy
  python run_federation.py --rounds 20 --local-epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")    # non-interactive backend (works on Kaggle/headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from client import (
    VariantDataset,
    evaluate,
    get_parameters,
    make_class_weights,
    set_parameters,
    train_local,
)
from model import LocalGenomeNet
from strategy import FedGenomeStrategy, _weighted_average

try:
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg
    _FLWR_AVAILABLE = True
except ImportError:
    _FLWR_AVAILABLE = False

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
SITES       = ["site_1", "site_2", "site_3"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_site(site: str) -> List[Dict]:
    with (DATA_DIR / f"{site}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def site_loader(records: List[Dict], shuffle: bool = True) -> DataLoader:
    return DataLoader(VariantDataset(records), BATCH_SIZE, shuffle=shuffle)


def evaluate_all_sites(model: torch.nn.Module, site_data: Dict[str, List]) -> Dict[str, Dict]:
    results = {}
    for site, records in site_data.items():
        loader     = site_loader(records, shuffle=False)
        auc, prec  = evaluate(model, loader)
        results[site] = {"auc": round(auc, 4), "precision": round(prec, 4)}
    return results


# ─── Centralized baseline ────────────────────────────────────────────────────

def run_centralized(site_data: Dict[str, List], epochs: int = 10) -> Dict:
    all_records = [r for recs in site_data.values() for r in recs]
    model       = LocalGenomeNet().to(DEVICE)
    cw          = make_class_weights(all_records)
    loader      = DataLoader(VariantDataset(all_records), BATCH_SIZE, shuffle=True)
    train_local(model, loader, epochs=epochs, class_weights=cw)
    per_site = evaluate_all_sites(model, site_data)
    aucs     = [v["auc"] for v in per_site.values()]
    return {
        "per_site":   per_site,
        "global_auc": round(float(np.mean(aucs)), 4),
        "equity_gap": round(float(np.std(aucs)), 4),
        "history":    [],
    }


# ─── Aggregation helpers ───────────────────────────────────────────────────────

class _FitResult:
    """Minimal stand-in for Flower FitRes when running manual simulation."""

    def __init__(self, parameters, num_examples: int, metrics: Dict) -> None:
        self.parameters   = parameters
        self.num_examples = num_examples
        self.metrics      = metrics


def _aggregate_round(
    strategy:     str,
    rnd:          int,
    local_params: List,
    weights_list: List[float],
    site_records: Dict[str, List],
    sites_order:  List[str],
) -> List:
    """Aggregate local models; uses FedGenomeStrategy.aggregate_fit() for fedgenome."""
    if strategy == "fedgenome" and _FLWR_AVAILABLE:
        results = []
        for site, params, w in zip(sites_order, local_params, weights_list):
            n = len(site_records[site])
            results.append((
                None,
                _FitResult(
                    parameters=ndarrays_to_parameters(params),
                    num_examples=n,
                    metrics={"precision": w},
                ),
            ))
        strategy_obj = FedGenomeStrategy()
        aggregated_params, _ = strategy_obj.aggregate_fit(rnd, results, [])
        return parameters_to_ndarrays(aggregated_params)
    return _weighted_average(local_params, weights_list)


# ─── Federated simulation ────────────────────────────────────────────────────

def run_federated(
    site_data:    Dict[str, List],
    strategy:     str,
    rounds:       int,
    local_epochs: int = 2,
) -> Dict:
    """
    Manual federated simulation (no Flower dependency).

    Each round:
      1. Broadcast global model weights to all sites
      2. Each site trains locally for `local_epochs` epochs
      3. Collect local parameters + precision scores
      4. Aggregate with chosen strategy
      5. Evaluate global model on each site
    """
    global_model = LocalGenomeNet().to(DEVICE)
    history: List[float] = []

    print(f"\n{'─'*60}")
    print(f"Strategy: {strategy}  |  Rounds: {rounds}  |  Local epochs: {local_epochs}")
    print(f"Sites: {list(site_data.keys())}  |  Device: {DEVICE}")
    print("─" * 60)

    for rnd in range(1, rounds + 1):
        local_params: List = []
        weights_list: List[float] = []
        site_aucs: List[float] = []

        for site, records in site_data.items():
            local_model = LocalGenomeNet().to(DEVICE)
            set_parameters(local_model, get_parameters(global_model))

            loader = site_loader(records, shuffle=True)
            cw     = make_class_weights(records)

            # Pass global model for FedProx proximal term
            global_ref = global_model if strategy == "fedprox" else None
            auc, prec  = train_local(
                local_model, loader, local_epochs, cw, global_model=global_ref
            )
            site_aucs.append(auc)
            local_params.append(get_parameters(local_model))

            if strategy == "fedgenome":
                weights_list.append(max(float(prec), 0.01))
            else:
                weights_list.append(float(len(records)))   # FedAvg / FedProx

        sites_order = list(site_data.keys())
        aggregated  = _aggregate_round(
            strategy, rnd, local_params, weights_list, site_data, sites_order
        )
        set_parameters(global_model, aggregated)

        mean_auc = float(np.mean(site_aucs))
        history.append(mean_auc)

        if rnd % 5 == 0 or rnd == 1 or rnd == rounds:
            per_site = evaluate_all_sites(global_model, site_data)
            aucs_str = "  ".join(
                f"{s}={v['auc']:.4f}" for s, v in per_site.items()
            )
            print(f"  Round {rnd:3d}/{rounds}  mean_AUC={mean_auc:.4f}  [{aucs_str}]")

    per_site = evaluate_all_sites(global_model, site_data)
    aucs     = [v["auc"] for v in per_site.values()]
    return {
        "per_site":   per_site,
        "global_auc": round(float(np.mean(aucs)), 4),
        "equity_gap": round(float(np.std(aucs)), 4),
        "history":    history,
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_convergence(summary: Dict, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    colors = {"fedavg": "#2e86c1", "fedprox": "#27ae60", "fedgenome": "#e74c3c"}
    for key, val in summary.items():
        if key == "centralized" or not val.get("history"):
            continue
        c = colors.get(key, "#7d3c98")
        plt.plot(val["history"], label=key, color=c, linewidth=2)
    plt.xlabel("Communication Round")
    plt.ylabel("Mean AUC-ROC (3 sites)")
    plt.title("FedGenome: Convergence Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "convergence.png", dpi=150)
    plt.close()
    print(f"  Convergence plot saved → {out / 'convergence.png'}")


def plot_ablation(summary: Dict, out: Path) -> None:
    strategies = list(summary.keys())
    global_aucs = [summary[s].get("global_auc", 0) for s in strategies]
    equity_gaps = [summary[s].get("equity_gap", 0) for s in strategies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#2e86c1", "#27ae60", "#e74c3c", "#f39c12"][: len(strategies)]

    ax1.bar(strategies, global_aucs, color=colors)
    ax1.set_ylabel("Global AUC-ROC")
    ax1.set_title("Global Performance")
    ax1.set_ylim(0, 1)
    for i, v in enumerate(global_aucs):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    ax2.bar(strategies, equity_gaps, color=colors)
    ax2.set_ylabel("Equity Gap (std of site AUCs)")
    ax2.set_title("Equity (lower = better)")
    for i, v in enumerate(equity_gaps):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out / "ablation.png", dpi=150)
    plt.close()
    print(f"  Ablation bar chart saved → {out / 'ablation.png'}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="FedGenome federated learning simulation")
    parser.add_argument(
        "--strategy", default="all",
        choices=["all", "central", "fedavg", "fedprox", "fedgenome"],
    )
    parser.add_argument("--rounds",       type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    if not (DATA_DIR / "site_1.json").exists():
        raise FileNotFoundError(
            "Site data not found.\n"
            "Run: python scripts/download_data.py && python prepare_data.py"
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    # Load all site data once
    site_data = {s: load_site(s) for s in SITES}
    for s, recs in site_data.items():
        print(f"  {s}: {len(recs)} variants")

    summary: Dict = {}

    if args.strategy in ("all", "central"):
        print("\n[Centralized baseline]")
        summary["centralized"] = run_centralized(site_data, epochs=args.rounds)
        r = summary["centralized"]
        print(f"  global_AUC={r['global_auc']:.4f}  equity_gap={r['equity_gap']:.4f}")

    for strat in ("fedavg", "fedprox", "fedgenome"):
        if args.strategy not in ("all", strat):
            continue
        summary[strat] = run_federated(site_data, strat, args.rounds, args.local_epochs)
        r = summary[strat]
        print(f"\n  {strat}: global_AUC={r['global_auc']:.4f}  equity_gap={r['equity_gap']:.4f}")

    # Save JSON results
    out_json = RESULTS_DIR / "ablation.json"
    with out_json.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults saved → {out_json}")

    # Print ablation table
    print("\n" + "─" * 72)
    print(f"{'Strategy':<14} | {'Global AUC':>10} | {'Site1 AUC':>9} | {'Site2 AUC':>9} | {'Site3 AUC':>9} | {'Equity Gap':>10}")
    print("─" * 72)
    for strat, r in summary.items():
        ps = r.get("per_site", {})
        s1 = ps.get("site_1", {}).get("auc", 0)
        s2 = ps.get("site_2", {}).get("auc", 0)
        s3 = ps.get("site_3", {}).get("auc", 0)
        print(
            f"{strat:<14} | {r['global_auc']:>10.4f} | {s1:>9.4f} | "
            f"{s2:>9.4f} | {s3:>9.4f} | {r['equity_gap']:>10.4f}"
        )
    print("─" * 72)

    # Plots
    plot_convergence(summary, RESULTS_DIR)
    plot_ablation(summary, RESULTS_DIR)


if __name__ == "__main__":
    main()
