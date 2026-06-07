"""
FedGenome Flower Client
========================
Wraps LocalGenomeNet training for federated simulation.
Each client handles one hospital site (TCGA cancer-type partition).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import LocalGenomeNet, NucleotideTransformerNet, one_hot_encode

DATA_DIR     = Path(__file__).parent / "data"
BATCH_SIZE   = 64
LOCAL_EPOCHS = 2
LR           = 1e-3
PROX_MU      = 0.01    # FedProx proximal term coefficient
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Dataset ─────────────────────────────────────────────────────────────────

class VariantDataset(Dataset):
    def __init__(self, records: List[Dict]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        r = self.records[idx]
        return one_hot_encode(r["context"]), int(r["label"])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    model.load_state_dict(state, strict=True)


def make_class_weights(records: List[Dict]) -> torch.Tensor:
    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = len(records) - n_pos
    w     = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)
    return w.to(DEVICE)


# ─── Training / evaluation ───────────────────────────────────────────────────

def train_local(
    model:         nn.Module,
    loader:        DataLoader,
    epochs:        int = LOCAL_EPOCHS,
    class_weights: Optional[torch.Tensor] = None,
    global_model:  Optional[nn.Module] = None,   # for FedProx
) -> Tuple[float, float]:
    model.train()
    opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    global_params = None
    if global_model is not None:
        global_params = [p.data.clone() for p in global_model.parameters()]

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)

            # FedProx proximal regularisation
            if global_params is not None:
                prox = sum(
                    ((p - gp) ** 2).sum()
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (PROX_MU / 2) * prox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    return evaluate(model, loader)


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    ys, probs, preds = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb      = xb.to(DEVICE)
            logits  = model(xb)
            prob    = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred    = logits.argmax(1).cpu().numpy()
            ys.extend(yb.numpy())
            probs.extend(prob)
            preds.extend(pred)

    try:
        auc = float(roc_auc_score(ys, probs))
    except ValueError:
        auc = 0.5

    prec = float(precision_score(ys, preds, zero_division=0))
    return auc, prec


# ─── Flower client (requires flwr) ───────────────────────────────────────────

try:
    import flwr as fl
    from flwr.client import Client
    from flwr.common import Context

    class FedGenomeClient(Client):
        def __init__(self, site: str, records: List[Dict], use_transformer: bool = False) -> None:
            self.site    = site
            self.records = records
            ModelClass   = NucleotideTransformerNet if use_transformer else LocalGenomeNet
            self.model   = ModelClass().to(DEVICE)
            self.cw      = make_class_weights(records)
            self.loader  = DataLoader(VariantDataset(records), BATCH_SIZE, shuffle=True)

        def get_parameters(self, config):
            return get_parameters(self.model)

        def fit(self, parameters, config):
            set_parameters(self.model, parameters)
            auc, prec = train_local(self.model, self.loader, LOCAL_EPOCHS, self.cw)
            return get_parameters(self.model), len(self.records), {
                "auc": float(auc), "precision": float(prec), "site": self.site,
            }

        def evaluate(self, parameters, config):
            set_parameters(self.model, parameters)
            auc, prec = evaluate(self.model, self.loader)
            return float(1.0 - auc), len(self.records), {
                "auc": float(auc), "precision": float(prec),
            }

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        sites        = ["site_1", "site_2", "site_3"]
        site         = sites[partition_id % 3]
        path         = DATA_DIR / f"{site}.json"
        with path.open() as fh:
            records = json.load(fh)
        return FedGenomeClient(site, records).to_client()

    app = fl.client.ClientApp(client_fn=client_fn)

except ImportError:
    pass   # Flower not installed; manual simulation in run_federation.py will be used
