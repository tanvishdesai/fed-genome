"""
FedGenome Aggregation Strategies
==================================
  FedAvg      — size-weighted average (baseline)
  FedProx     — FedAvg + proximal term in client loss (handled in client.py)
  FedGenome   — precision-weighted FedAvg (FedAlert ported to genomics)

The FedGenomeStrategy is the core contribution:
  Each client reports its local precision score alongside model weights.
  The server weights gradient contributions proportionally to precision —
  clients with high local precision (reliable classification) contribute more
  to the global model update.

This mirrors FedAlert's precision-weighted consensus aggregation,
adapted for variant pathogenicity classification.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import flwr as fl
    from flwr.common import (
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.strategy import FedAvg
    _FLWR_AVAILABLE = True
except ImportError:
    _FLWR_AVAILABLE = False
    FedAvg = object   # type: ignore


def _weighted_average(params_list: List[List[np.ndarray]], weights: List[float]) -> List[np.ndarray]:
    """Precision-weighted parameter aggregation."""
    total = sum(weights)
    w     = [x / total for x in weights]
    return [
        sum(wi * layer for wi, layer in zip(w, layer_params))
        for layer_params in zip(*params_list)
    ]


if _FLWR_AVAILABLE:
    class FedGenomeStrategy(FedAvg):
        """
        Precision-weighted FedAvg for multi-site variant classification.

        Each Flower client sends:
          fit_res.metrics["precision"] — local val precision for the round
        The server weights client updates by this precision score.
        Clients with high precision (reliable sites) get stronger votes.
        """

        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}

            params_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            precisions  = [
                max(float(fit_res.metrics.get("precision", 0.5)), 0.01)
                for _, fit_res in results
            ]

            aggregated = _weighted_average(params_list, precisions)
            print(
                f"  [FedGenome] Round {server_round}  "
                f"precision weights={[round(p,3) for p in precisions]}"
            )
            return ndarrays_to_parameters(aggregated), {
                "precision_weights": str(precisions),
            }

    class FedProxStrategy(FedAvg):
        """
        Standard FedAvg — FedProx proximal penalty is applied in the client
        training loop (see client.py), not here.
        """
        pass

else:
    class FedGenomeStrategy:   # type: ignore
        """Stub when Flower is not installed — used by manual simulation."""
        pass

    class FedProxStrategy:     # type: ignore
        pass
