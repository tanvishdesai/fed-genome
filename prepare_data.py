"""
FedGenome Data Preparation
===========================
Reads ClinVar variant_summary.txt, creates pathogenic/benign labels,
extracts real ±20 bp genomic context from Ensembl GRCh38, and partitions
variants into 3 non-IID hospital sites using Dirichlet allocation (α=0.5)
over cancer-gene subtypes (breast / lung / colorectal panels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from genomic_context import GenomicContextFetcher

DATA_DIR    = Path(__file__).parent / "data"
CONTEXT_LEN = 41   # ±20 bp around the variant
SEED        = 42
DIRICHLET_ALPHA = 0.5

# Cancer-gene panels mirroring TCGA BRCA / LUAD / COAD mutation profiles
CANCER_GENE_PANELS: Dict[str, Set[str]] = {
    "breast": {
        "BRCA1", "BRCA2", "ERBB2", "TP53", "PIK3CA", "GATA3", "CDH1",
        "MAP3K1", "PTEN", "AKT1", "ESR1", "FOXA1", "RB1", "MYC",
    },
    "lung": {
        "EGFR", "KRAS", "ALK", "MET", "STK11", "KEAP1", "NF1", "BRAF",
        "ROS1", "RET", "ERBB2", "TP53", "RB1", "MYC", "CDKN2A",
    },
    "colorectal": {
        "APC", "KRAS", "TP53", "BRAF", "PIK3CA", "SMAD4", "MLH1", "MSH2",
        "MSH6", "PMS2", "NRAS", "FBXW7", "SOX9", "TCF7L2", "CTNNB1",
    },
}

SITE_NAMES = ["site_1", "site_2", "site_3"]

PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic"}
BENIGN_TERMS     = {"Benign", "Likely benign"}


def load_clinvar(path: Path, max_per_class: int = 5_000) -> pd.DataFrame:
    print(f"Loading ClinVar from {path.name} …")
    usecols = [
        "Name", "Chromosome", "ClinicalSignificance",
        "GeneSymbol", "Type", "Start", "Stop",
    ]
    df = pd.read_csv(
        path, sep="\t",
        usecols=lambda c: c in usecols,
        low_memory=False,
        on_bad_lines="skip",
    )

    df["Chromosome"] = df["Chromosome"].astype(str).str.replace("chr", "", regex=False)
    df["Start"]      = pd.to_numeric(df["Start"], errors="coerce")
    df["Stop"]       = pd.to_numeric(df["Stop"], errors="coerce")

    def _label(sig: str) -> int:
        if any(p in sig for p in PATHOGENIC_TERMS):
            return 1
        if any(b in sig for b in BENIGN_TERMS):
            return 0
        return -1

    df["label"] = df["ClinicalSignificance"].fillna("").apply(_label)
    df = df[df["label"] >= 0]
    df = df[df["Type"].astype(str).str.contains("single nucleotide", case=False, na=False)]
    df = df.dropna(subset=["Start", "Chromosome"])
    df["Start"] = df["Start"].astype(int)
    df["Stop"]  = df["Stop"].fillna(df["Start"]).astype(int)

    pos = df[df["label"] == 1].head(max_per_class)
    neg = df[df["label"] == 0].head(max_per_class)
    df  = pd.concat([pos, neg], ignore_index=True)
    print(f"  Candidate variants: {len(df)}  (pathogenic={len(pos)}, benign={len(neg)})")
    return df


def _cancer_subtype(gene: str) -> str:
    g = str(gene).upper()
    for subtype, panel in CANCER_GENE_PANELS.items():
        if g in panel:
            return subtype
    return "other"


def attach_genomic_context(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch real reference sequences from Ensembl; drop variants we cannot resolve."""
    fetcher = GenomicContextFetcher(context_len=CONTEXT_LEN, flank=20)
    contexts: List[str] = []
    kept_idx: List[int] = []
    failed = 0

    for i, row in df.iterrows():
        ctx = fetcher.get_context(
            str(row["Chromosome"]),
            int(row["Start"]),
            int(row["Stop"]),
        )
        if ctx:
            contexts.append(ctx)
            kept_idx.append(i)
        else:
            failed += 1
        if (len(contexts) + failed) % 200 == 0:
            print(f"  Context fetch progress: {len(contexts)} ok, {failed} failed …")

    fetcher.save_cache()
    out = df.loc[kept_idx].copy()
    out["context"] = contexts
    print(f"  Resolved {len(out)} variants with real Ensembl context ({failed} failed).")
    if len(out) < 100:
        raise RuntimeError(
            "Too few variants with real genomic context. "
            "Check network access to rest.ensembl.org and ClinVar Start positions."
        )
    return out.reset_index(drop=True)


def dirichlet_partition(df: pd.DataFrame, alpha: float = DIRICHLET_ALPHA) -> pd.DataFrame:
    """
    Non-IID partition: for each cancer subtype pool, split variants across
    3 sites using Dirichlet(α, α, α) proportions (FedAlert-style heterogeneity).
    """
    rng  = np.random.default_rng(SEED)
    df   = df.copy()
    df["cancer_subtype"] = df["GeneSymbol"].apply(_cancer_subtype)
    df["site"] = ""

    for subtype in list(CANCER_GENE_PANELS.keys()) + ["other"]:
        pool = df.index[df["cancer_subtype"] == subtype].tolist()
        n    = len(pool)
        if n == 0:
            continue
        rng.shuffle(pool)
        props = rng.dirichlet([alpha] * 3)
        counts = rng.multinomial(n, props)
        pos = 0
        for site_name, count in zip(SITE_NAMES, counts):
            for idx in pool[pos: pos + count]:
                df.at[idx, "site"] = site_name
            pos += count

    unassigned = df["site"] == ""
    if unassigned.any():
        fallback = [SITE_NAMES[i % 3] for i in range(unassigned.sum())]
        df.loc[unassigned, "site"] = fallback

    for site in SITE_NAMES:
        n = (df["site"] == site).sum()
        print(f"  {site}: {n} variants")
    return df


def save_partitions(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = df[[
        "context", "label", "site", "GeneSymbol", "Chromosome",
        "cancer_subtype", "Start",
    ]].to_dict("records")

    with (DATA_DIR / "variants.json").open("w") as fh:
        json.dump(records, fh)
    print(f"  Saved {len(records)} total variants → {DATA_DIR / 'variants.json'}")

    for site in SITE_NAMES:
        subset = [r for r in records if r["site"] == site]
        with (DATA_DIR / f"{site}.json").open("w") as fh:
            json.dump(subset, fh)
        n_pos = sum(1 for r in subset if r["label"] == 1)
        n_neg = sum(1 for r in subset if r["label"] == 0)
        print(f"  {site}: {len(subset)} variants  (P={n_pos}, B={n_neg})")


def main() -> None:
    path = DATA_DIR / "variant_summary.txt"
    if not path.exists():
        raise FileNotFoundError(
            "ClinVar data not found.\n"
            "Run: python scripts/download_data.py"
        )
    df = load_clinvar(path)
    df = attach_genomic_context(df)
    df = dirichlet_partition(df)
    save_partitions(df)
    print("\nData preparation complete. Run: python run_federation.py")


if __name__ == "__main__":
    main()
