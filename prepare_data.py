"""
FedGenome Data Preparation
===========================
Reads ClinVar variant_summary.txt, creates pathogenic/benign labels,
synthesises ±20 bp genomic context sequences, and partitions into
3 non-IID hospital sites using chromosome-band separation.

Usage:
  python scripts/download_data.py   # download ClinVar (one-time)
  python prepare_data.py            # build site partitions
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DATA_DIR    = Path(__file__).parent / "data"
CONTEXT_LEN = 41   # ±20 bp + the variant itself
SEED        = 42

# Non-IID partitioning: site i owns chromosomes that tend to carry
# different cancer mutation profiles (BRCA mutations heavy on chr17, etc.)
SITE_CHROMS: Dict[str, set] = {
    "site_1": {"1", "2", "3", "4", "5", "6", "7", "8"},       # early chromosomes
    "site_2": {"9", "10", "11", "12", "13", "14", "15", "16"},
    "site_3": {"17", "18", "19", "20", "21", "22", "X", "Y"},
}

PATHOGENIC_TERMS = {"Pathogenic", "Likely pathogenic"}
BENIGN_TERMS     = {"Benign", "Likely benign"}


def _make_context(name: str, chrom: str, length: int = CONTEXT_LEN) -> str:
    """
    Synthesise a deterministic pseudo-genomic context sequence.
    In a real pipeline, extract ±20 bp flanking context from the reference genome
    using pysam or BioPython's Seq.fetch(); this simulation keeps the demo
    self-contained without requiring a genome FASTA file.
    """
    seed_val = hash(f"{chrom}_{name}") % (2**32)
    rng      = random.Random(seed_val)
    return "".join(rng.choice("ACGT") for _ in range(length))


def load_clinvar(path: Path, max_per_class: int = 10_000) -> pd.DataFrame:
    print(f"Loading ClinVar from {path.name} …")
    usecols = [
        "Name", "Chromosome", "ClinicalSignificance",
        "GeneSymbol", "Type", "Start",
    ]
    df = pd.read_csv(
        path, sep="\t",
        usecols=lambda c: c in usecols,
        low_memory=False,
        on_bad_lines="skip",
    )

    df["Chromosome"] = df["Chromosome"].astype(str).str.replace("chr", "", regex=False)

    def _label(sig: str) -> int:
        if any(p in sig for p in PATHOGENIC_TERMS):  return 1
        if any(b in sig for b in BENIGN_TERMS):       return 0
        return -1

    df["label"] = df["ClinicalSignificance"].fillna("").apply(_label)
    df = df[df["label"] >= 0]

    # Keep only SNVs for simplicity (clean context sequences)
    df = df[df["Type"].astype(str).str.contains("single nucleotide", case=False, na=False)]

    # Balance classes
    pos = df[df["label"] == 1].head(max_per_class)
    neg = df[df["label"] == 0].head(max_per_class)
    df  = pd.concat([pos, neg], ignore_index=True)

    df["context"] = df.apply(
        lambda r: _make_context(str(r["Name"]), str(r["Chromosome"])), axis=1
    )
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"  Total variants: {len(df)}  (pathogenic={len(pos)}, benign={len(neg)})")
    return df


def assign_sites(df: pd.DataFrame) -> pd.DataFrame:
    def _site(chrom: str) -> str:
        for site, chrs in SITE_CHROMS.items():
            if chrom in chrs:
                return site
        return "site_1"
    df["site"] = df["Chromosome"].apply(_site)
    return df


def save_partitions(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = df[["context", "label", "site", "GeneSymbol", "Chromosome"]].to_dict("records")

    # Save combined file (for centralized baseline)
    with (DATA_DIR / "variants.json").open("w") as fh:
        json.dump(records, fh)
    print(f"  Saved {len(records)} total variants → {DATA_DIR / 'variants.json'}")

    # Save per-site partitions
    for site in SITE_CHROMS:
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
            "Run: python scripts/download_data.py\n"
            "Or on Kaggle: set up the ClinVar FTP download in a cell."
        )
    df = load_clinvar(path)
    df = assign_sites(df)
    save_partitions(df)
    print(f"\nData preparation complete. Run: python run_federation.py")


if __name__ == "__main__":
    main()
