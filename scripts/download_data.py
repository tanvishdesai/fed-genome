"""
FedGenome Data Downloader
==========================
Downloads ClinVar variant_summary.txt from NCBI FTP.

Usage:
  python scripts/download_data.py           # full ClinVar (~400 MB)
  python scripts/download_data.py --sample  # sample (faster for testing)

On Kaggle:
  The ClinVar conflicting dataset (kevinarvai/clinvar-conflicting) is available
  directly. Add it to your notebook and use --kaggle flag.
"""

from __future__ import annotations

import argparse
import gzip
import io
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)


def _download_and_decompress(url: str, dest: Path) -> None:
    print(f"Downloading {url} …")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = dest.with_suffix(".txt.gz")
    with urllib.request.urlopen(url) as resp, open(gz_path, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    print(f"  Decompressing …")
    with gzip.open(gz_path, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz_path.unlink()
    size_mb = dest.stat().st_size // 1_000_000
    print(f"  Saved to {dest}  ({size_mb} MB)")


def copy_from_kaggle() -> None:
    """
    On Kaggle, use the ClinVar conflicting dataset.
    Add: https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting
    Path: /kaggle/input/clinvar-conflicting/clinvar_conflicting.csv
    """
    src = Path("/kaggle/input/clinvar-conflicting/clinvar_conflicting.csv")
    if not src.exists():
        print("Kaggle ClinVar dataset not found at /kaggle/input/clinvar-conflicting/")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / "variant_summary.txt"
    # Convert CSV to TSV-like format that prepare_data.py expects
    import pandas as pd
    df = pd.read_csv(src, low_memory=False)
    # Map columns to expected names
    rename = {
        "gene_symbol": "GeneSymbol",
        "CLNSIG":      "ClinicalSignificance",
        "CHROM":       "Chromosome",
        "Type":        "Type",
        "Name":        "Name",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Add missing columns
    for col in ["GeneSymbol", "ClinicalSignificance", "Chromosome", "Type", "Name"]:
        if col not in df.columns:
            df[col] = ""
    df.to_csv(dest, sep="\t", index=False)
    print(f"Copied Kaggle ClinVar → {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kaggle", action="store_true",
        help="Copy from Kaggle input dataset instead of downloading",
    )
    args = parser.parse_args()

    dest = DATA_DIR / "variant_summary.txt"
    if dest.exists():
        print(f"ClinVar already present at {dest}")
    elif args.kaggle:
        copy_from_kaggle()
    else:
        _download_and_decompress(CLINVAR_URL, dest)

    print("\nDone. Run: python prepare_data.py")
