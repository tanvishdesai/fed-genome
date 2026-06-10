"""
Real genomic context extraction via the Ensembl REST API (GRCh38).

Fetches ±flank bp around each ClinVar variant position and caches results
locally to avoid repeated network calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

ENSEMBL_SERVER = "https://rest.ensembl.org"
CACHE_PATH = Path(__file__).parent / "data" / "context_cache.json"
RATE_LIMIT_SLEEP = 0.15   # Ensembl allows ~15 req/s; stay conservative


def _normalise_chrom(chrom: str) -> str:
    c = str(chrom).replace("chr", "").strip()
    if c.upper() == "MT":
        return "MT"
    return c


def _fetch_region_sequence(chrom: str, start: int, end: int, flank: int) -> Optional[str]:
    """Fetch genomic sequence for chrom:start..end with flank expansion."""
    chrom = _normalise_chrom(chrom)
    region = f"{chrom}:{start}..{end}:1"
    params = urllib.parse.urlencode({
        "expand_5prime": str(flank),
        "expand_3prime": str(flank),
    })
    url = f"{ENSEMBL_SERVER}/sequence/region/human/{region}?{params}"
    req = urllib.request.Request(url, headers={"Content-Type": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            seq = resp.read().decode("utf-8").strip().upper()
            return seq.replace("N", "A") if seq else None
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            return None
        raise
    except urllib.error.URLError:
        return None


class GenomicContextFetcher:
    """Disk-cached Ensembl fetcher for variant flanking sequences."""

    def __init__(self, context_len: int = 41, flank: int = 20) -> None:
        self.context_len = context_len
        self.flank       = flank
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if CACHE_PATH.exists():
            with CACHE_PATH.open(encoding="utf-8") as fh:
                self._cache = json.load(fh)

    def save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(self._cache, fh)

    def _cache_key(self, chrom: str, start: int, end: int) -> str:
        return f"{_normalise_chrom(chrom)}:{start}-{end}"

    def get_context(self, chrom: str, start: int, end: Optional[int] = None) -> Optional[str]:
        """
        Return a fixed-length DNA context string centred on the variant.

        Uses Ensembl GRCh38 reference sequence. Returns None if the fetch fails.
        """
        if end is None or end < start:
            end = start
        key = self._cache_key(chrom, start, end)
        if key in self._cache:
            return self._cache[key]

        time.sleep(RATE_LIMIT_SLEEP)
        seq = _fetch_region_sequence(chrom, start, end, self.flank)
        if not seq:
            return None

        # Centre-trim or pad to exact context_len
        if len(seq) >= self.context_len:
            mid   = len(seq) // 2
            half  = self.context_len // 2
            seq   = seq[max(0, mid - half): mid - half + self.context_len]
        else:
            seq = seq.ljust(self.context_len, "N").replace("N", "A")

        self._cache[key] = seq
        return seq
