from __future__ import annotations

import pathlib
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _fred(series: str) -> pd.Series:
    cache = ROOT / "data/oil_price/raw" / f"fred_{series}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                        headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
                    cache.write_bytes(r.read())
                break
            except Exception as exc:
                last = exc
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"FRED download failed for {series}: {last}")
    frame = pd.read_csv(cache)
    frame.columns = ["date", "value"]
    frame = frame[pd.to_numeric(frame.value, errors="coerce").notna()]
    return pd.Series(frame.value.astype(float).to_numpy(), index=pd.to_datetime(frame.date))


def oil_growth() -> pd.Series:
    wti, cpi = _fred("WTISPLC"), _fred("CPIAUCSL")
    q = lambda s: s.groupby(s.index.to_period("Q")).mean()
    real = (q(wti) / q(cpi)).dropna()
    g = 100 * (np.log(real) - np.log(real.shift(4)))
    g.index = g.index.astype(str)
    return g.dropna()


if __name__ == "__main__":
    g = oil_growth()
    print(f"real oil growth: {g.index.min()}-{g.index.max()}, n={len(g)}")
    print(g.loc[["2008Q3", "2008Q4", "2009Q1", "2020Q2", "2021Q2"]].round(1).to_string())
