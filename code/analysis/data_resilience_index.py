from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import study_protocol as P

ISO2 = {"USA": "US", "UK": "GB", "Japan": "JP", "India": "IN", "Germany": "DE",
        "France": "FR", "Italy": "IT", "Canada": "CA"}
ISO3 = {"USA": "USA", "UK": "GBR", "Japan": "JPN", "India": "IND", "Germany": "DEU",
        "France": "FRA", "Italy": "ITA", "Canada": "CAN"}


def _get(url: str) -> bytes:
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r:
                return r.read()
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def fred(root: pathlib.Path, series: str) -> pd.Series:
    cache = root / "data/resilience/raw" / f"fred_{series}.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"))
    frame = pd.read_csv(cache)
    frame.columns = ["date", "value"]
    frame = frame[pd.to_numeric(frame.value, errors="coerce").notna()]
    return pd.Series(frame.value.astype(float).to_numpy(),
                     index=pd.to_datetime(frame.date))


def wdi(root: pathlib.Path, indicator: str) -> pd.DataFrame:
    cache = root / "data/resilience/raw" / f"wdi_{indicator}.json"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://api.worldbank.org/v2/country/{';'.join(ISO3.values())}"
               f"/indicator/{indicator}?format=json&per_page=20000&date=1990:2021")
        cache.write_bytes(_get(url))
    rows = json.loads(cache.read_text(encoding="utf-8"))[1] or []
    out = [{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
           for r in rows if r["value"] is not None]
    return pd.DataFrame(out)


def components(root: pathlib.Path) -> pd.DataFrame:
    cpi = wdi(root, "FP.CPI.TOTL.ZG")
    une = wdi(root, "SL.UEM.TOTL.ZS")
    rows = []
    for country in P.COUNTRIES:
        debt = fred(root, f"GGGDTA{ISO2[country]}A188N")
        debt_y = debt.groupby(debt.index.year).last()
        hh = fred(root, f"Q{ISO2[country]}HAM770A")
        hh_q4 = hh[hh.index.month == 10]
        hh_y = pd.Series(hh_q4.to_numpy(), index=hh_q4.index.year)
        for shock, spec in P.SHOCKS.items():
            y = spec["resilience_year"]
            c = cpi[(cpi.iso3 == ISO3[country]) & (cpi.year == y)].value
            u = une[(une.iso3 == ISO3[country]) & (une.year == y)].value
            rows.append({
                "country": country, "shock": shock, "year": y,
                "gov_debt": float(debt_y.get(y, np.nan)),
                "hh_credit_change": (float(hh_y[y] - hh_y[y - 5])
                                     if (y in hh_y.index and y - 5 in hh_y.index)
                                     else np.nan),
                "inflation": float(c.iloc[0]) if len(c) else np.nan,
                "unemployment": float(u.iloc[0]) if len(u) else np.nan,
            })
    frame = pd.DataFrame(rows)
    frame["inflation_gap"] = (frame.inflation - 2.0).abs()
    return frame


def index(frame: pd.DataFrame) -> pd.DataFrame:
    spec = P.RESILIENCE
    names = list(spec["components"])
    out = []
    for shock, block in frame.groupby("shock"):
        block = block.copy()
        z = pd.DataFrame(index=block.index)
        for name in names:
            col = block[name]
            sd = col.std(ddof=1)
            z[name] = (spec["components"][name]["sign"] * (col - col.mean()) / sd
                       if sd and np.isfinite(sd) else np.nan)
        block["n_components"] = z.notna().sum(axis=1)
        raw = z.mean(axis=1, skipna=True)
        raw[block.n_components < spec["min_components"]] = np.nan
        block["resilience_raw"] = raw
        block["resilience"] = (raw - raw.mean()) / raw.std(ddof=1)
        for name in names:
            block[f"z_{name}"] = z[name]
        out.append(block)
    return pd.concat(out).sort_values(["shock", "country"]).reset_index(drop=True)


def main(base: str) -> pd.DataFrame:
    root = pathlib.Path(base)
    table = index(components(root))
    table.to_csv(root / "data/resilience/resilience_index.csv", index=False)
    show = table[["shock", "country", "gov_debt", "hh_credit_change", "inflation",
                  "unemployment", "n_components", "resilience"]].round(2)
    print("pre-shock resilience (higher = more resilient; z across the eight):")
    print(show.to_string(index=False))
    short = table[table.n_components < 4][["shock", "country", "n_components"]]
    if len(short):
        print("\ncomponents missing in the source (index uses the remaining ones):")
        print(short.to_string(index=False))
    return table


if __name__ == "__main__":
    main(sys.argv[1])
