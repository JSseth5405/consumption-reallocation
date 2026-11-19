from __future__ import annotations

import io
import pathlib
import sys
import time
import urllib.request
import zipfile

import pandas as pd

URL = "https://www150.statcan.gc.ca/n1/tbl/csv/36100124-eng.zip"
SA = "Seasonally adjusted at quarterly rates"
NOMINAL, REAL = "Current prices", "2017 constant prices"

DIVISION = {
    "Food and non-alcoholic beverages": "food_nonalc",
    "Alcoholic beverages, tobacco and cannabis": "alcohol_tobacco",
    "Clothing and footwear": "clothing_footwear",
    "Housing, water, electricity, gas and other fuels": "housing_utilities",
    ("Furnishings, household equipment and other goods and services related to "
     "the dwelling and property"): "household_goods",
    "Health": "health",
    "Transport": "transport",
    "Communications": "communication",
    "Recreation and culture": "recreation_culture",
    "Education": "education",
    "Food, beverage and accommodation services": "restaurants_hotels",
    "Insurance and financial services": "miscellaneous",
    "Miscellaneous goods and services": "miscellaneous",
}
TOTAL = "Household final consumption expenditure"


def raw_table(root: pathlib.Path) -> pd.DataFrame:
    cache = root / "data/raw/statcan/36100124-eng.zip"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(
                        URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r:
                    cache.write_bytes(r.read())
                break
            except Exception as exc:
                last = exc
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"StatCan download failed: {last}")
    with zipfile.ZipFile(cache) as z:
        frame = pd.read_csv(io.BytesIO(z.read("36100124.csv")), low_memory=False)
    frame = frame[(frame["Seasonal adjustment"] == SA)
                  & frame["Prices"].isin([NOMINAL, REAL])
                  & frame["VALUE"].notna()].copy()
    frame["period"] = (frame.REF_DATE.str.slice(0, 4) + "Q"
                       + ((frame.REF_DATE.str.slice(5, 7).astype(int) - 1) // 3 + 1)
                       .astype(str))
    return frame


def build(root: pathlib.Path) -> pd.DataFrame:
    frame = raw_table(root)
    div = frame[frame.Estimates.isin(DIVISION)].copy()
    div["category"] = div.Estimates.map(DIVISION)
    div["measure"] = div.Prices.map({NOMINAL: "nominal_mcad", REAL: "real_mcad"})
    long = (div.groupby(["period", "category", "measure"], as_index=False)["VALUE"].sum())
    wide = long.pivot_table(index=["period", "category"], columns="measure",
                            values="VALUE").reset_index().dropna()
    order = sorted(wide.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    rank = {p: i for i, p in enumerate(order)}
    wide = wide.sort_values(["category", "period"],
                            key=lambda s: s.map(rank) if s.name == "period" else s)
    return wide.reset_index(drop=True)


def reconcile(root: pathlib.Path, panel: pd.DataFrame) -> pd.DataFrame:
    frame = raw_table(root)
    nom = frame[frame.Prices == NOMINAL]
    total = nom[nom.Estimates == TOTAL].set_index("period")["VALUE"]
    excluded = (nom[nom.Estimates.isin(["Net expenditure abroad",
                                        "Household final consumption expenditure, adjusting entry"])]
                .groupby("period")["VALUE"].sum())
    target = (total - excluded.reindex(total.index).fillna(0.0)).rename("published")
    built = panel.groupby("period").nominal_mcad.sum().rename("built")
    out = pd.concat([target, built], axis=1).dropna()
    out["deviation_pc"] = 100 * (out.built - out.published) / out.published
    return out.reset_index().rename(columns={"index": "period"})


def main(base: str) -> None:
    root = pathlib.Path(base)
    panel = build(root)
    (root / "data/interim").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    panel.to_csv(root / "data/interim/canada_quarterly_divisions.csv", index=False)
    per = sorted(panel.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    print("Canada quarterly divisions:", panel.shape, "|", panel.category.nunique(),
          "divisions |", len(per), "quarters", per[0], "-", per[-1])
    rec = reconcile(root, panel)
    rec.to_csv(root / "metadata/canada_reconciliation.csv", index=False)
    print(f"twelve divisions vs StatCan's published total (net of expenditure "
          f"abroad and the adjusting entry): max absolute deviation "
          f"{rec.deviation_pc.abs().max():.6f}% over {len(rec)} quarters")


if __name__ == "__main__":
    main(sys.argv[1])
