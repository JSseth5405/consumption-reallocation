from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
from country import GEO, PREFIX
NOMINAL, REAL = "CP_MEUR", "CLV15_MEUR"

DIVISION = {
    "CP01": "food_nonalc",
    "CP02": "alcohol_tobacco",
    "CP03": "clothing_footwear",
    "CP04": "housing_utilities",
    "CP05": "household_goods",
    "CP06": "health",
    "CP07": "transport",
    "CP08": "communication",
    "CP09": "recreation_culture",
    "CP10": "education",
    "CP11": "restaurants_hotels",
    "CP12": "miscellaneous",
    "CP13": "miscellaneous",
}
LEGACY = {f"CP{i:02d}": v for i, v in enumerate(
    ["food_nonalc", "alcohol_tobacco", "clothing_footwear", "housing_utilities",
     "household_goods", "health", "transport", "communication",
     "recreation_culture", "education", "restaurants_hotels", "miscellaneous"],
    start=1)}


def fetch(root: pathlib.Path, dataset: str, params: dict, tag: str) -> dict:
    cache = root / "data/raw/eurostat" / GEO / f"{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    query = [("format", "JSON"), ("lang", "EN"), ("geo", GEO)]
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            query.extend((key, v) for v in value)
        else:
            query.append((key, value))
    url = f"{API}/{dataset}?" + urllib.parse.urlencode(query)
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last = exc
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"Eurostat fetch failed after 5 attempts: {last}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload))
    return payload


def tidy(payload: dict, code_dim: str) -> pd.DataFrame:
    dims = payload["dimension"]
    order = payload.get("id") or list(dims.keys())
    sizes = payload.get("size") or [len(dims[d]["category"]["index"]) for d in order]
    labels = {}
    for d in order:
        index = dims[d]["category"]["index"]
        labels[d] = {v: k for k, v in index.items()}
    rows = []
    for flat, value in payload["value"].items():
        rest = int(flat)
        position = {}
        for d, size in zip(reversed(order), reversed(sizes)):
            position[d] = labels[d][rest % size]
            rest //= size
        rows.append({"time": position["time"], "code": position[code_dim],
                     "value": float(value)})
    return pd.DataFrame(rows)


def annual_divisions(root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for measure, unit in (("nominal", NOMINAL), ("real", REAL)):
        payload = fetch(root, "nama_10_cp18",
                        {"unit": unit, "coicop18": sorted(set(DIVISION))},
                        f"nama_10_cp18_{unit}")
        frame = tidy(payload, "coicop18")
        frame["category"] = frame.code.map(DIVISION)
        frame["measure"] = measure
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    nominal = (long[long.measure == "nominal"]
               .groupby(["time", "category"], as_index=False)["value"].sum())
    real = real_from_parts(long)
    wide = (nominal.rename(columns={"value": "nominal_meur"})
            .merge(real.rename(columns={"value": "real_meur"}),
                   on=["time", "category"], how="inner"))
    wide["year"] = wide.time.astype(int)
    audit = (long.groupby(["measure", "category"])
             .agg(years=("time", "nunique"), codes=("code", "nunique"))
             .reset_index())
    return wide.sort_values(["category", "year"]).reset_index(drop=True), audit


def real_from_parts(long: pd.DataFrame) -> pd.DataFrame:
    nom = long[long.measure == "nominal"].set_index(["time", "code"])["value"]
    rea = long[long.measure == "real"].set_index(["time", "code"])["value"]
    rows = []
    for category in sorted(set(DIVISION.values())):
        codes = [c for c, v in DIVISION.items() if v == category]
        if len(codes) == 1:
            part = rea.xs(codes[0], level="code")
            rows.extend({"time": t, "category": category, "value": float(v)}
                        for t, v in part.items())
            continue
        times = set.intersection(*[set(rea.xs(c, level="code").index) for c in codes])
        for t in sorted(times):
            n = sum(float(nom.loc[(t, c)]) for c in codes)
            r = sum(float(rea.loc[(t, c)]) for c in codes)
            rows.append({"time": t, "category": category,
                         "value": n / (n / r) if r else float("nan")})
    return pd.DataFrame(rows)


def quarterly_indicator(root: pathlib.Path) -> pd.DataFrame:
    out = {}
    for measure, unit in (("nominal", NOMINAL), ("real", REAL)):
        payload = fetch(root, "namq_10_fcs",
                        {"unit": unit, "s_adj": "SCA", "na_item": "P31_S14"},
                        f"namq_10_fcs_{unit}")
        frame = tidy(payload, "na_item")
        frame["period"] = frame.time.str.replace("-", "", regex=False)
        out[measure] = frame.set_index("period")["value"]
    joined = pd.DataFrame(out).dropna().reset_index()
    joined["q"] = pd.PeriodIndex(joined.period, freq="Q")
    return joined.sort_values("q").drop(columns=["q"]).reset_index(drop=True)


def reconcile(root: pathlib.Path, annual: pd.DataFrame) -> pd.DataFrame:
    payload = fetch(root, "nama_10_cp18",
                    {"unit": NOMINAL, "coicop18": "TOTAL"},
                    f"nama_10_cp18_{NOMINAL}_TOTAL")
    published = tidy(payload, "coicop18").set_index("time")["value"]
    built = annual.groupby("time").nominal_meur.sum()
    out = pd.concat([published.rename("published"), built.rename("built")],
                    axis=1).dropna()
    out["deviation_pc"] = 100 * (out.built - out.published) / out.published
    return out.reset_index().rename(columns={"index": "time"})


def validate(root: pathlib.Path, annual: pd.DataFrame) -> pd.DataFrame:
    payload = fetch(root, "nama_10_co3_p3",
                    {"unit": NOMINAL, "coicop": sorted(LEGACY)},
                    f"nama_10_co3_p3_{NOMINAL}")
    old = tidy(payload, "coicop")
    old["category"] = old.code.map(LEGACY)
    old = old.groupby(["time", "category"], as_index=False)["value"].sum()
    joined = annual.merge(old.rename(columns={"value": "legacy_meur"}),
                          on=["time", "category"], how="inner")
    if joined.empty:
        return pd.DataFrame()
    joined["gap_pc"] = (100 * (joined.nominal_meur - joined.legacy_meur)
                        / joined.legacy_meur)
    return (joined.groupby("category")
            .agg(years=("time", "size"),
                 max_abs_gap_pc=("gap_pc", lambda s: float(s.abs().max())),
                 mean_gap_pc=("gap_pc", "mean"))
            .reset_index())


def main(base: str) -> None:
    root = pathlib.Path(base)
    annual, audit = annual_divisions(root)
    indicator = quarterly_indicator(root)

    check = validate(root, annual)
    if not check.empty:
        check.to_csv(root / "metadata/coicop2018_vs_coicop1999.csv", index=False)
        print("COICOP 2018 (used) vs the retired COICOP 1999 series, mean gap "
              "per division over their shared years - a concordance report, "
              "not a test; see validate() for why each gap is there:")
        print(check.sort_values("mean_gap_pc")
              .assign(mean_gap_pc=lambda d: d.mean_gap_pc.round(1))
              [["category", "years", "mean_gap_pc"]].to_string(index=False))

    annual.to_csv(root / f"data/interim/{PREFIX}_annual_divisions.csv", index=False)
    indicator.to_csv(root / f"data/interim/{PREFIX}_quarterly_indicator.csv",
                     index=False)
    audit.to_csv(root / "metadata/eurostat_series_used.csv", index=False)

    years = sorted(annual.year.unique())
    print(f"{PREFIX} annual divisions:", annual.shape, "|",
          annual.category.nunique(), "divisions |", len(years), "years",
          years[0], "-", years[-1])
    print("quarterly indicator (observed):", len(indicator), "quarters",
          indicator.period.iloc[0], "-", indicator.period.iloc[-1])

    rec = reconcile(root, annual)
    rec.to_csv(root / f"metadata/{PREFIX}_reconciliation.csv", index=False)
    print(f"twelve divisions vs Eurostat's published total household "
          f"consumption: max absolute deviation "
          f"{rec.deviation_pc.abs().max():.6f}% over {len(rec)} years")


if __name__ == "__main__":
    main(sys.argv[1])
