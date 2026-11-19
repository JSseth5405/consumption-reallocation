from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import disaggregate as DA
from country import PREFIX

MEASURES = {"nominal_meur": "nominal", "real_meur": "real"}


def indicator_series(root: pathlib.Path) -> dict[str, pd.Series]:
    frame = pd.read_csv(root / f"data/interim/{PREFIX}_quarterly_indicator.csv")
    frame["q"] = pd.PeriodIndex(frame.period, freq="Q")
    frame = frame.sort_values("q").set_index("q")
    return {"nominal": frame["nominal"].astype(float),
            "real": frame["real"].astype(float)}


def division_indicators(root: pathlib.Path) -> dict[tuple[str, str], pd.Series]:
    path = root / f"data/interim/{PREFIX}_division_indicators.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame["q"] = pd.PeriodIndex(frame.period, freq="Q")
    frame = frame.sort_values("q")
    out = {}
    for (category,), block in frame.groupby(["category"]):
        for measure in ("nominal", "real"):
            out[(category, measure)] = (block.set_index("q")[measure]
                                        .astype(float).dropna())
    return out


def build(root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual = pd.read_csv(root / f"data/interim/{PREFIX}_annual_divisions.csv")
    indicators = indicator_series(root)
    per_division = division_indicators(root)
    observed = {str(q) for q in indicators["nominal"].index}

    rows, log = [], []
    for category, block in annual.groupby("category"):
        for column, label in MEASURES.items():
            series = (block.dropna(subset=[column])
                      .set_index("year")[column].astype(float).sort_index())
            if series.empty:
                continue
            flow = (label == "nominal")
            own = per_division.get((category, label))
            indicator = own if own is not None else indicators[label]
            quarterly = DA.denton_cholette(series, indicator, flow=flow)
            for period, value in quarterly.items():
                rows.append({"period": str(period), "category": category,
                             "measure": label, "value": float(value)})
            got = quarterly.groupby(quarterly.index.year).sum()
            got = got.reindex(series.index).dropna()
            if not flow:
                got = got / 4.0
            want = series.reindex(got.index)
            worst = (float(((got - want) / want).abs().max() * 100)
                     if len(got) else np.nan)
            fit = DA.chow_lin_check(series, indicator)
            base = DA.chow_lin_check(series, indicators[label])
            log.append({"category": category, "measure": label,
                        "years_benchmarked": int(len(got)),
                        "max_abs_benchmark_error_pc": worst,
                        "indicator": ("own_durability_blend" if own is not None
                                      else "aggregate"),
                        "indicator_r2": fit["indicator_r2"],
                        "indicator_corr_growth": fit["indicator_corr_growth"],
                        "aggregate_indicator_r2": base["indicator_r2"],
                        "aggregate_corr_growth": base["indicator_corr_growth"]})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["period", "category"], columns="measure",
                            values="value").reset_index()
    wide = wide.rename(columns={"nominal": "nominal_meur", "real": "real_meur"})
    wide["observed_indicator"] = wide.period.isin(observed).astype(int)
    wide["year"] = wide.period.str.slice(0, 4).astype(int)
    order = sorted(wide.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    rank = {p: i for i, p in enumerate(order)}
    wide = wide.sort_values(["category", "period"],
                            key=lambda s: s.map(rank) if s.name == "period" else s)
    return wide.reset_index(drop=True), pd.DataFrame(log)


def main(base: str) -> None:
    root = pathlib.Path(base)
    panel, log = build(root)
    panel.to_csv(root / f"data/interim/{PREFIX}_quarterly_divisions.csv", index=False)
    log.to_csv(root / "metadata/disaggregation_benchmark_check.csv", index=False)
    quarters = sorted(panel.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    print("quarterly divisions:", panel.shape, "|",
          panel.category.nunique(), "divisions |", len(quarters), "quarters",
          quarters[0], "-", quarters[-1])
    worst = float(log.max_abs_benchmark_error_pc.max())
    print(f"annual benchmark reproduced to {worst:.10f}% at worst "
          f"({int(log.years_benchmarked.min())}-{int(log.years_benchmarked.max())} "
          f"years per division)")
    share = 100 * panel.observed_indicator.mean()
    print(f"quarters whose benchmarking indicator was OBSERVED, not itself "
          f"generated: {share:.1f}%")
    used = log.indicator.value_counts().to_dict()
    print("benchmarking indicator used:", used)
    fit = log.groupby("measure")[["indicator_r2", "aggregate_indicator_r2"]].mean()
    print("indicator fit (annualised indicator vs the annual benchmark, mean "
          "R^2 over divisions):")
    for measure, row in fit.iterrows():
        print(f"  {measure:8s} own durability blend {row.indicator_r2:.4f}   "
              f"vs single aggregate {row.aggregate_indicator_r2:.4f}")
    g = log.groupby("measure")[["indicator_corr_growth",
                                "aggregate_corr_growth"]].mean()
    print("growth-rate correlation with the annual benchmark:")
    for measure, row in g.iterrows():
        print(f"  {measure:8s} own durability blend "
              f"{row.indicator_corr_growth:.4f}   vs single aggregate "
              f"{row.aggregate_corr_growth:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
