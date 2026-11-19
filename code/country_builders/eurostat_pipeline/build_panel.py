from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import protocol as P
from country import PREFIX


def main(base: str) -> None:
    root = pathlib.Path(base)
    raw = pd.read_csv(root / f"data/interim/{PREFIX}_quarterly_divisions.csv")
    nominal = raw.pivot_table(index="period", columns="category",
                              values="nominal_meur")
    real = raw.pivot_table(index="period", columns="category", values="real_meur")
    order = sorted(nominal.index, key=lambda q: pd.Period(q, freq="Q"))
    nominal, real = nominal.reindex(order), real.reindex(order)

    shares = nominal.div(nominal.sum(axis=1), axis=0)
    log_shares = np.log(shares)
    clr = log_shares.sub(log_shares.mean(axis=1), axis=0)

    growth4 = 100 * (np.log(real) - np.log(real.shift(4)))
    growth1 = 100 * (np.log(real) - np.log(real.shift(1)))

    weights = pd.read_csv(root / "config/exposure_weights.csv")
    intensity = weights.set_index("category")

    rows = []
    for period in order:
        for category in nominal.columns:
            row = {
                "period": period, "category": category,
                "nominal_meur": float(nominal.at[period, category]),
                "real_index": float(real.at[period, category]),
                "share": float(100 * shares.at[period, category]),
                "clr_share": float(clr.at[period, category]),
                "real_growth": float(growth4.at[period, category]),
                "real_growth_qoq": float(growth1.at[period, category]),
                "weight": float(shares.at[period, category]),
                "necessity": int(category in P.NECESSITY),
            }
            for column in intensity.columns:
                if column.endswith("_intensity"):
                    row[column] = float(intensity.at[category, column])
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel["year"] = panel.period.str.slice(0, 4).astype(int)

    for shock in P.SHOCKS:
        impact = set(P.impact_periods(shock))
        window = set(P.window_periods(shock))
        panel[f"impact_{shock}"] = panel.period.isin(impact).astype(int)
        panel[f"window_{shock}"] = panel.period.isin(window).astype(int)
        panel[f"exposed_{shock}"] = panel[P.EXPOSURE_MARGIN[shock]]
    for name, spec in P.SAMPLES.items():
        panel[f"in_{name}"] = ((panel.period >= spec["start"])
                               & (panel.period <= spec["end"])).astype(int)

    panel.to_csv(root / f"data/final/{PREFIX}_quarterly_category_panel.csv", index=False)
    print(f"{PREFIX} quarterly category panel:", panel.shape,
          "|", panel.category.nunique(), "divisions |",
          panel.period.nunique(), "quarters",
          panel.period.min(), "-", panel.period.max())
    check = panel.groupby("period")["share"].sum().round(6)
    print("share sums:", check.min(), "-", check.max())
    for name, spec in P.SAMPLES.items():
        sub = panel[panel[f"in_{name}"] == 1]
        usable = sub.dropna(subset=["real_growth"])
        print(f"  {name:13s} {spec['start']}+: {sub.period.nunique():3d} quarters, "
              f"{len(usable):5d} usable category-quarters")

    print("\nfour-quarter real growth in each impact phase (per cent):")
    for shock in P.SHOCKS:
        block = panel[panel.period.isin(P.impact_periods(shock))]
        agg = (block.dropna(subset=["real_growth"])
               .groupby("period")
               .apply(lambda d: float((d.real_growth * d.weight).sum()
                                      / d.weight.sum()), include_groups=False))
        if not agg.empty:
            print(f"  {shock:8s} " + "  ".join(f"{q} {v:6.2f}"
                                               for q, v in agg.items()))


if __name__ == "__main__":
    main(sys.argv[1])
