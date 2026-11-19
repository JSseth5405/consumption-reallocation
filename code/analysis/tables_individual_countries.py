from __future__ import annotations

import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import study_protocol as P

SETTLED = ("supported", "rejected_opposite_sign", "negligible_effect")
MAIN = ["H1", "H2", "H3", "H4"]


def counts(ind: pd.DataFrame) -> pd.DataFrame:
    main = ind[ind.hypothesis.isin(MAIN)]
    rows = []
    for (country, timeline), blk in main.groupby(["country", "timeline"], sort=False):
        row = {"country": country, "timeline": timeline, "window": blk.window.iloc[0],
               "quarters": int(blk.quarters.iloc[0]), "time_cluster": blk.time_cluster.iloc[0]}
        for spec in P.SPECS:
            s = blk[blk.spec == spec]
            row[f"{spec}_settled"] = int(s.decision_headline.isin(SETTLED).sum())
            row[f"{spec}_supported"] = int((s.decision_headline == "supported").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def changes(ind: pd.DataFrame) -> pd.DataFrame:
    main = ind[ind.hypothesis.isin(MAIN)]
    a = main[main.spec == "core"].set_index(["country", "timeline", "pool", "hypothesis"])
    b = main[main.spec == "clean"].set_index(["country", "timeline", "pool", "hypothesis"])
    j = a[["estimate", "std_error", "decision_headline"]].join(
        b[["estimate", "std_error", "decision_headline"]], lsuffix="_core", rsuffix="_clean")
    return j[j.decision_headline_core != j.decision_headline_clean].reset_index()


def main(base: str) -> None:
    root = pathlib.Path(base)
    out = root / "results/tables"
    ind = pd.read_csv(out / "individual_results.csv")
    c = counts(ind)
    c.to_csv(out / "individual_counts_by_spec.csv", index=False)
    ch = changes(ind)
    ch.to_csv(out / "individual_core_vs_clean_changes.csv", index=False)
    comb = pd.read_csv(out / "individual_results_combined.csv")
    cc = []
    for (country, timeline), blk in comb.groupby(["country", "timeline"], sort=False):
        row = {"country": country, "timeline": timeline, "window": blk.window.iloc[0]}
        for spec in P.SPECS:
            s = blk[blk.spec == spec]
            row[f"{spec}_settled"] = int(s.decision_headline.isin(SETTLED).sum())
            row[f"{spec}_supported"] = int((s.decision_headline == "supported").sum())
        cc.append(row)
    cc = pd.DataFrame(cc)
    cc.to_csv(out / "individual_combined_counts_by_spec.csv", index=False)
    print("settled cells out of 12 (H1-H4 x 3 shocks), by specification:")
    print(c[["country", "timeline", "window"] + [f"{s}_settled" for s in P.SPECS]].to_string(index=False))
    tot = c[[f"{s}_settled" for s in P.SPECS]].sum()
    print("totals:", tot.to_dict(), f"of {12 * len(c)}")
    print(f"\ncells whose verdict changes from core to clean: {len(ch)}")
    tc = cc[[f"{s}_supported" for s in P.SPECS]].sum()
    print("three shocks together, supported of", 4 * len(cc), ":", tc.to_dict())


if __name__ == "__main__":
    main(sys.argv[1])
