from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import inference_tests as inf
import models_all_countries as M
import study_protocol as P

SUBSETS = {
    "quarterly_observed": ["USA", "UK", "Japan", "Canada"],
    "benchmarked": ["Germany", "France", "Italy", "India"],
}
POOLS = list(P.SHOCKS) + ["combined"]


def main(base: str, log=print) -> pd.DataFrame:
    root = pathlib.Path(base)
    data = M.load_panel()
    rows = []
    for name, countries in SUBSETS.items():
        sub = data[data.country.isin(countries)].copy()
        log(f"{name}: {', '.join(countries)} ({len(sub)} rows)")
        for pool in POOLS:
            for r in M.run_pool(sub, pool, P.HEADLINE_SPEC, log=lambda *a: None, h5_test=False):
                r.update({"subset": name, "countries": ", ".join(countries)})
                rows.append(r)
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for _, blk in table.groupby(["subset", "pool"]):
        ok = blk.p_parametric.notna()
        if ok.any():
            table.loc[blk.index[ok], "p_holm"] = inf.holm(blk.loc[ok, "p_parametric"]).to_numpy()
    out = root / "results/tables/robustness_quarterly_observed.csv"
    table.to_csv(out, index=False)
    full = pd.read_csv(root / "results/tables/pooled_results.csv")
    full = full[full.spec == P.HEADLINE_SPEC].set_index(["pool", "hypothesis"]).decision_headline
    log("\nverdicts (all eight | quarterly-observed four | benchmarked four):")
    for pool in POOLS:
        for h in ["H1", "H2", "H3", "H4"]:
            q = table[(table.subset == "quarterly_observed") & (table.pool == pool) & (table.hypothesis == h)].iloc[0]
            b = table[(table.subset == "benchmarked") & (table.pool == pool) & (table.hypothesis == h)].iloc[0]
            log(f"  {pool:8s} {h}  {full[(pool, h)]:22s} | {q.decision_headline:22s} {q.estimate:+9.3f} "
                f"({q.std_error:.3f}) | {b.decision_headline:22s} {b.estimate:+9.3f} ({b.std_error:.3f})")
    log(f"written {out}")
    return table


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent))
