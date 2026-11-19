from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import study_protocol as P


def meta_regression(ctab: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    h1 = ctab[ctab.hypothesis == "H1"][["country", "pool", "estimate", "std_error"]]
    h1 = h1.rename(columns={"pool": "shock"}).merge(
        res[["country", "shock", "resilience"]], on=["country", "shock"])
    out = []
    for shock, blk in list(h1.groupby("shock")) + [("combined", h1)]:
        blk = blk.dropna()
        w = 1.0 / blk.std_error.to_numpy() ** 2
        if shock == "combined":
            X = pd.get_dummies(blk.shock, dtype=float)
            X["resilience"] = blk.resilience.to_numpy()
        else:
            X = pd.DataFrame({"const": 1.0, "resilience": blk.resilience.to_numpy()})
        Xa, y = X.to_numpy(float), blk.estimate.to_numpy(float)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(Xa * sw[:, None], y * sw, rcond=None)
        resid = (y - Xa @ beta) * sw
        df = len(y) - Xa.shape[1]
        s2 = float(resid @ resid) / df
        cov = s2 * np.linalg.inv((Xa * sw[:, None]).T @ (Xa * sw[:, None]))
        j = list(X.columns).index("resilience")
        est, se = float(beta[j]), float(np.sqrt(cov[j, j]))
        out.append({"shock": shock, "n_country_shocks": len(y), "slope": est, "std_error": se,
                    "df": df, "p_one_sided_positive": float(stats.t.sf(est / se, df))})
    return pd.DataFrame(out)


def main(base: str) -> None:
    root = pathlib.Path(base)
    tab = pd.read_csv(root / "results/tables/pooled_results.csv")
    ctab = pd.read_csv(root / "results/tables/per_country_results.csv")
    res = pd.read_csv(root / "data/resilience/resilience_index.csv")
    meta = meta_regression(ctab, res)
    meta.to_csv(root / "results/tables/h5_meta_regression.csv", index=False)
    head = tab[tab.spec == P.HEADLINE_SPEC]
    print("HEADLINE (clean) pooled verdicts:")
    print(head.pivot(index="pool", columns="hypothesis", values="decision_headline")
          .reindex(list(P.SHOCKS) + ["combined"]).to_string())
    print("\nH5 two-step meta-regression (slope of country H1 effect on resilience):")
    print(meta.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
