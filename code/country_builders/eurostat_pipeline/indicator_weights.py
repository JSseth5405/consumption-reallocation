from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import optimize

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import extract_eurostat as E

CLASSES = ["durable", "semi_durable", "non_durable", "services"]
NA_ITEM = {"durable": "P311_S14", "semi_durable": "P312_S14",
           "non_durable": "P313_S14", "services": "P314_S14"}

DURABILITY = {
    "CP011": "non_durable",
    "CP012": "non_durable",
    "CP013": "services",
    "CP021": "non_durable",
    "CP022": "services",
    "CP023": "non_durable",
    "CP024": "non_durable",
    "CP031": "semi_durable",
    "CP032": "semi_durable",
    "CP041": "services",
    "CP042": "services",
    "CP043": "services",
    "CP044": "services",
    "CP045": "non_durable",
    "CP051": "durable",
    "CP052": "semi_durable",
    "CP053": "durable",
    "CP054": "semi_durable",
    "CP055": "semi_durable",
    "CP056": "services",
    "CP061": "non_durable",
    "CP062": "services",
    "CP063": "services",
    "CP064": "services",
    "CP071": "durable",
    "CP072": "non_durable",
    "CP073": "services",
    "CP074": "services",
    "CP081": "durable",
    "CP082": "semi_durable",
    "CP083": "services",
    "CP091": "durable",
    "CP092": "semi_durable",
    "CP093": "non_durable",
    "CP094": "services",
    "CP095": "semi_durable",
    "CP096": "services",
    "CP097": "semi_durable",
    "CP098": "services",
    "CP101": "services", "CP102": "services", "CP103": "services",
    "CP104": "services", "CP105": "services",
    "CP111": "services",
    "CP112": "services",
    "CP121": "services",
    "CP122": "services",
    "CP131": "non_durable",
    "CP132": "semi_durable",
    "CP133": "services",
    "CP139": "services",
}
RIDGE = 1e-3


def starting_matrix(root: pathlib.Path) -> pd.DataFrame:
    payload = E.fetch(root, "nama_10_cp18",
                      {"unit": E.NOMINAL, "coicop18": sorted(DURABILITY)},
                      f"nama_10_cp18_{E.NOMINAL}_subitems")
    detail = E.tidy(payload, "coicop18")
    detail["division"] = detail.code.map(lambda c: E.DIVISION[c[:4]])
    detail["dur"] = detail.code.map(DURABILITY)
    share = (detail.groupby(["division", "dur"])["value"].sum()
             .unstack(fill_value=0.0).reindex(columns=CLASSES, fill_value=0.0))
    return share.div(share.sum(axis=1), axis=0)


def published(root: pathlib.Path, freq: str, unit: str) -> pd.DataFrame:
    dataset = "nama_10_fcs" if freq == "a" else "namq_10_fcs"
    params = {"unit": unit}
    if freq == "q":
        params["s_adj"] = "SCA"
    out = {}
    for label, na in NA_ITEM.items():
        payload = E.fetch(root, dataset, {**params, "na_item": na},
                          f"{dataset}_{unit}_{na}" + ("_q" if freq == "q" else ""))
        frame = E.tidy(payload, "na_item")
        if freq == "q":
            frame["time"] = frame.time.str.replace("-", "", regex=False)
        out[label] = frame.set_index("time")["value"]
    return pd.DataFrame(out).dropna()


def calibrate(annual: pd.DataFrame, durability: pd.DataFrame,
              start: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    divisions = list(start.index)
    A = (annual.pivot_table(index="time", columns="category",
                            values="nominal_meur")[divisions].dropna())
    D = durability.reindex(A.index).dropna()
    A = A.reindex(D.index)
    Am, Dm, W0 = A.to_numpy(float), D[CLASSES].to_numpy(float), start.to_numpy(float)
    scale = Am.sum(axis=1, keepdims=True)
    Am, Dm = Am / scale, Dm / scale
    n_c, n_d = W0.shape

    def residual(flat):
        W = flat.reshape(n_c, n_d)
        return ((Am @ W - Dm) ** 2).sum() / Dm.size + RIDGE * ((W - W0) ** 2).sum()

    def jac(flat):
        W = flat.reshape(n_c, n_d)
        g = 2.0 * Am.T @ (Am @ W - Dm) / Dm.size + 2.0 * RIDGE * (W - W0)
        return g.ravel()

    rows = np.repeat(np.arange(n_c), n_d)
    rowsum = np.zeros((n_c, n_c * n_d))
    rowsum[rows, np.arange(n_c * n_d)] = 1.0
    constraint = optimize.LinearConstraint(rowsum, 1.0, 1.0)
    result = optimize.minimize(residual, W0.ravel(), jac=jac, method="SLSQP",
                               bounds=[(0.0, 1.0)] * (n_c * n_d),
                               constraints=[constraint],
                               options={"maxiter": 5000, "ftol": 1e-14})
    W = pd.DataFrame(result.x.reshape(n_c, n_d), index=divisions, columns=CLASSES)
    W = W.clip(lower=0.0)
    W = W.div(W.sum(axis=1), axis=0)
    if residual(W.to_numpy(float).ravel()) > residual(W0.ravel()):
        W = start.copy()
        result.success = False

    def fit(matrix):
        implied = Am @ matrix.to_numpy(float)
        dev = 100 * (implied - Dm) / Dm
        return {f"max_abs_dev_{c}_pc": float(np.abs(dev[:, i]).max())
                for i, c in enumerate(CLASSES)}

    report = {"years": int(len(A)), "converged": bool(result.success)}
    report.update({f"before_{k}": v for k, v in fit(start).items()})
    report.update({f"after_{k}": v for k, v in fit(W).items()})
    return W, report


def build(root: pathlib.Path):
    annual = pd.read_csv(root / f"data/interim/{E.PREFIX}_annual_divisions.csv")
    annual["time"] = annual.year.astype(str)
    start = starting_matrix(root)
    W, report = calibrate(annual, published(root, "a", E.NOMINAL), start)

    indicators = {}
    for measure, unit in (("nominal", E.NOMINAL), ("real", E.REAL)):
        quarterly = published(root, "q", unit)
        blend = quarterly[CLASSES].to_numpy(float) @ W[CLASSES].to_numpy(float).T
        indicators[measure] = pd.DataFrame(blend, index=quarterly.index,
                                           columns=list(W.index))
    return W, report, indicators, start


def main(base: str) -> None:
    root = pathlib.Path(base)
    W, report, indicators, start = build(root)
    W.round(6).to_csv(root / "config/indicator_weights.csv")
    start.round(6).to_csv(root / "metadata/indicator_weights_concordance.csv")
    pd.DataFrame([report]).to_csv(root / "metadata/indicator_weights_fit.csv",
                                  index=False)
    long = []
    for measure, frame in indicators.items():
        melted = frame.reset_index().melt(id_vars="time", var_name="category",
                                          value_name="value")
        melted["measure"] = measure
        long.append(melted)
    out = (pd.concat(long).pivot_table(index=["time", "category"],
                                       columns="measure", values="value")
           .reset_index().rename(columns={"time": "period"}))
    out.to_csv(root / f"data/interim/{E.PREFIX}_division_indicators.csv", index=False)

    print("division-by-durability composition W (calibrated to the published "
          "annual durability aggregates):")
    print(W.round(3).to_string())
    print(f"\nfit against Destatis' published annual durability totals "
          f"({report['years']} years), max absolute deviation:")
    for c in CLASSES:
        print(f"  {c:13s} concordance alone {report[f'before_max_abs_dev_{c}_pc']:6.2f}%"
              f"   ->  calibrated {report[f'after_max_abs_dev_{c}_pc']:6.2f}%")
    per = sorted(out.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    print(f"\nper-division quarterly indicators: {out.category.nunique()} "
          f"divisions x {len(per)} quarters  {per[0]} - {per[-1]}")


if __name__ == "__main__":
    main(sys.argv[1])
