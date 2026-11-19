from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260917)


def _difference_matrix(n: int) -> np.ndarray:
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    return D


def _aggregation_matrix(years: np.ndarray, flow: bool) -> np.ndarray:
    unique = np.unique(years)
    C = np.zeros((len(unique), len(years)))
    for i, year in enumerate(unique):
        mask = years == year
        C[i, mask] = 1.0 if flow else 1.0 / mask.sum()
    return C


def denton_cholette(annual: pd.Series, indicator: pd.Series, *,
                    flow: bool = True, ridge: float = 1e-8) -> pd.Series:
    indicator = indicator.dropna()
    if indicator.empty:
        raise ValueError("indicator is empty")
    years_q = indicator.index.year.to_numpy()
    usable = sorted(set(annual.dropna().index).intersection(set(np.unique(years_q))))
    usable = [y for y in usable if (years_q == y).sum() == 4]
    if not usable:
        raise ValueError("no complete overlapping year")
    mask = np.isin(years_q, usable)
    z = indicator.to_numpy(float)[mask]
    periods = indicator.index[mask]
    years = years_q[mask]
    a = annual.loc[usable].to_numpy(float)

    shift = 0.0
    if z.min() <= 0:
        shift = abs(z.min()) + 0.01 * (np.ptp(z) if np.ptp(z) > 0 else 1.0)
        z = z + shift

    n = len(z)
    D = _difference_matrix(n)
    Z = np.diag(z)
    C = _aggregation_matrix(years, flow) @ Z
    H = 2.0 * (D.T @ D) + ridge * np.eye(n)
    k = C.shape[0]
    KKT = np.block([[H, C.T], [C, np.zeros((k, k))]])
    rhs = np.concatenate([np.zeros(n), a])
    solution, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
    u = solution[:n]
    return pd.Series(z * u, index=periods)


def smooth_benchmark(annual: pd.Series, periods: pd.PeriodIndex, *,
                     flow: bool = True) -> pd.Series:
    indicator = pd.Series(1.0, index=periods)
    return denton_cholette(annual, indicator, flow=flow)


def monte_carlo_band(annual: pd.Series, indicator: pd.Series, *,
                     flow: bool = True, draws: int = 300) -> pd.DataFrame:
    base = denton_cholette(annual, indicator, flow=flow)
    clean = indicator.dropna()
    growth = np.diff(np.log(np.abs(clean.to_numpy(float)) + 1e-12))
    if len(growth) < 8:
        return pd.DataFrame({"value": base, "ensemble_sd": 0.0})
    phi = float(np.corrcoef(growth[:-1], growth[1:])[0, 1])
    phi = np.clip(phi if np.isfinite(phi) else 0.0, -0.9, 0.9)
    sigma = float(np.std(growth - phi * np.concatenate([[growth[0]], growth[:-1]])))

    paths = []
    for _ in range(draws):
        shocks = RNG.normal(0.0, sigma, size=len(growth))
        simulated = np.empty_like(growth)
        previous = 0.0
        for i in range(len(growth)):
            previous = phi * previous + shocks[i]
            simulated[i] = growth[i] + previous
        rebuilt = np.concatenate([[clean.to_numpy(float)[0]],
                                  clean.to_numpy(float)[0] * np.exp(np.cumsum(simulated))])
        try:
            path = denton_cholette(annual, pd.Series(rebuilt, index=clean.index),
                                   flow=flow)
        except Exception:
            continue
        paths.append(path.reindex(base.index).to_numpy(float))
    if not paths:
        return pd.DataFrame({"value": base, "ensemble_sd": 0.0})
    stacked = np.vstack(paths)
    return pd.DataFrame({"value": base, "ensemble_sd": stacked.std(axis=0)},
                        index=base.index)


def chow_lin_check(annual: pd.Series, indicator: pd.Series) -> dict:
    clean = indicator.dropna()
    annualised = clean.groupby(clean.index.year).sum()
    joined = pd.concat([annual.rename("a"), annualised.rename("z")], axis=1).dropna()
    if len(joined) < 5:
        return {"indicator_r2": np.nan, "indicator_corr_growth": np.nan, "n_years": len(joined)}
    x = np.column_stack([np.ones(len(joined)), joined["z"].to_numpy(float)])
    y = joined["a"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    ga = np.diff(np.log(np.abs(y) + 1e-12))
    gz = np.diff(np.log(np.abs(joined["z"].to_numpy(float)) + 1e-12))
    corr = float(np.corrcoef(ga, gz)[0, 1]) if len(ga) > 3 else np.nan
    return {"indicator_r2": r2, "indicator_corr_growth": corr, "n_years": len(joined)}
