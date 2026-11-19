from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SEED = 20260917
RNG = np.random.default_rng(SEED)


def flip_matrix(n_units: int, reps: int) -> np.ndarray:
    return np.random.default_rng(SEED).choice([-1.0, 1.0], size=(reps, n_units))


def one_sided_p(estimate: float, se: float, df: float, direction: str) -> float:
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        return np.nan
    t = estimate / se
    return float(stats.t.cdf(t, df) if direction == "negative"
                 else stats.t.sf(t, df))


def romano_wolf(t_stats: np.ndarray, boot_t: np.ndarray) -> np.ndarray:
    t_stats = np.asarray(t_stats, float)
    boot_t = np.asarray(boot_t, float)
    k = len(t_stats)
    order = np.argsort(-t_stats)
    adjusted = np.empty(k)
    remaining = list(order)
    previous = 0.0
    for position, index in enumerate(order):
        subset = boot_t[:, remaining]
        if subset.size == 0 or not np.isfinite(t_stats[index]):
            adjusted[index] = np.nan
            if remaining:
                remaining.pop(0)
            continue
        maxima = np.nanmax(subset, axis=1)
        p = float((np.sum(maxima >= t_stats[index]) + 1) / (len(maxima) + 1))
        previous = max(previous, p)
        adjusted[index] = min(previous, 1.0)
        remaining.pop(0)
    return adjusted


def romano_wolf_by_domain(t_obs: dict, boots: dict,
                          domains: dict) -> dict:
    usable = [k for k in domains if k in boots and len(boots[k])]
    if not usable:
        return {}
    reps = min(len(boots[k]) for k in usable)
    matrix = np.column_stack([boots[k][:reps] for k in usable])
    values = romano_wolf(np.array([t_obs[k] for k in usable]), matrix)
    return dict(zip(usable, values))


def holm(pvalues: pd.Series) -> pd.Series:
    valid = pvalues.dropna()
    if valid.empty:
        return pvalues
    return pd.Series(multipletests(valid.to_numpy(), method="holm")[1],
                     index=valid.index).reindex(pvalues.index)


def tost_equivalence(estimate: float, se: float, df: float,
                     bound: float) -> dict:
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        return {"tost_p": np.nan, "equivalent": False,
                "ci90_low": np.nan, "ci90_high": np.nan}
    t_lower = (estimate + bound) / se
    t_upper = (estimate - bound) / se
    p = max(float(stats.t.sf(t_lower, df)), float(stats.t.cdf(t_upper, df)))
    crit = stats.t.ppf(0.95, df)
    low, high = estimate - crit * se, estimate + crit * se
    return {"tost_p": p, "equivalent": bool(low > -bound and high < bound),
            "ci90_low": low, "ci90_high": high}


def _hac_se(X, resid, idx, lag):
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]
    meat = u.T @ u
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        g = u[l:].T @ u[:-l]
        meat += w * (g + g.T)
    cov = xtx_inv @ meat @ xtx_inv * (n / max(n - k, 1))
    return float(np.sqrt(max(cov[idx, idx], 1e-18)))


def year_block_bootstrap_t(frame: pd.DataFrame, ycol: str, xcols: list[str],
                           term: str, year_col: str = "year",
                           reps: int = 999) -> tuple[float, np.ndarray]:
    work = frame[[ycol, *xcols, year_col]].dropna()
    if work.empty or term not in xcols:
        return np.nan, np.array([])
    X = work[xcols].to_numpy(float)
    y = work[ycol].to_numpy(float)
    idx = xcols.index(term)
    lag = max(1, int(np.floor(4 * (len(y) / 100.0) ** (2.0 / 9.0))))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    se = _hac_se(X, resid, idx, lag)
    t_obs = beta[idx] / se

    years = work[year_col].to_numpy()
    unique = np.unique(years)
    blocks = {y_: np.where(years == y_)[0] for y_ in unique}
    draws = []
    for _ in range(reps):
        picked = RNG.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([blocks[y_] for y_ in picked])
        Xb, yb = X[index], y[index]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        try:
            bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            rb = yb - Xb @ bb
            seb = _hac_se(Xb, rb, idx, lag)
            if seb > 0:
                draws.append((bb[idx] - beta[idx]) / seb)
        except Exception:
            continue
    return float(t_obs), np.asarray(draws)


def wild_cluster_bootstrap_t(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                             idx: int, reps: int = 999) -> tuple[float, np.ndarray]:
    c = np.zeros(X.shape[1])
    c[idx] = 1.0
    xtx_inv = np.linalg.pinv(X.T @ X)
    denom = float(c @ xtx_inv @ c)
    if denom <= 1e-14:
        return np.nan, np.array([])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a = xtx_inv @ c
    beta0 = beta - a * (float(c @ beta) / denom)
    fitted0 = X @ beta0
    resid0 = y - fitted0
    codes = pd.factorize(groups)[0]

    def cluster_se(resid, params_resid):
        scores = X * params_resid[:, None]
        n_g = codes.max() + 1
        summed = np.zeros((n_g, X.shape[1]))
        np.add.at(summed, codes, scores)
        meat = summed.T @ summed
        n = X.shape[0]
        corr = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - X.shape[1], 1))
        cov = corr * xtx_inv @ meat @ xtx_inv
        return float(np.sqrt(max(cov[idx, idx], 1e-18)))

    resid_obs = y - X @ beta
    t_obs = beta[idx] / cluster_se(resid_obs, resid_obs)
    units = np.unique(codes)
    hat = xtx_inv @ X.T
    weights = flip_matrix(len(units), reps)
    draws = []
    for flips in weights:
        ys = fitted0 + resid0 * flips[codes]
        params = hat @ ys
        rs = ys - X @ params
        se = cluster_se(rs, rs)
        if se > 0:
            draws.append(params[idx] / se)
    return float(t_obs), np.asarray(draws)


def wild_cluster_bootstrap_contrast(X, y, groups, contrast, reps: int = 999):
    c = np.asarray(contrast, float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    denom = float(c @ xtx_inv @ c)
    if denom <= 1e-14:
        return np.nan, np.array([])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    a = xtx_inv @ c
    beta0 = beta - a * (float(c @ beta) / denom)
    fitted0 = X @ beta0
    resid0 = y - fitted0
    codes = pd.factorize(groups)[0]

    def contrast_se(res):
        n_g = codes.max() + 1
        summed = np.zeros((n_g, X.shape[1]))
        np.add.at(summed, codes, X * res[:, None])
        corr = (n_g / max(n_g - 1, 1)) * ((len(y) - 1) / max(len(y) - X.shape[1], 1))
        cov = corr * xtx_inv @ (summed.T @ summed) @ xtx_inv
        return float(np.sqrt(max(c @ cov @ c, 1e-18)))

    t_obs = float(c @ beta) / contrast_se(y - X @ beta)
    units = np.unique(codes)
    hat = xtx_inv @ X.T
    draws = []
    for flips in flip_matrix(len(units), reps):
        ys = fitted0 + resid0 * flips[codes]
        params = hat @ ys
        se = contrast_se(ys - X @ params)
        if se > 0:
            draws.append(float(c @ params) / se)
    return t_obs, np.asarray(draws)


def decide(estimate: float, direction: str, p_adjusted: float,
           p_adjusted_opposite: float, equivalence: dict,
           alpha: float = 0.05) -> str:
    if not np.isfinite(estimate) or not np.isfinite(p_adjusted):
        return "not_estimable"
    sign = np.sign(estimate)
    wanted = -1 if direction == "negative" else 1
    if p_adjusted < alpha and sign == wanted:
        return "supported"
    if (np.isfinite(p_adjusted_opposite) and p_adjusted_opposite < alpha
            and sign == -wanted):
        return "rejected_opposite_sign"
    if equivalence.get("equivalent"):
        return "negligible_effect"
    return "inconclusive"
