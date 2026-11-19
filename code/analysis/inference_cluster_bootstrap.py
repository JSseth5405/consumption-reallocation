from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20260917
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def weight_matrix(n_clusters: int, reps: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    if kind == "rademacher":
        return rng.choice([-1.0, 1.0], size=(reps, n_clusters))
    if kind == "webb":
        return rng.choice(WEBB, size=(reps, n_clusters))
    raise ValueError(kind)


def _meat(Xa: np.ndarray, resid: np.ndarray, codes: np.ndarray) -> np.ndarray:
    n_g = codes.max() + 1
    summed = np.zeros((n_g, Xa.shape[1]))
    np.add.at(summed, codes, Xa * resid[:, None])
    return summed.T @ summed


def oneway_cov(Xa, resid, xtx_inv, codes) -> tuple[np.ndarray, int]:
    n, k = Xa.shape
    n_g = int(codes.max() + 1)
    corr = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return corr * xtx_inv @ _meat(Xa, resid, codes) @ xtx_inv, n_g


def twoway_cov(Xa, resid, xtx_inv, codes_a, codes_b) -> tuple[np.ndarray, int]:
    both = pd.factorize(pd.Series(list(zip(codes_a, codes_b))))[0]

    def part(codes):
        n, k = Xa.shape
        n_g = codes.max() + 1
        corr = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
        return corr * xtx_inv @ _meat(Xa, resid, codes) @ xtx_inv

    cov = part(codes_a) + part(codes_b) - part(both)
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    cov = v @ np.diag(np.clip(w, 0, None)) @ v.T
    return cov, int(min(codes_a.max() + 1, codes_b.max() + 1))


def wcr_bootstrap(Xa: np.ndarray, y: np.ndarray, codes: np.ndarray, idx: int | None = None,
                  kind: str = "rademacher", reps: int = 999,
                  xtx_inv: np.ndarray | None = None,
                  contrast: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    n, k = Xa.shape
    if xtx_inv is None:
        xtx_inv = np.linalg.pinv(Xa.T @ Xa)
    if contrast is None:
        c = np.zeros(k)
        c[idx] = 1.0
    else:
        c = np.asarray(contrast, float)
    a = xtx_inv @ c
    denom = float(c @ a)
    if denom <= 1e-14:
        return np.nan, np.array([])
    beta = xtx_inv @ (Xa.T @ y)
    beta0 = beta - a * (float(c @ beta) / denom)
    resid0 = y - Xa @ beta0
    resid = y - Xa @ beta
    n_g = int(codes.max() + 1)
    corr = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    z = Xa @ a

    def per_cluster(v):
        out = np.zeros(n_g)
        np.add.at(out, codes, v)
        return out

    se_obs = np.sqrt(max(corr * float((per_cluster(z * resid) ** 2).sum()), 1e-18))
    t_obs = float(c @ beta) / se_obs
    A = per_cluster(z * resid0)
    S = np.zeros((n_g, k))
    np.add.at(S, codes, Xa * resid0[:, None])
    B = np.zeros((n_g, k))
    np.add.at(B, codes, Xa * z[:, None])
    W = weight_matrix(n_g, reps, kind)
    beta_star = W @ A
    V = (W @ S) @ xtx_inv.T
    per_g = W * A[None, :] - V @ B.T
    se_star = np.sqrt(np.maximum(corr * (per_g ** 2).sum(axis=1), 1e-18))
    return t_obs, beta_star / se_star


def boot_p(t_obs: float, draws: np.ndarray, direction: str) -> tuple[float, float]:
    draws = np.asarray(draws, float)
    draws = draws[np.isfinite(draws)]
    if not np.isfinite(t_obs) or len(draws) == 0:
        return np.nan, np.nan
    lower = (1.0 + float((draws <= t_obs).sum())) / (1.0 + len(draws))
    upper = (1.0 + float((draws >= t_obs).sum())) / (1.0 + len(draws))
    return (lower, upper) if direction == "negative" else (upper, lower)
