from __future__ import annotations

import numpy as np
import pandas as pd


def quiet_periods(panel: pd.DataFrame, shocks: dict, buffer: int = 2,
                  extra: set | None = None) -> list[str]:
    periods = sorted(panel.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    index = {p: i for i, p in enumerate(periods)}
    blocked = set()
    for spec in shocks.values():
        lo, hi = spec["window"]
        for period in periods:
            if lo <= period <= hi:
                k = index[period]
                for j in range(max(k - buffer, 0), min(k + buffer + 1, len(periods))):
                    blocked.add(periods[j])
    for period in (extra or set()):
        if period in index:
            k = index[period]
            for j in range(max(k - buffer, 0), min(k + buffer + 1, len(periods))):
                blocked.add(periods[j])
    return [p for p in periods if p not in blocked]


def randomisation(estimate: float, placebos, direction: str) -> dict:
    draws = np.asarray([v for v in placebos if np.isfinite(v)], float)
    out = {"n_placebo": int(len(draws)), "placebo_sd": np.nan,
           "placebo_mean": np.nan, "p_randomisation": np.nan}
    if len(draws) < 6 or not np.isfinite(estimate):
        return out
    out["placebo_sd"] = float(draws.std(ddof=1))
    out["placebo_mean"] = float(draws.mean())
    centred = draws - draws.mean()
    stat = estimate - draws.mean()
    lower = int((centred <= stat).sum())
    upper = int((centred >= stat).sum())
    hit, miss = (lower, upper) if direction == "negative" else (upper, lower)
    out["p_randomisation"] = (1.0 + hit) / (1.0 + len(draws))
    out["p_randomisation_opposite"] = (1.0 + miss) / (1.0 + len(draws))
    return out
