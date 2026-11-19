from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import inference_tests as inf
import inference_cluster_bootstrap as ip
import study_protocol as P

HOME = pathlib.Path.home()
ROOT = HERE.parent.parent
SETTLED = ("supported", "rejected_opposite_sign", "negligible_effect")


def add_volume_shares(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    tot = d.groupby(["country", "period"]).real_index.transform("sum")
    lv = np.log(d.real_index / tot)
    d["clr_vol"] = lv - lv.groupby([d.country, d.period]).transform("mean")
    return d


def load_panel() -> pd.DataFrame:
    keep = ["period", "category", "share", "clr_share", "real_growth",
            "real_growth_qoq", "real_index", "weight", "necessity",
            "contact_intensity", "durable_intensity", "discretionary_intensity",
            "energy_intensity"]
    frames = []
    for country, spec in P.COUNTRIES.items():
        d = pd.read_csv(ROOT / spec["panel"])
        cols = keep + [f"exposed_{s}" for s in P.SHOCKS]
        missing = [c for c in cols if c not in d.columns]
        if missing:
            raise SystemExit(f"{country} panel lacks {missing}")
        d = d[cols].copy()
        d["country"] = country
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data = data[(data.period >= P.COMMON_START) & (data.period <= P.COMMON_END)].copy()
    data["unit"] = data.country + "|" + data.category
    data["year"] = data.period.str.slice(0, 4).astype(int)
    q = pd.PeriodIndex(data.period, freq="Q")
    data["t"] = q.astype("int64").to_numpy(float)
    data["t"] -= data["t"].mean()
    for s in P.SHOCKS:
        data[f"impact_{s}"] = data.period.isin(P.impact_periods(s)).astype(float)
    data["impact_any"] = data[[f"impact_{s}" for s in P.SHOCKS]].max(axis=1)
    import data_oil_price as oil
    data["oil4q"] = data.period.map(oil.oil_growth()).astype(float)
    res = pd.read_csv(ROOT / "data/resilience/resilience_index.csv")
    for s in P.SHOCKS:
        m = res[res.shock == s].set_index("country")["resilience"]
        data[f"res_{s}"] = data.country.map(m).astype(float)
    data = add_volume_shares(data)
    return data.sort_values(["unit", "period"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(ROOT / "data/country_events.csv")
    ev["periods"] = [set(P.periods(a, b)) for a, b in zip(ev.start, ev.end)]
    ev["cats"] = [set(str(x).split(";")) if isinstance(x, str) and x else set()
                  for x in ev.categories.fillna("")]
    return ev


EVENTS = None


def events() -> pd.DataFrame:
    global EVENTS
    if EVENTS is None:
        EVENTS = load_events()
    return EVENTS


def disturbed(country: str, spec: str) -> set:
    if spec == "core":
        return set()
    ev = events()
    rows = ev[(ev.country == country) & (ev.scope == "aggregate")
              & ev.type.isin(["C", "X"]) & (ev.tier == "primary")]
    return set().union(*rows.periods) if len(rows) else set()


def first_x(country: str, shock: str):
    ev = events()
    rows = ev[(ev.country == country) & (ev.type == "X") & (ev.context == shock)]
    return min(pd.Period(s, freq="Q") for s in rows.start) if len(rows) else None


def event_dummies(data: pd.DataFrame, spec: str, focal: set,
                  include_aggregate: bool) -> pd.DataFrame:
    cols = {}
    if spec == "core":
        return pd.DataFrame(index=data.index)
    ev = events()
    for row in ev.itertuples():
        if row.tier == "secondary" and spec != "extended":
            continue
        if row.type == "M" and spec != "mechanism":
            continue
        if row.scope == "aggregate":
            if not include_aggregate:
                continue
            if spec in ("clean", "extended") and (row.periods & focal):
                continue
            flag = (data.country == row.country) & data.period.isin(row.periods)
        else:
            flag = ((data.country == row.country) & data.period.isin(row.periods)
                    & data.category.isin(row.cats))
        if flag.any():
            cols[f"ev_{row.country}_{row.event}"] = flag.astype(float)
    return pd.DataFrame(cols, index=data.index)


def controls_level(sub: pd.DataFrame, spec: str, focal: set) -> pd.DataFrame:
    ev = event_dummies(sub, spec, focal, include_aggregate=True)
    if spec == "mechanism":
        ev = ev.assign(oil4q=sub["oil4q"].astype(float))
    return ev


def controls_share(sub: pd.DataFrame, spec: str, focal: set) -> pd.DataFrame:
    ev = event_dummies(sub, spec, focal, include_aggregate=False)
    if spec == "mechanism":
        ev = ev.assign(oil_x_energy=sub["oil4q"].astype(float) * sub["energy_intensity"].astype(float))
    return ev


def all_windows() -> set:
    out = set()
    for s in P.SHOCKS:
        out |= set(P.window_periods(s))
    return out


def sesoi_level(dc: pd.DataFrame, quiet_excl: set) -> float:
    quiet = dc[~dc.period.isin(quiet_excl)].dropna(subset=["real_growth"])
    per = quiet.groupby("period").apply(
        lambda d: float((d.real_growth * d.weight).sum() / d.weight.sum()),
        include_groups=False)
    return float(per.std(ddof=1)) if len(per) > 4 else np.nan


def sesoi_differential(dc: pd.DataFrame, quiet_excl: set, column: str, group: str) -> float:
    quiet = dc[~dc.period.isin(quiet_excl)].dropna(subset=[column])
    weight = quiet[group].astype(float)
    high = weight > weight.median()
    if high.sum() == 0 or (~high).sum() == 0:
        high = weight > (weight.max() + weight.min()) / 2.0
    if high.sum() == 0 or (~high).sum() == 0:
        return np.nan
    per = quiet.assign(_hi=high).groupby(["period", "_hi"])[column].mean().unstack()
    if per.shape[1] < 2:
        return np.nan
    diff = (per[True] - per[False]).dropna()
    if column in ("clr_share", "clr_vol"):
        diff = diff.diff(4).dropna()
    return float(diff.std(ddof=1)) if len(diff) > 3 else np.nan


def country_sesoi(data: pd.DataFrame, country: str, hyp: str, shock: str) -> float:
    dc = data[data.country == country]
    excl = all_windows() | disturbed(country, "clean")
    if hyp in ("H1", "H4", "H5", "H4_local"):
        return sesoi_level(dc, excl)
    if hyp == "H2":
        return sesoi_differential(dc, excl, "clr_share", "necessity")
    if hyp == "H2_volume":
        return sesoi_differential(dc, excl, "clr_vol", "necessity")
    if hyp == "H3":
        shocks = list(P.SHOCKS) if shock == "combined" else [shock]
        vals = [sesoi_differential(dc, excl, "real_growth", f"exposed_{s}") for s in shocks]
        return float(np.nanmean(vals))
    raise ValueError(hyp)


def pooled_sesoi(data: pd.DataFrame, hyp: str, shock: str) -> tuple[float, dict]:
    per = {c: country_sesoi(data, c, hyp, shock) for c in data.country.unique()}
    return float(np.nanmean(list(per.values()))), per


def _drop_zero(X: pd.DataFrame) -> pd.DataFrame:
    return X.loc[:, (X != 0).any(axis=0)]


def design_level(data: pd.DataFrame, regressors: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    units = pd.get_dummies(data["unit"], prefix="u", dtype=float)
    trends = units.mul(data["t"], axis=0).add_prefix("tr_")
    return _drop_zero(pd.concat([units, trends, ev, regressors], axis=1))


def design_share(data: pd.DataFrame, regressors: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    units = pd.get_dummies(data["unit"], prefix="u", dtype=float)
    cp = data.country + "@" + data.period
    first_p = data.groupby("country").period.transform("min")
    cpd = pd.get_dummies(cp.where(data.period != first_p), prefix="cp", dtype=float)
    first_c = data.groupby("country").category.transform("min")
    tr_units = pd.get_dummies(data["unit"].where(data.category != first_c),
                              prefix="tr", dtype=float).mul(data["t"], axis=0)
    return _drop_zero(pd.concat([units, cpd, tr_units, ev, regressors], axis=1))


def fit(X: pd.DataFrame, y: np.ndarray, w: np.ndarray, target: str,
        data: pd.DataFrame, cluster: str, boot_kind: str, bootstrap: bool = True,
        contrast: dict | None = None, homogeneity: list | None = None,
        time_key: str = "year") -> dict:
    ww = np.sqrt(w)
    ww = ww / ww.mean()
    Xa = X.to_numpy(float) * ww[:, None]
    ya = y * ww
    idx = list(X.columns).index(target)
    xtx_inv = np.linalg.pinv(Xa.T @ Xa)
    beta = xtx_inv @ (Xa.T @ ya)
    resid = ya - Xa @ beta
    if cluster == "twoway":
        cu = pd.factorize(data["unit"])[0]
        cy = pd.factorize(data[time_key])[0]
        cov, g = ip.twoway_cov(Xa, resid, xtx_inv, cu, cy)
        boot_codes = cu
    else:
        codes = pd.factorize(data[cluster])[0]
        cov, g = ip.oneway_cov(Xa, resid, xtx_inv, codes)
        boot_codes = codes
    cols = list(X.columns)
    c = np.zeros(len(cols))
    if contrast:
        for col, wt in contrast.items():
            c[cols.index(col)] = wt
    else:
        c[idx] = 1.0
    est = float(c @ beta)
    se = float(np.sqrt(max(float(c @ cov @ c), 0.0)))
    out = {"estimate": est, "std_error": se, "df": float(g - 1), "n_clusters": g,
           "nobs": int(len(ya)), "k": int(Xa.shape[1])}
    if contrast:
        out["components"] = {col: float(beta[cols.index(col)]) for col in contrast}
    if homogeneity:
        j = [cols.index(col) for col in homogeneity]
        R = np.zeros((len(j) - 1, len(cols)))
        for r in range(len(j) - 1):
            R[r, j[r]], R[r, j[r + 1]] = 1.0, -1.0
        Rb = R @ beta
        RVR = R @ cov @ R.T
        try:
            wstat = float(Rb @ np.linalg.solve(RVR, Rb))
            q = R.shape[0]
            out["homogeneity_F"] = wstat / q
            out["homogeneity_p"] = float(stats.f.sf(wstat / q, q, max(g - 1, 1)))
        except np.linalg.LinAlgError:
            out["homogeneity_F"], out["homogeneity_p"] = np.nan, np.nan
    if bootstrap:
        t_obs, draws = ip.wcr_bootstrap(Xa, ya, boot_codes, kind=boot_kind,
                                        reps=P.REPS, xtx_inv=xtx_inv, contrast=c)
        out["t_boot_obs"], out["boot_draws"] = t_obs, draws
    return out


def focal_set(shock: str) -> set:
    if shock in ("combined", "combined_common"):
        out = set()
        for s in P.SHOCKS:
            out |= set(P.impact_periods(s))
        return out
    return set(P.impact_periods(shock))


def treat(data: pd.DataFrame, shock: str, periods: set | None = None) -> pd.Series:
    if periods is not None:
        return data.period.isin(periods).astype(float)
    return data["impact_any"] if shock in ("combined", "combined_common") \
        else data[f"impact_{shock}"]


def mean_exposure(data: pd.DataFrame) -> pd.Series:
    return data[[f"exposed_{s}" for s in P.SHOCKS]].mean(axis=1)


SHOCK_COLS = [f"x_{s}" for s in P.SHOCKS]
MEAN = {c: 1.0 / len(SHOCK_COLS) for c in SHOCK_COLS}


def h1(data, shock, spec, periods=None, cluster="twoway", boot="rademacher", bootstrap=True,
       time_key="year"):
    sub = data.dropna(subset=["real_growth"])
    ev = controls_level(sub, spec, focal_set(shock))
    if shock == "combined" and periods is None:
        reg = pd.DataFrame({f"x_{s}": sub[f"impact_{s}"] for s in P.SHOCKS}, index=sub.index)
        X = design_level(sub, reg, ev)
        return fit(X, sub.real_growth.to_numpy(float), sub.weight.to_numpy(float), SHOCK_COLS[0],
                   sub, cluster, boot, bootstrap, contrast=MEAN, homogeneity=SHOCK_COLS,
                   time_key=time_key)
    reg = pd.DataFrame({"x": treat(sub, shock, periods)}, index=sub.index)
    X = design_level(sub, reg, ev)
    return fit(X, sub.real_growth.to_numpy(float), sub.weight.to_numpy(float), "x",
               sub, cluster, boot, bootstrap, time_key=time_key)


OUTCOME = {"H2": "clr_share", "H2_volume": "clr_vol", "H3": "real_growth"}


def _inter(sub, kind, s, per=None):
    base = sub.period.isin(per).astype(float) if per is not None else sub[f"impact_{s}"]
    return base * (sub["necessity"].astype(float) if kind in ("H2", "H2_volume")
                   else sub[f"exposed_{s}"].astype(float))


def h23(data, shock, spec, kind, periods=None, cluster="twoway", boot="rademacher",
        bootstrap=True, placebo_margin=None):
    outcome = OUTCOME[kind]
    sub = data.dropna(subset=[outcome])
    ev = controls_share(sub, spec, focal_set(shock))
    y, w = sub[outcome].to_numpy(float), sub.weight.to_numpy(float)
    actual = {f"x_{s}": _inter(sub, kind, s) for s in P.SHOCKS}
    if periods is not None:
        margin = placebo_margin or (shock if shock in P.SHOCKS else "covid")
        reg = pd.DataFrame({**actual, "x": _inter(sub, kind, margin, periods)}, index=sub.index)
        return fit(design_share(sub, reg, ev), y, w, "x", sub, cluster, boot, bootstrap)
    if shock == "combined":
        reg = pd.DataFrame(actual, index=sub.index)
        return fit(design_share(sub, reg, ev), y, w, SHOCK_COLS[0], sub, cluster, boot,
                   bootstrap, contrast=MEAN, homogeneity=SHOCK_COLS)
    if shock == "combined_common":
        x = (sub["impact_any"] * sub["necessity"].astype(float) if kind in ("H2", "H2_volume")
             else sum(_inter(sub, kind, s) for s in P.SHOCKS))
        reg = pd.DataFrame({"x": x}, index=sub.index)
        return fit(design_share(sub, reg, ev), y, w, "x", sub, cluster, boot, bootstrap)
    reg = pd.DataFrame({("x" if s == shock else f"other_{s}"): v.to_numpy()
                        for s, v in ((k[2:], c) for k, c in actual.items())}, index=sub.index)
    return fit(design_share(sub, reg, ev), y, w, "x", sub, cluster, boot, bootstrap)


def h5(data, shock, spec, periods=None, bootstrap=True):
    sub = data.dropna(subset=["real_growth"])
    if shock == "combined":
        if periods is None:
            reg = {f"imp_{s}": sub[f"impact_{s}"] for s in P.SHOCKS}
            reg["x"] = sum(sub[f"impact_{s}"] * sub[f"res_{s}"] for s in P.SHOCKS)
        else:
            base = sub.period.isin(periods).astype(float)
            reg = {"imp": base,
                   "x": base * sub[[f"res_{s}" for s in P.SHOCKS]].mean(axis=1)}
    else:
        base = treat(sub, shock, periods)
        reg = {"imp": base, "x": base * sub[f"res_{shock}"]}
    reg = pd.DataFrame(reg, index=sub.index)
    ev = controls_level(sub, spec, focal_set(shock))
    X = design_level(sub, reg, ev)
    return fit(X, sub.real_growth.to_numpy(float), sub.weight.to_numpy(float), "x",
               sub, "country", "webb", bootstrap)


def clean_anchor(country: str, shock: str, spec: str) -> pd.Period:
    blocked = disturbed(country, spec) | set(P.window_periods(shock))
    cand = pd.Period(P.SHOCKS[shock]["window"][0], freq="Q") - 1
    for _ in range(12):
        if str(cand) not in blocked:
            return cand
        cand -= 1
    return cand


def h4_rows(data: pd.DataFrame, shock: str, spec: str, anchor=None, truncate=True,
            local: bool = False, horizons=None, clear_path: bool = False) -> pd.DataFrame:
    end = pd.Period(P.COMMON_END, freq="Q")
    windows = all_windows()
    rows = []
    for country, dc in data.groupby("country"):
        excl = all_windows() | disturbed(country, spec)
        quiet = set(p for p in dc.period.unique() if p not in excl)
        a = anchor if anchor is not None else clean_anchor(country, shock, spec)
        stop = first_x(country, shock) if (truncate and spec != "core") else None
        for category, block in dc.groupby("category"):
            block = block.dropna(subset=["real_index"])
            pool_q = quiet
            if local:
                pool_q = {p for p in quiet if a - 40 <= pd.Period(p, freq="Q") < a}
            drift = block[block.period.isin(pool_q)]["real_growth_qoq"].dropna()
            at = block[block.period == str(a)]
            if len(drift) < 20 or at.empty:
                continue
            rate = float(drift.mean()) / 100.0
            base = float(np.log(at.real_index.iloc[0]))
            for h in (horizons or P.H4_HORIZONS):
                tq = a + h
                if tq > end or (stop is not None and tq >= stop):
                    continue
                if clear_path and any(str(a + k) in windows for k in range(1, h + 1)):
                    continue
                tgt = block[block.period == str(tq)]
                if tgt.empty:
                    continue
                rows.append({"country": country, "category": category, "horizon": h,
                             "gap": float(100 * (np.log(tgt.real_index.iloc[0]) - base - rate * h)),
                             "weight": float(tgt.weight.iloc[0]), "anchor": str(a)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["weight"] = frame.weight / frame.groupby("country").horizon.transform("nunique")
    return frame


def h4(data, shock, spec, anchor=None, truncate=True, bootstrap=True, single=False, local=False,
       clear_path=False):
    shocks = list(P.SHOCKS) if shock == "combined" else [shock]
    frames = [h4_rows(data, s, spec, anchor, truncate, local, clear_path=clear_path).assign(shock=s)
              for s in shocks]
    nonempty = [f for f in frames if not f.empty]
    frame = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if frame.empty or len(frame) < 10:
        return None, frame
    X = pd.DataFrame({"x": np.ones(len(frame))})
    cluster, boot = ("category", "rademacher") if single else ("country", "webb")
    res = fit(X, frame.gap.to_numpy(float), frame.weight.to_numpy(float), "x",
              frame, cluster, boot, bootstrap)
    return res, frame


def placebo_blocks(data: pd.DataFrame) -> list[list[str]]:
    per = sorted(data.period.unique(), key=lambda q: pd.Period(q, freq="Q"))
    index = {p: i for i, p in enumerate(per)}
    blocked = set()
    for s in P.SHOCKS:
        for p in P.window_periods(s):
            if p in index:
                k = index[p]
                blocked.update(per[max(k - P.BUFFER, 0): k + P.BUFFER + 1])
    quiet = [p for p in per if p not in blocked]
    qs = set(quiet)
    out = []
    for p in quiet[::P.PLACEBO_STRIDE]:
        blk = [str(pd.Period(p, freq="Q") + k) for k in range(2)]
        if all(b in qs for b in blk):
            out.append(blk)
    return out


def placebo_draws(fn, data) -> np.ndarray:
    draws = []
    for blk in placebo_blocks(data):
        try:
            r = fn(set(blk))
        except Exception:
            r = None
        if r is not None and np.isfinite(r):
            draws.append(float(r))
    return np.asarray(draws, float)


def randomisation(est: float, draws: np.ndarray, direction: str) -> dict:
    import placebo_tests as PL
    return PL.randomisation(est, draws, direction)


def three_way(est, se, df, sesoi, direction):
    if not all(np.isfinite([est, se, df, sesoi])) or se <= 0:
        return "", np.nan
    pn = stats.t.cdf((-sesoi - est) / se, df)
    pp = stats.t.sf((sesoi - est) / se, df)
    pc = 1.0 - pn - pp
    pa, pr = (pn, pp) if direction == "negative" else (pp, pn)
    call, prob = max([("supported", pa), ("near_zero", pc), ("rejected_opposite_sign", pr)],
                     key=lambda x: x[1])
    return call, float(prob)


def assemble_row(pool, spec, hyp, res, sesoi, sesoi_by_country, rand, use_boot_p, extra=None):
    direction = P.DIRECTION[hyp]
    row = {"pool": pool, "spec": spec, "hypothesis": hyp, "expected_sign": direction,
           "statement": P.STATEMENT[hyp], "sesoi": sesoi}
    if res is None:
        row["decision_headline"] = "not_estimable"
        row.update(extra or {})
        return row
    est, se, df = res["estimate"], res["std_error"], res["df"]
    p_an = inf.one_sided_p(est, se, df, direction)
    pb, pb_opp = ip.boot_p(res.get("t_boot_obs", np.nan), res.get("boot_draws", np.array([])),
                           direction)
    p_par, p_par_opp = (pb, pb_opp) if use_boot_p else (p_an, 1.0 - p_an)
    eq = inf.tost_equivalence(est, se, df, sesoi)
    d_par = inf.decide(est, direction, p_par, p_par_opp, {"equivalent": eq["equivalent"]}, P.ALPHA)
    d_rand = inf.decide(est, direction, rand.get("p_randomisation", np.nan),
                        rand.get("p_randomisation_opposite", np.nan), {"equivalent": False}, P.ALPHA)
    head = d_par if d_par in SETTLED else (d_rand if d_rand in SETTLED else "inconclusive")
    call, prob = three_way(est, se, df, sesoi, direction)
    row.update({"estimate": est, "std_error": se, "df": df, "n_clusters": res["n_clusters"],
                "nobs": res["nobs"], "p_analytic": p_an, "p_bootstrap": pb,
                "p_parametric": p_par, "p_parametric_opposite": p_par_opp,
                "parametric_rule": "wild cluster bootstrap (Webb), G-1 df" if use_boot_p
                else "two-way cluster-robust t, min(G)-1 df",
                "tost_p": eq["tost_p"], "ci90_low": eq["ci90_low"], "ci90_high": eq["ci90_high"],
                "min_equivalence_bound": abs(est) + stats.t.ppf(0.95, df) * se,
                "n_placebo": rand.get("n_placebo", 0), "p_randomisation": rand.get("p_randomisation", np.nan),
                "decision_parametric": d_par, "decision_randomisation": d_rand,
                "decision_headline": head, "three_way": call, "three_way_prob": prob,
                "sesoi_by_country": "; ".join(f"{k} {v:.3f}" for k, v in sesoi_by_country.items())})
    row.update(extra or {})
    return row


def run_pool(data: pd.DataFrame, pool: str, spec: str, log=print, h5_test: bool = True) -> list[dict]:
    rows = []
    sesoi_pool = "combined" if pool == "combined_common" else pool

    def comb_extra(res):
        if "homogeneity_p" not in res:
            return {}
        return {"homogeneity_F": res["homogeneity_F"], "homogeneity_p": res["homogeneity_p"],
                "shock_specific": "; ".join(f"{k[2:]} {v:+.4f}" for k, v in res["components"].items())}

    res = h1(data, pool, spec)
    draws = placebo_draws(lambda per: h1(data, pool, spec, per, bootstrap=False)["estimate"], data)
    s, sc = pooled_sesoi(data, "H1", sesoi_pool)
    rows.append(assemble_row(pool, spec, "H1", res, s, sc,
                             randomisation(res["estimate"], draws, "negative"), False, comb_extra(res)))
    for kind in ("H2", "H3"):
        res = h23(data, pool, spec, kind)
        if pool in ("combined", "combined_common") and kind == "H3":
            fn = lambda per, k=kind: float(np.mean([h23(data, "combined", spec, k, per, bootstrap=False,
                                                         placebo_margin=m)["estimate"]
                                                    for m in P.SHOCKS]))
        else:
            fn = lambda per, k=kind: h23(data, pool, spec, k, per, bootstrap=False)["estimate"]
        draws = placebo_draws(fn, data)
        s, sc = pooled_sesoi(data, kind, sesoi_pool)
        rows.append(assemble_row(pool, spec, kind, res, s, sc,
                                 randomisation(res["estimate"], draws, P.DIRECTION[kind]), False,
                                 comb_extra(res)))
    if pool == "combined_common":
        for r in rows:
            log(f"  {pool:15s} {spec:9s} {r['hypothesis']}  est {r.get('estimate', np.nan):+9.4f}  "
                f"se {r.get('std_error', np.nan):8.4f}  -> {r['decision_headline']}")
        return rows
    res, frame = h4(data, pool, spec)
    if res is not None:
        def h4p(per):
            r, _ = h4(data, pool if pool != "combined" else "covid", spec,
                      anchor=pd.Period(sorted(per)[0], freq="Q"), truncate=False, bootstrap=False, clear_path=True)
            return None if r is None else r["estimate"]
        draws = placebo_draws(h4p, data)
        rand = randomisation(res["estimate"], draws, "negative")
        extra = {"h4_rows": len(frame),
                 "h4_horizons_by_country": "; ".join(
                     f"{c}:{','.join(str(h) for h in sorted(g.horizon.unique()))}"
                     for c, g in frame.groupby("country"))}
    else:
        rand, extra = {}, {}
    s, sc = pooled_sesoi(data, "H4", pool)
    rows.append(assemble_row(pool, spec, "H4", res, s, sc, rand, True, extra))
    if not h5_test:
        for r in rows:
            log(f"  {pool:8s} {spec:9s} {r['hypothesis']}  est {r.get('estimate', np.nan):+9.4f}  "
                f"se {r.get('std_error', np.nan):8.4f}  sesoi {r['sesoi']:7.4f}  -> {r['decision_headline']}")
        return rows
    res = h5(data, pool, spec)
    draws = placebo_draws(lambda per: h5(data, pool, spec, per, bootstrap=False)["estimate"], data)
    s, sc = pooled_sesoi(data, "H5", pool)
    rows.append(assemble_row(pool, spec, "H5", res, s, sc, randomisation(res["estimate"], draws, "positive"), True))
    for r in rows:
        log(f"  {pool:8s} {spec:9s} {r['hypothesis']}  est {r.get('estimate', np.nan):+9.4f}  "
            f"se {r.get('std_error', np.nan):8.4f}  sesoi {r['sesoi']:7.4f}  -> {r['decision_headline']}")
    return rows


def run_country(data: pd.DataFrame, country: str, shock: str, spec: str = "clean") -> list[dict]:
    dc = data[data.country == country].copy()
    rows = []
    for hyp in ("H1", "H2", "H3", "H4"):
        if hyp == "H1":
            res = h1(dc, shock, spec)
            fn = lambda per: h1(dc, shock, spec, per, bootstrap=False)["estimate"]
        elif hyp in ("H2", "H3"):
            res = h23(dc, shock, spec, hyp)
            fn = lambda per, k=hyp: h23(dc, shock, spec, k, per, bootstrap=False)["estimate"]
        else:
            res, _ = h4(dc, shock, spec, single=True)

            def fn(per):
                r, _ = h4(dc, shock if shock != "combined" else "covid", spec,
                          anchor=pd.Period(sorted(per)[0], freq="Q"), truncate=False,
                          bootstrap=False, single=True, clear_path=True)
                return None if r is None else r["estimate"]
        draws = placebo_draws(fn, dc) if res is not None else np.array([])
        sesoi = country_sesoi(data, country, hyp, shock)
        rand = randomisation(res["estimate"], draws, P.DIRECTION[hyp]) if res is not None else {}
        row = assemble_row(shock, spec, hyp, res, sesoi, {country: sesoi}, rand, False)
        row["country"] = country
        rows.append(row)
    return rows


def h5_components(data: pd.DataFrame, log=print) -> pd.DataFrame:
    import data_oil_price as oil
    data["oil4q"] = data.period.map(oil.oil_growth()).astype(float)
    res = pd.read_csv(ROOT / "data/resilience/resilience_index.csv")
    rows = []
    for comp in P.RESILIENCE["components"]:
        z = res.set_index(["country", "shock"])[f"z_{comp}"]
        d = data.copy()
        for s in P.SHOCKS:
            d[f"res_{s}"] = [z.get((c, s), np.nan) for c in d.country]
        for pool in list(P.SHOCKS) + ["combined"]:
            dd = d.dropna(subset=[f"res_{s}" for s in P.SHOCKS]) if pool == "combined" \
                else d.dropna(subset=[f"res_{pool}"])
            r = h5(dd, pool, P.HEADLINE_SPEC)
            pb, pb_opp = ip.boot_p(r["t_boot_obs"], r["boot_draws"], "positive")
            rows.append({"component": comp, "pool": pool, "estimate": r["estimate"],
                         "std_error": r["std_error"], "n_countries": int(dd.country.nunique()),
                         "p_bootstrap_positive": pb, "p_bootstrap_negative": pb_opp})
            log(f"  H5 component {comp:17s} {pool:8s} est {r['estimate']:+.4f} "
                f"se {r['std_error']:.4f}  p(+) {pb:.3f}  p(-) {pb_opp:.3f}  "
                f"[{int(dd.country.nunique())} countries]")
    return pd.DataFrame(rows)


def main(base: str, log=print) -> None:
    root = pathlib.Path(base)
    data = load_panel()
    check = data.groupby(["country", "period"]).weight.sum()
    if (check - 1).abs().max() > 1e-6:
        raise SystemExit("within-country weights do not sum to one")
    log(f"pooled panel: {data.country.nunique()} countries x {data.category.nunique()} divisions "
        f"x {data.period.nunique()} quarters ({data.period.min()}-{data.period.max()}), "
        f"{len(data)} rows; placebo phases: {len(placebo_blocks(data))}")
    rows = []
    for pool in list(P.SHOCKS) + ["combined", "combined_common"]:
        for spec in P.SPECS:
            rows += run_pool(data, pool, spec, log)
    for pool in list(P.SHOCKS) + ["combined"]:
        spec = P.HEADLINE_SPEC
        res = h23(data, pool, spec, "H2_volume")
        draws = placebo_draws(lambda per: h23(data, pool, spec, "H2_volume", per,
                                              bootstrap=False)["estimate"], data)
        s, sc = pooled_sesoi(data, "H2_volume", pool)
        rows.append(assemble_row(pool, spec, "H2_volume", res, s, sc,
                                 randomisation(res["estimate"], draws, "positive"), False))
        res, _ = h4(data, pool, spec, local=True)
        if res is not None:
            draws = placebo_draws(lambda per: (lambda r: None if r is None else r["estimate"])(
                h4(data, pool if pool != "combined" else "covid", spec,
                   anchor=pd.Period(sorted(per)[0], freq="Q"), truncate=False,
                   bootstrap=False, local=True, clear_path=True)[0]), data)
            rand = randomisation(res["estimate"], draws, "negative")
        else:
            rand = {}
        s, sc = pooled_sesoi(data, "H4_local", pool)
        rows.append(assemble_row(pool, spec, "H4_local", res, s, sc, rand, True))
        log(f"  {pool:8s} robustness H2_volume / H4_local done")
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    for (pool, spec), blk in table[table.hypothesis.isin(["H1", "H2", "H3", "H4", "H5"])].groupby(["pool", "spec"]):
        ok = blk.p_parametric.notna()
        if ok.any():
            table.loc[blk.index[ok], "p_holm"] = inf.holm(blk.loc[ok, "p_parametric"]).to_numpy()
    out = root / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pooled_results.csv", index=False)
    log("\nper-country results on the common timeline (clean specification)")
    crow = []
    for country in P.COUNTRIES:
        for shock in P.SHOCKS:
            crow += run_country(data, country, shock)
    ctab = pd.DataFrame(crow)
    ctab.to_csv(out / "per_country_results.csv", index=False)
    log("\nH5 by resilience component (pre-registered robustness, clean specification)")
    h5_components(data, log).to_csv(out / "h5_by_component.csv", index=False)
    log(f"written {out/'pooled_results.csv'} and {out/'per_country_results.csv'}")


if __name__ == "__main__":
    main(sys.argv[1])
