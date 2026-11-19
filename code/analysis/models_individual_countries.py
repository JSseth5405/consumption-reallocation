from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import models_all_countries as M
import placebo_tests as PL
import study_protocol as P

ROOT = HERE.parent.parent
TIMELINES = ("full", "matched")
SPECS = P.SPECS
ROBUSTNESS = ("H2_volume", "H4_local")


def load_country(country: str, timeline: str) -> pd.DataFrame:
    start = P.FULL_START[country] if timeline == "full" else P.COMMON_START
    d = pd.read_csv(ROOT / P.COUNTRIES[country]["panel"])
    d = d[(d.period >= start) & (d.period <= P.COMMON_END)].copy()
    d["country"] = country
    d["unit"] = country + "|" + d.category
    d["year"] = d.period.str.slice(0, 4).astype(int)
    q = pd.PeriodIndex(d.period, freq="Q")
    d["t"] = q.astype("int64").to_numpy(float)
    d["t"] -= d["t"].mean()
    for s in P.SHOCKS:
        d[f"impact_{s}"] = d.period.isin(P.impact_periods(s)).astype(float)
    d["impact_any"] = d[[f"impact_{s}" for s in P.SHOCKS]].max(axis=1)
    if P.TIME_CLUSTER[country] == "fy" and "fy" not in d.columns:
        raise SystemExit(f"{country} needs a financial-year column")
    import data_oil_price as oil
    d["oil4q"] = d.period.map(oil.oil_growth()).astype(float)
    if d.oil4q.isna().any():
        raise SystemExit(f"{country}: oil series does not cover {d[d.oil4q.isna()].period.min()}")
    d = M.add_volume_shares(d)
    return d.sort_values(["unit", "period"]).reset_index(drop=True)


def placebo_blocks(dc: pd.DataFrame, country: str, spec: str) -> list[list[str]]:
    extra = M.disturbed(country, spec)
    quiet = PL.quiet_periods(dc, P.SHOCKS, buffer=2, extra=extra)
    qs = set(quiet)
    stride = 4 if P.TIME_CLUSTER[country] != "period" else 1
    out = []
    for p in quiet[::stride]:
        blk = [str(pd.Period(p, freq="Q") + k) for k in range(2)]
        if all(b in qs for b in blk):
            out.append(blk)
    return out


class Projector:

    def __init__(self, X_base: pd.DataFrame, y: np.ndarray, w: np.ndarray):
        ww = np.sqrt(w)
        self.ww = ww / ww.mean()
        Xa = X_base.to_numpy(float) * self.ww[:, None]
        U, s, _ = np.linalg.svd(Xa, full_matrices=False)
        tol = s.max() * max(Xa.shape) * np.finfo(float).eps
        self.U = U[:, s > tol]
        ya = y * self.ww
        self.y_res = ya - self.U @ (self.U.T @ ya)

    def coef(self, x: np.ndarray) -> float:
        xa = x * self.ww
        xr = xa - self.U @ (self.U.T @ xa)
        den = float(xr @ xr)
        return float(xr @ self.y_res) / den if den > 1e-12 else np.nan


def h1_placebos(dc, shock, spec, blocks) -> np.ndarray:
    sub = dc.dropna(subset=["real_growth"])
    ev = M.controls_level(sub, spec, M.focal_set(shock))
    base = M.design_level(sub, pd.DataFrame(index=sub.index), ev)
    pr = Projector(base, sub.real_growth.to_numpy(float), sub.weight.to_numpy(float))
    return np.array([pr.coef(sub.period.isin(b).to_numpy(float)) for b in blocks])


def h23_placebos(dc, shock, spec, kind, blocks, margin=None) -> np.ndarray:
    outcome = M.OUTCOME[kind]
    sub = dc.dropna(subset=[outcome])
    ev = M.controls_share(sub, spec, M.focal_set(shock))
    actual = pd.DataFrame({f"x_{s}": M._inter(sub, kind, s) for s in P.SHOCKS}, index=sub.index)
    base = M.design_share(sub, actual, ev)
    pr = Projector(base, sub[outcome].to_numpy(float), sub.weight.to_numpy(float))
    m = margin or shock
    return np.array([pr.coef(M._inter(sub, kind, m, set(b)).to_numpy(float)) for b in blocks])


def verify_placebo_identity(log=print) -> float:
    worst = 0.0
    for country in ("USA", "Germany"):
        dc = load_country(country, "matched")
        blocks = placebo_blocks(dc, country, "clean")[:4]
        tk = P.TIME_CLUSTER[country]
        fast = h1_placebos(dc, "gfc", "clean", blocks)
        slow = [M.h1(dc, "gfc", "clean", set(b), bootstrap=False, time_key=tk)["estimate"] for b in blocks]
        worst = max(worst, float(np.max(np.abs(fast - np.array(slow)))))
        for kind in ("H2", "H3"):
            fast = h23_placebos(dc, "covid", "clean", kind, blocks)
            slow = [M.h23(dc, "covid", "clean", kind, set(b), cluster="unit", bootstrap=False)["estimate"]
                    for b in blocks]
            worst = max(worst, float(np.max(np.abs(fast - np.array(slow)))))
        fast = h1_placebos(dc, "combined", "clean", blocks)
        slow = [M.h1(dc, "combined", "clean", set(b), bootstrap=False, time_key=tk)["estimate"] for b in blocks]
        worst = max(worst, float(np.max(np.abs(fast - np.array(slow)))))
        for m in P.SHOCKS:
            fast = h23_placebos(dc, "combined", "clean", "H3", blocks, margin=m)
            slow = [M.h23(dc, "combined", "clean", "H3", set(b), cluster="unit", bootstrap=False,
                          placebo_margin=m)["estimate"] for b in blocks]
            worst = max(worst, float(np.max(np.abs(fast - np.array(slow)))))
    log(f"placebo identity: fast FWL coefficients vs full re-fits, worst gap {worst:.2e}")
    return worst


def combined_rows(dc, country, timeline, tk, spec, blocks, h4_draws) -> list[dict]:
    out = []
    for hyp in ("H1", "H2", "H3", "H4"):
        extra = {}
        if hyp == "H1":
            res = M.h1(dc, "combined", spec, time_key=tk)
            draws = h1_placebos(dc, "combined", spec, blocks)
        elif hyp in ("H2", "H3"):
            res = M.h23(dc, "combined", spec, hyp, cluster="unit")
            if hyp == "H3":
                draws = np.mean([h23_placebos(dc, "combined", spec, hyp, blocks, margin=m)
                                 for m in P.SHOCKS], axis=0)
            else:
                draws = h23_placebos(dc, "combined", spec, hyp, blocks, margin="covid")
        else:
            res, frame = M.h4(dc, "combined", spec, single=True)
            draws = h4_draws
            if res is not None:
                extra = {"h4_horizons": "; ".join(
                    f"{s}:{','.join(str(h) for h in sorted(g.horizon.unique()))}"
                    for s, g in frame.groupby("shock"))}
        if res is not None and "homogeneity_p" in res:
            extra.update({"homogeneity_p": res["homogeneity_p"],
                          "shock_specific": "; ".join(f"{k[2:]} {v:+.4f}" for k, v in res["components"].items())})
        sesoi = M.country_sesoi(dc, country, hyp, "combined")
        rand = (M.randomisation(res["estimate"], draws[np.isfinite(draws)], P.DIRECTION[hyp])
                if res is not None else {})
        row = M.assemble_row("combined", spec, hyp, res, sesoi, {country: sesoi}, rand, False, extra)
        row.update({"country": country, "timeline": timeline,
                    "window": f"{dc.period.min()}-{dc.period.max()}",
                    "quarters": int(dc.period.nunique()), "time_cluster": tk})
        out.append(row)
    return out


def run(country: str, timeline: str, log=print) -> tuple[list[dict], list[dict]]:
    dc = load_country(country, timeline)
    tk = P.TIME_CLUSTER[country]
    rows, comb = [], []
    for spec in SPECS:
        blocks = placebo_blocks(dc, country, spec)
        h4_draws = None
        for shock in P.SHOCKS:
            for hyp in ("H1", "H2", "H3", "H4"):
                if hyp == "H1":
                    res = M.h1(dc, shock, spec, time_key=tk)
                    draws = h1_placebos(dc, shock, spec, blocks)
                elif hyp in ("H2", "H3"):
                    res = M.h23(dc, shock, spec, hyp, cluster="unit")
                    draws = h23_placebos(dc, shock, spec, hyp, blocks)
                else:
                    res, _ = M.h4(dc, shock, spec, single=True)
                    if h4_draws is None:
                        h4_draws = []
                        for b in blocks:
                            r, _ = M.h4(dc, shock, spec, anchor=pd.Period(b[0], freq="Q"),
                                        truncate=False, clear_path=True, bootstrap=False, single=True)
                            h4_draws.append(np.nan if r is None else r["estimate"])
                        h4_draws = np.asarray(h4_draws, float)
                    draws = h4_draws
                sesoi = M.country_sesoi(dc, country, hyp, shock)
                rand = (M.randomisation(res["estimate"], draws[np.isfinite(draws)], P.DIRECTION[hyp])
                        if res is not None else {})
                extra = {}
                if hyp == "H4" and res is not None:
                    _, frame = M.h4(dc, shock, spec, single=True, bootstrap=False)
                    extra = {"h4_horizons": ",".join(str(h) for h in sorted(frame.horizon.unique())),
                             "h4_anchor": frame.anchor.iloc[0]}
                row = M.assemble_row(shock, spec, hyp, res, sesoi, {country: sesoi}, rand, False, extra)
                row.update({"country": country, "timeline": timeline,
                            "window": f"{dc.period.min()}-{dc.period.max()}",
                            "quarters": int(dc.period.nunique()), "time_cluster": tk})
                rows.append(row)
        comb += combined_rows(dc, country, timeline, tk, spec, blocks, h4_draws)
        n = sum(r["decision_headline"] in M.SETTLED for r in rows
                if r["spec"] == spec and r["hypothesis"] in ("H1", "H2", "H3", "H4"))
        log(f"  {country:8s} {timeline:7s} {spec:9s} {n:2d}/12  "
            f"({dc.period.min()}-{dc.period.max()}, {len(blocks)} placebo phases)")
    spec = P.HEADLINE_SPEC
    blocks = placebo_blocks(dc, country, spec)
    local_draws = None
    for shock in P.SHOCKS:
        res = M.h23(dc, shock, spec, "H2_volume", cluster="unit")
        draws = h23_placebos(dc, shock, spec, "H2_volume", blocks)
        rows.append(_row(dc, country, timeline, tk, shock, spec, "H2_volume", res, draws))
        res, _ = M.h4(dc, shock, spec, single=True, local=True)
        if local_draws is None:
            local_draws = []
            for b in blocks:
                r, _ = M.h4(dc, shock, spec, anchor=pd.Period(b[0], freq="Q"), truncate=False, clear_path=True,
                            bootstrap=False, single=True, local=True)
                local_draws.append(np.nan if r is None else r["estimate"])
            local_draws = np.asarray(local_draws, float)
        rows.append(_row(dc, country, timeline, tk, shock, spec, "H4_local", res, local_draws))
    return rows, comb


def _row(dc, country, timeline, tk, shock, spec, hyp, res, draws):
    sesoi = M.country_sesoi(dc, country, hyp, shock)
    rand = (M.randomisation(res["estimate"], draws[np.isfinite(draws)], P.DIRECTION[hyp])
            if res is not None else {})
    row = M.assemble_row(shock, spec, hyp, res, sesoi, {country: sesoi}, rand, False)
    row.update({"country": country, "timeline": timeline,
                "window": f"{dc.period.min()}-{dc.period.max()}",
                "quarters": int(dc.period.nunique()), "time_cluster": tk})
    return row


def main(base: str, log=print) -> None:
    root = pathlib.Path(base)
    gap = verify_placebo_identity(log)
    if gap > 1e-8:
        raise SystemExit("fast placebo coefficients disagree with full re-fits")
    rows, comb = [], []
    for country in P.COUNTRIES:
        for timeline in TIMELINES:
            if timeline == "full" and P.FULL_START[country] == P.COMMON_START:
                log(f"  {country:8s} full     = matched (record begins {P.COMMON_START})")
                continue
            r, c = run(country, timeline, log)
            rows += r
            comb += c
    out = root / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "individual_results.csv", index=False)
    pd.DataFrame(comb).to_csv(out / "individual_results_combined.csv", index=False)
    log(f"written {out / 'individual_results.csv'} and individual_results_combined.csv")


if __name__ == "__main__":
    main(sys.argv[1])
