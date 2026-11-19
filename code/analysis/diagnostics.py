from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import models_all_countries as M
import study_protocol as P

ORDER = ["USA", "UK", "Japan", "Canada", "Germany", "France", "Italy", "India"]
NAME = {"USA": "United States", "UK": "United Kingdom", "Japan": "Japan", "Canada": "Canada",
        "Germany": "Germany", "France": "France", "Italy": "Italy", "India": "India"}
CATEGORY = {"food_nonalc": "Food", "alcohol_tobacco": "Alcohol, tobacco",
            "clothing_footwear": "Clothing", "housing_utilities": "Housing, utilities",
            "household_goods": "Furnishings", "health": "Health", "transport": "Transport",
            "communication": "Communication", "recreation_culture": "Recreation",
            "education": "Education", "restaurants_hotels": "Restaurants, hotels",
            "miscellaneous": "Miscellaneous"}
RECONCILIATION = {"France": "countries/France/metadata/france_reconciliation.csv",
                  "Italy": "countries/Italy/metadata/italy_reconciliation.csv",
                  "Canada": "countries/Canada/metadata/canada_reconciliation.csv"}
POOLS = list(P.SHOCKS) + ["combined"]
HYPS = ["H1", "H2", "H3", "H4"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold"})


def coverage(root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for c in ORDER:
        d = pd.read_csv(root / P.COUNTRIES[c]["panel"])
        full = d[(d.period >= P.FULL_START[c]) & (d.period <= P.COMMON_END)]
        common = d[(d.period >= P.COMMON_START) & (d.period <= P.COMMON_END)]
        wsum = common.groupby("period").weight.sum()
        rec = np.nan
        if c in RECONCILIATION:
            rec = float(pd.read_csv(root / RECONCILIATION[c]).deviation_pc.abs().max())
        rows.append({
            "country": c, "data_source": P.COUNTRIES[c]["source"],
            "categories_published": P.COUNTRIES[c]["observed"],
            "full_timeline": f"{full.period.min()}-{full.period.max()}",
            "full_quarters": full.period.nunique(),
            "common_quarters": common.period.nunique(),
            "categories": common.category.nunique(),
            "first_quarter_with_4q_growth": str(full[full.real_growth.notna()].period.min()),
            "rows_without_4q_growth_full": int(full.real_growth.isna().sum()),
            "rows_without_4q_growth_common": int(common.real_growth.isna().sum()),
            "missing_shares_common": int(common.clr_share.isna().sum()),
            "max_weight_sum_error": float((wsum - 1).abs().max()),
            "max_reconciliation_error_pc": rec,
            "time_cluster_individual": {"period": "quarter", "year": "calendar year",
                                        "fy": "financial year"}[P.TIME_CLUSTER[c]],
        })
    return pd.DataFrame(rows)


def exposure_concentration(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ORDER:
        dc = data[data.country == c]
        w = dc.groupby("category").weight.mean()
        w = w / w.sum()
        nec = dc.groupby("category").necessity.max().astype(float)
        row = {"country": c, "necessity_share_pc": float(100 * (w * nec).sum())}
        for s in P.SHOCKS:
            e = dc.groupby("category")[f"exposed_{s}"].mean()
            we = w * e
            row[f"{s}_exposed_share_pc"] = float(100 * we.sum())
            row[f"{s}_effective_exposed_categories"] = (float(we.sum() ** 2 / (we ** 2).sum())
                                                         if (we ** 2).sum() > 0 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _placebo_fn(data, pool, hyp):
    spec = P.HEADLINE_SPEC
    if hyp == "H1":
        return lambda per: M.h1(data, pool, spec, per, bootstrap=False)["estimate"]
    if hyp in ("H2", "H3"):
        if pool == "combined" and hyp == "H3":
            return lambda per: float(np.mean([M.h23(data, "combined", spec, hyp, per, bootstrap=False,
                                                    placebo_margin=m)["estimate"] for m in P.SHOCKS]))
        return lambda per: M.h23(data, pool, spec, hyp, per, bootstrap=False)["estimate"]

    def h4p(per):
        r, _ = M.h4(data, pool if pool != "combined" else "covid", spec,
                    anchor=pd.Period(sorted(per)[0], freq="Q"), truncate=False, clear_path=True, bootstrap=False)
        return None if r is None else r["estimate"]
    return h4p


def placebo_distributions(data: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    blocks = M.placebo_blocks(data)
    rows = []
    for pool in POOLS:
        for hyp in HYPS:
            fn = _placebo_fn(data, pool, hyp)
            pub = pooled[(pooled.pool == pool) & (pooled.hypothesis == hyp)].iloc[0]
            kept = []
            for blk in blocks:
                try:
                    v = fn(set(blk))
                except Exception:
                    v = None
                if v is not None and np.isfinite(v):
                    kept.append((blk, float(v)))
            draws = np.asarray([v for _, v in kept], float)
            if not np.array_equal(draws, M.placebo_draws(fn, data)):
                raise SystemExit(f"placebo draws out of step for {pool} {hyp}")
            again = M.randomisation(pub.estimate, draws, P.DIRECTION[hyp])["p_randomisation"]
            if not np.isclose(again, pub.p_randomisation, rtol=0, atol=1e-12):
                raise SystemExit(f"placebo distribution does not reproduce {pool} {hyp}")
            for blk, v in kept:
                rows.append({"pool": pool, "hypothesis": hyp, "placebo_phase": "-".join(blk),
                             "placebo_estimate": float(v), "actual_estimate": float(pub.estimate),
                             "p_randomisation": float(pub.p_randomisation)})
    return pd.DataFrame(rows)


def h4_placebo_rule_comparison(pooled_data: pd.DataFrame) -> pd.DataFrame:
    import models_individual_countries as MI
    spec = P.HEADLINE_SPEC
    rows = []

    def verdicts(res, sesoi, sc, draws_old, draws_new, use_boot):
        out = {}
        for tag, dr in (("before", draws_old), ("after", draws_new)):
            dr = np.asarray([v for v in dr if v is not None and np.isfinite(v)], float)
            rand = M.randomisation(res["estimate"], dr, "negative")
            row = M.assemble_row("x", spec, "H4", res, sesoi, sc, rand, use_boot)
            out[f"n_placebo_{tag}"] = rand.get("n_placebo", 0)
            out[f"p_placebo_{tag}"] = rand.get("p_randomisation", np.nan)
            out[f"verdict_{tag}"] = row["decision_headline"]
        return out

    for country in P.COUNTRIES:
        for timeline in MI.TIMELINES:
            if timeline == "full" and P.FULL_START[country] == P.COMMON_START:
                continue
            dc = MI.load_country(country, timeline)
            blocks = MI.placebo_blocks(dc, country, spec)
            draws = {}
            for cp in (False, True):
                d = []
                for b in blocks:
                    r, _ = M.h4(dc, "gfc", spec, anchor=pd.Period(b[0], freq="Q"), truncate=False,
                                bootstrap=False, single=True, clear_path=cp)
                    d.append(np.nan if r is None else r["estimate"])
                draws[cp] = d
            for s in P.SHOCKS:
                res, _ = M.h4(dc, s, spec, single=True, bootstrap=False)
                if res is None:
                    continue
                sesoi = M.country_sesoi(dc, country, "H4", s)
                rows.append({"analysis": "individual", "country": country, "timeline": timeline,
                             "shock": s, "estimate": res["estimate"],
                             **verdicts(res, sesoi, {country: sesoi}, draws[False], draws[True], False)})
    for pool in POOLS:
        res, _ = M.h4(pooled_data, pool, spec)
        draws = {}
        for cp in (False, True):
            draws[cp] = list(M.placebo_draws(lambda per, cp=cp: (lambda r: None if r is None else r["estimate"])(
                M.h4(pooled_data, pool if pool != "combined" else "covid", spec,
                     anchor=pd.Period(sorted(per)[0], freq="Q"), truncate=False, bootstrap=False,
                     clear_path=cp)[0]), pooled_data))
        s_, sc = M.pooled_sesoi(pooled_data, "H4", pool)
        rows.append({"analysis": "all countries", "country": "all", "timeline": "common", "shock": pool,
                     "estimate": res["estimate"], **verdicts(res, s_, sc, draws[False], draws[True], True)})
    return pd.DataFrame(rows)


def figure_timelines(root: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for s, v in P.SHOCKS.items():
        a, b = (pd.Period(q, freq="Q") for q in v["window"])
        ax.axvspan(a.year + (a.quarter - 1) / 4, b.year + b.quarter / 4, color="#f2d7d5", lw=0)
        ax.text((a.year + b.year + 1) / 2, len(ORDER) - 0.35, P.SHOCKS[s]["label"].replace(
            "Global financial crisis", "GFC"), ha="center", fontsize=7.5, color="#7b241c")
    start_c = pd.Period(P.COMMON_START, freq="Q")
    end = pd.Period(P.COMMON_END, freq="Q")
    for i, c in enumerate(reversed(ORDER)):
        s = pd.Period(P.FULL_START[c], freq="Q")
        quarterly = P.COUNTRIES[c]["observed"] == "quarterly"
        ax.barh(i, end.year + 1 - (s.year + (s.quarter - 1) / 4), left=s.year + (s.quarter - 1) / 4,
                height=0.55, color="#9fb6cd" if quarterly else "#c8d6c5")
        ax.barh(i, end.year + 1 - (start_c.year + (start_c.quarter - 1) / 4),
                left=start_c.year + (start_c.quarter - 1) / 4, height=0.25,
                color="#1f3b5c" if quarterly else "#2e5e2a")
        ax.text(s.year + (s.quarter - 1) / 4 - 0.6, i, f"{NAME[c]} ({P.FULL_START[c]})", ha="right",
                va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(1945, 2023.5)
    ax.set_ylim(-0.6, len(ORDER) - 0.1)
    ax.set_xlabel("Year")
    ax.set_title("Timelines: full record (wide bar) and common timeline 1999Q2-2021Q4 (narrow bar)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#9fb6cd"), plt.Rectangle((0, 0), 1, 1, color="#c8d6c5"),
               plt.Rectangle((0, 0), 1, 1, color="#f2d7d5")]
    ax.legend(handles, ["categories published quarterly", "categories published annually "
                        "(quarters benchmarked)", "shock windows"], loc="lower left", fontsize=7.5,
              frameon=False)
    fig.tight_layout()
    fig.savefig(root / "results/diagnostics/diag_timelines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_spending_shares(root: pathlib.Path, data: pd.DataFrame) -> None:
    w = data.groupby(["country", "category"]).weight.mean().unstack().reindex(ORDER)
    w = 100 * w.div(w.sum(axis=1), axis=0)
    cats = list(CATEGORY)
    colors = plt.get_cmap("tab20").colors
    fig, ax = plt.subplots(figsize=(10, 3.8))
    left = np.zeros(len(ORDER))
    for k, cat in enumerate(cats):
        vals = w[cat].to_numpy()
        ax.barh(range(len(ORDER)), vals, left=left, color=colors[k], label=CATEGORY[cat])
        left += vals
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NAME[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of household spending, 1999Q2-2021Q4 average (%)")
    ax.set_title("Spending composition by country")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=6, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(root / "results/diagnostics/diag_spending_shares.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_placebos(root: pathlib.Path, pl: pd.DataFrame) -> None:
    label = {"dotcom": "Dot-com", "gfc": "GFC", "covid": "COVID-19", "combined": "All three"}
    fig, axes = plt.subplots(len(POOLS), len(HYPS), figsize=(11, 8))
    rng = np.random.default_rng(P.SEED)
    for i, pool in enumerate(POOLS):
        for j, hyp in enumerate(HYPS):
            ax = axes[i, j]
            b = pl[(pl.pool == pool) & (pl.hypothesis == hyp)]
            ax.scatter(b.placebo_estimate, rng.uniform(-0.3, 0.3, len(b)), s=14, color="#7f8c8d",
                       alpha=0.8, label="placebo periods")
            act = float(b.actual_estimate.iloc[0])
            ax.axvline(act, color="#c0392b", lw=1.8, label="actual estimate")
            ax.axvline(0, color="black", lw=0.5)
            lo = min(b.placebo_estimate.min(), act, 0.0)
            hi = max(b.placebo_estimate.max(), act, 0.0)
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ax.set_xlim(lo - pad, hi + pad)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.set_yticks([])
            ax.set_ylim(-1, 1)
            right = act < (lo + hi) / 2
            ax.text(0.97 if right else 0.03, 0.86, f"p = {b.p_randomisation.iloc[0]:.3f}",
                    transform=ax.transAxes, fontsize=7.5, ha="right" if right else "left", zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
            if i == 0:
                ax.set_title(hyp)
            if j == 0:
                ax.set_ylabel(label[pool], rotation=0, ha="right", va="center", fontweight="bold")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Placebo test, all countries together (headline): the actual estimate against the "
                 "same model run on every normal two-quarter period", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(root / "results/diagnostics/diag_placebo_distributions.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main(base: str, log=print) -> None:
    root = pathlib.Path(base)
    out = root / "results/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    data = M.load_panel()
    pooled = pd.read_csv(root / "results/tables/pooled_results.csv")
    pooled = pooled[pooled.spec == P.HEADLINE_SPEC]
    cov = coverage(root)
    cov.to_csv(out / "data_coverage.csv", index=False)
    exp = exposure_concentration(data)
    exp.to_csv(out / "exposure_concentration.csv", index=False)
    pl = placebo_distributions(data, pooled)
    pl.to_csv(out / "placebo_distributions_pooled.csv", index=False)
    h4c = h4_placebo_rule_comparison(data)
    h4c.to_csv(out / "h4_placebo_rule_comparison.csv", index=False)
    figure_timelines(root)
    figure_spending_shares(root, data)
    figure_placebos(root, pl)
    log(f"diagnostics: coverage ({len(cov)} countries), exposure concentration, "
        f"{len(pl)} placebo estimates reproducing every published placebo p-value, 3 figures")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent))
