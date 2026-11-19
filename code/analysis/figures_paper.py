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
COLOR = dict(zip(ORDER, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                         "#8c564b", "#e377c2", "#17becf", "#ff7f0e"]))
TITLE = {"dotcom": "Dot-com bust", "gfc": "Global financial crisis", "covid": "COVID-19"}
XLABEL = {"dotcom": "discretionary intensity", "gfc": "durable-goods intensity",
          "covid": "contact intensity"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def _save(fig, root, name):
    out = root / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return out / f"{name}.png"


def _wls_slope(x, y, w):
    X = np.column_stack([np.ones_like(x), x])
    W = np.sqrt(w)
    b = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
    return b


def figure1_data(data: pd.DataFrame) -> pd.DataFrame:
    d = data.dropna(subset=["real_growth"]).copy()
    avg = d.groupby(["country", "period"]).apply(
        lambda b: np.average(b.real_growth, weights=b.weight), include_groups=False)
    d["rel"] = d.real_growth - d.set_index(["country", "period"]).index.map(avg).to_numpy()
    rows = []
    for c, dc in d.groupby("country"):
        normal = ~dc.period.isin(M.all_windows() | M.disturbed(c, P.HEADLINE_SPEC))
        base = dc[normal].groupby("category").rel.mean()
        for s in P.SHOCKS:
            imp = dc[dc.period.isin(P.impact_periods(s))].groupby("category")
            for cat, b in imp:
                rows.append({"country": c, "shock": s, "category": cat,
                             "exposure": float(b[f"exposed_{s}"].iloc[0]),
                             "weight": float(dc[dc.category == cat].weight.mean()),
                             "relative_impact": float(b.rel.mean() - base[cat])})
    return pd.DataFrame(rows)


def figure1(root, data, pooled) -> pathlib.Path:
    f = figure1_data(data)
    f.to_csv(root / "results/tables/figure_data_2_reallocation.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    for ax, s in zip(axes, P.SHOCKS):
        b = f[f.shock == s]
        for c in ORDER:
            bc = b[b.country == c]
            ax.scatter(bc.exposure, bc.relative_impact, s=12 + 900 * bc.weight, color=COLOR[c],
                       alpha=0.75, edgecolor="white", linewidth=0.5, label=NAME[c])
        a0, a1 = _wls_slope(b.exposure.to_numpy(), b.relative_impact.to_numpy(), b.weight.to_numpy())
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, a0 + a1 * xs, color="black", lw=1.4)
        ax.axhline(0, color="grey", lw=0.6)
        est = pooled[(pooled.pool == s) & (pooled.hypothesis == "H3")].iloc[0]
        ax.set_title(TITLE[s])
        ax.set_xlabel(f"Exposure to this shock ({XLABEL[s]})")
        ax.text(0.02, 0.03, f"line through points: {a1:+.1f}\nmodel estimate (H3): {est.estimate:+.1f} "
                f"({est.std_error:.1f})", transform=ax.transAxes, fontsize=7.5, va="bottom")
    axes[0].set_ylabel("Category growth in impact quarters, relative to\nthe country average "
                       "and to normal times (pp)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=8, fontsize=7.5, markerscale=0.6,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return _save(fig, root, "fig2_reallocation_h3")


def figure2_data(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in P.SHOCKS:
        fr = M.h4_rows(data, s, P.HEADLINE_SPEC, horizons=tuple(range(1, 17)))
        for (c, h), b in fr.groupby(["country", "horizon"]):
            rows.append({"shock": s, "country": c, "horizon": int(h), "anchor": b.anchor.iloc[0],
                         "gap": float(np.average(b.gap, weights=b.weight))})
    return pd.DataFrame(rows)


def figure2(root, data, pooled) -> pathlib.Path:
    f = figure2_data(data)
    f.to_csv(root / "results/tables/figure_data_3_recovery_gaps.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True,
                             gridspec_kw={"width_ratios": [16, 16, 10]})
    for ax, s in zip(axes, P.SHOCKS):
        b = f[f.shock == s]
        for c in ORDER:
            bc = b[b.country == c].sort_values("horizon")
            ax.plot(bc.horizon, bc.gap, color=COLOR[c], lw=1.1, alpha=0.85, label=NAME[c])
        avg = b.groupby("horizon").gap.mean()
        n = b.groupby("horizon").country.nunique()
        avg = avg[n == n.max()]
        ax.plot(avg.index, avg.values, color="black", lw=2.4, label="Average of countries")
        hmax = int(b.horizon.max())
        for h in P.H4_HORIZONS:
            if h <= hmax:
                ax.axvline(h, color="grey", lw=0.6, ls=":")
        ax.axhline(0, color="grey", lw=0.8)
        est = pooled[(pooled.pool == s) & (pooled.hypothesis == "H4")].iloc[0]
        ax.set_title(TITLE[s])
        ax.set_xlabel("Quarters after the last pre-shock quarter")
        ax.set_xlim(0.5, hmax + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.text(0.02, 0.03, f"H4 estimate: {est.estimate:+.2f} ({est.std_error:.2f})",
                transform=ax.transAxes, fontsize=7.5)
    axes[0].set_ylabel("Average category gap from its\nnormal-growth path (log points x 100)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=9, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return _save(fig, root, "fig3_recovery_gaps_h4")


def figure3_data(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in P.SHOCKS:
        a = pd.Period(P.SHOCKS[s]["window"][0], freq="Q") - 1
        for c, dc in data.groupby("country"):
            at = dc[dc.period == str(a)].set_index("category")
            if at.empty:
                continue
            w = at.weight / at.weight.sum()
            for h in range(-4, 17):
                q = a + h
                if str(q) < P.COMMON_START or str(q) > P.COMMON_END:
                    continue
                lvl = dc[dc.period == str(q)].set_index("category").real_index
                rows.append({"shock": s, "country": c, "h": h, "period": str(q),
                             "change": float(100 * ((w * lvl / at.real_index).sum() - 1))})
    return pd.DataFrame(rows)


def figure3(root, data) -> pathlib.Path:
    f = figure3_data(data)
    f.to_csv(root / "results/tables/figure_data_1_consumption_paths.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [19, 21, 13]})
    for ax, s in zip(axes, P.SHOCKS):
        a = pd.Period(P.SHOCKS[s]["window"][0], freq="Q") - 1
        w0, w1 = ((pd.Period(q, freq="Q") - a).n for q in P.SHOCKS[s]["window"])
        i0, i1 = ((pd.Period(q, freq="Q") - a).n for q in P.SHOCKS[s]["impact_quarters"])
        ax.axvspan(w0 - 0.5, w1 + 0.5, color="#eef2f6", lw=0)
        ax.axvspan(i0 - 0.5, i1 + 0.5, color="#d9e1ea", lw=0)
        b = f[f.shock == s]
        for c in ORDER:
            bc = b[b.country == c].sort_values("h")
            ax.plot(bc.h, bc.change, color=COLOR[c], lw=1.3, label=NAME[c])
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_title(TITLE[s])
        ax.set_xlabel(f"Quarters since {a}")
        ax.set_xlim(int(b.h.min()) - 0.5, int(b.h.max()) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Real household consumption (fixed pre-shock\nweights), % change since that quarter")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=8, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    return _save(fig, root, "fig1_consumption_paths")


def main(base: str, log=print) -> list[pathlib.Path]:
    root = pathlib.Path(base)
    data = M.load_panel()
    pooled = pd.read_csv(root / "results/tables/pooled_results.csv")
    pooled = pooled[pooled.spec == P.HEADLINE_SPEC]
    out = [figure3(root, data), figure1(root, data, pooled), figure2(root, data, pooled)]
    log("paper figures: " + ", ".join(p.name for p in out))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent))
