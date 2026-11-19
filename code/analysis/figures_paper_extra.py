from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

NAVY, RED, GREY = "#1f4e79", "#8b1e1e", "#9aa5b1"
ORDER = ["USA", "UK", "Japan", "Canada", "Germany", "France", "Italy", "India"]
CODE = {"USA": "US", "UK": "UK", "Japan": "JP", "Canada": "CA", "Germany": "DE", "France": "FR",
        "Italy": "IT", "India": "IN"}
NAME = {"USA": "United States", "UK": "United Kingdom", "Japan": "Japan", "Canada": "Canada",
        "Germany": "Germany", "France": "France", "Italy": "Italy", "India": "India"}
TITLE = {"dotcom": "Dot-com bust", "gfc": "Global financial crisis", "covid": "COVID-19"}
SHOCKS = ["dotcom", "gfc", "covid"]
OFFSET = {("covid", "Germany"): (-14, 4), ("covid", "France"): (4, -9), ("gfc", "UK"): (-16, -2)}


def style() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#444444", "xtick.color": "#444444",
                         "ytick.color": "#444444"})


def save(fig, out: pathlib.Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def country_vs_pooled(ind: pd.DataFrame, pooled: pd.DataFrame, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, s in zip(axes, SHOCKS):
        m = ind[(ind.spec == "clean") & (ind.timeline == "matched") & (ind.pool == s)
                & (ind.hypothesis == "H1")].set_index("country")
        for i, c in enumerate(ORDER):
            r = m.loc[c]
            t = stats.t.ppf(0.95, r.df)
            col = NAVY if r.decision_headline == "supported" else GREY
            ax.errorbar(r.estimate, i, xerr=t * r.std_error, fmt="o", color=col, ms=4, capsize=2, lw=1)
        q = pooled[(pooled.spec == "clean") & (pooled.pool == s) & (pooled.hypothesis == "H1")].iloc[0]
        t = stats.t.ppf(0.95, q.df)
        y = len(ORDER) + 0.6
        ax.fill([q.estimate - t * q.std_error, q.estimate, q.estimate + t * q.std_error, q.estimate],
                [y, y + 0.32, y, y - 0.32], color=RED)
        ax.axvline(0, color="black", lw=0.6)
        ax.axvline(q.estimate, color=RED, lw=0.6, ls=":")
        ax.set_title(TITLE[s], fontweight="bold")
        ax.set_xlabel("Impact on 4-quarter real consumption growth (pp)")
    axes[0].set_yticks(list(range(len(ORDER))) + [len(ORDER) + 0.6])
    axes[0].set_yticklabels([NAME[c] for c in ORDER] + ["All countries (pooled)"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, out, "fig4_country_vs_pooled_h1")


def resilience_vs_impact(ind: pd.DataFrame, res: pd.DataFrame, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, s in zip(axes, SHOCKS):
        m = ind[(ind.spec == "clean") & (ind.timeline == "matched") & (ind.pool == s)
                & (ind.hypothesis == "H1")].set_index("country")
        r = res[res.shock == s].set_index("country").resilience
        x = np.array([r[c] for c in ORDER])
        y = np.array([m.loc[c, "estimate"] for c in ORDER])
        sup = np.array([m.loc[c, "decision_headline"] == "supported" for c in ORDER])
        ax.axhline(0, color="#444444", lw=0.6)
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 10)
        ax.plot(xs, b[1] + b[0] * xs, color=GREY, lw=1.2, ls="--")
        ax.scatter(x[sup], y[sup], s=34, color=NAVY, marker="o", edgecolor="white", linewidth=0.8,
                   zorder=3, label="Impact supported")
        ax.scatter(x[~sup], y[~sup], s=34, color=GREY, marker="s", edgecolor="white", linewidth=0.8,
                   zorder=3, label="Impact not supported")
        for xi, yi, c in zip(x, y, ORDER):
            ax.annotate(CODE[c], (xi, yi), xytext=OFFSET.get((s, c), (4, 3)), textcoords="offset points",
                        fontsize=7.5, color="#222222")
        slope = f"{b[0]:+.2f}".replace("-", "−")
        ax.set_title(f"{TITLE[s]}  (fitted slope {slope})", fontweight="bold", fontsize=9)
        ax.set_xlabel("Pre-shock resilience index (s.d.)")
    axes[0].set_ylabel("Impact on consumption growth, H1 (pp)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, out, "fig_h5_resilience_vs_impact")


def exposure_shares(expc: pd.DataFrame, out: pathlib.Path) -> None:
    cols = [("necessity_share_pc", "Necessities (H2)"), ("dotcom_exposed_share_pc", "Dot-com: discretionary"),
            ("gfc_exposed_share_pc", "Financial crisis: durables"),
            ("covid_exposed_share_pc", "COVID-19: contact-intensive")]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=True)
    ypos = np.arange(len(ORDER))
    for ax, (col, lab) in zip(axes, cols):
        v = np.array([expc.loc[c, col] for c in ORDER])
        ax.barh(ypos, v, color=NAVY, height=0.62)
        for yi, vi in zip(ypos, v):
            ax.text(vi + 0.8, yi, f"{vi:.0f}", va="center", fontsize=7.5, color="#222222")
        ax.set_title(lab, fontweight="bold", fontsize=9)
        ax.set_xlim(0, max(60, v.max() * 1.18))
        ax.set_xlabel("Share of spending (%)")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([NAME[c] for c in ORDER])
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, out, "figA_exposure_shares")


def placebo_vs_clustered(ind: pd.DataFrame, out: pathlib.Path) -> None:
    c = ind[(ind.spec == "clean") & ind.hypothesis.isin(["H1", "H2", "H3", "H4"])].copy()
    c = c[np.isfinite(c.p_parametric) & np.isfinite(c.p_randomisation)]
    hi, lo = c[c.n_placebo >= 19], c[c.n_placebo < 19]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(lo.p_parametric.clip(1e-4), lo.p_randomisation, s=14, color=GREY, marker="s", alpha=0.75,
               label="Fewer than 19 placebo periods")
    ax.scatter(hi.p_parametric.clip(1e-4), hi.p_randomisation, s=16, color=NAVY, marker="o", alpha=0.8,
               label="19 or more placebo periods")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(0.05, color=RED, lw=0.8, ls="--")
    ax.axhline(0.05, color=RED, lw=0.8, ls="--")
    ax.set_xlabel("Clustered test, one-sided p-value (log scale)")
    ax.set_ylabel("Placebo test, one-sided p-value (log scale)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, out, "figA_placebo_vs_clustered")


def main(root: str, log=print) -> None:
    base = pathlib.Path(root)
    out = base / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    ind = pd.read_csv(base / "results/tables/individual_results.csv")
    pooled = pd.read_csv(base / "results/tables/pooled_results.csv")
    res = pd.read_csv(base / "data/resilience/resilience_index.csv")
    expc = pd.read_csv(base / "results/diagnostics/exposure_concentration.csv").set_index("country")
    country_vs_pooled(ind, pooled, out)
    resilience_vs_impact(ind, res, out)
    exposure_shares(expc, out)
    placebo_vs_clustered(ind, out)
    log("  paper figures 4, 5, B1 and E1 written to results/figures")


if __name__ == "__main__":
    main(str(pathlib.Path(__file__).resolve().parents[2]))
