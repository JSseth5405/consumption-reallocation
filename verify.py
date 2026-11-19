from __future__ import annotations

import hashlib
import importlib.metadata as md
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "analysis"))
TABLES = ["results/tables/individual_results.csv", "results/tables/individual_counts_by_spec.csv",
          "results/tables/individual_core_vs_clean_changes.csv",
          "results/tables/individual_results_combined.csv",
          "results/tables/individual_combined_counts_by_spec.csv", "results/tables/pooled_results.csv",
          "results/tables/per_country_results.csv", "results/tables/h5_by_component.csv",
          "results/tables/h5_meta_regression.csv",
          "results/tables/robustness_quarterly_observed.csv",
          "results/tables/figure_data_1_consumption_paths.csv",
          "results/tables/figure_data_2_reallocation.csv",
          "results/tables/figure_data_3_recovery_gaps.csv",
          "results/diagnostics/data_coverage.csv", "results/diagnostics/exposure_concentration.csv",
          "results/diagnostics/placebo_distributions_pooled.csv",
          "results/diagnostics/h4_placebo_rule_comparison.csv",
          "data/resilience/resilience_index.csv",
          "data/all_countries_pooled_analysis_data.csv", "data/data_dictionary.csv",
          "countries/USA/usa_analysis_data.csv", "countries/UK/uk_analysis_data.csv",
          "countries/Japan/japan_analysis_data.csv", "countries/Canada/canada_analysis_data.csv",
          "countries/Germany/germany_analysis_data.csv", "countries/France/france_analysis_data.csv",
          "countries/Italy/italy_analysis_data.csv", "countries/India/india_analysis_data.csv"]
REPORT = []


def say(*parts) -> None:
    line = " ".join(str(x) for x in parts)
    print(line)
    REPORT.append(line)


def check_environment() -> list[str]:
    problems = []
    say("environment")
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if not m:
            continue
        name, want = m.groups()
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            problems.append(f"{name} not installed")
            continue
        flag = "" if have == want else f"  (pinned {want})"
        say(f"  {name:12s} {have}{flag}")
        if have.split(".")[0] != want.split(".")[0]:
            problems.append(f"{name} {have}: major version differs from pinned {want}")
    return problems


def check_inputs() -> list[str]:
    problems = []
    say("\ninputs")
    man = pd.read_csv(ROOT / "countries/MANIFEST.csv")
    for r in man.itertuples():
        path = ROOT / "countries" / r.file
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        ok = got == r.sha256
        say(f"  {r.file:52s} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            problems.append(f"input {r.file} does not match its manifest hash")
    return problems


def check_data() -> list[str]:
    import models_all_countries as M
    import study_protocol as P
    problems = []
    say("\ndata")
    d = M.load_panel()
    shape = (d.country.nunique(), d.category.nunique(), d.period.nunique())
    say(f"  {shape[0]} countries x {shape[1]} divisions x {shape[2]} quarters, "
          f"{d.period.min()}-{d.period.max()}")
    if shape != (8, 12, 91) or d.period.min() != P.COMMON_START or d.period.max() != P.COMMON_END:
        problems.append(f"panel shape {shape} / window differs from the protocol")
    w = d.groupby(["country", "period"]).weight.sum()
    say(f"  within-country weights sum to {w.min():.9f} - {w.max():.9f}")
    if (w - 1).abs().max() > 1e-6:
        problems.append("within-country weights do not sum to one")
    for name, path, tol in (
            ("France", "countries/France/metadata/france_reconciliation.csv", 0.01),
            ("Italy", "countries/Italy/metadata/italy_reconciliation.csv", 0.01),
            ("Canada", "countries/Canada/metadata/canada_reconciliation.csv", 0.01)):
        rec = pd.read_csv(ROOT / path)
        worst = float(rec.deviation_pc.abs().max())
        say(f"  {name:7s} divisions vs published total: max deviation {worst:.6f}%")
        if worst > tol:
            problems.append(f"{name} reconciliation drifted to {worst:.6f}%")
    res = pd.read_csv(ROOT / "data/resilience/resilience_index.csv")
    if res.resilience.isna().any():
        problems.append("resilience index has missing values")
    say(f"  resilience index: {len(res)} country-shocks, "
          f"min components {int(res.n_components.min())}")
    return problems


def check_algorithm() -> list[str]:
    import inference_tests as inf
    import inference_cluster_bootstrap as ip
    problems = []
    say("\nalgorithm")
    rng = np.random.default_rng(7)
    codes = np.repeat(np.arange(14), 70)
    n = len(codes)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, 25))])
    y = X @ rng.normal(size=26) + rng.normal(size=n) * (1 + codes % 3)
    t1, d1 = inf.wild_cluster_bootstrap_t(X, y, codes, 4)
    t2, d2 = ip.wcr_bootstrap(X, y, codes, 4)
    c = np.zeros(26)
    c[[3, 4, 5]] = 1 / 3
    t3, d3 = inf.wild_cluster_bootstrap_contrast(X, y, codes, c)
    t4, d4 = ip.wcr_bootstrap(X, y, codes, contrast=c)
    e1 = max(abs(t1 - t2), float(np.max(np.abs(d1 - d2))))
    e2 = max(abs(t3 - t4), float(np.max(np.abs(d3 - d4))))
    say(f"  fast bootstrap vs reference routine: coefficient {e1:.1e}, contrast {e2:.1e}")
    if e1 > 1e-8 or e2 > 1e-8:
        problems.append("fast bootstrap differs from the reference routine")
    return problems


def check_placebo_identity() -> list[str]:
    import models_individual_countries as MI
    say("\nplacebo identity (fast projection vs full re-fits)")
    gap = MI.verify_placebo_identity(log=lambda m: say("  " + m))
    return [] if gap <= 1e-8 else [f"fast placebo coefficients differ from re-fits ({gap:.2e})"]


def check_regression() -> list[str]:
    problems = []
    say("\nregression check (France/Italy pipeline on frozen German data)")
    work = ROOT / "countries/Germany/reproduction_check"
    raw = work / "data/raw/eurostat/DE"
    if not raw.exists() or not any(raw.glob("*.json")):
        say("  frozen German raw cache not shipped; skipped")
        return problems
    subprocess.run([sys.executable, str(ROOT / "code/country_builders/build_eurostat_country.py"),
                    "DE", "germany_check", "Germany/reproduction_check"], check=True, capture_output=True)
    a = pd.read_csv(ROOT / "countries/Germany/data/final/germany_quarterly_category_panel.csv")
    b = pd.read_csv(work / "data/final/germany_check_quarterly_category_panel.csv")
    m = a.merge(b, on=["period", "category"], suffixes=("_a", "_b"))
    worst = 0.0
    for col in ["share", "clr_share", "real_growth", "real_index", "weight",
                "contact_intensity", "durable_intensity", "discretionary_intensity"]:
        worst = max(worst, float(np.nanmax(np.abs(m[col + "_a"] - m[col + "_b"]))))
    say(f"  {len(m)} rows compared, worst difference {worst:.1e}")
    if len(m) != len(a) or worst > 1e-9:
        problems.append(f"parameterised pipeline no longer reproduces Germany ({worst:.2e})")
    return problems


def check_results() -> list[str]:
    problems = []
    say("\nresults (full re-estimation of everything, compared with the shipped files)")
    keep = ROOT / "results/_published"
    if keep.exists():
        shutil.rmtree(keep)
    keep.mkdir(parents=True)
    for name in TABLES:
        shutil.copyfile(ROOT / name, keep / name.replace("/", "__"))
    quiet = lambda *a, **k: None
    import data_resilience_index as resilience
    import models_individual_countries as MI
    import models_all_countries as M
    import robustness_quarterly_vs_annual as RS
    import tables_all_countries as R
    import tables_individual_countries as RI
    import data_export as EX
    import diagnostics as DG
    import figures_paper as FP
    import figures_paper_extra as FX
    resilience.main(str(ROOT))
    MI.main(str(ROOT), log=quiet)
    M.main(str(ROOT), log=quiet)
    RS.main(str(ROOT), log=quiet)
    R.main(str(ROOT))
    RI.main(str(ROOT))
    EX.main(str(ROOT), log=quiet)
    DG.main(str(ROOT), log=quiet)
    FP.main(str(ROOT), log=quiet)
    FX.main(str(ROOT), log=quiet)
    for name in TABLES:
        a, b = pd.read_csv(keep / name.replace("/", "__")), pd.read_csv(ROOT / name)
        if a.shape != b.shape:
            problems.append(f"{name}: shape {a.shape} -> {b.shape}")
            say(f"  {name:66s} DIFFERS (shape)")
            continue
        num = a.select_dtypes("number").columns
        worst = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if len(num) else 0.0
        nan_ok = (a[num].isna().to_numpy() == b[num].isna().to_numpy()).all()
        txt = [c for c in a.columns if c not in num and not
               a[c].astype("string").fillna("<NA>").eq(b[c].astype("string").fillna("<NA>")).all()]
        ok = worst <= 1e-9 and nan_ok and not txt
        say(f"  {name:66s} {'reproduces exactly' if ok else f'DIFFERS (max {worst:.2e}, text {txt})'}")
        if not ok:
            problems.append(f"{name} does not reproduce")
    shutil.rmtree(keep)
    return problems


def main() -> None:
    problems = check_environment() + check_inputs() + check_data() + check_algorithm()
    problems += check_placebo_identity() + check_regression()
    if "--all" in sys.argv:
        problems += check_results()
    say("\n" + "=" * 60)
    if problems:
        say("VERIFICATION FAILED")
        for p in problems:
            say("  -", p)
    else:
        say("VERIFICATION PASSED" + (" - every published number reproduces exactly."
                                     if "--all" in sys.argv else
                                     " (fast checks; run with --all to re-estimate)"))
    out = ROOT / "results/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    name = "verification_report_full.txt" if "--all" in sys.argv else "verification_report_fast.txt"
    (out / name).write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
