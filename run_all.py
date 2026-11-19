from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "analysis"))
BUILDERS = ROOT / "code" / "country_builders"


def main() -> None:
    for folder in ("data/resilience", "results/tables", "results/figures", "results/diagnostics",
                   "results/reports"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    if "--panels" in sys.argv:
        print("[0/6] rebuilding the France, Italy and Canada panels from their raw files")
        for geo, prefix, folder in (("FR", "france", "France"), ("IT", "italy", "Italy")):
            subprocess.run([sys.executable, str(BUILDERS / "build_eurostat_country.py"), geo, prefix,
                            folder], check=True)
        subprocess.run([sys.executable, str(BUILDERS / "build_canada.py")], check=True)
    import data_resilience_index
    import models_individual_countries
    import models_all_countries
    import robustness_quarterly_vs_annual
    import tables_all_countries
    import tables_individual_countries
    import data_export
    import diagnostics
    import figures_paper
    import figures_paper_extra
    import report_results
    print("[1/6] pre-shock resilience index (H5)")
    data_resilience_index.main(str(ROOT))
    print("\n[2/6] individual countries: H1-H4, 8 countries, full and matched timelines, 4 specifications")
    models_individual_countries.main(str(ROOT))
    print("\n[3/6] all countries together: H1-H5, 3 shocks and combined, 4 specifications")
    models_all_countries.main(str(ROOT))
    print("\n      data check: quarterly-observed countries only, benchmarked countries only")
    robustness_quarterly_vs_annual.main(str(ROOT))
    print("\n[4/6] result tables and comparisons")
    tables_all_countries.main(str(ROOT))
    tables_individual_countries.main(str(ROOT))
    print("\n[5/6] data files (8 country files, 1 all-country file) and diagnostics")
    data_export.main(str(ROOT))
    diagnostics.main(str(ROOT))
    print("\n[6/6] figures and the results document")
    figures_paper.main(str(ROOT))
    figures_paper_extra.main(str(ROOT))
    report_results.main(str(ROOT))
    print("\ndone")


if __name__ == "__main__":
    main()
