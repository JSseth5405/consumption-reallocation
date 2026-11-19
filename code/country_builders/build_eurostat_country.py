from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
COUNTRIES = HERE.parent.parent / "countries"


def main(geo: str, prefix: str, folder: str) -> None:
    root = COUNTRIES / folder
    for sub in ("data/interim", "data/final", "data/raw/eurostat", "metadata", "config"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    cfg = root / "_country"
    cfg.mkdir(exist_ok=True)
    (cfg / "country.py").write_text(f'GEO = "{geo}"\nPREFIX = "{prefix}"\n', encoding="utf-8")
    sys.path[:0] = [str(cfg), str(HERE / "eurostat_pipeline")]
    import extract_eurostat
    import exposure_weights
    import indicator_weights
    import disaggregate_categories
    import build_panel
    assert extract_eurostat.GEO == geo, "country config not picked up"
    print(f"===== {folder} ({geo}) =====")
    extract_eurostat.main(str(root))
    exposure_weights.main(str(root))
    indicator_weights.main(str(root))
    disaggregate_categories.main(str(root))
    build_panel.main(str(root))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
