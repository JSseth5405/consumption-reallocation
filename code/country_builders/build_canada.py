from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "statcan_pipeline"))
import extract_statcan, exposure_weights, build_panel


def main() -> None:
    root = HERE.parent.parent / "countries" / "Canada"
    for sub in ("data/interim", "data/final", "metadata", "config"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    extract_statcan.main(str(root))
    exposure_weights.main(str(root))
    build_panel.main(str(root))


if __name__ == "__main__":
    main()
