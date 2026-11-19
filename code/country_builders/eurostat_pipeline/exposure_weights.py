from __future__ import annotations

import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import extract_eurostat as E

MARGINS = ["energy", "contact", "durable", "discretionary"]

SUBITEM = {
    "CP011": ("food_nonalc", 0, 0, 0, 0),
    "CP012": ("food_nonalc", 0, 0, 0, 0),
    "CP013": ("food_nonalc", 0, 0, 0, 0),
    "CP021": ("alcohol_tobacco", 0, 0, 0, 1),
    "CP022": ("alcohol_tobacco", 0, 0, 0, 1),
    "CP023": ("alcohol_tobacco", 0, 0, 0, 1),
    "CP024": ("alcohol_tobacco", 0, 0, 0, 1),
    "CP031": ("clothing_footwear", 0, 0, 0, 1),
    "CP032": ("clothing_footwear", 0, 0, 0, 1),
    "CP041": ("housing_utilities", 0, 0, 0, 0),
    "CP042": ("housing_utilities", 0, 0, 0, 0),
    "CP043": ("housing_utilities", 0, 1, 0, 0),
    "CP044": ("housing_utilities", 0, 0, 0, 0),
    "CP045": ("housing_utilities", 1, 0, 0, 0),
    "CP051": ("household_goods", 0, 0, 1, 1),
    "CP052": ("household_goods", 0, 0, 0, 1),
    "CP053": ("household_goods", 0, 0, 1, 1),
    "CP054": ("household_goods", 0, 0, 0, 1),
    "CP055": ("household_goods", 0, 0, 1, 1),
    "CP056": ("household_goods", 0, 1, 0, 1),
    "CP061": ("health", 0, 0, 0, 0),
    "CP062": ("health", 0, 1, 0, 0),
    "CP063": ("health", 0, 1, 0, 0),
    "CP064": ("health", 0, 1, 0, 0),
    "CP071": ("transport", 0, 0, 1, 1),
    "CP072": ("transport", 1, 0, 0, 0),
    "CP073": ("transport", 0, 1, 0, 0),
    "CP074": ("transport", 0, 0, 0, 0),
    "CP081": ("communication", 0, 0, 1, 1),
    "CP082": ("communication", 0, 0, 0, 1),
    "CP083": ("communication", 0, 0, 0, 0),
    "CP091": ("recreation_culture", 0, 0, 1, 1),
    "CP092": ("recreation_culture", 0, 0, 0, 1),
    "CP093": ("recreation_culture", 0, 0, 0, 1),
    "CP094": ("recreation_culture", 0, 1, 0, 1),
    "CP095": ("recreation_culture", 0, 0, 0, 1),
    "CP096": ("recreation_culture", 0, 1, 0, 1),
    "CP097": ("recreation_culture", 0, 0, 0, 1),
    "CP098": ("recreation_culture", 0, 1, 0, 1),
    "CP101": ("education", 0, 1, 0, 0),
    "CP102": ("education", 0, 1, 0, 0),
    "CP103": ("education", 0, 1, 0, 0),
    "CP104": ("education", 0, 1, 0, 0),
    "CP105": ("education", 0, 1, 0, 0),
    "CP111": ("restaurants_hotels", 0, 1, 0, 1),
    "CP112": ("restaurants_hotels", 0, 1, 0, 1),
    "CP121": ("miscellaneous", 0, 0, 0, 0),
    "CP122": ("miscellaneous", 0, 0, 0, 0),
    "CP131": ("miscellaneous", 0, 1, 0, 1),
    "CP132": ("miscellaneous", 0, 0, 0, 1),
    "CP133": ("miscellaneous", 0, 1, 0, 0),
    "CP139": ("miscellaneous", 0, 0, 0, 0),
}


def read_subitems(root: pathlib.Path) -> pd.DataFrame:
    payload = E.fetch(root, "nama_10_cp18",
                      {"unit": E.NOMINAL, "coicop18": sorted(SUBITEM)},
                      f"nama_10_cp18_{E.NOMINAL}_subitems")
    frame = E.tidy(payload, "coicop18")
    return frame.rename(columns={"time": "year"})


def build(root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = read_subitems(root)
    missing = sorted(set(SUBITEM) - set(detail.code))
    for code in missing:
        division = SUBITEM[code][0]
        siblings = {SUBITEM[c][1:] for c in SUBITEM if SUBITEM[c][0] == division}
        if len(siblings) > 1:
            raise SystemExit(
                f"Eurostat publishes no series for {code} and division "
                f"'{division}' has sub-items with differing exposure flags, so "
                f"its intensity cannot be measured from published figures")
    if missing:
        print(f"note: Eurostat publishes no separate series for {missing}; "
              "every sub-item of the affected division carries identical "
              "exposure flags, so the measured intensity is unchanged")
    detail["division"] = detail.code.map(lambda c: SUBITEM[c][0])
    for i, margin in enumerate(MARGINS):
        detail[margin] = detail.code.map(lambda c: float(SUBITEM[c][1 + i]))

    rows = []
    for division, block in detail.groupby("division"):
        total = block.groupby("year")["value"].sum()
        row = {"category": division, "n_subitems": block.code.nunique(),
               "years_used": block.year.nunique(), "source": "measured"}
        for margin in MARGINS:
            exposed = (block[block[margin] == 1].groupby("year")["value"].sum()
                       .reindex(total.index).fillna(0.0))
            row[f"{margin}_intensity"] = float((exposed / total).mean())
        rows.append(row)
    weights = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    published = pd.read_csv(root / f"data/interim/{E.PREFIX}_annual_divisions.csv")
    published["year"] = published.year.astype(str)
    built = detail.groupby(["division", "year"])["value"].sum().rename("built")
    check = (published.rename(columns={"category": "division"})
             .merge(built.reset_index(), on=["division", "year"], how="inner"))
    check["deviation_pc"] = (100 * (check.built - check.nominal_meur)
                             / check.nominal_meur)
    summary = (check.groupby("division")
               .agg(years=("deviation_pc", "size"),
                    max_abs_deviation_pc=("deviation_pc",
                                          lambda s: float(s.abs().max())))
               .reset_index())
    return weights, summary


def main(base: str) -> None:
    root = pathlib.Path(base)
    weights, check = build(root)
    (root / "config").mkdir(exist_ok=True)
    weights.drop(columns=["source"]).to_csv(root / "config/exposure_weights.csv",
                                            index=False)
    weights.to_csv(root / "metadata/exposure_weights_with_source.csv", index=False)
    check.to_csv(root / "metadata/exposure_partition_check.csv", index=False)
    print("measured exposure intensities (share of each division that is exposed):")
    print(weights.drop(columns=["years_used"]).round(3).to_string(index=False))
    print("\n3-digit sub-items vs the published division total:")
    print(check.round(5).to_string(index=False))
    print(f"worst deviation: {check.max_abs_deviation_pc.max():.5f}%")


if __name__ == "__main__":
    main(sys.argv[1])
