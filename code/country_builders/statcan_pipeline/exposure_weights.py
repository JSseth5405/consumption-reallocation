from __future__ import annotations

import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import extract_statcan as E

MARGINS = ["energy", "contact", "durable", "discretionary"]

SUBITEM = {
    "Food": ("food_nonalc", 0, 0, 0, 0),
    "Non-alcoholic beverages": ("food_nonalc", 0, 0, 0, 0),
    "Alcoholic beverages": ("alcohol_tobacco", 0, 0, 0, 1),
    "Tobacco": ("alcohol_tobacco", 0, 0, 0, 1),
    "Cannabis products for non-medical use (licensed)": ("alcohol_tobacco", 0, 0, 0, 1),
    "Cannabis products for non-medical use (unlicensed)": ("alcohol_tobacco", 0, 0, 0, 1),
    "Clothing materials, other articles of clothing and clothing accessories":
        ("clothing_footwear", 0, 0, 0, 1),
    "Garments": ("clothing_footwear", 0, 0, 0, 1),
    "Cleaning of clothing": ("clothing_footwear", 0, 0, 0, 1),
    "Footwear": ("clothing_footwear", 0, 0, 0, 1),
    "Paid rental fees for housing": ("housing_utilities", 0, 0, 0, 0),
    "Imputed rental fees for housing": ("housing_utilities", 0, 0, 0, 0),
    "Materials for the maintenance and repair of the dwelling": ("housing_utilities", 0, 0, 0, 0),
    "Services for the maintenance and repair of the dwelling": ("housing_utilities", 0, 1, 0, 0),
    "Water supply and sanitation services": ("housing_utilities", 0, 0, 0, 0),
    "Electricity": ("housing_utilities", 1, 0, 0, 0),
    "Gas": ("housing_utilities", 1, 0, 0, 0),
    "Other fuels": ("housing_utilities", 1, 0, 0, 0),
    "Furniture and furnishings": ("household_goods", 0, 0, 1, 1),
    "Carpets and other floor coverings": ("household_goods", 0, 0, 1, 1),
    "Household textiles": ("household_goods", 0, 0, 0, 1),
    "Major household appliances": ("household_goods", 0, 0, 1, 1),
    "Small electric household appliances": ("household_goods", 0, 0, 0, 1),
    "Major tools and equipment": ("household_goods", 0, 0, 1, 1),
    "Small tools and miscellaneous accessories": ("household_goods", 0, 0, 0, 1),
    "Other semi-durable household goods": ("household_goods", 0, 0, 0, 1),
    "Other non-durable household goods": ("household_goods", 0, 0, 0, 1),
    "Repair of personal and household goods except vehicles": ("household_goods", 0, 1, 0, 1),
    "Renting and leasing of personal and household goods except passenger vehicles":
        ("household_goods", 0, 0, 0, 1),
    "Other services related to the dwelling and property": ("household_goods", 0, 1, 0, 0),
    "Pharmaceutical products and other medical products (except cannabis)": ("health", 0, 0, 0, 0),
    "Cannabis products for medical use": ("health", 0, 0, 0, 0),
    "Therapeutic appliances and equipment": ("health", 0, 0, 0, 0),
    "Out-patient services": ("health", 0, 1, 0, 0),
    "Hospital services": ("health", 0, 1, 0, 0),
    "New passenger cars": ("transport", 0, 0, 1, 1),
    "New trucks, vans and sport utility vehicles": ("transport", 0, 0, 1, 1),
    "Used motor vehicles": ("transport", 0, 0, 1, 1),
    "Other vehicles": ("transport", 0, 0, 1, 1),
    "Spare parts and accessories for vehicles": ("transport", 0, 0, 0, 0),
    "Fuels and lubricants": ("transport", 1, 0, 0, 0),
    "Maintenance and repair of vehicles": ("transport", 0, 0, 0, 0),
    "Parking": ("transport", 0, 0, 0, 0),
    "Passenger vehicle renting": ("transport", 0, 0, 0, 0),
    "Other services related to the operation of transport equipment": ("transport", 0, 0, 0, 0),
    "Railway transport": ("transport", 0, 1, 0, 0),
    "Urban transit": ("transport", 0, 1, 0, 0),
    "Interurban bus": ("transport", 0, 1, 0, 0),
    "Taxi and limousine": ("transport", 0, 1, 0, 0),
    "Air transport": ("transport", 0, 1, 0, 0),
    "Water transport": ("transport", 0, 1, 0, 0),
    "Other transport services": ("transport", 0, 1, 0, 0),
    "Postal services": ("communication", 0, 0, 0, 0),
    "Telecommunication equipment": ("communication", 0, 0, 1, 1),
    "Telecommunication services": ("communication", 0, 0, 0, 0),
    "Audio-visual and photographic equipment": ("recreation_culture", 0, 0, 1, 1),
    "Information processing equipment": ("recreation_culture", 0, 0, 1, 1),
    "Recording media": ("recreation_culture", 0, 0, 0, 1),
    "Major durables for outdoor recreation": ("recreation_culture", 0, 0, 1, 1),
    "Musical instruments and major durables for indoor recreation": ("recreation_culture", 0, 0, 1, 1),
    "Games, toys and hobbies": ("recreation_culture", 0, 0, 0, 1),
    "Equipment for sport, camping and open-air recreation": ("recreation_culture", 0, 0, 0, 1),
    "Garden products, plants and flowers": ("recreation_culture", 0, 0, 0, 1),
    "Pets and pet food": ("recreation_culture", 0, 0, 0, 1),
    "Veterinary and other services for pets": ("recreation_culture", 0, 0, 0, 1),
    "Recreational and sporting services": ("recreation_culture", 0, 1, 0, 1),
    "Cable, satellite and other program distribution services": ("recreation_culture", 0, 0, 0, 1),
    "Cinemas": ("recreation_culture", 0, 1, 0, 1),
    "Photographic services": ("recreation_culture", 0, 1, 0, 1),
    "Other cultural services": ("recreation_culture", 0, 1, 0, 1),
    "Games of chance": ("recreation_culture", 0, 0, 0, 1),
    "Books": ("recreation_culture", 0, 0, 0, 1),
    "Newspapers and periodicals": ("recreation_culture", 0, 0, 0, 1),
    "Miscellaneous printed matter and stationery and drawing materials":
        ("recreation_culture", 0, 0, 0, 1),
    "University education": ("education", 0, 1, 0, 0),
    "Other education": ("education", 0, 1, 0, 0),
    "Food and non-alcoholic beverage services": ("restaurants_hotels", 0, 1, 0, 1),
    "Alcoholic beverage services": ("restaurants_hotels", 0, 1, 0, 1),
    "Accommodation services": ("restaurants_hotels", 0, 1, 0, 1),
    "Life insurance": ("miscellaneous", 0, 0, 0, 0),
    "Property insurance": ("miscellaneous", 0, 0, 0, 0),
    "Health insurance": ("miscellaneous", 0, 0, 0, 0),
    "Insurance related to transport": ("miscellaneous", 0, 0, 0, 0),
    "Implicit loan charges": ("miscellaneous", 0, 0, 0, 0),
    "Implicit deposit charges": ("miscellaneous", 0, 0, 0, 0),
    "Stock and bond commissions": ("miscellaneous", 0, 0, 0, 0),
    "Other actual financial charges": ("miscellaneous", 0, 0, 0, 0),
    "Trusteed pension funds": ("miscellaneous", 0, 0, 0, 0),
    "Mutual funds": ("miscellaneous", 0, 0, 0, 0),
    "Personal grooming services": ("miscellaneous", 0, 1, 0, 1),
    "Electrical appliances for personal care": ("miscellaneous", 0, 0, 0, 1),
    "Other appliances, articles and products for personal care": ("miscellaneous", 0, 0, 0, 1),
    "Jewellery, clocks and watches": ("miscellaneous", 0, 0, 0, 1),
    "Other personal effects": ("miscellaneous", 0, 0, 0, 1),
    "Child care services outside the home": ("miscellaneous", 0, 1, 0, 0),
    "Child care services in the home": ("miscellaneous", 0, 1, 0, 0),
    "Other social services": ("miscellaneous", 0, 1, 0, 0),
    "Undertaking and other funeral services": ("miscellaneous", 0, 0, 0, 0),
    "Legal and other services": ("miscellaneous", 0, 0, 0, 0),
}


def build(root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = E.raw_table(root)
    detail = frame[(frame.Prices == E.NOMINAL) & frame.Estimates.isin(SUBITEM)].copy()
    missing = sorted(set(SUBITEM) - set(detail.Estimates))
    if missing:
        raise SystemExit(f"StatCan publishes no current-price series for {missing}")
    detail["division"] = detail.Estimates.map(lambda c: SUBITEM[c][0])
    for i, margin in enumerate(MARGINS):
        detail[margin] = detail.Estimates.map(lambda c: float(SUBITEM[c][1 + i]))
    rows = []
    for division, block in detail.groupby("division"):
        total = block.groupby("period")["VALUE"].sum()
        row = {"category": division, "n_subitems": block.Estimates.nunique(),
               "quarters_used": block.period.nunique()}
        for margin in MARGINS:
            exposed = (block[block[margin] == 1].groupby("period")["VALUE"].sum()
                       .reindex(total.index).fillna(0.0))
            row[f"{margin}_intensity"] = float((exposed / total).mean())
        rows.append(row)
    weights = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)

    published = pd.read_csv(root / "data/interim/canada_quarterly_divisions.csv")
    built = detail.groupby(["division", "period"])["VALUE"].sum().rename("built")
    check = (published.rename(columns={"category": "division"})
             .merge(built.reset_index(), on=["division", "period"], how="inner"))
    check["deviation_pc"] = 100 * (check.built - check.nominal_mcad) / check.nominal_mcad
    summary = (check.groupby("division")
               .agg(quarters=("deviation_pc", "size"),
                    max_abs_deviation_pc=("deviation_pc", lambda s: float(s.abs().max())),
                    mean_deviation_pc=("deviation_pc", "mean"))
               .reset_index())
    return weights, summary


def main(base: str) -> None:
    root = pathlib.Path(base)
    weights, check = build(root)
    (root / "config").mkdir(exist_ok=True)
    weights.to_csv(root / "config/exposure_weights.csv", index=False)
    check.to_csv(root / "metadata/exposure_partition_check.csv", index=False)
    print("measured exposure intensities (share of each division that is exposed):")
    print(weights.drop(columns=["quarters_used"]).round(3).to_string(index=False))
    print("\nsub-items (excluding StatCan adjusting entries) vs the published division total:")
    print(check.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
