from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import models_individual_countries as MI
import models_all_countries as M
import study_protocol as P

POOLED_FILE = "data/all_countries_pooled_analysis_data.csv"
DICTIONARY_FILE = "data/data_dictionary.csv"
COUNTRY_FILE = "countries/{folder}/{prefix}_analysis_data.csv"
FOLDER = {"USA": ("USA", "usa"), "UK": ("UK", "uk"), "Japan": ("Japan", "japan"),
          "Canada": ("Canada", "canada"), "Germany": ("Germany", "germany"),
          "France": ("France", "france"), "Italy": ("Italy", "italy"), "India": ("India", "india")}

COUNTRY_NAME = {"USA": "United States", "UK": "United Kingdom", "Japan": "Japan",
                "Canada": "Canada", "Germany": "Germany", "France": "France", "Italy": "Italy",
                "India": "India"}
CATEGORY_NAME = {"food_nonalc": "Food and non-alcoholic beverages",
                 "alcohol_tobacco": "Alcohol and tobacco",
                 "clothing_footwear": "Clothing and footwear",
                 "housing_utilities": "Housing, water, electricity and gas",
                 "household_goods": "Furnishings and household equipment", "health": "Health",
                 "transport": "Transport", "communication": "Communication",
                 "recreation_culture": "Recreation and culture", "education": "Education",
                 "restaurants_hotels": "Restaurants and hotels",
                 "miscellaneous": "Miscellaneous goods and services"}
NOMINAL = {"USA": ("nominal_musd", "USD million"), "UK": ("nominal_mgbp", "GBP million"),
           "Japan": ("nominal_bnyen", "JPY billion"), "Canada": ("nominal_mcad", "CAD million"),
           "Germany": ("nominal_meur", "EUR million"), "France": ("nominal_meur", "EUR million"),
           "Italy": ("nominal_meur", "EUR million"), "India": ("nominal_crore", "INR crore")}

COLUMNS = [
    ("country", "Country code used in the code", "", "country"),
    ("country_name", "Country name", "", ""),
    ("period", "Quarter, YYYYQn", "", "period"),
    ("year", "Calendar year", "", "year"),
    ("quarter", "Quarter of the year (1-4)", "", ""),
    ("category", "COICOP division code used in the code", "", "category"),
    ("category_name", "COICOP division name", "", ""),
    ("data_source", "Statistical source of the category data", "", ""),
    ("category_data_published", "How the statistical office publishes spending by category: "
     "quarterly, or annually (quarters benchmarked onto an official quarterly indicator by "
     "proportional Denton-Cholette)", "", ""),
    ("in_common_timeline", "1 = row in the common timeline 1999Q2-2021Q4 (the all-country "
     "analysis and each country's matched timeline); every row of the all-country file is 1", "0/1",
     ""),
    ("time_cluster", "Time unit used for clustering in the individual analysis: the quarter, the "
     "calendar year, or India's April-March financial year", "", "period / year / fy"),
    ("time_cluster_value", "Value of that time unit for this row", "", ""),
    ("nominal_spending", "Household spending on the category at current prices", "see "
     "nominal_unit", "nominal_*"),
    ("nominal_unit", "Unit of nominal_spending", "", ""),
    ("real_index", "Real (volume) spending index of the category", "index", "real_index"),
    ("real_growth_4q", "Four-quarter log growth of real spending, x100 (H1, H3 outcome)",
     "per cent", "real_growth"),
    ("real_growth_qoq", "Quarter-on-quarter log growth of real spending, x100 (H4 normal growth)",
     "per cent", "real_growth_qoq"),
    ("budget_share", "Category share of nominal household spending", "per cent", "share"),
    ("clr_budget_share", "Centred log-ratio of the budget share (H2 outcome)", "log ratio",
     "clr_share"),
    ("clr_volume_share", "Centred log-ratio of the real-volume share (H2 robustness outcome)",
     "log ratio", "clr_vol"),
    ("spending_weight", "Regression weight: category share of its country's spending in the "
     "quarter", "fraction", "weight"),
    ("necessity", "1 = necessity category (food; housing, water, electricity and gas; health)",
     "0/1", "necessity"),
    ("exposure_dotcom", "Exposure to the dot-com bust (discretionary intensity), measured from "
     "sub-items", "0-1", "exposed_dotcom"),
    ("exposure_gfc", "Exposure to the financial crisis (durable-goods intensity), measured from "
     "sub-items", "0-1", "exposed_gfc"),
    ("exposure_covid", "Exposure to COVID-19 (contact intensity), measured from sub-items "
     "(India: the 2020 lockdown order)", "0-1", "exposed_covid"),
    ("energy_intensity", "Energy intensity of the category (oil interaction in the policy "
     "specification)", "0-1", "energy_intensity"),
    ("impact_dotcom", "1 = dot-com impact quarter (2001Q1-2001Q2)", "0/1", "impact_dotcom"),
    ("impact_gfc", "1 = financial-crisis impact quarter (2008Q4-2009Q1)", "0/1", "impact_gfc"),
    ("impact_covid", "1 = COVID-19 impact quarter (2020Q2-2020Q3)", "0/1", "impact_covid"),
    ("window_dotcom", "1 = dot-com window quarter (2000Q1-2002Q4)", "0/1", ""),
    ("window_gfc", "1 = financial-crisis window quarter (2008Q1-2009Q4)", "0/1", ""),
    ("window_covid", "1 = COVID-19 window quarter (2020Q1-2021Q4)", "0/1", ""),
    ("real_oil_price_growth_4q", "Four-quarter log growth of the real WTI oil price, x100 "
     "(policy specification control)", "per cent", "oil4q"),
    ("resilience_dotcom", "Country pre-shock resilience index for the dot-com bust (1999), z-score, "
     "higher = more resilient (H5)", "z", "res_dotcom"),
    ("resilience_gfc", "Country pre-shock resilience index for the financial crisis (2007)", "z",
     "res_gfc"),
    ("resilience_covid", "Country pre-shock resilience index for COVID-19 (2019)", "z", "res_covid"),
    ("active_events", "Country events from data/country_events.csv active for this row: event code "
     "(type, tier). Type C confounder, S structural regime, M policy response, X second shock",
     "", "built from data/country_events.csv"),
]


def _events_for(country: str, frame: pd.DataFrame) -> pd.Series:
    ev = pd.read_csv(HERE.parent.parent / "data/country_events.csv")
    ev = ev[ev.country == country]
    out = pd.Series([[] for _ in range(len(frame))], index=frame.index)
    for e in ev.itertuples():
        per = set(P.periods(e.start, e.end))
        mask = frame.period.isin(per)
        if e.scope == "category":
            mask &= frame.category.isin(str(e.categories).split(";"))
        for i in frame.index[mask]:
            out[i].append(f"{e.event} ({e.type}, {e.tier})")
    return out.map(lambda v: "; ".join(v))


def build() -> pd.DataFrame:
    res = pd.read_csv(HERE.parent.parent / "data/resilience/resilience_index.csv")
    frames = []
    for country in P.COUNTRIES:
        d = MI.load_country(country, "full")
        col, unit = NOMINAL[country]
        tc = P.TIME_CLUSTER[country]
        out = pd.DataFrame({
            "country": country, "country_name": COUNTRY_NAME[country], "period": d.period,
            "year": d.year, "quarter": d.period.str[-1].astype(int), "category": d.category,
            "category_name": d.category.map(CATEGORY_NAME),
            "data_source": P.COUNTRIES[country]["source"],
            "category_data_published": ("quarterly" if P.COUNTRIES[country]["observed"] == "quarterly"
                                        else "annually (quarters benchmarked)"),
            "in_common_timeline": ((d.period >= P.COMMON_START) & (d.period <= P.COMMON_END)).astype(int),
            "time_cluster": {"period": "quarter", "year": "calendar year",
                             "fy": "financial year (April-March)"}[tc],
            "time_cluster_value": d["period" if tc == "period" else tc].astype(str),
            "nominal_spending": d[col], "nominal_unit": unit, "real_index": d.real_index,
            "real_growth_4q": d.real_growth, "real_growth_qoq": d.real_growth_qoq,
            "budget_share": d.share, "clr_budget_share": d.clr_share, "clr_volume_share": d.clr_vol,
            "spending_weight": d.weight, "necessity": d.necessity.astype(int),
            "exposure_dotcom": d.exposed_dotcom, "exposure_gfc": d.exposed_gfc,
            "exposure_covid": d.exposed_covid, "energy_intensity": d.energy_intensity,
        })
        for s in P.SHOCKS:
            out[f"impact_{s}"] = d[f"impact_{s}"].astype(int)
        for s in P.SHOCKS:
            out[f"window_{s}"] = d.period.isin(P.window_periods(s)).astype(int)
        out["real_oil_price_growth_4q"] = d.oil4q
        for s in P.SHOCKS:
            out[f"resilience_{s}"] = float(res[(res.country == country) & (res.shock == s)].resilience.iloc[0])
        out["active_events"] = _events_for(country, out)
        frames.append(out)
    data = pd.concat(frames, ignore_index=True)
    data = data[[c[0] for c in COLUMNS]].sort_values(["country", "category", "period"])
    return data.reset_index(drop=True)


def check_against_models(data: pd.DataFrame) -> float:
    keys = ["country", "period", "category"]
    pooled = M.load_panel()
    pooled = pooled.rename(columns={c: f"model_{c}" for c in pooled.columns if c not in keys})
    common = data[data.in_common_timeline == 1]
    j = common.merge(pooled, on=keys, how="outer", indicator=True)
    if (j["_merge"] != "both").any() or len(j) != len(pooled):
        raise SystemExit("the all-country file does not cover the model data row for row")
    pairs = [("real_growth_4q", "real_growth"), ("real_growth_qoq", "real_growth_qoq"),
             ("clr_budget_share", "clr_share"), ("clr_volume_share", "clr_vol"),
             ("spending_weight", "weight"), ("exposure_dotcom", "exposed_dotcom"),
             ("exposure_gfc", "exposed_gfc"), ("exposure_covid", "exposed_covid"),
             ("real_oil_price_growth_4q", "oil4q"), ("resilience_dotcom", "res_dotcom"),
             ("resilience_gfc", "res_gfc"), ("resilience_covid", "res_covid"),
             ("impact_dotcom", "impact_dotcom"), ("impact_gfc", "impact_gfc"),
             ("impact_covid", "impact_covid")]
    worst = 0.0
    for a, b in pairs:
        x, y = j[a].to_numpy(float), j[f"model_{b}"].to_numpy(float)
        if not (np.isnan(x) == np.isnan(y)).all():
            raise SystemExit(f"missing values differ for {a}")
        worst = max(worst, float(np.nanmax(np.abs(x - y))))
    if worst > 1e-12:
        raise SystemExit(f"the all-country file differs from the model data ({worst:.2e})")
    return worst


def main(base: str, log=print) -> pathlib.Path:
    root = pathlib.Path(base)
    data = build()
    worst = check_against_models(data)
    for country, (folder, prefix) in FOLDER.items():
        out = root / COUNTRY_FILE.format(folder=folder, prefix=prefix)
        out.parent.mkdir(parents=True, exist_ok=True)
        data[data.country == country].to_csv(out, index=False)
    pooled = data[data.in_common_timeline == 1].sort_values(["country", "category", "period"])
    (root / "data").mkdir(parents=True, exist_ok=True)
    pooled.to_csv(root / POOLED_FILE, index=False)
    pd.DataFrame(COLUMNS, columns=["column", "description", "unit", "name_in_code"]).to_csv(
        root / DICTIONARY_FILE, index=False)
    log(f"data files: 8 country files ({len(data)} rows in all, each country's full record) and the "
        f"all-country file ({len(pooled)} rows, 1999Q2-2021Q4), {data.shape[1]} columns each; "
        f"identical to the model data (largest difference {worst:.1e})")
    return root / POOLED_FILE


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent))
