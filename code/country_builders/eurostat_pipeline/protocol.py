from __future__ import annotations

SHOCKS = {
    "dotcom": {
        "label": "Dot-com / early-2000s slowdown",
        "impact_quarters": ("2001Q1", "2001Q2"),
        "window": ("2000Q1", "2002Q4"),
        "event": "NBER dates the business-cycle peak at March 2001; the "
                 "technology bust and the equity decline reach household "
                 "spending through wealth and employment",
    },
    "gfc": {
        "label": "Global financial crisis",
        "impact_quarters": ("2008Q4", "2009Q1"),
        "window": ("2008Q1", "2009Q4"),
        "event": "Lehman Brothers fails in September 2008; the impact phase is "
                 "the two quarters that follow",
    },
    "covid": {
        "label": "COVID-19",
        "impact_quarters": ("2020Q2", "2020Q3"),
        "window": ("2020Q1", "2021Q4"),
        "event": "national emergency declared 13 March 2020 and state stay-at-"
                 "home orders follow through late March and April; the impact "
                 "phase is the lockdown quarter and the quarter after it",
    },
    "energy": {
        "label": "Energy / inflation",
        "impact_quarters": ("2022Q2", "2022Q3"),
        "window": ("2022Q1", "2023Q4"),
        "event": "Russia invades Ukraine in February 2022; crude and retail "
                 "gasoline prices peak in the following two quarters",
    },
}

NECESSITY = ["food_nonalc", "housing_utilities", "health"]
NECESSITY_BROAD = NECESSITY + ["education"]

EXPOSURE_MARGIN = {
    "dotcom": "discretionary_intensity",
    "gfc": "durable_intensity",
    "covid": "contact_intensity",
    "energy": "energy_intensity",
}
EXPOSURE_NOTE = {
    "dotcom": "postponable discretionary spending",
    "gfc": "postponable durable and semi-durable purchases",
    "covid": "consumption requiring physical co-presence, measured from "
             "Destatis three-digit COICOP 2018 as the published expenditure "
             "share of outpatient, inpatient and other health services, all "
             "five education levels, recreational and cultural services, "
             "package holidays, passenger transport services, food and "
             "beverage serving, accommodation, personal care, social "
             "protection and in-home maintenance within each division",
    "energy": "electricity, gas and other household fuels, and motor fuel",
}

TIME_CLUSTER = "year"

DIRECTION = {"H1": "negative", "H2": "positive", "H3": "negative", "H4": "negative"}
STATEMENT = {
    "H1": "The shock reduces real household consumption during its impact phase.",
    "H2": "Necessity categories gain budget share during the shock.",
    "H3": "Categories exposed to that particular shock contract more than others.",
    "H4": "Consumption stays below its no-shock counterfactual after the impact.",
}

SESOI_RULE = ("one standard deviation of the outcome's own variation in "
              "non-shock quarters, computed from non-shock data only")
SESOI_UNITS = {
    "H1": "pp of four-quarter real category growth",
    "H2": "log points of CLR share",
    "H3": "pp of four-quarter real category growth per unit of exposure",
    "H4": "log % of the consumption level",
}

ALPHA = 0.05
FAMILY = "shock"
PRIMARY_ADJUSTMENT = "romano_wolf"

H4_HORIZONS = (8, 12, 16)

SAMPLES = {
    "full_1995": {
        "start": "1995Q1", "end": "2024Q4",
        "label": "1995Q1-2024Q4, the full Destatis consumption-by-purpose record"},
    "matched_1999": {
        "start": "1999Q2", "end": "2024Q4",
        "label": "1999Q2-2024Q4, the cross-country span (Germany's record ends "
                 "one quarter earlier than India's)"},
}
PRIMARY_SAMPLE = "full_1995"
COMPARISON_SAMPLE = "matched_1999"


def impact_periods(shock: str) -> list[str]:
    import pandas as pd
    lo, hi = SHOCKS[shock]["impact_quarters"]
    return [str(q) for q in pd.period_range(lo, hi, freq="Q")]


def window_periods(shock: str) -> list[str]:
    import pandas as pd
    lo, hi = SHOCKS[shock]["window"]
    return [str(q) for q in pd.period_range(lo, hi, freq="Q")]


EXTRA_DISTURBANCES = {
    "vat_2007_rise": ("2006Q4", "2007Q1"),
    "flood_2013": ("2013Q2", "2013Q3"),
    "oil_collapse_2015": ("2014Q4", "2015Q2"),
    "refugee_influx_2015": ("2015Q3", "2016Q2"),
}


def disturbed_periods() -> set:
    import pandas as pd
    out = set()
    for lo, hi in EXTRA_DISTURBANCES.values():
        out |= {str(q) for q in pd.period_range(lo, hi, freq="Q")}
    return out
