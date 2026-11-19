from __future__ import annotations

import pandas as pd

COUNTRIES = {
    "USA": {"panel": "countries/USA/data/final/usa_quarterly_category_panel.csv",
            "source": "BEA NIPA 2.3/2.4", "observed": "quarterly"},
    "UK": {"panel": "countries/UK/data/final/uk_quarterly_category_panel.csv",
           "source": "ONS Consumer Trends", "observed": "quarterly"},
    "Japan": {"panel": "countries/Japan/data/final/japan_quarterly_category_panel.csv",
              "source": "ESRI SNA table 12", "observed": "quarterly"},
    "Canada": {"panel": "countries/Canada/data/final/canada_quarterly_category_panel.csv",
               "source": "Statistics Canada 36-10-0124-01", "observed": "quarterly"},
    "Germany": {"panel": "countries/Germany/data/final/germany_quarterly_category_panel.csv",
                "source": "Destatis via Eurostat", "observed": "annual, benchmarked"},
    "France": {"panel": "countries/France/data/final/france_quarterly_category_panel.csv",
               "source": "INSEE via Eurostat", "observed": "annual, benchmarked"},
    "Italy": {"panel": "countries/Italy/data/final/italy_quarterly_category_panel.csv",
              "source": "ISTAT via Eurostat", "observed": "annual, benchmarked"},
    "India": {"panel": "countries/India/data/final/india_quarterly_category_panel.csv",
              "source": "MoSPI", "observed": "annual, benchmarked"},
}

COMMON_START, COMMON_END = "1999Q2", "2021Q4"

FULL_START = {"USA": "1959Q1", "UK": "1985Q1", "Japan": "1994Q1", "Canada": "1981Q1",
              "Germany": "1995Q1", "France": "1980Q1", "Italy": "1996Q1", "India": "1999Q2"}
TIME_CLUSTER = {"USA": "period", "UK": "period", "Japan": "period", "Canada": "period",
                "Germany": "year", "France": "year", "Italy": "year", "India": "fy"}

SHOCKS = {
    "dotcom": {"label": "Dot-com", "impact_quarters": ("2001Q1", "2001Q2"),
               "window": ("2000Q1", "2002Q4"), "resilience_year": 1999},
    "gfc": {"label": "Global financial crisis", "impact_quarters": ("2008Q4", "2009Q1"),
            "window": ("2008Q1", "2009Q4"), "resilience_year": 2007},
    "covid": {"label": "COVID-19", "impact_quarters": ("2020Q2", "2020Q3"),
              "window": ("2020Q1", "2021Q4"), "resilience_year": 2019},
}

NECESSITY = ["food_nonalc", "housing_utilities", "health"]
EXPOSURE_MARGIN = {"dotcom": "discretionary_intensity", "gfc": "durable_intensity",
                   "covid": "contact_intensity"}
H4_HORIZONS = (8, 12, 16)

DIRECTION = {"H1": "negative", "H2": "positive", "H3": "negative",
             "H4": "negative", "H5": "positive",
             "H2_volume": "positive", "H4_local": "negative"}
STATEMENT = {
    "H1": "The shock reduces real household consumption during its impact phase.",
    "H2": "Necessity categories gain budget share during the shock.",
    "H3": "Categories exposed to that particular shock contract more than others.",
    "H4": "Consumption stays below its no-shock counterfactual after the impact.",
    "H5": "Greater pre-shock resilience attenuates the negative effect of the "
          "shock on consumption.",
    "H2_volume": "H2 on real-volume budget shares, so a price rise cannot raise a "
                 "necessity share on its own (robustness).",
    "H4_local": "H4 with each division's drift taken from quiet quarters in the ten "
                "years before the anchor - a local trend (robustness).",
}
ALPHA = 0.05
SPECS = ("core", "clean", "mechanism", "extended")
HEADLINE_SPEC = "clean"
PLACEBO_STRIDE = 4
BUFFER = 2
SEED = 20260917
REPS = 999

RESILIENCE = {
    "components": {
        "gov_debt": {"label": "general government gross debt, % of GDP",
                     "source": "IMF World Economic Outlook via FRED (GGGDTA{ISO2}A188N)",
                     "sign": -1},
        "hh_credit_change": {"label": "5-year change in household credit, pp of GDP",
                             "source": "BIS total credit statistics via FRED (Q{ISO2}HAM770A)",
                             "sign": -1},
        "inflation_gap": {"label": "absolute deviation of CPI inflation from 2%",
                          "source": "World Bank WDI FP.CPI.TOTL.ZG", "sign": -1},
        "unemployment": {"label": "unemployment rate, %",
                         "source": "World Bank WDI SL.UEM.TOTL.ZS (ILO modelled)",
                         "sign": -1},
    },
    "min_components": 3,
}


def periods(lo: str, hi: str) -> list[str]:
    return [str(q) for q in pd.period_range(lo, hi, freq="Q")]


def impact_periods(shock: str) -> list[str]:
    return periods(*SHOCKS[shock]["impact_quarters"])


def window_periods(shock: str) -> list[str]:
    return periods(*SHOCKS[shock]["window"])
