from __future__ import annotations

import datetime as dt
import pathlib
import re
import sys
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import study_protocol as P

_TTF = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                   ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-M", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(_TTF / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

NAVY, GREY = colors.HexColor("#1f3b5c"), colors.HexColor("#666666")
ST = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=21, leading=26, textColor=NAVY,
                            spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=12.5, leading=17, textColor=GREY),
    "H1": ParagraphStyle("H1", fontName="DV-B", fontSize=16, leading=20, textColor=NAVY,
                         spaceBefore=4, spaceAfter=8),
    "H2": ParagraphStyle("H2", fontName="DV-B", fontSize=12, leading=15, textColor=NAVY,
                         spaceBefore=10, spaceAfter=5),
    "H3": ParagraphStyle("H3", fontName="DV-B", fontSize=9.6, leading=12.5, textColor=colors.black,
                         spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.8, leading=12.4, spaceAfter=4),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.4, leading=9.6, textColor=GREY,
                            spaceAfter=3),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=8.8, leading=12.2, leftIndent=12,
                             bulletIndent=2, spaceAfter=2.5),
    "eq": ParagraphStyle("eq", fontName="DV-M", fontSize=7.9, leading=11.2, leftIndent=8,
                         spaceAfter=2, textColor=colors.HexColor("#111111")),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.3, leading=9.1),
    "cellc": ParagraphStyle("cellc", fontName="DV", fontSize=7.3, leading=9.1, alignment=TA_CENTER),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=7.4, leading=9.2, textColor=colors.white),
    "headc": ParagraphStyle("headc", fontName="DV-B", fontSize=7.4, leading=9.2,
                            textColor=colors.white, alignment=TA_CENTER),
    "box": ParagraphStyle("box", fontName="DV", fontSize=8.6, leading=12, spaceAfter=2),
}
W = A4[0] - 2 * 1.7 * cm

COUNTRY = {"USA": "United States", "UK": "United Kingdom", "Japan": "Japan", "Canada": "Canada",
           "Germany": "Germany", "France": "France", "Italy": "Italy", "India": "India"}
ORDER = ["USA", "UK", "Japan", "Canada", "Germany", "France", "Italy", "India"]
SHOCK = {"dotcom": "Dot-com bust", "gfc": "Global financial crisis", "covid": "COVID-19",
         "combined": "All three shocks"}
SHOCK_SHORT = {"dotcom": "Dot-com", "gfc": "GFC", "covid": "COVID-19", "combined": "All three"}
DIV = {"food_nonalc": "Food and non-alcoholic beverages", "alcohol_tobacco": "Alcohol and tobacco",
       "clothing_footwear": "Clothing and footwear",
       "housing_utilities": "Housing, water, electricity and gas",
       "household_goods": "Furnishings and household equipment", "health": "Health",
       "transport": "Transport", "communication": "Communication",
       "recreation_culture": "Recreation and culture", "education": "Education",
       "restaurants_hotels": "Restaurants and hotels",
       "miscellaneous": "Miscellaneous goods and services"}
TYPE = {"C": "Confounder", "S": "Structural regime", "M": "Policy response", "X": "Second shock"}
SPEC = {"core": "Without effects", "clean": "With effects (headline)",
        "mechanism": "With effects + policy responses and oil",
        "extended": "With effects + optional events"}
TCLUSTER = {"period": "quarter", "year": "calendar year", "fy": "Indian financial year (April-March)"}
SYM = {"supported": "✓", "rejected_opposite_sign": "✗", "negligible_effect": "≈0",
       "inconclusive": "?", "not_estimable": "n/a"}
WORD = {"supported": "supported", "rejected_opposite_sign": "rejected (opposite sign)",
        "negligible_effect": "no effect", "inconclusive": "undecided", "not_estimable": "not estimable"}
FG = {"supported": "#17773a", "rejected_opposite_sign": "#b42318", "negligible_effect": "#1d5aa6",
      "inconclusive": "#555555", "not_estimable": "#888888"}
BG = {"supported": "#e3f3e8", "rejected_opposite_sign": "#fbe6e4", "negligible_effect": "#e5eef9",
      "inconclusive": "#f3f3f3", "not_estimable": "#ffffff"}
SETTLED = ("supported", "rejected_opposite_sign", "negligible_effect")
MAIN = ["H1", "H2", "H3", "H4"]
SOURCE_LONG = {
    "USA": "Bureau of Economic Analysis, National Income and Product Accounts (personal consumption "
           "expenditures by type, tables 2.3 and 2.4), mapped to the twelve COICOP divisions",
    "UK": "Office for National Statistics, Consumer Trends (household final consumption by COICOP "
          "division)",
    "Japan": "Cabinet Office (ESRI), quarterly national accounts, household final consumption by purpose",
    "Canada": "Statistics Canada, table 36-10-0124-01, household final consumption expenditure by "
              "purpose",
    "Germany": "Destatis national accounts, household consumption by purpose, via Eurostat",
    "France": "INSEE national accounts, household consumption by purpose, via Eurostat",
    "Italy": "ISTAT national accounts, household consumption by purpose, via Eurostat",
    "India": "Ministry of Statistics and Programme Implementation (MoSPI), National Accounts "
             "Statistics, private final consumption expenditure by purpose",
}
SOURCE_NOTE = {
    "France": "France is built with exactly the same code as Germany (the code reproduces "
              "Germany's own panel bit for bit).",
    "Italy": "Italy is built with exactly the same code as Germany (the code reproduces "
             "Germany's own panel bit for bit).",
    "India": "India's division record starts in 1999Q2, so its full record and the matched "
             "timeline are the same and it has one timeline.",
}

WHY_EFFECTS = {
    ("USA", "full", "dotcom", "H4"):
        "The declared disturbances (1973-75 and 1979-80 oil shocks, 1980 credit controls, "
        "1981-82 recession, September 11) no longer count as normal quarters, so average normal "
        "growth rises (0.88% to 0.94% a quarter) and the shortfall grows (-2.32 to -3.04); now "
        "significant.",
    ("UK", "matched", "dotcom", "H2"):
        "The 2001 foot-and-mouth outbreak (restaurants and hotels, recreation, transport) is now "
        "controlled in the impact quarters. Less information remains, the interval widens "
        "(standard error 0.012 to 0.021), and 'no effect' can no longer be proven.",
    ("Japan", "full", "dotcom", "H1"):
        "With Japan's declared events controlled (tax rises of 1997, 2014 and 2019, the 2001 "
        "appliance-recycling law, the 2010 subsidy expiry, the 2011 earthquake) the wrong-sign "
        "result (+1.27) is no longer significant (+1.18, standard error 0.63 to 0.92): it depended "
        "on those events.",
    ("Japan", "full", "dotcom", "H3"):
        "With Japan's events controlled, the estimate moves from -0.23 to -0.54 and 'no effect' "
        "is no longer proven.",
    ("Japan", "full", "dotcom", "H4"):
        "The tax-rise quarters, the 2010 subsidy expiry and the 2011 earthquake no longer count "
        "as normal quarters, so average normal growth rises (0.21% to 0.35% a quarter); the "
        "wrong-sign result (+2.53) shrinks to +0.80 and is no longer significant.",
    ("Japan", "full", "gfc", "H4"):
        "Same higher normal-growth path, and the recovery check now stops before the March 2011 "
        "earthquake (8 and 12 quarters only): shortfall -2.59 to -4.02, now significant.",
    ("Canada", "full", "dotcom", "H1"):
        "With Canada's event dummies added, the estimate moves from +0.12 to -0.30 and the "
        "interval widens, so 'no effect' is no longer proven.",
    ("Germany", "full", "dotcom", "H1"):
        "With Germany's event dummies added (the 2001 tax reform is set aside because it falls in "
        "the impact quarters) the estimate moves from -0.11 to -0.24 and the interval widens; "
        "'no effect' is no longer proven.",
    ("Germany", "full", "gfc", "H4"):
        "The declared disturbances (2001 tax reform, euro changeover, 2002 floods, 2007 VAT rise, "
        "2013 floods, 2014-15 oil collapse, 2015-16 refugee influx) no longer count as normal "
        "quarters. Average normal growth eases slightly (0.33% to 0.31% a quarter), the shortfall "
        "shrinks slightly (-2.08 to -1.87) and the p-value moves from 0.033 to 0.057.",
    ("France", "full", "covid", "H4"):
        "The December 2019 pension strike is a declared disturbance, so the recovery clock starts "
        "in 2019Q3 and the 8-quarter check lands in 2021Q3, when restaurants and hotels (-19%) and "
        "transport (-15%) were still far below normal. The average shortfall is similar (-2.64 to "
        "-2.98) but divisions diverge strongly, so the standard error triples (0.86 to 2.78).",
    ("France", "matched", "covid", "H4"):
        "Same reason as the full record: the recovery clock starts in 2019Q3, the check lands in "
        "2021Q3, and the standard error triples (0.85 to 2.73) while the shortfall is similar "
        "(-1.92 to -2.10).",
    ("Italy", "full", "dotcom", "H4"):
        "The euro changeover, the 2002 Argentine default and the 2011-13 sovereign-debt crisis no "
        "longer count as normal quarters, so average normal growth rises (0.20% to 0.32% a "
        "quarter); shortfall -0.31 to -1.81, now significant.",
    ("Japan", "full", "covid", "H4"):
        "The October 2019 consumption-tax rise is a declared disturbance, so the recovery clock "
        "starts in 2019Q2 instead of the tax-distorted 2019Q4; the shortfall grows from -1.63 to "
        "-7.16. The clustered test gives p = 0.066; the placebo test supports it (p = 0.027).",
    ("Japan", "matched", "covid", "H4"):
        "Same reason as the full record: the recovery clock starts in 2019Q2, before the October "
        "2019 tax rise; the shortfall grows from -1.11 to -6.70 (clustered test p = 0.083; placebo "
        "test p = 0.036).",
    ("Japan", "matched", "dotcom", "H4"):
        "The tax-rise quarters, the 2010 subsidy expiry and the 2011 earthquake no longer count as "
        "normal quarters, so the normal-growth path is steeper; the wrong-sign result (+3.25) shrinks "
        "to +1.35 and is no longer significant.",
    ("India", "matched", "gfc", "H4"):
        "The 2001 earthquake, the 2002 and 2009 droughts, demonetisation, the GST launch and the "
        "2018-19 shadow-bank (NBFC) stress no longer count as normal quarters, so average normal "
        "growth rises (1.72% to 1.84% a quarter); shortfall -2.30 to -3.71, now significant.",
}
WHY_SPEC = {
    ("mechanism", "UK", "gfc", "H1"):
        "The UK VAT cut (2008Q4-2009Q4) and the real oil-price collapse fall in the same quarters "
        "as the impact, so these controls absorb the shock itself. Not evidence against H1.",
    ("mechanism", "France", "gfc", "H1"):
        "The real oil-price collapse coincides with the impact quarters and absorbs the shock "
        "itself. Not evidence against H1.",
    ("mechanism", "Italy", "gfc", "H1"):
        "The wage-support scheme (Cassa Integrazione, from 2009Q1) and the real oil-price collapse "
        "coincide with the impact quarters and absorb the shock itself. Not evidence against H1.",
    ("mechanism", "Canada", "dotcom", "H1"):
        "With the oil-price control, the small estimate is now proven to be below the threshold.",
    ("mechanism", "Canada", "dotcom", "H3"):
        "With the oil-price x energy-intensity control, the wrong-sign result is no longer "
        "significant.",
    ("mechanism", "France", "dotcom", "H1"):
        "The 35-hour-week regime (2000-02) overlaps the impact quarters, so it enters only here; "
        "with it the wrong-sign result disappears and the effect is shown to be small.",
    ("extended", "Germany", "covid", "H2"):
        "Adding the optional 2018 Rhine low-water event moves the p-value from 0.048 to 0.051: "
        "the headline result is borderline.",
}


PROBLEMS = [
    "Russia cannot meet the common timeline (its trusted spending-by-purpose accounts cover 2011-2019 only); it is excluded with evidence rather than spliced from a survey of a different concept.",
    "IMF DataMapper is blocked from the build network; the same IMF World Economic Outlook and BIS series were taken from FRED, and inflation and unemployment from the World Bank.",
    "BIS household credit for India includes household-enterprise credit (45% of GDP in 2007); H5 therefore uses the five-year change in household credit, which does not depend on the level definition.",
    "Two resilience components do not reach back far enough for three country-shocks (US government debt 1999; India household credit 1999 and 2007); the index uses the remaining three, as fixed in advance.",
    "Germany's Eurostat cache paths did not contain the country code, so cloning the pipeline for France would have silently reused German data; the country is now part of every cache and output path, and the code reproduces Germany's panel bit for bit.",
    "H1 and H5 include every category's own trend. An earlier version dropped one category's trend, which, without period effects, forces that category's trend to zero; corrected before any result in this document was produced.",
    "Whole-economy events that fall in a shock's impact quarters cannot be separated from that shock by a dummy; a uniform rule moves them to the policy specification for that shock's tests: Germany's 2001 tax reform, India's 2001 Gujarat earthquake, France's 35-hour-week regime and Japan's bad-loan and deflation regime (dot-com), and Canada's 2020 oil collapse (COVID-19).",
    "With eight countries, country-clustered inference (H4, H5) uses the wild cluster bootstrap with Webb six-point weights; the fast exact bootstrap is verified draw for draw against a reference implementation.",
    "The first combined-pool design imposed one common coefficient across the three shocks. Their effects differ by an order of magnitude and equality is rejected at p < 0.001 for H1-H3, so the common slope inflated the standard error (combined H3 standard error 9.2 against 1-4 per shock). The combined result now estimates shock-specific effects and reports their equal-weight mean; the common-coefficient form is reported as robustness. This change was made after the first results were seen, and is disclosed as such.",
    "The first build read the country panels from home-directory paths, so it would not have run on another computer. All eight panels now ship in countries/ with SHA-256 fingerprints in countries/MANIFEST.csv, checked by verify.py.",
    "An audit of the event list against the design memo found gaps: structural regimes (France 35-hour week, Japan bad loans and deflation, UK post-referendum, Italy 2018-19), three policy responses (France 2009 household support, Japan 2009 cash handout, Canada wage subsidy), the oil-price control, the H2 relative-price concern and the H4 local-trend suggestion. All are implemented: regimes as type S, oil and policy responses in the policy specification, volume-share H2 and local-trend H4 as robustness checks, and the memo's optional events in a separate specification.",
    "Full records reach back before the memo's events for France (1980), Italy (1996) and Canada (1981). Pre-1999 events were screened with a fixed rule (documented, outside the shock windows, at least one normal-quarter standard deviation): six qualify (Canada's 1981-82 and 1990-91 recessions, France's 1983 austerity turn and 1992-93 recession, Italy's 1997 and 1998 car-scrappage schemes) and six are rejected (documentation/pre1999_events_rejected.csv). Italy's 1997 eurotax was rejected after testing on spending excluding transport: its apparent deviation was the car-scrappage boom.",
    "With energy dropped as a focal shock, every timeline ends at 2021Q4, so the 2022 war and energy regime is neither counted as normal nor read as COVID-19 recovery.",
    "H4 failed instead of returning 'not estimable' when no category had a usable local trend at an early placebo date; fixed.",
    "Final audit: the placebo version of H4 started from normal quarters but its 8-16-quarter path could run into another shock's window (for example a 2004 start ending inside the financial crisis), so some placebo gaps contained real shock effects. Placebo recovery paths now must not cross any shock window. The change affects only placebo p-values for H4; Appendix C reports its effect.",
    "Final audit: the verdict rule fixed before estimation counts a result as supported if either the clustered test or the placebo test is significant. Appendix C reports how many verdicts rest on the placebo test alone.",
    "Final audit: the package said Python 3.10 was enough, but the pinned scipy needs 3.12, and the Statistics Canada raw file had been left out of the package; both fixed and tested on a fresh Python 3.12 installation with the network blocked.",
]


def esc(s) -> str:
    return escape(str(s))


def minus(s: str) -> str:
    return re.sub(r"(?<![\w.])-(?=\.?\d)", "−", s)


def num(x, hyp="H1", sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    d = 3 if str(hyp).startswith("H2") else 2
    return minus(f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def para(text, style="body") -> Paragraph:
    return Paragraph(text, ST[style])


def bullets(items, style="bullet") -> list:
    return [Paragraph(t, ST[style], bulletText="•") for t in items]


def quarters(a, b) -> str:
    return a if a == b else f"{a}–{b}"


def grid(rows, widths, header_rows=1, extra=None, zebra=True, font=7.3) -> Table:
    data = []
    for i, r in enumerate(rows):
        out = []
        for v in r:
            if isinstance(v, str):
                out.append(Paragraph(v, ST["head" if i < header_rows else "cell"]))
            else:
                out.append(v)
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [("BACKGROUND", (0, 0), (-1, header_rows - 1), NAVY),
             ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cc")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(style + (extra or [])))
    return t


def vcell(dec, est, se, hyp, show_se=True):
    dec = dec if isinstance(dec, str) else "not_estimable"
    if dec == "not_estimable" or est is None or not np.isfinite(est):
        return Paragraph("n/a", ST["cellc"]), BG["not_estimable"]
    txt = f'<font color="{FG[dec]}"><b>{SYM[dec]}</b></font>&nbsp;{num(est, hyp)}'
    if show_se and se is not None and np.isfinite(se):
        txt += f'<br/><font size="6.4" color="#666666">({num(se, hyp, sign=False)})</font>'
    return Paragraph(txt, ST["cellc"]), BG[dec]


def legend() -> Paragraph:
    return para('<font color="#17773a"><b>✓</b></font> supported &nbsp; '
                '<font color="#b42318"><b>✗</b></font> rejected (significant, opposite sign) &nbsp; '
                '<font color="#1d5aa6"><b>≈0</b></font> no effect (proven smaller than the threshold) &nbsp; '
                '<b>?</b> undecided &nbsp;·&nbsp; each cell: estimate (standard error)', "small")


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                         topMargin=1.7 * cm, bottomMargin=1.6 * cm, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("DV", 7)
        canv.setFillColor(GREY)
        if doc.page > 1:
            canv.drawString(self.leftMargin, A4[1] - 1.05 * cm,
                            "Consumption Reallocation and Macroeconomic Resilience - Results")
            canv.setStrokeColor(colors.HexColor("#c9d2db"))
            canv.line(self.leftMargin, A4[1] - 1.15 * cm, A4[0] - self.rightMargin, A4[1] - 1.15 * cm)
        canv.drawRightString(A4[0] - self.rightMargin, 0.95 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("H1", "H2"):
            level = 0 if f.style.name == "H1" else 1
            key = f"h{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, f.getPlainText(), self.page, key))


class Data:
    def __init__(self, root: pathlib.Path):
        t = root / "results/tables"
        self.root = root
        self.ind = pd.read_csv(t / "individual_results.csv")
        self.counts = pd.read_csv(t / "individual_counts_by_spec.csv")
        self.changes = pd.read_csv(t / "individual_core_vs_clean_changes.csv")
        self.comb = pd.read_csv(t / "individual_results_combined.csv")
        self.comb_counts = pd.read_csv(t / "individual_combined_counts_by_spec.csv")
        self.h4rule = pd.read_csv(root / "results/diagnostics/h4_placebo_rule_comparison.csv")
        self.pool = pd.read_csv(t / "pooled_results.csv")
        self.h5c = pd.read_csv(t / "h5_by_component.csv")
        self.meta = pd.read_csv(t / "h5_meta_regression.csv")
        self.res = pd.read_csv(root / "data/resilience/resilience_index.csv")
        self.ev = pd.read_csv(root / "data/country_events.csv")
        self.rej = pd.read_csv(root / "documentation/pre1999_events_rejected.csv")
        self.sub = pd.read_csv(t / "robustness_quarterly_observed.csv")
        dg = root / "results/diagnostics"
        self.cov = pd.read_csv(dg / "data_coverage.csv")
        self.expc = pd.read_csv(dg / "exposure_concentration.csv")
        self.dictionary = pd.read_csv(root / "data/data_dictionary.csv")
        self.pooled_data = pd.read_csv(root / "data/all_countries_pooled_analysis_data.csv",
                                       usecols=["country", "period"])
        self.country_rows = {c: len(pd.read_csv(root / f"countries/{c}/{c.lower()}_analysis_data.csv",
                                                usecols=["period"])) for c in ORDER}
        rep = dg / "verification_report_full.txt"
        self.report = rep.read_text(encoding="utf-8").splitlines() if rep.exists() else []
        self.ev["periods"] = [set(P.periods(a, b)) for a, b in zip(self.ev.start, self.ev.end)]
        self.check()
        self.check_prose()

    def cell(self, country, timeline, spec, pool, hyp):
        r = self.ind[(self.ind.country == country) & (self.ind.timeline == timeline)
                     & (self.ind.spec == spec) & (self.ind.pool == pool) & (self.ind.hypothesis == hyp)]
        return r.iloc[0] if len(r) else None

    def ccell(self, country, timeline, spec, hyp):
        r = self.comb[(self.comb.country == country) & (self.comb.timeline == timeline)
                      & (self.comb.spec == spec) & (self.comb.hypothesis == hyp)]
        return r.iloc[0] if len(r) else None

    def timelines(self, country):
        return [t for t in ("full", "matched") if ((self.ind.country == country)
                                                    & (self.ind.timeline == t)).any()]

    def check_prose(self):
        c = self.ind[(self.ind.spec == "clean") & self.ind.hypothesis.isin(MAIN)]
        sup = c.assign(s=c.decision_headline.eq("supported")).pivot_table(
            index="pool", columns="hypothesis", values="s", aggfunc="sum")
        want = {("covid", "H1"): 15, ("covid", "H2"): 15, ("covid", "H3"): 15, ("covid", "H4"): 9,
                ("gfc", "H1"): 12, ("gfc", "H2"): 6, ("gfc", "H3"): 10, ("gfc", "H4"): 12,
                ("dotcom", "H1"): 0}
        bad = {k: int(sup.loc[k]) for k, v in want.items() if int(sup.loc[k]) != v}
        gfc_h1 = set(map(tuple, c[(c.pool == "gfc") & (c.hypothesis == "H1")
                                  & (c.decision_headline != "supported")][["country", "timeline"]].values))
        covid_h4 = set(c[(c.pool == "covid") & (c.hypothesis == "H4")
                         & (c.decision_headline == "supported")].country)
        pc = self.pool[self.pool.pool.isin(list(P.SHOCKS) + ["combined"])
                       & self.pool.hypothesis.isin(MAIN + ["H5"])]
        pcl = pc[pc.spec == "clean"]
        same = (pc[pc.spec == "core"].set_index(["pool", "hypothesis"]).decision_headline
                == pcl.set_index(["pool", "hypothesis"]).decision_headline).all()
        h5 = self.h5c[(self.h5c.component == "hh_credit_change") & (self.h5c.pool == "gfc")]
        checks = [not bad, gfc_h1 == {("Germany", "full"), ("Germany", "matched"), ("India", "matched")},
                  covid_h4 == {"UK", "Germany", "Italy", "Japan", "Canada"},
                  int((pcl.decision_headline == "supported").sum()) == 11,
                  pcl[pcl.decision_headline == "negligible_effect"][["pool", "hypothesis"]].values.tolist()
                  == [["dotcom", "H5"]],
                  len(self.changes) == 16, bool(same),
                  round(float(h5.p_bootstrap_positive.iloc[0]), 3) == 0.063,
                  self.cell("USA", "matched", "clean", "covid", "H4").decision_headline
                  == "rejected_opposite_sign"]
        s = self.sub.set_index(["subset", "pool", "hypothesis"])
        full_sup = [(r.pool, r.hypothesis) for r in pcl[pcl.hypothesis.isin(MAIN)
                                                       & (pcl.decision_headline == "supported")].itertuples()]
        hold = {k: sum(s.loc[(k, a, b)].decision_headline == "supported" for a, b in full_sup)
                for k in ("quarterly_observed", "benchmarked")}
        checks += [len(full_sup) == 11, hold == {"quarterly_observed": 10, "benchmarked": 10},
                   s.loc[("quarterly_observed", "covid", "H4")].decision_headline == "inconclusive",
                   round(float(s.loc[("quarterly_observed", "covid", "H4")].p_parametric), 3) == 0.079,
                   s.loc[("benchmarked", "gfc", "H3")].decision_headline == "inconclusive",
                   s.loc[("quarterly_observed", "gfc", "H2")].decision_headline == "supported",
                   s.loc[("benchmarked", "gfc", "H2")].decision_headline == "inconclusive",
                   s.loc[("quarterly_observed", "dotcom", "H4")].decision_headline == "rejected_opposite_sign",
                   bool((self.sub[(self.sub.decision_headline == "supported")
                                  & (self.sub.subset == "quarterly_observed")].p_holm < 0.05).all())]
        if not all(checks):
            raise SystemExit(f"results changed; update the prose of results_pdf.py ({checks}, {bad})")

    def check(self):
        ch = {(r.country, r.timeline, r.pool, r.hypothesis) for r in self.changes.itertuples()}
        missing = ch - set(WHY_EFFECTS)
        stale = set(WHY_EFFECTS) - ch
        if missing or stale:
            raise SystemExit(f"explanations out of date: {missing} {stale}")
        for (spec, country, pool, hyp) in WHY_SPEC:
            for tl in self.timelines(country):
                a, b = self.cell(country, tl, "clean", pool, hyp), self.cell(country, tl, spec, pool, hyp)
                if a is not None and a.decision_headline != b.decision_headline:
                    break
            else:
                raise SystemExit(f"specification explanation out of date: {spec} {country} {pool} {hyp}")


def results_grid(D: Data, country: str, spec: str) -> Table:
    head = ["Timeline", "Shock"] + [f"{h}" for h in MAIN]
    rows, extra = [head], []
    for tl in D.timelines(country):
        first = D.cell(country, tl, spec, "dotcom", "H1")
        label = (f"<b>{'Full record' if tl == 'full' else 'Matched'}</b><br/>"
                 f"{first.window.replace('-', '–')}<br/>({int(first.quarters)} quarters)")
        if country == "India":
            label = (f"<b>Full record = matched</b><br/>{first.window.replace('-', '–')}<br/>"
                     f"({int(first.quarters)} quarters)")
        start = len(rows)
        for s in P.SHOCKS:
            row = [label if s == "dotcom" else "", SHOCK[s]]
            for j, h in enumerate(MAIN):
                r = D.cell(country, tl, spec, s, h)
                p, bg = vcell(r.decision_headline, r.estimate, r.std_error, h)
                row.append(p)
                extra.append(("BACKGROUND", (2 + j, len(rows)), (2 + j, len(rows)), colors.HexColor(bg)))
            rows.append(row)
        extra.append(("SPAN", (0, start), (0, start + 2)))
        extra.append(("VALIGN", (0, start), (0, start + 2), "MIDDLE"))
    hdr = [("ALIGN", (2, 0), (-1, 0), "CENTER")]
    return grid(rows, [3.3 * cm, 3.1 * cm] + [(W - 6.4 * cm) / 4] * 4, extra=hdr + extra, zebra=False)


def tally(D: Data, country: str, spec: str) -> str:
    parts = []
    for tl in D.timelines(country):
        b = D.ind[(D.ind.country == country) & (D.ind.timeline == tl) & (D.ind.spec == spec)
                  & D.ind.hypothesis.isin(MAIN)]
        sup = int((b.decision_headline == "supported").sum())
        rej = int((b.decision_headline == "rejected_opposite_sign").sum())
        neg = int((b.decision_headline == "negligible_effect").sum())
        und = int((b.decision_headline == "inconclusive").sum())
        name = "full record" if tl == "full" else "matched"
        if country == "India":
            name = "the single timeline"
        parts.append(f"<b>{name}</b>: {sup} supported, {rej} rejected, {neg} no effect, "
                     f"{und} undecided (of 12)")
    return "; ".join(parts) + "."


def h4_note(D: Data, country: str, spec: str) -> str:
    out = []
    for tl in D.timelines(country):
        bits = []
        for s in P.SHOCKS:
            r = D.cell(country, tl, spec, s, "H4")
            if r is None or not isinstance(r.h4_anchor, str):
                bits.append(f"{SHOCK_SHORT[s]}: not estimable")
                continue
            hz = [int(h) for h in str(r.h4_horizons).split(",")]
            default = str(pd.Period(P.SHOCKS[s]["window"][0], freq="Q") - 1)
            why = []
            if r.h4_anchor != default:
                blocked = [e for e in D.ev[(D.ev.country == country) & (D.ev.tier == "primary")
                                           & (D.ev.scope == "aggregate")
                                           & D.ev.type.isin(["C", "X"])].itertuples()
                           if default in e.periods]
                name = blocked[0].description.split(" (")[0] if blocked else "a declared disturbance"
                why.append(f"anchor moved back past: {esc(name)}")
            missing = [h for h in P.H4_HORIZONS if h not in hz]
            if missing:
                end = pd.Period(P.COMMON_END, freq="Q")
                a = pd.Period(r.h4_anchor, freq="Q")
                if all(a + h > end for h in missing):
                    why.append("later horizons fall after 2021Q4")
                else:
                    x = D.ev[(D.ev.country == country) & (D.ev.type == "X") & (D.ev.context == s)]
                    if spec != "core" and len(x):
                        why.append(f"stops before the second shock: {esc(x.iloc[0].description.split(' (')[0])}")
                    else:
                        why.append("later horizons fall after 2021Q4")
            txt = f"{SHOCK_SHORT[s]}: anchor {r.h4_anchor}, horizons {', '.join(map(str, hz))}"
            if why:
                txt += f" ({'; '.join(why)})"
            bits.append(txt)
        name = "Full record" if tl == "full" else "Matched"
        if country == "India":
            name = "Timeline"
        out.append(f"<b>{name}</b> - " + " · ".join(bits))
    return "<br/>".join(out)


def event_rows(D: Data, country: str) -> list:
    ev = D.ev[D.ev.country == country].copy()
    ev = ev.sort_values(["tier", "start"], key=lambda c: c.map({"primary": 0, "secondary": 1})
                        if c.name == "tier" else c)
    rows = []
    for e in ev.itertuples():
        overl = [s for s in P.SHOCKS if e.periods & set(P.impact_periods(s))]
        if e.tier == "secondary":
            use = "Optional-events specification only"
        elif e.type == "M":
            use = "Policy specification only (the headline measures the total effect)"
        elif e.scope == "aggregate":
            use = "Headline and all with-effects specifications"
            if overl:
                use = (f"Headline, except in the {', '.join(SHOCK_SHORT[s] for s in overl)} tests "
                       "(falls in its impact quarters; moved to the policy specification there)")
        else:
            use = "Headline and all with-effects specifications (named divisions only)"
        if e.type in ("C", "X") and e.tier == "primary" and e.scope == "aggregate":
            use += "; its quarters are not treated as normal"
        if e.type == "S":
            use += "; regime dummy only, quarters stay normal"
        if e.type == "X":
            use += f"; ends the {SHOCK_SHORT.get(e.context, e.context)} recovery check before {e.start}"
        if e.end < P.COMMON_START:
            use += ". Full record only (before 1999Q2)"
        scope = ("Whole economy" if e.scope == "aggregate"
                 else "; ".join(DIV[c] for c in str(e.categories).split(";")))
        title = (f"<b>{esc(e.description)}</b><br/><font size='6.3' color='#666666'>"
                 f"Source: {esc(e.source)}</font>")
        typ = TYPE[e.type] + (" (optional)" if e.tier == "secondary" else "")
        rows.append([title, quarters(e.start, e.end), typ, esc(scope), esc(use)])
    return rows


def exposure_table(D: Data, country: str) -> Table:
    d = pd.read_csv(D.root / P.COUNTRIES[country]["panel"])
    d = d[(d.period >= P.COMMON_START) & (d.period <= P.COMMON_END)]
    g = d.groupby("category").agg(weight=("weight", "mean"), necessity=("necessity", "max"),
                                  dotx=("exposed_dotcom", "mean"), gfc=("exposed_gfc", "mean"),
                                  covid=("exposed_covid", "mean"))
    g = g.reindex(list(DIV))
    rows = [["Division", "Share of spending, 1999-2021 avg (%)", "Necessity (H2)",
             "Dot-com exposure", "GFC exposure", "COVID exposure"]]
    for cat, r in g.iterrows():
        rows.append([DIV[cat], f"{100 * r.weight:.1f}", "yes" if r.necessity else "",
                     f"{r.dotx:.2f}", f"{r.gfc:.2f}", f"{r.covid:.2f}"])
    t = grid(rows, [5.4 * cm, 2.6 * cm, 2.0 * cm] + [(W - 10 * cm) / 3] * 3,
             extra=[("ALIGN", (1, 1), (-1, -1), "CENTER")])
    return t


def front(D: Data) -> list:
    S = [Spacer(1, 2.2 * cm),
         para("Consumption Reallocation and Macroeconomic Resilience After Major Economic Shocks",
              "title"),
         para("Results: eight countries one by one, and all eight together", "subtitle"),
         Spacer(1, 0.5 * cm),
         para("<b>Shocks:</b> the dot-com bust (2001), the global financial crisis (2008-09) and "
              "COVID-19 (2020).<br/><b>Countries:</b> United States, United Kingdom, Japan, Canada, "
              "Germany, France, Italy and India.<br/><b>Data:</b> quarterly household consumption "
              "in twelve spending categories (COICOP divisions), every sample ending 2021Q4.",
              "body"),
         para(f"Generated {dt.date.today():%d %B %Y} from the published result tables of the "
              "replication package; every number here can be reproduced with "
              "<font name='DV-M'>python verify.py --all</font>.", "small"),
         Spacer(1, 0.6 * cm)]
    box = [para("<b>How to read the result tables</b>", "box")] + bullets([
        '<font color="#17773a"><b>✓ Supported</b></font> - the effect has the predicted sign and is '
        'statistically significant (5%, one-sided).',
        '<font color="#b42318"><b>✗ Rejected</b></font> - significant, but in the opposite direction '
        'to the prediction.',
        '<font color="#1d5aa6"><b>≈0 No effect</b></font> - the effect is proven to be too small to '
        'matter: its 90% confidence interval lies inside plus or minus one ordinary quarterly '
        'fluctuation of that country (the "smallest effect of interest").',
        '<b>? Undecided</b> - not significant, but not proven small either: the data cannot tell. '
        'Undecided is not the same as "no effect".',
        "Each cell shows the estimate and, in brackets, its standard error.",
        "Units - H1 and H3: percentage points of four-quarter real growth; H2: change in the "
        "log-ratio budget share of necessities; H4: per cent gap between actual consumption and "
        "its no-shock path; H5: change in the impact effect per one standard deviation of "
        "pre-shock resilience."], "bullet")
    bt = Table([[box]], colWidths=[W])
    bt.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, NAVY),
                            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f7fa")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    S += [bt, PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9.5, leading=13, leftIndent=0),
                       ParagraphStyle("t1", fontName="DV", fontSize=8.4, leading=11, leftIndent=14)]
    S += [para("<b>Contents</b>", "H3"), toc, PageBreak()]
    return S


def at_a_glance(D: Data) -> list:
    clean = D.ind[(D.ind.spec == "clean") & D.ind.hypothesis.isin(MAIN)]
    n = clean.groupby(["country", "timeline"]).ngroups
    sup = clean.assign(s=clean.decision_headline.eq("supported")).pivot_table(
        index="pool", columns="hypothesis", values="s", aggfunc="sum")
    pc = D.pool[D.pool.spec == "clean"]
    rows = [["Shock", "H1 consumption falls", "H2 necessities gain share",
             "H3 exposed divisions fall more", "H4 stays below trend", "H5 resilience helps"]]
    extra = []
    for i, s in enumerate(list(P.SHOCKS) + ["combined"], start=1):
        row = [SHOCK[s]]
        for j, h in enumerate(MAIN + ["H5"]):
            r = pc[(pc.pool == s) & (pc.hypothesis == h)].iloc[0]
            if s in P.SHOCKS and h in MAIN:
                k = int(sup.loc[s, h])
            elif s == "combined" and h in MAIN:
                cb = D.comb[(D.comb.spec == "clean") & (D.comb.hypothesis == h)]
                k = int((cb.decision_headline == "supported").sum())
            else:
                k = None
            countries = (f"<br/><font size='6.4' color='#444444'>countries: {k} of {n}</font>"
                         if k is not None else "")
            p = Paragraph(f'<font color="{FG[r.decision_headline]}"><b>{SYM[r.decision_headline]}</b></font>'
                          f"&nbsp;pooled {num(r.estimate, h)}{countries}", ST["cellc"])
            row.append(p)
            extra.append(("BACKGROUND", (1 + j, i), (1 + j, i), colors.HexColor(BG[r.decision_headline])))
        rows.append(row)
    t = grid(rows, [3.0 * cm] + [(W - 3.0 * cm) / 5] * 5, extra=extra, zebra=False)
    tot = D.counts[[f"{s}_supported" for s in P.SPECS] + [f"{s}_settled" for s in P.SPECS]].sum()
    S = [para("Results at a glance", "H1"),
         para("The table gives the all-country (pooled) verdict for each shock and hypothesis, and, "
              f"underneath, in how many of the {n} individual country-timelines the hypothesis is "
              "supported (eight countries; full record and matched timeline; India has one). All "
              "with the country-specific effects included (the headline).", "body"),
         t, legend(), Spacer(1, 4)]
    S += bullets([
        "<b>COVID-19</b> is the clearest case: consumption fell in every country and timeline (H1), "
        "necessities gained budget share everywhere (H2), and contact-intensive categories fell "
        "most everywhere (H3). Whether spending was still below trend (H4) can only be checked "
        "eight quarters out, before the 2022 energy shock, and is supported in 9 of 15.",
        "<b>The global financial crisis</b> cut consumption (H1, 12 of 15; not Germany or India), "
        "hit durable-heavy categories hardest (H3, 10 of 15) and left spending below trend (H4, "
        "12 of 15). The shift towards necessities (H2) is weaker (6 of 15).",
        "<b>The dot-com bust</b> did not measurably reduce household consumption in any country "
        "(H1, 0 of 15). It was mainly an equity and investment shock.",
        "<b>Pooled across all eight countries</b>, 11 of the 20 tests are supported and one more is "
        "proven to be no effect (resilience, dot-com). Pre-shock resilience (H5) is undecided "
        "for the financial crisis, COVID-19 and all three combined.",
        "<b>Adding the country-specific effects</b> changes 16 of 180 country results: supported "
        f"{int(tot['core_supported'])} to {int(tot['clean_supported'])}, settled "
        f"{int(tot['core_settled'])} to {int(tot['clean_settled'])}. They move results rather "
        "than add to them, and they remove wrong-sign results in Japan that were caused by "
        "uncontrolled events."])
    return S + [PageBreak()]


def design(D: Data) -> list:
    S = [para("Part 1. How the study works", "H1"),
         para("This part explains, once, everything the country and all-country results share: the "
              "hypotheses, the shocks, the data, the country-specific effects and how a verdict is "
              "reached. Each country section then repeats what is specific to that country.", "body")]
    S += [para("1.1 Hypotheses", "H2")]
    rows = [["", "Hypothesis", "Predicted", "How it is measured"],
            ["H1", "The shock reduces real household consumption during its impact phase.", "negative",
             "Effect of the impact quarters on four-quarter real growth of every category"],
            ["H2", "Necessities gain budget share during the shock.", "positive",
             "Impact x necessity on log-ratio budget shares, comparing categories within the same quarter"],
            ["H3", "Categories exposed to that particular shock contract more than others.", "negative",
             "Impact x the category's exposure to this shock, on real growth, within the same quarter"],
            ["H4", "Consumption stays below its no-shock path after the impact.", "negative",
             "Average gap between actual spending and its normal-growth path 8, 12 and 16 quarters "
             "after the last pre-shock quarter"],
            ["H5", "Greater pre-shock resilience attenuates the shock (all-country analysis only).",
             "positive", "Impact x the country's pre-shock resilience index"]]
    S += [grid(rows, [0.9 * cm, 6.4 * cm, 1.7 * cm, W - 9.0 * cm])]
    S += [para("1.2 The three shocks", "H2")]
    rows = [["Shock", "Shock window", "Impact quarters", "Recovery check (H4) starts after",
             "Resilience measured in"]]
    for s, v in P.SHOCKS.items():
        a = str(pd.Period(v["window"][0], freq="Q") - 1)
        rows.append([SHOCK[s], quarters(*v["window"]), quarters(*v["impact_quarters"]), a,
                     str(v["resilience_year"])])
    S += [grid(rows, [4.0 * cm, 2.8 * cm, 2.8 * cm, 4.2 * cm, W - 13.8 * cm]),
          para("The 2022 energy-price shock is not studied. Every sample ends in 2021Q4, the last "
               "quarter of the COVID-19 window, so that the 2022 war and energy regime is neither "
               "counted as a normal period nor read as COVID-19 recovery. For COVID-19 this leaves "
               "only the 8-quarter recovery point (2021Q4).", "small")]
    S += [para("1.3 Countries, data and timelines", "H2")]
    rows = [["Country", "Source", "Categories observed", "Full timeline", "Matched timeline",
             "Time unit for clustering"]]
    for c in ORDER:
        w = D.ind[(D.ind.country == c)]
        full = w[w.timeline == "full"]
        fl = (f"{full.window.iloc[0].replace('-', '–')} ({int(full.quarters.iloc[0])} q)"
              if len(full) else "= matched")
        rows.append([COUNTRY[c], P.COUNTRIES[c]["source"], P.COUNTRIES[c]["observed"], fl,
                     "1999Q2–2021Q4 (91 q)", TCLUSTER[P.TIME_CLUSTER[c]]])
    S += [grid(rows, [2.3 * cm, 3.6 * cm, 2.6 * cm, 3.3 * cm, 2.8 * cm, W - 14.6 * cm])]
    S += [Spacer(1, 4), _figure(D, "results/diagnostics/diag_timelines.png", 0.92)]
    S += bullets([
        "<b>Full timeline</b>: each country's whole quarterly record, so the estimates use as much "
        "history as exists. <b>Matched timeline</b>: 1999Q2-2021Q4 for everyone, so countries are "
        "compared over identical years. India's record starts in 1999Q2 and binds the common start.",
        "<b>Observed quarterly</b> (USA, UK, Japan, Canada): the statistical office publishes "
        "spending by category every quarter.",
        "<b>Annual, benchmarked</b> (Germany, France, Italy, India): spending by category is published "
        "once a year (total consumption is published quarterly). Each year's official figure is "
        "spread over its four quarters following an official quarterly series - household "
        "consumption by durability group (durable, semi-durable, non-durable, services) for "
        "Germany, France and Italy, blended by each category's mix; total household consumption "
        "for India, the same series for all twelve categories - by proportional Denton-Cholette "
        "benchmarking, the method of the IMF Quarterly National Accounts Manual. The four quarters "
        "add up exactly to the published annual value, so every annual level is official. Within-"
        "year timing partly comes from the indicator, so these countries' tests count each year "
        "(India: each April-March financial year) as one unit of independent information, and "
        "section 3.9 shows the pooled results without them."])
    S += [para("1.4 Variables", "H2")]
    S += bullets([
        "<b>Twelve categories</b> (COICOP divisions): " + "; ".join(DIV.values()) + ".",
        "<b>Real growth</b>: four-quarter change in the log of each category's real (volume) "
        "spending, in per cent.",
        "<b>Budget shares</b>: each category's share of nominal spending, in centred log-ratio form "
        "so the twelve shares can be modelled without the adding-up constraint.",
        "<b>Weights</b>: each category's share of its country's spending, so large categories "
        "count more; in the all-country analysis every country carries equal total weight.",
        "<b>Necessities</b> (H2): food and non-alcoholic beverages; housing, water, electricity and "
        "gas; health.",
        "<b>Exposure</b> (H3), measured from each country's own sub-items before any estimation: "
        "dot-com - discretionary intensity; financial crisis - durable-goods intensity; COVID-19 - "
        "contact (face-to-face) intensity. Each country section lists its own values."])
    S += [para("1.5 Country-specific effects", "H2"),
          para("Every country event in the design memo was dated, sourced and classified before "
               "estimation (data/country_events.csv, 97 events). The type decides how it enters:", "body")]
    rows = [["Type", "What it is", "How it enters the model"],
            ["C - Confounder", "An independent event that moved consumption on its own (tax change, "
             "strike, disaster, another crisis).",
             "Own dummy. Its quarters are also removed from the normal (quiet) quarters used for "
             "trends, the threshold and the placebo test."],
            ["S - Structural regime", "A multi-year change in the economic environment.",
             "Regime dummy only; its quarters still count as normal."],
            ["M - Policy response", "A measure taken because of the shock (stimulus cheques, furlough, "
             "car scrappage).",
             "Not in the headline, which measures the shock's total effect including the policy "
             "cushion. Added in the policy specification."],
            ["X - Second shock", "A new shock during the recovery.",
             "Treated as a confounder, and the recovery check (H4) stops before it."]]
    S += [grid(rows, [3.0 * cm, 6.2 * cm, W - 9.2 * cm])]
    S += bullets([
        "<b>Scope</b>: whole economy (all twelve categories) or named categories only. H2 and H3 "
        "compare categories within the same quarter, so only category-specific events can enter "
        "them; whole-economy events enter H1 and the normal-quarter sets.",
        "<b>Overlap rule</b>: a whole-economy event falling in a shock's own impact quarters cannot "
        "be told apart from the shock by a dummy (it would absorb the shock itself). For that "
        "shock's tests it moves to the policy specification. Category events are unaffected.",
        "<b>Primary and optional</b>: the memo's optional events (11) enter only a separate "
        "specification, so the headline is not tuned by them.",
        "<b>Events before 1999</b> for the long records were admitted only under a fixed rule: "
        "documented, outside the shock windows, and at least one normal-quarter standard deviation "
        "in size. The candidates that failed are listed in the appendix."])
    S += [para("1.6 Four specifications on identical models", "H2")]
    rows = [["Specification", "Country effects included"],
            [SPEC["core"], "None. No quarter is declared disturbed."],
            [SPEC["clean"], "Primary confounders (C), structural regimes (S) and second shocks (X)."],
            [SPEC["mechanism"], "Headline + policy responses (M) + whole-economy events that overlap "
             "the impact quarters + real oil-price growth (H1) or oil x category energy intensity "
             "(H2, H3)."],
            [SPEC["extended"], "Headline + the memo's optional events."]]
    S += [grid(rows, [6.0 * cm, W - 6.0 * cm]),
          para("Only the set of effects differs; the equations, data and timelines are the same, so "
               "the difference between 'with' and 'without' is the pure contribution of the effects.",
               "small")]
    S += [para("1.7 How each verdict is reached", "H2")]
    S += bullets([
        "<b>Estimation</b>: weighted least squares, weights = spending shares.",
        "<b>Individual countries</b>: H1 standard errors are clustered two ways, by category and by "
        "time (quarter where categories are observed quarterly; calendar year for Germany, France "
        "and Italy; financial year for India). H2-H4 are clustered by category. Tests use the "
        "t-distribution with 11 degrees of freedom (12 categories minus one).",
        "<b>Second, assumption-free test</b>: an exact placebo test. The same model is re-estimated "
        "pretending the shock hit in each normal two-quarter period of the sample, and the real "
        "estimate is ranked against these placebo estimates.",
        "<b>Verdict</b>: supported if either test is significant at 5% in the predicted direction "
        "(fixed before estimation); rejected if significant in the opposite direction; no "
        "effect if the 90% interval lies inside plus or minus the threshold (equivalence test); "
        "otherwise undecided.",
        "<b>Threshold</b> (smallest effect of interest): one standard deviation of the country's own "
        "normal-quarter variation in the same statistic - 'smaller than an ordinary quarter's "
        "fluctuation'. The same threshold is used in every specification.",
        "<b>All countries together</b>: H1-H3 clustered by country x category and by calendar year "
        "(22 degrees of freedom); H4 and H5 clustered by country with the wild cluster bootstrap "
        "(Webb weights, 7 degrees of freedom), because eight clusters are too few for the usual "
        "formula alone; placebo test on 11 year-long phases; threshold = average of the eight "
        "country thresholds."])
    S += [para("1.8 Robustness checks requested in the design memo", "H2")]
    S += bullets([
        "<b>H2 on real-volume shares</b>: budget shares recomputed from real spending, so a price "
        "rise alone cannot raise a necessity's share.",
        "<b>H4 with a local trend</b>: each category's normal growth taken from the ten years "
        "before the shock (at least 20 normal quarters) instead of the whole record."])
    return S


def country_section(D: Data, c: str, k: int) -> list:
    S = [PageBreak(), para(f"2.{k} {COUNTRY[c]}", "H2")]
    S += [para(f"<b>Data.</b> {SOURCE_LONG[c]}. Categories are {P.COUNTRIES[c]['observed']}. "
               + SOURCE_NOTE.get(c, ""), "body")]
    S += [para("A. Timeline", "H3")]
    rows = [["Timeline", "Period", "Quarters", "Time unit (H1 clustering)",
             "Placebo periods with / without effects"]]
    for tl in D.timelines(c):
        r = D.cell(c, tl, "clean", "dotcom", "H1")
        nph = {s: int(D.ind[(D.ind.country == c) & (D.ind.timeline == tl)
                            & (D.ind.spec == s)].n_placebo.max()) for s in ("clean", "core")}
        name = {"full": "Full record", "matched": "Matched"}[tl] if c != "India" else "Full record = matched"
        rows.append([name, r.window.replace("-", "–"), str(int(r.quarters)), TCLUSTER[P.TIME_CLUSTER[c]],
                     f"{nph['clean']} / {nph['core']}"])
    S += [grid(rows, [3.4 * cm, 3.3 * cm, 1.8 * cm, 4.6 * cm, W - 13.1 * cm])]
    ev = D.ev[(D.ev.country == c)]
    n_agg = int(((ev.tier == "primary") & ev.type.isin(["C", "S", "X"]) & (ev.scope == "aggregate")).sum())
    n_cat = int(((ev.tier == "primary") & ev.type.isin(["C", "S", "X"]) & (ev.scope == "category")).sum())
    S += [para("B. Equations", "H3"),
          para("j = category, t = quarter, s = shock. g = four-quarter real growth (%), clr = log-ratio "
               "budget share, w = spending weight.", "small")]
    S += [para(minus(t), "eq") for t in [
        "H1:  g_jt   = α_j + τ_j·t + β·Impact_t + Σ_k γ_k·D_kt + ε_jt",
        "H2:  clr_jt = α_j + λ_t + τ_j·t + β·(Impact_t × Necessity_j)",
        "              + Σ_(other shocks) θ·(Impact'_t × Necessity_j) + Σ_k γ_k·(D_kt × Cat_kj) + ε_jt",
        "H3:  g_jt   = α_j + λ_t + τ_j·t + β·(Impact_t × Exposure^s_j)",
        "              + Σ_(other shocks) θ·(Impact'_t × Exposure'_j) + Σ_k γ_k·(D_kt × Cat_kj) + ε_jt",
        "H4:  gap_jh = 100·[ ln L_j,a+h − ln L_j,a − r_j·h ],   gap_jh = β + ε_jh"]]
    S += [para(
        "α_j category effect; τ_j·t category trend (every category has its own); λ_t quarter effect, "
        "so H2 and H3 compare categories within the same quarter; Impact_t = 1 in the shock's two "
        "impact quarters; D_kt the country's event dummies (Cat_kj restricts a category event to its "
        "categories); L real spending level; a the last pre-shock quarter (anchor); h = 8, 12, 16 "
        "quarters; r_j the category's average quarterly growth over normal quarters. β is the "
        f"tested effect. For {COUNTRY[c]}, the headline model has {n_agg} whole-economy and {n_cat} "
        "category-specific events (listed in C). H1 errors are clustered by category and "
        f"{TCLUSTER[P.TIME_CLUSTER[c]]}; H2-H4 by category.", "small")]
    S += [para("Recovery check (H4) in the headline:", "small"), para(h4_note(D, c, "clean"), "small")]
    S += [para("C. Effects accounted for", "H3")]
    rows = [["Event", "Quarters", "Type", "Applies to", "How it is used"]] + event_rows(D, c)
    S += [grid(rows, [5.7 * cm, 2.45 * cm, 2.05 * cm, 2.9 * cm, W - 13.1 * cm])]
    S += [Spacer(1, 4), para("Necessity set and measured exposures (the H2 and H3 inputs):", "small"),
          exposure_table(D, c)]
    S += [CondPageBreak(7 * cm), para("D. Results with effects (headline)", "H3"),
          results_grid(D, c, "clean"), legend(), para(tally(D, c, "clean"), "small")]
    S += [CondPageBreak(7 * cm), para("E. Results without effects", "H3"),
          results_grid(D, c, "core"), legend(), para(tally(D, c, "core"), "small"),
          para("Recovery check without effects: " + h4_note(D, c, "core"), "small")]
    S += [CondPageBreak(4 * cm), para("F. What adding the effects changed", "H3")]
    cnt = D.counts[D.counts.country == c]
    rows = [["Timeline", "Supported: without → with", "Settled (✓ ✗ ≈0): without → with"]]
    for r in cnt.itertuples():
        rows.append([r.timeline if c != "India" else "single timeline",
                     f"{r.core_supported} → {r.clean_supported}", f"{r.core_settled} → {r.clean_settled}"])
    S += [grid(rows, [4 * cm, 5 * cm, W - 9 * cm])]
    ch = D.changes[D.changes.country == c]
    if len(ch):
        rows = [["Timeline", "Shock", "Test", "Without effects", "With effects", "Why"]]
        extra = []
        for i, r in enumerate(ch.itertuples(), start=1):
            a, ba = vcell(r.decision_headline_core, r.estimate_core, r.std_error_core, r.hypothesis)
            b, bb = vcell(r.decision_headline_clean, r.estimate_clean, r.std_error_clean, r.hypothesis)
            extra += [("BACKGROUND", (3, i), (3, i), colors.HexColor(ba)),
                      ("BACKGROUND", (4, i), (4, i), colors.HexColor(bb))]
            rows.append([r.timeline if c != "India" else "single", SHOCK_SHORT[r.pool], r.hypothesis, a, b,
                         esc(minus(WHY_EFFECTS[(r.country, r.timeline, r.pool, r.hypothesis)]))])
        S += [Spacer(1, 3), grid(rows, [1.6 * cm, 1.7 * cm, 1.0 * cm, 2.2 * cm, 2.2 * cm, W - 8.7 * cm],
                                 extra=extra, zebra=False)]
    else:
        S += [para("No verdict changes: every cell has the same verdict with and without the effects.",
                   "body")]
    S += [CondPageBreak(4 * cm), para("G. Further checks", "H3")]
    rows, extra = [["Variant", "Timeline", "Shock", "Test", "Headline", "Variant result", "Why"]], []
    for spec in ("mechanism", "extended"):
        for tl in D.timelines(c):
            for s in P.SHOCKS:
                for h in MAIN:
                    a, b = D.cell(c, tl, "clean", s, h), D.cell(c, tl, spec, s, h)
                    if a.decision_headline != b.decision_headline:
                        i = len(rows)
                        pa, ba = vcell(a.decision_headline, a.estimate, a.std_error, h)
                        pb, bb = vcell(b.decision_headline, b.estimate, b.std_error, h)
                        extra += [("BACKGROUND", (4, i), (4, i), colors.HexColor(ba)),
                                  ("BACKGROUND", (5, i), (5, i), colors.HexColor(bb))]
                        rows.append([{"mechanism": "Policy + oil", "extended": "Optional events"}[spec],
                                     tl if c != "India" else "single", SHOCK_SHORT[s], h, pa, pb,
                                     esc(minus(WHY_SPEC.get((spec, c, s, h), "")))])
    S += [para("<b>Other specifications.</b> Cells whose verdict differs from the headline when "
               "policy responses and oil are added, or when the optional events are added "
               "(all other cells are identical in verdict):", "body")]
    if len(rows) > 1:
        S += [grid(rows, [2.1 * cm, 1.5 * cm, 1.5 * cm, 0.9 * cm, 1.9 * cm, 1.9 * cm, W - 9.8 * cm],
                   extra=extra, zebra=False)]
    else:
        S += [para("None.", "body")]
    rows, extra = [["Timeline", "Shock", "H2 headline", "H2 on volume shares", "H4 headline",
                    "H4 with local trend"]], []
    for tl in D.timelines(c):
        for s in P.SHOCKS:
            i = len(rows)
            row = [tl if c != "India" else "single", SHOCK_SHORT[s]]
            for j, h in enumerate(["H2", "H2_volume", "H4", "H4_local"]):
                r = D.cell(c, tl, "clean", s, h)
                p, bg = vcell(r.decision_headline, r.estimate, r.std_error, h)
                row.append(p)
                extra.append(("BACKGROUND", (2 + j, i), (2 + j, i), colors.HexColor(bg)))
            rows.append(row)
    S += [Spacer(1, 4), para("<b>Robustness checks from the design memo</b> (headline effects):", "body"),
          KeepTogether(grid(rows, [1.8 * cm, 2.2 * cm] + [(W - 4 * cm) / 4] * 4, extra=extra, zebra=False)),
          para("'n/a' for H4 with a local trend: fewer than 20 normal quarters exist in the ten years "
               "before the anchor.", "small")]
    S += [Spacer(1, 4), para("<b>H. All three shocks together</b> - one regression with a separate effect "
                             "for each shock; the reported effect is their equal-weight average, as in "
                             "the all-country combined result. Equal effects: p-value of the test that "
                             "the three shock effects are the same.", "body")]
    rows, extra = [["Timeline", "Specification", "H1", "H2", "H3", "H4", "Equal effects? (H1/H2/H3)"]], []
    for tl in D.timelines(c):
        for spec in ("clean", "core"):
            i = len(rows)
            row = [tl if c != "India" else "single", {"clean": "with effects", "core": "without effects"}[spec]]
            hom = []
            for j, h in enumerate(MAIN):
                r = D.ccell(c, tl, spec, h)
                pc, bg = vcell(r.decision_headline, r.estimate, r.std_error, h)
                row.append(pc)
                extra.append(("BACKGROUND", (2 + j, i), (2 + j, i), colors.HexColor(bg)))
                if h != "H4":
                    hom.append("n/a" if pd.isna(r.get("homogeneity_p", np.nan)) else f"{r.homogeneity_p:.3f}")
            row.append(" / ".join(hom))
            rows.append(row)
    S += [grid(rows, [1.6 * cm, 2.3 * cm] + [(W - 7.6 * cm) / 4] * 4 + [3.7 * cm], extra=extra, zebra=False)]
    S += [Spacer(1, 4), para("<b>In short</b> (headline): " + in_short(D, c), "body")]
    return S


def in_short(D: Data, c: str) -> str:
    out = []
    for tl in D.timelines(c):
        bits = []
        for s in P.SHOCKS:
            sup = [h for h in MAIN if D.cell(c, tl, "clean", s, h).decision_headline == "supported"]
            rej = [h for h in MAIN if D.cell(c, tl, "clean", s, h).decision_headline == "rejected_opposite_sign"]
            neg = [h for h in MAIN if D.cell(c, tl, "clean", s, h).decision_headline == "negligible_effect"]
            t = f"{SHOCK_SHORT[s]} supports {', '.join(sup) if sup else 'none'}"
            if neg:
                t += f"; no effect for {', '.join(neg)}"
            if rej:
                t += f"; opposite sign for {', '.join(rej)}"
            bits.append(t)
        name = {"full": "full record", "matched": "matched"}[tl] if c != "India" else "single timeline"
        out.append(f"<i>{name}</i> - " + " · ".join(bits))
    return "<br/>".join(out)


def countries(D: Data) -> list:
    S = [PageBreak(), para("Part 2. Results country by country", "H1"),
         para("Each country follows the same order: <b>A</b> timeline, <b>B</b> equations, <b>C</b> "
              "effects accounted for, <b>D</b> results with the effects (the headline), <b>E</b> "
              "results without them, <b>F</b> what the effects changed and why, <b>G</b> further "
              "checks (other specifications and the memo's robustness checks), <b>H</b> the three "
              "shocks estimated together.", "body")]
    rows = [["Country", "Timeline", "Without effects", "With effects (headline)",
             "+ policy and oil", "+ optional events"]]
    for r in D.counts.itertuples():
        rows.append([COUNTRY[r.country], r.timeline if r.country != "India" else "single"] +
                    [f"{getattr(r, f'{s}_supported')} / {getattr(r, f'{s}_settled')}" for s in P.SPECS])
    tot = D.counts[[f"{s}_{k}" for s in P.SPECS for k in ("supported", "settled")]].sum()
    rows.append(["<b>Total (of 180)</b>", ""] +
                [f"<b>{int(tot[f'{s}_supported'])} / {int(tot[f'{s}_settled'])}</b>" for s in P.SPECS])
    S += [para("Overview: supported / settled cells (of 12 per row) in each specification", "H3"),
          grid(rows, [3.0 * cm, 1.9 * cm] + [(W - 4.9 * cm) / 4] * 4,
               extra=[("ALIGN", (2, 1), (-1, -1), "CENTER")]),
          para("Settled = supported, rejected or no effect. 15 country-timelines x 4 hypotheses x 3 "
               "shocks = 180 cells per specification.", "small")]
    cc = D.comb_counts
    S += [para("<b>All three shocks together</b> (section H of each country): supported in "
               + ", ".join(f"{int(cc[s + '_supported'].sum())} of {4 * len(cc)} ({SPEC[s].lower()})"
                           for s in P.SPECS)
               + ". The three shock effects differ significantly from each other in most countries, "
               "so this average mixes very different shocks: it shows whether the shocks on average "
               "moved consumption, not that each shock did.", "body")]
    for k, c in enumerate(ORDER, start=1):
        S += country_section(D, c, k)
    return S


def pooled(D: Data) -> list:
    pc = D.pool
    S = [PageBreak(), para("Part 3. All eight countries together", "H1"),
         para("The pooled analysis asks the same questions of all eight countries at once, adds H5 "
              "(does pre-shock resilience soften the shock?), and adds a fourth pool combining all "
              "three shocks.", "body")]
    S += [para("3.1 Timeline", "H2"),
          para("One common timeline for every country: <b>1999Q2-2021Q4</b>, 91 quarters; 8 countries "
               "x 12 categories x 91 quarters = 8,736 observations. India's category record sets the "
               "start; the end is the last COVID-19 quarter before the 2022 regime.", "body")]
    S += [para("3.2 Equations", "H2"),
          para("c = country, j = category, t = quarter. Each country carries equal total weight.", "small")]
    S += [para(minus(t), "eq") for t in [
        "H1:  g_cjt   = α_cj + τ_cj·t + β·Impact_t + Σ_k γ_k·D_ckt + ε_cjt",
        "H2:  clr_cjt = α_cj + λ_ct + τ_cj·t + β·(Impact_t × Necessity_j)",
        "               + other shocks + Σ_k γ_k·(D_ckt × Cat_kj) + ε_cjt",
        "H3:  g_cjt   = α_cj + λ_ct + τ_cj·t + β·(Impact_t × Exposure^s_cj)",
        "               + other shocks + Σ_k γ_k·(D_ckt × Cat_kj) + ε_cjt",
        "H4:  gap_cjh = 100·[ ln L_cj,a+h − ln L_cj,a − r_cj·h ],   gap_cjh = β + ε_cjh",
        "H5:  g_cjt   = α_cj + τ_cj·t + β·Impact_t + δ·(Impact_t × Resilience_c)",
        "               + Σ_k γ_k·D_ckt + ε_cjt          (H5 is δ > 0)"]]
    hp = pc[(pc.pool == "combined") & (pc.spec == "clean") & pc.hypothesis.isin(["H1", "H2", "H3"])]
    S += [para(
        "λ_ct is a country x quarter effect, so H2 and H3 compare categories within the same country "
        "and quarter. <b>All three shocks combined</b>: one regression with a separate effect for each "
        "shock; the reported effect is the equal-weight average of the three. A single common "
        "coefficient is rejected by the data (test that the three effects are equal: p "
        + ("< 0.001 for each of H1, H2 and H3" if (hp.homogeneity_p < 0.001).all() else
           "= " + ", ".join(f"{p:.4f} for {h}" for h, p in zip(hp.hypothesis, hp.homogeneity_p)))
        + "), so the common-coefficient version is shown only as a robustness check.", "small")]
    S += [para("3.3 Effects accounted for", "H2"),
          para("All 97 country events of Part 2 enter with the same rules and the same four "
               "specifications, each event affecting only its own country. H5 adds the pre-shock "
               "resilience index below, measured in the year before each shock window, standardised "
               "across the eight countries and signed so that higher = more resilient (at least 3 of "
               "4 components required).", "body")]
    rows = [["Component", "Why it matters", "Source"],
            ["Government gross debt, % of GDP", "Fiscal space to respond", "IMF World Economic Outlook (via FRED)"],
            ["5-year change in household credit, pp of GDP", "Household leverage build-up",
             "BIS total credit statistics (via FRED)"],
            ["Distance of CPI inflation from 2%", "Price stability (deflation is not scored as strength)",
             "World Bank WDI"],
            ["Unemployment rate, %", "Labour-market slack", "World Bank WDI (ILO modelled)"]]
    S += [grid(rows, [5.6 * cm, 5.4 * cm, W - 11 * cm])]
    rows = [["Shock", "Country", "Gov. debt (% GDP)", "Household credit, 5-year change (pp)",
             "Inflation (%)", "Unemploy-<br/>ment (%)", "Compo-<br/>nents", "Resilience (z)"]]
    r = D.res.copy()
    r["o"] = r.shock.map({s: i for i, s in enumerate(P.SHOCKS)})
    for x in r.sort_values(["o", "resilience"], ascending=[True, False]).itertuples():
        f = lambda v: "n/a" if pd.isna(v) else minus(f"{v:.1f}")
        rows.append([SHOCK_SHORT[x.shock], COUNTRY[x.country], f(x.gov_debt), f(x.hh_credit_change),
                     f(x.inflation), f(x.unemployment), str(int(x.n_components)),
                     minus(f"{x.resilience:+.2f}")])
    S += [Spacer(1, 4), grid(rows, [1.8 * cm, 2.6 * cm, 2.2 * cm, 3.0 * cm, 1.9 * cm, 2.1 * cm,
                                    1.7 * cm, W - 15.3 * cm],
                             extra=[("ALIGN", (2, 1), (-1, -1), "CENTER")]),
          para("n/a: the source series does not reach back to that year (US debt 1999; India "
               "household credit 1999 and 2007); the index uses the remaining three components, as "
               "pre-specified.", "small")]

    def pgrid(spec):
        rows, extra = [["Pool"] + MAIN + ["H5"]], []
        for i, s in enumerate(list(P.SHOCKS) + ["combined"], start=1):
            row = [SHOCK[s]]
            for j, h in enumerate(MAIN + ["H5"]):
                x = pc[(pc.pool == s) & (pc.spec == spec) & (pc.hypothesis == h)].iloc[0]
                p, bg = vcell(x.decision_headline, x.estimate, x.std_error, h)
                row.append(p)
                extra.append(("BACKGROUND", (1 + j, i), (1 + j, i), colors.HexColor(bg)))
            rows.append(row)
        return grid(rows, [3.6 * cm] + [(W - 3.6 * cm) / 5] * 5, extra=extra, zebra=False)

    def ptally(spec):
        b = pc[(pc.spec == spec) & pc.pool.isin(list(P.SHOCKS) + ["combined"])
               & pc.hypothesis.isin(MAIN + ["H5"])]
        return (f"{int((b.decision_headline == 'supported').sum())} supported, "
                f"{int((b.decision_headline == 'rejected_opposite_sign').sum())} rejected, "
                f"{int((b.decision_headline == 'negligible_effect').sum())} no effect, "
                f"{int((b.decision_headline == 'inconclusive').sum())} undecided (of 20).")

    S += [CondPageBreak(8 * cm), para("3.4 Results with effects (headline)", "H2"), pgrid("clean"),
          legend(), para(ptally("clean"), "small")]
    S += [CondPageBreak(8 * cm), para("3.5 Results without effects", "H2"), pgrid("core"), legend(),
          para(ptally("core"), "small")]
    ch = []
    a, b = pc[pc.spec == "core"], pc[pc.spec == "clean"]
    for s in list(P.SHOCKS) + ["combined"]:
        for h in MAIN + ["H5"]:
            x = a[(a.pool == s) & (a.hypothesis == h)].iloc[0]
            y = b[(b.pool == s) & (b.hypothesis == h)].iloc[0]
            if x.decision_headline != y.decision_headline:
                ch.append(f"{SHOCK[s]} {h}: {WORD[x.decision_headline]} → {WORD[y.decision_headline]}")
    h4a = a[(a.pool == "combined") & (a.hypothesis == "H4")].estimate.iloc[0]
    h4b = b[(b.pool == "combined") & (b.hypothesis == "H4")].estimate.iloc[0]
    S += [para("<b>What adding the effects changed:</b> "
               + ("; ".join(ch) if ch else "no pooled verdict changes; the estimates move little "
                  f"(for example the combined H4 shortfall from {num(h4a)} to {num(h4b)}).") , "body")]
    S += [CondPageBreak(7 * cm), para("3.6 Other specifications", "H2"),
          para("<b>With effects + policy responses and oil</b>", "H3"), pgrid("mechanism"),
          para(ptally("mechanism") + " With policy responses and the oil price controlled, the "
               "financial-crisis impact (H1) is no longer separable: the oil collapse and the policy "
               "measures coincide with the impact quarters. This is a decomposition, not a test of "
               "H1.", "small"),
          para("<b>With effects + optional events</b>", "H3"), pgrid("extended"),
          para(ptally("extended"), "small")]
    rows, extra = [["Pool", "H2 headline", "H2 on volume shares", "H4 headline", "H4 with local trend"]], []
    for i, s in enumerate(list(P.SHOCKS) + ["combined"], start=1):
        row = [SHOCK[s]]
        for j, h in enumerate(["H2", "H2_volume", "H4", "H4_local"]):
            x = pc[(pc.pool == s) & (pc.spec == "clean") & (pc.hypothesis == h)].iloc[0]
            p, bg = vcell(x.decision_headline, x.estimate, x.std_error, h)
            row.append(p)
            extra.append(("BACKGROUND", (1 + j, i), (1 + j, i), colors.HexColor(bg)))
        rows.append(row)
    S += [CondPageBreak(6 * cm), para("3.7 Robustness checks", "H2"),
          grid(rows, [3.6 * cm] + [(W - 3.6 * cm) / 4] * 4, extra=extra, zebra=False),
          para("'n/a' for the dot-com local trend: the common timeline starts in 1999Q2, so fewer "
               "than 20 normal quarters precede the 1999Q4 anchor.", "small")]
    rows, extra = [["Specification", "H1", "H2", "H3"]], []
    for i, spec in enumerate(P.SPECS, start=1):
        row = [SPEC[spec]]
        for j, h in enumerate(["H1", "H2", "H3"]):
            x = pc[(pc.pool == "combined_common") & (pc.spec == spec) & (pc.hypothesis == h)].iloc[0]
            p, bg = vcell(x.decision_headline, x.estimate, x.std_error, h)
            row.append(p)
            extra.append(("BACKGROUND", (1 + j, i), (1 + j, i), colors.HexColor(bg)))
        rows.append(row)
    S += [para("<b>All three shocks with one common coefficient</b> (robustness; misspecified because "
               "the three effects differ, which inflates the standard errors):", "body"),
          grid(rows, [6.0 * cm] + [(W - 6.0 * cm) / 3] * 3, extra=extra, zebra=False)]
    S += [CondPageBreak(7 * cm), para("3.8 H5 in detail", "H2"),
          para("<b>Each resilience component on its own</b> (headline specification). p(+) is the "
               "bootstrap p-value for the predicted positive effect; p(−) for the opposite.", "body")]
    lab = {"gov_debt": "Government debt", "hh_credit_change": "Household credit build-up",
           "inflation_gap": "Inflation distance from 2%", "unemployment": "Unemployment"}
    rows = [["Component", "Shock", "Estimate (SE)", "p(+)", "p(−)", "Countries"]]
    for x in D.h5c.itertuples():
        rows.append([lab[x.component], SHOCK_SHORT[x.pool], f"{num(x.estimate)} ({num(x.std_error, sign=False)})",
                     f"{x.p_bootstrap_positive:.3f}", f"{x.p_bootstrap_negative:.3f}", str(int(x.n_countries))])
    S += [grid(rows, [4.4 * cm, 2.2 * cm, 3.4 * cm, 2.0 * cm, 2.0 * cm, W - 14 * cm],
               extra=[("ALIGN", (2, 1), (-1, -1), "CENTER")]),
          para("Read as exploratory: four components x four pools are 16 tests, so single p-values "
               "near 0.05 are expected by chance. The closest to the prediction is household credit "
               "for the financial crisis (p = 0.063). One component points the other way: for the "
               "financial crisis, countries with lower pre-crisis unemployment had larger consumption "
               "falls (p(−) = 0.010).", "small")]
    rows = [["Shock", "Country-shocks", "Slope (SE)", "Degrees of freedom", "p (slope > 0)"]]
    meta = D.meta.set_index("shock").reindex(list(P.SHOCKS) + ["combined"]).reset_index()
    for x in meta.itertuples():
        rows.append([SHOCK_SHORT[x.shock], str(int(x.n_country_shocks)),
                     f"{num(x.slope)} ({num(x.std_error, sign=False)})", str(int(x.df)),
                     f"{x.p_one_sided_positive:.3f}"])
    S += [Spacer(1, 3), KeepTogether([
        para("<b>Two-step check</b>: each country's own H1 effect regressed on its resilience score.",
             "body"),
        grid(rows, [3.0 * cm, 2.8 * cm, 3.6 * cm, 3.2 * cm, W - 12.6 * cm],
             extra=[("ALIGN", (1, 1), (-1, -1), "CENTER")])])]
    S += subsample_section(D)
    return S


def subsample_section(D: Data) -> list:
    sub = D.sub.set_index(["subset", "pool", "hypothesis"])
    full = D.pool[D.pool.spec == "clean"].set_index(["pool", "hypothesis"])
    S = [PageBreak(), para("3.9 Check: only the countries that publish quarterly categories", "H2"),
         para("<b>Why.</b> Germany, France, Italy and India publish spending by category only once a "
              "year, so their quarters are benchmarked onto an official quarterly series (section 1.3). "
              "Their yearly levels are official, but the exact quarter in which a category moved is "
              "partly the indicator's. A referee will ask whether the results depend on that.", "body"),
         para("<b>What was done.</b> The pooled H1-H4 were re-estimated with nothing changed except the "
              "countries: once on the four that publish categories every quarter (United States, United "
              "Kingdom, Japan, Canada), and once on the four benchmarked ones. Same equations, events, "
              "timeline, clustering and placebo test; the threshold is the mean of the included "
              "countries' own thresholds. H5 is not re-estimated, because it compares countries by "
              "resilience and four countries per shock are too few. With four countries, H4 is "
              "clustered on four countries (3 degrees of freedom), so it has little power by "
              "construction.", "body")]
    rows, extra = [["Pool", "Test", "All eight countries", "Quarterly-observed four",
                    "Benchmarked four"]], []
    for pool in list(P.SHOCKS) + ["combined"]:
        for h in MAIN:
            i = len(rows)
            row = [SHOCK[pool] if h == "H1" else "", h]
            cells = [full.loc[(pool, h)], sub.loc[("quarterly_observed", pool, h)],
                     sub.loc[("benchmarked", pool, h)]]
            for j, r in enumerate(cells):
                pc, bg = vcell(r.decision_headline, r.estimate, r.std_error, h)
                row.append(pc)
                extra.append(("BACKGROUND", (2 + j, i), (2 + j, i), colors.HexColor(bg)))
            rows.append(row)
    for k in range(4):
        extra.append(("SPAN", (0, 1 + 4 * k), (0, 4 + 4 * k)))
    S += [grid(rows, [3.6 * cm, 1.2 * cm] + [(W - 4.8 * cm) / 3] * 3, extra=extra, zebra=False),
          legend()]
    q = sub.loc[("quarterly_observed", "covid", "H4")]
    S += [para("<b>What it gives.</b>", "body")] + bullets([
        "<b>The conclusions do not rest on the constructed quarters.</b> Of the 11 results supported "
        "with all eight countries, 10 hold using only the four countries that publish categories "
        "quarterly, and all of these survive the Holm correction for multiple testing. The eleventh, "
        f"COVID-19 recovery (H4), has the same size ({num(q.estimate)} against "
        f"{num(full.loc[('covid', 'H4')].estimate)}) but p = {q.p_parametric:.3f}, because four "
        "country clusters leave only 3 degrees of freedom.",
        "<b>The two halves agree.</b> The benchmarked four alone also support 10 of the 11 (the "
        "financial-crisis H3 is smaller there and undecided). No supported result changes sign.",
        "<b>One result is stronger where categories are measured quarterly:</b> the shift towards "
        "necessities in the financial crisis (H2) is supported in the quarterly-observed four "
        f"({num(sub.loc[('quarterly_observed', 'gfc', 'H2')].estimate, 'H2')}) but not in the "
        f"benchmarked four ({num(sub.loc[('benchmarked', 'gfc', 'H2')].estimate, 'H2')}), which "
        "leaves it undecided in the full pool. Two explanations fit and cannot be separated here: "
        "benchmarking smooths category differences within a year (in India all categories follow "
        "one indicator), which weakens H2 and H3 in that half; and that half includes Germany and "
        "India, whose consumption barely fell in the crisis. Either way, the full-pool H2 and H3 "
        "results are, if anything, conservative.",
        "<b>Dot-com recovery (H4) is significantly positive in the quarterly-observed four</b> "
        f"({num(sub.loc[('quarterly_observed', 'dotcom', 'H4')].estimate)}): their spending grew "
        "faster in 2000-03 than its 1999-2021 average. This matches the full-pool reading that "
        "the dot-com bust left no consumption shortfall."])
    S += [para("<b>How to write it.</b> One sentence in the results and the table in an appendix: "
               "<i>“The results do not depend on the quarterly paths constructed for the four "
               "countries that publish categories annually: re-estimating the pooled model on the "
               "United States, United Kingdom, Japan and Canada alone, 10 of the 11 supported results "
               "remain supported and the eleventh keeps the same magnitude (Appendix Table X).”</i>",
               "body")]
    return S


FIGURES = [
    ("fig1_consumption_paths",
     "Figure 1. Real household consumption around each shock",
     "Total real household consumption in each country (fixed spending weights of the last "
     "pre-shock quarter), per cent change since that quarter. Light shading: the shock window; dark "
     "shading: the two impact quarters. Common timeline 1999Q2-2021Q4.",
     "Consumption kept growing through the dot-com bust in every country. In the financial crisis "
     "it dipped in the UK, Italy, Japan, the United States and France, stayed flat in Germany and "
     "kept rising in Canada and India. COVID-19 produced a deep, fast V everywhere, deepest in the "
     "UK and India (about -24% in 2020Q2).",
     "In the stylised-facts section, before any regression: it shows the reader the three shocks "
     "are different in kind."),
    ("fig2_reallocation_h3",
     "Figure 2. Spending moves away from the categories each shock exposes (H3)",
     "Each dot is one spending category in one country (size: its share of spending). Vertical: "
     "the category's growth in the shock's two impact quarters, relative to its own country's "
     "average category in the same quarters and to its normal relative growth. Horizontal: the "
     "category's exposure to that shock, measured before estimation (dot-com: discretionary share; "
     "financial crisis: durable-goods share; COVID-19: face-to-face share). Black line: "
     "spending-weighted fit; the formal H3 estimate is shown for comparison.",
     "No pattern for the dot-com bust; a clear downward slope for the financial crisis (a fully "
     "durable category grew about 7 points less than one with no durables); a steep one for "
     "COVID-19 (a fully face-to-face category grew about 31-34 points less than one with no "
     "face-to-face spending). The simple line through the points almost equals the model estimate "
     "in every panel, so the H3 result is visible in the raw data, not produced by the model.",
     "Next to the H3 results; this is the paper's main figure."),
    ("fig3_recovery_gaps_h4",
     "Figure 3. Spending relative to its normal-growth path after each shock (H4)",
     "Spending-weighted average gap between each category's real spending and the path it would "
     "have followed at its normal growth rate, from the last clean pre-shock quarter (Japan and "
     "France start earlier for COVID-19, before their 2019 tax rise and pension strike). Dotted "
     "lines: the 8, 12 and 16-quarter points tested by H4. A line stops before a declared second "
     "shock or at 2021Q4. Thick line: average where all countries are observed.",
     "No lasting gap after the dot-com bust (except India); a persistent gap of about 4-5% after "
     "the financial crisis, largest in the UK and the United States; after COVID-19 a gap still "
     "open at eight quarters on average, but closed in the United States.",
     "Next to the H4 results."),
]


def figures(D: Data) -> list:
    S = [PageBreak(), para("Part 4. Figures for the paper", "H1"),
         para("Three figures, computed from the same data, definitions and code as the tests "
              "(code/analysis/figures_paper.py). Each comes as a 300-dpi PNG and a vector PDF in "
              "results/figures/, and the numbers plotted are in results/tables/figure_data_*.csv. "
              "For each: the figure, a caption ready for the paper, what it shows, and where it goes.",
              "body")]
    for name, title, caption, shows, where in FIGURES:
        path = D.root / "results/figures" / f"{name}.png"
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(str(path)).getSize()
        S += [CondPageBreak(10 * cm), para(title, "H2"),
              Image(str(path), width=W, height=W * ih / iw),
              para(f"<b>Caption.</b> {esc(minus(caption))}", "small"),
              para(f"<b>What it shows.</b> {esc(minus(shows))}", "body"),
              para(f"<b>Where it goes.</b> {esc(where)}", "body")]
    return S


def summary(D: Data) -> list:
    S = [PageBreak(), para("Part 5. Overall summary", "H1")]
    clean = D.ind[D.ind.hypothesis.isin(MAIN)]
    rows = [["Shock", "Specification"] + MAIN]
    for s in P.SHOCKS:
        for spec in ("core", "clean"):
            b = clean[(clean.pool == s) & (clean.spec == spec)]
            row = [SHOCK[s] if spec == "core" else "", SPEC[spec]]
            for h in MAIN:
                x = b[b.hypothesis == h]
                row.append(f"{int((x.decision_headline == 'supported').sum())} of {len(x)}")
            rows.append(row)
    ex = [("SPAN", (0, 1 + 2 * i), (0, 2 + 2 * i)) for i in range(3)]
    S += [para("5.1 How often each hypothesis is supported across countries", "H2"),
          grid(rows, [4.2 * cm, 4.4 * cm] + [(W - 8.6 * cm) / 4] * 4,
               extra=ex + [("ALIGN", (2, 1), (-1, -1), "CENTER")], zebra=False),
          para("Counts over the 15 country-timelines: 7 full records and 8 matched timelines "
               "(India's full record is its matched timeline).", "small")]
    tot = D.counts[[f"{s}_{k}" for s in P.SPECS for k in ("supported", "settled")]].sum()
    pc = D.pool[D.pool.pool.isin(list(P.SHOCKS) + ["combined"]) & D.pool.hypothesis.isin(MAIN + ["H5"])]
    rows = [["Specification", "Individual countries: supported / settled (of 180)",
             "All countries: supported / settled (of 20)"]]
    for spec in P.SPECS:
        b = pc[pc.spec == spec]
        rows.append([SPEC[spec], f"{int(tot[f'{spec}_supported'])} / {int(tot[f'{spec}_settled'])}",
                     f"{int((b.decision_headline == 'supported').sum())} / "
                     f"{int(b.decision_headline.isin(SETTLED).sum())}"])
    S += [para("5.2 With and without the country-specific effects", "H2"),
          grid(rows, [6.0 * cm, 6.0 * cm, W - 12 * cm], extra=[("ALIGN", (1, 1), (-1, -1), "CENTER")])]
    S += [para("5.3 All countries together versus country by country", "H2"),
          para("The all-country estimate against the simple average of the eight countries' own "
               "estimates on the same timeline (matched, 1999Q2-2021Q4, headline):", "body")]
    m = D.ind[(D.ind.spec == "clean") & (D.ind.timeline == "matched")]
    rows = [["Shock", "Test", "All countries", "Average of 8 countries", "Range across countries",
             "Countries supporting (of 15)"]]
    for s in P.SHOCKS:
        for h in MAIN:
            x = m[(m.pool == s) & (m.hypothesis == h)]
            pe = D.pool[(D.pool.spec == "clean") & (D.pool.pool == s) & (D.pool.hypothesis == h)].iloc[0]
            n15 = D.ind[(D.ind.spec == "clean") & (D.ind.pool == s) & (D.ind.hypothesis == h)]
            rows.append([SHOCK_SHORT[s] if h == "H1" else "", h,
                         f"{SYM[pe.decision_headline]} {num(pe.estimate, h)}", num(x.estimate.mean(), h),
                         f"{num(x.estimate.min(), h)} to {num(x.estimate.max(), h)}",
                         str(int((n15.decision_headline == "supported").sum()))])
    S += [grid(rows, [2.2 * cm, 1.2 * cm, 2.8 * cm, 3.4 * cm, 4.0 * cm, W - 13.6 * cm],
               extra=[("ALIGN", (2, 1), (-1, -1), "CENTER")]),
          para("For H1, H2 and H4 the all-country estimate is essentially the average of the countries. "
               "For H3 it is smaller than the average, because within each quarter it weights each country "
               "by how much its categories differ in exposure. Where countries agree, both analyses agree; "
               "where they disagree (financial-crisis H2) the all-country verdict is undecided; and pooling "
               "detects the COVID-19 recovery gap that some single countries are too small to show.",
               "small")]
    S += [para("5.4 What the results say", "H2")]
    S += bullets([
        "<b>The impact is real and large for the two deep recessions.</b> COVID-19 reduced "
        "consumption in all 15 country-timelines and the financial crisis in 12 (exceptions: "
        "Germany on both timelines and India), and both are supported pooled.",
        "<b>Spending is reallocated, not only cut.</b> In COVID-19 necessities gained share and "
        "contact-intensive categories fell most in every country; in the financial crisis "
        "durable-heavy categories fell most in 10 of 15. Pooled, H2 and H3 hold for COVID-19 and "
        "for the three shocks combined, and H3 for the financial crisis.",
        "<b>Recovery was incomplete after the financial crisis</b> (H4 supported in 12 of 15 and "
        "pooled). For COVID-19 the recovery can only be judged eight quarters out (2021Q4); the "
        "shortfall is supported pooled and in the UK, Germany, Italy and Japan (both timelines) and "
        "Canada (matched), while US spending was back at its trend (full record) or significantly "
        "above it (matched timeline).",
        "<b>The dot-com bust was not a consumption shock.</b> No country shows a significant fall "
        "in consumption (0 of 15), and the pooled H1-H4 tests are all undecided.",
        "<b>Resilience (H5)</b> is proven to make no difference for the dot-com bust and is "
        "undecided for the financial crisis, COVID-19 and all three combined: with eight countries, "
        "the study cannot tell a modest softening effect from none.",
        "<b>The results do not rest on the quarters built from annual data.</b> Using only the four "
        "countries that publish categories every quarter, 10 of the 11 supported pooled results "
        "hold; the eleventh (COVID-19 recovery) keeps the same size but has only three degrees of "
        "freedom. The four benchmarked countries alone also give 10 of 11.",
        "<b>The country-specific effects make the results more credible rather than more "
        "numerous.</b> They change 16 of 180 country cells, remove wrong-sign results that were "
        "caused by uncontrolled events (Japan), and leave the pooled verdicts unchanged."])
    S += [para("5.5 Reading 'undecided' correctly", "H2"),
          para("An undecided cell means the data cannot distinguish the effect from zero or from a "
               "meaningful effect; it is a statement about precision, not evidence that nothing "
               "happened. Where the data can rule out a meaningful effect, the cell says so "
               "explicitly (≈0). Most undecided cells sit where economics predicts weak or "
               "short-lived signals: the dot-com bust (an equity and investment shock), the "
               "necessity shift in the financial crisis (which hit durables rather than the "
               "necessity/non-necessity split), and COVID-19 recovery, which can only be observed "
               "for eight quarters before the 2022 energy shock.", "body")]
    return S


def appendix(D: Data) -> list:
    S = [PageBreak(), para("Appendix A. Everything that was done, step by step", "H1")]
    S += bullets([
        "<b>Country data.</b> The quarterly category panels of the United States, United Kingdom, "
        "Japan, Germany and India were built and verified in earlier country studies; they enter "
        "here as data only (fingerprinted in countries/MANIFEST.csv). Every result in this document "
        "was estimated anew in this study; the earlier studies' results are not used.",
        "<b>Three new countries.</b> France and Italy were built with Germany's exact code, "
        "parameterised by country (it reproduces Germany's panel bit for bit). Canada was built "
        "from Statistics Canada's quarterly table, with exposures measured from its sub-items.",
        "<b>Countries left out.</b> Russia: its only trusted spending-by-purpose accounts cover "
        "2011-2019, which contains none of the three shocks; including it would delete the dot-com "
        "shock for everyone. China: only eight merged categories, no sub-items, and a survey break "
        "in 2013.",
        "<b>Energy dropped as a shock; samples end 2021Q4.</b> The 2022 war and energy regime would "
        "otherwise count as normal quarters or as COVID-19 recovery.",
        "<b>Country effects.</b> The design memo's events were dated, sourced and classified "
        "(C, S, M, X; primary or optional) before estimation; an audit added the structural "
        "regimes, three missing policy responses, the oil-price control and the two robustness "
        "checks; pre-1999 events were screened by a fixed rule (Appendix B).",
        "<b>Four specifications on identical models</b>, so the contribution of the effects is "
        "measured directly, and the headline is fixed in advance (with effects, without policy "
        "responses).",
        "<b>Same framework twice.</b> The individual analysis clusters within a country (by category, "
        "and by time for H1); the pooled analysis adds cross-country clustering and H5. Both are run "
        "shock by shock and with the three shocks together.",
        "<b>Data check.</b> The pooled H1-H4 were re-estimated on the four countries that publish "
        "categories quarterly and, separately, on the four benchmarked ones (section 3.9).",
        "<b>Figures.</b> Three figures for the paper, computed from the same data and code as the "
        "tests (Part 4); the numbers behind them are in results/tables/figure_data_*.csv.",
        "<b>Diagnostics, data files and package.</b> Data and inference diagnostics (Appendix D), "
        "one analysis-data file per country and one all-country file with every variable used "
        "(Appendix E), and a package that rebuilds everything with one command on a fresh "
        "computer (Appendix F)."])
    S += [para("Problems found and how they were resolved", "H2")]
    S += bullets([esc(minus(p)) for p in PROBLEMS])
    S += [para("Appendix B. Pre-1999 events that were tested and rejected", "H1"),
          para("A candidate event for the long records was admitted only if documented, outside the "
               "shock windows, and at least one normal-quarter standard deviation in size. These "
               "failed the size test:", "body")]
    names = {"ice_storm_1998": "Quebec-Ontario ice storm 1998", "vat_1995": "VAT increase August 1995",
             "strikes_1995": "Public-sector strikes November-December 1995",
             "balladurette": "Balladurette car-scrappage premium 1994-95",
             "juppette": "Juppette car-scrappage premium 1995-96",
             "eurotax_1997": "Eurotax (euro-entry levy) 1997"}
    rows = [["Country", "Candidate", "Quarters", "Tested on", "Size (normal-quarter SDs)"]]
    for x in D.rej.itertuples():
        rows.append([COUNTRY[x.country], names.get(x.candidate, x.candidate), x.quarters,
                     esc(x.tested_on), f"{x.quiet_sd:.2f}"])
    S += [grid(rows, [2.3 * cm, 6.0 * cm, 2.8 * cm, 3.6 * cm, W - 14.7 * cm]),
          para("Italy's eurotax looked large (1.92 SD) on total spending, but that deviation was the "
               "car-scrappage boom; excluding transport it is 0.87 SD and positive, so it was not "
               "admitted (the scrappage schemes are included as their own transport events).", "small")]
    S += [para("Appendix C. Consistency checks and reproduction", "H1")]
    hc = D.ind[(D.ind.spec == "clean") & D.ind.hypothesis.isin(MAIN)]
    sup = hc[hc.decision_headline == "supported"]
    only = sup[sup.decision_parametric != "supported"]
    only_txt = "; ".join(f"{COUNTRY[r.country]} {r.timeline} {SHOCK_SHORT[r.pool]} {r.hypothesis}"
                         for r in only.itertuples())
    hr = D.h4rule
    chg = hr[hr.verdict_before != hr.verdict_after]
    chg_txt = "; ".join(f"{COUNTRY.get(r.country, 'all countries')} {r.timeline} {SHOCK_SHORT[r.shock]}: "
                        f"{WORD[r.verdict_before]} to {WORD[r.verdict_after]} (placebo p {r.p_placebo_before:.3f} "
                        f"to {r.p_placebo_after:.3f})" for r in chg.itertuples())
    pc = D.pool[(D.pool.spec == "clean") & D.pool.pool.isin(list(P.SHOCKS) + ["combined"])
                & D.pool.hypothesis.isin(MAIN + ["H5"])]
    pooled_only = int(((pc.decision_headline == "supported") & (pc.decision_parametric != "supported")).sum())
    cb = D.comb[D.comb.spec == "clean"]
    cb_only = int(((cb.decision_headline == "supported") & (cb.decision_parametric != "supported")).sum())
    S += bullets([
        f"<b>Which test decides.</b> The verdict rule fixed before estimation counts a result as supported "
        f"if either the clustered test or the placebo test is significant at 5%. Of the {len(sup)} "
        f"supported individual results (headline, shock by shock), {len(sup) - len(only)} are supported "
        f"by the clustered test itself and {len(only)} by the placebo test alone ({esc(only_txt)}). "
        f"{pooled_only} of the all-country results and {cb_only} of the three-shocks-together results "
        "rest on the placebo test alone.",
        f"<b>Correction of the H4 placebo test.</b> Placebo recovery paths may no longer cross a shock "
        f"window (Appendix A). Comparing the earlier and the corrected rule on the {len(hr)} headline H4 "
        f"results (results/diagnostics/h4_placebo_rule_comparison.csv), {len(chg)} verdicts change: "
        f"{esc(chg_txt)}. Estimates and clustered tests are unaffected.",
        "<b>All-country and individual results agree.</b> For H1, H2 and H4 the all-country estimate "
        "is essentially the simple average of the eight countries' estimates on the same timeline; "
        "for H3 it lies within the countries' range but below their average (section 5.3).",

        "<b>Computation checks</b> (verify.py): input files match their SHA-256 fingerprints; panel "
        "shape and weights; France, Italy and Canada add up to their published totals; the fast "
        "bootstrap equals the reference routine draw for draw; the fast placebo estimates equal "
        "full re-estimation (largest gap 3e-12); the France/Italy code reproduces Germany bit for "
        "bit.",
        "<b>Full reproduction</b>: <font name='DV-M'>python verify.py --all</font> re-estimates "
        "both analyses from the shipped inputs and confirms that every published number reproduces "
        "exactly; this was run on the build machine and again from a fresh copy of the package."])
    return S


def _figure(D: Data, rel: str, frac: float = 1.0) -> Image:
    from reportlab.lib.utils import ImageReader
    path = D.root / rel
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=W * frac, height=W * frac * ih / iw)


def appendix_diagnostics(D: Data) -> list:
    S = [PageBreak(), para("Appendix D. Diagnostics", "H1"),
         para("Checks on the data and on the inference, produced by code/analysis/diagnostics.py and "
              "verify.py; files in results/diagnostics/.", "body"),
         para("D.1 Data coverage", "H2")]
    rows = [["Country", "Categories published", "Full timeline", "Quar-<br/>ters", "First quarter "
             "with 4-quarter growth", "Rows without growth (common)", "Weights sum error",
             "Max error vs published total (%)"]]
    for r in D.cov.itertuples():
        rows.append([COUNTRY[r.country], r.categories_published, r.full_timeline.replace("-", "–"),
                     str(r.full_quarters), r.first_quarter_with_4q_growth,
                     str(r.rows_without_4q_growth_common), f"{r.max_weight_sum_error:.0e}",
                     "n/a" if pd.isna(r.max_reconciliation_error_pc) else f"{r.max_reconciliation_error_pc:.5f}"])
    S += [grid(rows, [2.4 * cm, 2.5 * cm, 2.6 * cm, 1.5 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, W - 15.4 * cm]),
          para("Every country has 12 categories and 91 quarters on the common timeline. The first year "
               "of each record has no four-quarter growth by construction; on the common timeline only "
               "India lacks it (its record starts in 1999Q2, so growth starts in 2000Q2). Spending "
               "weights add to one in every quarter. France, Italy and Canada add up to their published "
               "totals (the other five panels are fingerprinted inputs built and verified in earlier country studies).",
               "small")]
    S += [para("D.2 Spending composition and exposure", "H2"),
          _figure(D, "results/diagnostics/diag_spending_shares.png", 0.95)]
    rows = [["Country", "Necessity share (%)", "Dot-com exposed (%)", "Dot-com effective categories",
             "GFC exposed (%)", "GFC effective categories", "COVID exposed (%)",
             "COVID effective categories"]]
    for r in D.expc.itertuples():
        rows.append([COUNTRY[r.country], f"{r.necessity_share_pc:.1f}", f"{r.dotcom_exposed_share_pc:.1f}",
                     f"{r.dotcom_effective_exposed_categories:.1f}", f"{r.gfc_exposed_share_pc:.1f}",
                     f"{r.gfc_effective_exposed_categories:.1f}", f"{r.covid_exposed_share_pc:.1f}",
                     f"{r.covid_effective_exposed_categories:.1f}"])
    S += [Spacer(1, 4), grid(rows, [2.6 * cm] + [(W - 2.6 * cm) / 7] * 7,
                             extra=[("ALIGN", (1, 1), (-1, -1), "CENTER")]),
          para("Exposed share: spending-weighted average exposure. Effective categories: how many "
               "categories the exposed spending is spread over (inverse Herfindahl). H3 compares "
               "categories, so it has more information where exposure is spread over more categories: "
               "the financial-crisis exposure is concentrated in about two to three categories in "
               "every country, which is one reason its H3 is less precise than COVID-19's.", "small")]
    S += [PageBreak(), para("D.3 Placebo test, all countries together", "H2"),
          _figure(D, "results/diagnostics/diag_placebo_distributions.png", 1.0),
          para("Each grey dot is the same model estimated as if the shock had hit in one normal "
               "two-quarter period (one per year: 11 periods for H1-H3, and 9 for H4, whose 8-16-quarter "
               "placebo paths must not cross a shock window); the red line is the actual estimate. "
               "A test is supported by the placebo when the actual estimate lies beyond almost all "
               "placebo estimates. With 11 periods the smallest attainable placebo p-value is 1/12 = "
               "0.083 (1/10 = 0.100 with 9), so the placebo test corroborates the all-country results "
               "but cannot decide them on its own; the verdicts rest on the clustered tests. All "
               f"{len(pd.read_csv(D.root / 'results/diagnostics/placebo_distributions_pooled.csv'))} "
               "placebo estimates reproduce the published placebo p-values exactly (results/diagnostics/"
               "placebo_distributions_pooled.csv).", "small")]
    S += [para("D.4 Verification", "H2")]
    if D.report:
        keep = [x for x in D.report if x.strip() and not set(x.strip()) <= {"="}]
        S += [para(esc(x).replace(" ", "&nbsp;"), "eq") for x in keep]
    else:
        S += [para("Run <font name='DV-M'>python verify.py --all</font> to produce the full "
                   "verification report.", "body")]
    return S


def appendix_data(D: Data) -> list:
    S = [PageBreak(), para("Appendix E. The data files", "H1"),
         para("Every row and every variable the analyses use is in plain CSV files, written by "
              "code/analysis/data_export.py, which also checks that they equal, value for value, "
              "the data the models read. One row is one spending category of one country in one "
              "quarter, and all files share the same columns.", "body")]
    rows = [["File", "What it holds", "Rows"]]
    rows.append(["data/all_countries_pooled_analysis_data.csv",
                 "The data the all-country analysis ran on: 8 countries x 12 categories x 91 quarters, "
                 "1999Q2-2021Q4", f"{len(D.pooled_data):,}"])
    for c in ORDER:
        rows.append([f"countries/{c}/{c.lower()}_analysis_data.csv",
                     f"{COUNTRY[c]}: its full record to 2021Q4 (the individual analysis; filter "
                     "in_common_timeline = 1 for the matched timeline)", f"{D.country_rows[c]:,}"])
    S += [grid(rows, [6.4 * cm, W - 8.0 * cm, 1.6 * cm], extra=[("ALIGN", (2, 1), (2, -1), "RIGHT")]),
          para("The country events are listed by row in active_events and in full in "
               "data/country_events.csv. The columns, defined in data/data_dictionary.csv:", "body")]
    rows = [["Column", "Description", "Unit"]]
    for r in D.dictionary.itertuples():
        rows.append([f"<font name='DV-M' size='6.8'>{esc(r.column)}</font>", esc(r.description),
                     "" if pd.isna(r.unit) else esc(r.unit)])
    S += [grid(rows, [4.2 * cm, W - 6.4 * cm, 2.2 * cm])]
    return S


def appendix_run(D: Data) -> list:
    S = [PageBreak(), para("Appendix F. How to run the package", "H1"),
         para("<b>Requirements.</b> Python 3.12 or newer (the pinned scipy needs 3.12) and the "
              "packages in requirements.txt. No internet connection is needed: every raw download "
              "is cached in the package. Tested on a fresh Python 3.12 installation and on Python "
              "3.13, with network access blocked.", "body"),
         para("<b>Commands</b> (from the package folder):", "body")]
    S += [para(esc(x).replace(" ", "&nbsp;"), "eq") for x in [
        "python -m pip install -r requirements.txt",
        "python run_all.py              # everything: models, tables, data, diagnostics, figures, reports",
        "python run_all.py --panels     # also rebuilds France, Italy and Canada from the raw files",
        "python verify.py               # quick checks, about one minute",
        "python verify.py --all         # re-runs everything and compares every published number"]]
    S += [para("A full run takes 15-45 minutes, depending on the computer; <font name='DV-M'>verify.py --all"
               "</font> about the same.", "small"),
          para("<b>What is where.</b>", "body")]
    rows = [["Folder or file", "Contents"],
            ["run_all.py", "Runs the whole analysis in order"],
            ["verify.py", "Checks inputs, data and algorithms; with --all re-runs and compares everything"],
            ["code/analysis/", "All analysis code: protocol, models, inference, placebo tests, "
             "robustness, data export, diagnostics, figures, reports"],
            ["code/country_builders/", "The code that builds the France, Italy and Canada panels "
             "from their raw files (Eurostat and Statistics Canada pipelines)"],
            ["countries/USA/ ... India/", "One folder per country: its analysis-data file, its quarterly "
             "panel (data/final/), and for France, "
             "Italy and Canada the raw and intermediate files the panel is built from; Germany also "
             "holds the reproduction check"],
            ["countries/MANIFEST.csv", "SHA-256 fingerprints of every country panel"],
            ["data/all_countries_pooled_analysis_data.csv", "The data the all-country analysis ran on"],
            ["data/data_dictionary.csv", "Definitions of every column of the data files"],
            ["data/country_events.csv", "The 97 country events: dates, type, scope, source"],
            ["data/resilience/", "The pre-shock resilience index and its cached raw sources"],
            ["data/oil_price/", "The cached oil-price and CPI series"],
            ["documentation/", "The Russia note and the pre-1999 events that were tested and rejected"],
            ["results/tables/", "Every result table (CSV)"],
            ["results/figures/", "The three paper figures (PNG and PDF)"],
            ["results/diagnostics/", "Data coverage, exposure, placebo distributions, verification reports"],
            ["results/reports/", "This results document"]]
    S += [grid(rows, [6.6 * cm, W - 6.6 * cm])]
    return S


def main(base: str) -> pathlib.Path:
    root = pathlib.Path(base)
    D = Data(root)
    out = root / "results/reports" / "Results_Individual_and_All_Countries.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Doc(out, title="Consumption Reallocation and Macroeconomic Resilience - Results",
              author="Replication package")
    story = (front(D) + at_a_glance(D) + design(D) + countries(D) + pooled(D) + figures(D)
             + summary(D) + appendix(D) + appendix_diagnostics(D) + appendix_data(D)
             + appendix_run(D))
    doc.multiBuild(story)
    print(f"results document: {out}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent.parent))
