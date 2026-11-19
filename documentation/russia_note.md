# Why Russia is not in the pooled panel

Russia was in the nine-country design. It is excluded because it cannot meet the
study's binding constraint — **one common quarterly timeline for every country,
with the same twelve COICOP divisions from the national accounts** — and
including it would remove the dot-com shock for all eight other countries.

## What was checked, and what was found

| source | what it offers | why it cannot be used |
|---|---|---|
| OECD Table 5 (annual household consumption by COICOP purpose), `DSD_NAMAIN10@DF_TABLE5_T501`, `REF_AREA=RUS` | 12 divisions plus some sub-items | **2011–2019 only.** Contains none of the three shock windows; it ends before COVID. |
| OECD COICOP-2018 table (`DF_TABLE5A_T501`) | — | No Russian data (HTTP 404). |
| Rosstat national-accounts table 6.3, *Фактическое конечное потребление домашних хозяйств* | household actual final consumption, 2000 and 2005–2013 | Broken down by **source of financing** (own expenditure vs social transfers in kind), **not by purpose**. |
| Rosstat household budget survey, *Структура потребительских расходов домашних хозяйств в соответствии с КИПЦ-ДХ* | COICOP-based expenditure structure | Starts **2001** and is a **survey share**, not national accounts — a different concept from the other eight countries. |
| OECD quarterly household final consumption (via FRED `NAEXKP02RUQ661S`, `RUSPFCEQDSMEI`) | quarterly total | Starts **2003Q1** — after the dot-com window opens. |

Rosstat's own site (`rosstat.gov.ru`) fails TLS verification from outside Russia
because it uses a Russian national certificate authority; the federal data portal
(`fedstat.ru`) returned HTTP 403. Rosstat pages were read with certificate
verification disabled solely to establish what exists; no Russian number enters
any estimate.

## Consequence of including it anyway

The latest series start among the other eight is India's, 1999Q2. Russia's earliest
national-accounts quarterly total is 2003Q1 and its earliest by-purpose national
accounts on a trusted source are 2011. A common timeline that included Russia
would therefore begin no earlier than 2003–2011, deleting the **dot-com** shock
(window 2000Q1–2002Q4) for every country and, on the trusted source, the financial
crisis as well. Russia's own episodes are also the most confounded in the design —
the post-1998 recovery during dot-com, the oil collapse and capital reversal during
the financial crisis, and the 2020 oil collapse during COVID.

The pooled panel is therefore **eight countries on one common timeline,
1999Q2–2021Q4**, and every pooled result applies to those eight.
