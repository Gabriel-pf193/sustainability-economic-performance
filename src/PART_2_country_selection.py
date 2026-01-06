"""
Module 2: Selection of the sample of 50 countries for the analysis

This module:
- loads the full merged panel created in PART_1_data_preparation.py
- filters it down to the final sample of 50 countries
- saves the 50-country panel for later modeling
- saves summary tables as HTML in results/country_distribution
- saves two key figures as PDFs:
    * Percentage of missing values by indicator
    * Correlation heatmap with all indicators

Important:
- Rows with missing values are NOT deleted from the dataset here.
- Missing values will be handled later, separately for:
    * the fixed-effects regression
    * the machine learning models
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "country_selection"

FILE_FULL_PANEL = PROCESSED_DIR / "panel_full_unfiltered.csv"
FILE_PANEL_50 = PROCESSED_DIR / "panel_50_countries.csv"


# ---------- Loading ----------
def load_full_panel() -> pd.DataFrame:
    """
    Load the full merged panel dataset (all countries, all indicators).
    """
    return pd.read_csv(FILE_FULL_PANEL)

# ---------- List of selected countries (by name) ----------

""" 
This sample has been established by choosing countries so that every region 
(and even sub-regions, for example in Europe, I didn't only choose occidental countries, 
I also made sure to represent various regions of of Europe) and income level across the world is represented, while accounting for missing values (I did not choose a country with a percentage of data availability below 70%).
"""

SELECTED_COUNTRY_NAMES = [
    # North America
    "United States", "Canada",

    # South Asia
    "India", "Sri Lanka", "Nepal", "Bangladesh", "Maldives", "Bhutan",

    # MENA + Afghanistan & Pakistan
    "Israel", "Iran, Islamic Rep.", "Egypt, Arab Rep.", "Tunisia",
    "Saudi Arabia", "Pakistan", "Algeria",

    # Latin America & Caribbean
    "Brazil", "Colombia", "Mexico", "Costa Rica", "Uruguay", "Chile",
    "Honduras", "Bolivia", "Dominican Republic", "Peru",

    # East Asia & Pacific
    "Japan", "Korea, Rep.", "Australia", "China", "Indonesia",
    "Viet Nam", "Philippines", "Cambodia",

    # Sub-Saharan Africa
    "South Africa", "Mauritius", "Nigeria", "Ghana", "Kenya",
    "Madagascar", "Rwanda", "Burkina Faso",

    # Europe & Central Asia
    "Germany", "France", "United Kingdom", "Poland", "Romania",
    "Hungary", "Georgia", "Kazakhstan", "Uzbekistan",
]

# ---------- Build the new 50-country dataset ----------
def select_50_countries(df: pd.DataFrame, selected_names: list[str] | None = None) -> pd.DataFrame:
    """
    Filter the full panel to keep only the selected countries.
    """
    if selected_names is None:
        selected_names = SELECTED_COUNTRY_NAMES

    if not selected_names:
        raise ValueError("SELECTED_COUNTRY_NAMES is empty.")

    df_50 = df[df["Country Name"].isin(selected_names)].copy()

    # Sanity check on the number of unique countries
    n_unique = df_50["Country Name"].nunique()
    print(f"\nNumber of unique countries in filtered panel: {n_unique}")

    return df_50


def build_50_country_panel(save: bool = True, filename: str = "panel_50_countries.csv") -> pd.DataFrame:
    
    df_full = load_full_panel()
    df_50 = select_50_countries(df_full)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / filename
        df_50.to_csv(out_path, index=False)

    return df_50

# ---------- Tables (saved as HTML) ----------
def make_region_income_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Region x Income Group crosstab from a panel dataset.
    """
    countries_unique = df[["Country Name", "Region", "Income Group"]].drop_duplicates()

    table = pd.crosstab(
        countries_unique["Region"],
        countries_unique["Income Group"],
        margins=True,
        margins_name="Total",
    )
    table.index.name = None
    table.columns.name = None
    return table

def get_selected_countries_metadata(df_50: pd.DataFrame) -> pd.DataFrame:
    """
    Table listing the selected countries with:
    - Country Name, Country Code, Region, Income Group
    """
    meta = (
        df_50[["Country Name", "Country Code", "Region", "Income Group"]]
        .drop_duplicates()
        .sort_values(["Country Name"])
        .reset_index(drop=True)
    )
    meta.index = meta.index + 1  # start index at 1
    return meta

def save_table_html(df: pd.DataFrame, out_path: Path, title: str | None = None) -> None:
    """
    Save the tables as HTML files.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_table = df.to_html(border=0, justify="center")

    if title is None:
        title = out_path.stem.replace("_", " ").title()

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #999; padding: 6px 10px; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
<h2>{title}</h2>
{html_table}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

# ---------- Graphs (save as PDF) ----------
_INDICATOR_SHORT = {
    "Economic and Social Rights Performance Score": "Social Rights Score",
    "Gini index": "Gini index",
    "Research and development expenditure (% of GDP)": "R&D Expenditure (% GDP)",
    "Inflation, consumer prices (annual %)": "Inflation (annual %)",
    "Methane emissions (metric tons of CO2 equivalent per capita)": "Methane emissions (CO2 tons/cap)",
    "CO2 emissions (metric tons per capita)": "CO2 emissions (tons/cap)",
    "Nitrous oxide emissions (metric tons of CO2 equivalent per capita)": "Nitrous oxide emissions (CO2 tons/cap)",
    "Renewable electricity output (% of total electricity output)": "Renewable Elec (%)",
    "Renewable energy consumption (% of total final energy consumption)": "Renew. Energy (%)",
    "Fossil fuel energy consumption (% of total)": "Fossil Fuel consumption (%)",
    "Political Stability and Absence of Violence/Terrorism: Estimate": "Political Stability",
    "Control of Corruption: Estimate": "Corruption Control",
    "Foreign direct investment, net inflows (% of GDP)": "FDI (% GDP)",
    "GDP per capita (constant 2015 US$)": "GDP per capita",
    "GDP growth (annual %)": "GDP Growth (annual %)",
    "Unemployment, total (% of total labor force) (modeled ILO estimate)": "Unemployment (% labor force)",
}


def save_missing_values_histogram(df_50: pd.DataFrame, out_path: Path) -> None:
    """
    Bar chart of missing value percentage by indicator (saved as PDF).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing_pct = (
        df_50.groupby("Indicator")["Value"]
        .apply(lambda s: s.isna().mean() * 100)
        .sort_values(ascending=False)
    )

    missing_pct_short = missing_pct.rename(index=_INDICATOR_SHORT)

    # Horizontal bar chart (readability)
    fig, ax = plt.subplots(figsize=(12, 8))
    missing_pct_short.sort_values(ascending=True).plot(kind="bar", ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("Missing data (%)", fontsize = 14)
    ax.grid(axis="y", linestyle="--", alpha=1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def save_correlation_heatmap(df_50: pd.DataFrame, out_path: Path) -> None:
    """
    Correlation heatmap of all indicators (saved as pdf)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_wide = df_50.pivot_table(
        index=["Country Name", "Year"],
        columns="Indicator",
        values="Value",
        aggfunc="mean",
    ).rename(columns=_INDICATOR_SHORT)

    corr = df_wide.corr()

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        corr.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm" 
    )

    n = corr.shape[0]
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))

    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()

    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------- Main function for main.py ----------
def run_country_selection() -> pd.DataFrame:
    """
    Function to call from main.py.

    It will:
    1) build and save panel_50_countries.csv
    2) save the three tables as HTML
    3) save the two graphs as PDF
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load full (192-country) panel for tables, then build 50-country panel
    df_full = load_full_panel()
    df_50 = build_50_country_panel(save=True, filename="panel_50_countries.csv")

    # ---- Tables (HTML) ----
    table_192 = make_region_income_table(df_full)
    save_table_html(table_192, RESULTS_DIR / "region_income_table_192.html", title="Region x Income Group (All Countries)")

    table_50 = make_region_income_table(df_50)
    save_table_html(table_50, RESULTS_DIR / "region_income_table_50.html", title="Region x Income Group (50-country sample)")

    meta_50 = get_selected_countries_metadata(df_50)
    save_table_html(meta_50, RESULTS_DIR / "selected_countries_50.html", title="Selected countries (50-country sample)")

    # ---- Figures (PDF) ----
    save_missing_values_histogram(df_50, RESULTS_DIR / "hist_missing_values.pdf")
    save_correlation_heatmap(df_50, RESULTS_DIR / "correlation_heatmap.pdf")

    return df_50


# ---------- Test execution ----------
if __name__ == "__main__":
    run_country_selection()