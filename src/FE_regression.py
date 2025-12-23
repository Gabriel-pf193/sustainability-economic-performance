"""
MODULE 4: Fixed-Effects Regression

This module:
    1) Builds the regression dataset by:
        a) Standardizing every ESG indicator
        b) changing the direction of the indicators when necessary
           so that in increase in an indicators means better ESG performance
        c) Building an index for each ESG category
           (one index for environment, another for social, and one last for governance)
        d) Combine those indexes with the economic indicators
    2) Save the new dataset to data/processed/panel_FE_regression.csv
    3) Run Country + Year fixed effects regression (clustered SE by country)
    4) Save the regression table to results/regression/fixed_effects_regression.tex
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_REG_DIR = PROJECT_ROOT / "results" / "regression"

PANEL_50_PATH = PROCESSED_DIR / "panel_50_countries.csv"
FE_DATASET_PATH = PROCESSED_DIR / "panel_FE_regression.csv"
FE_TABLE_TEX_PATH = RESULTS_REG_DIR / "fixed_effects_regression.tex"

# ---------- ESG Category + Direction ----------
"""
ESG map (category + direction)
+1 = higher value is better
-1 = higher value is worse

List of economic indicators
"""

ESG_MAP = {
    # --- Environmental (E) ---
    "CO2 emissions (metric tons per capita)": {"category": "E", "direction": -1},
    "Methane emissions (metric tons of CO2 equivalent per capita)": {"category": "E", "direction": -1},
    "Nitrous oxide emissions (metric tons of CO2 equivalent per capita)": {"category": "E", "direction": -1},
    "Fossil fuel energy consumption (% of total)": {"category": "E", "direction": -1},
    "Renewable energy consumption (% of total final energy consumption)": {"category": "E", "direction": +1},
    "Renewable electricity output (% of total electricity output)": {"category": "E", "direction": +1},
    # --- Social (S) ---
    "Unemployment, total (% of total labor force) (modeled ILO estimate)": {"category": "S", "direction": -1},
    "Gini index": {"category": "S", "direction": -1},
    "Economic and Social Rights Performance Score": {"category": "S", "direction": +1},
    # --- Governance (G) ---
    "Control of Corruption: Estimate": {"category": "G", "direction": +1},
    "Political Stability and Absence of Violence/Terrorism: Estimate": {"category": "G", "direction": +1},
}

ECON_INDICATORS = [
    "GDP growth (annual %)",
    "GDP per capita (constant 2015 US$)",
    "Inflation, consumer prices (annual %)",
    "Foreign direct investment, net inflows (% of GDP)",
    "Research and development expenditure (% of GDP)",
]

# ---------- Build the regression dataset ----------
def load_panel_50(path: Path = PANEL_50_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find panel_50_countries.csv at: {path}")
    return pd.read_csv(path)

def build_esg_indices(df_50: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (Country Name, Country Code, Year) with:
    ENV_index, SOC_index, GOV_index
    """
    df_esg = df_50[df_50["Indicator"].isin(ESG_MAP.keys())].copy()

    # direction and signed values
    df_esg["direction"] = df_esg["Indicator"].map(lambda x: ESG_MAP[x]["direction"])
    df_esg["value_signed"] = df_esg["Value"] * df_esg["direction"]

    # z-score within each indicator (standardization)
    df_esg["value_z"] = (
        df_esg.groupby("Indicator")["value_signed"]
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    # map to ESG category
    df_esg["ESG_category"] = df_esg["Indicator"].map(lambda x: ESG_MAP[x]["category"])

    # average z within category (skipna by default via mean)
    esg_indices = (
        df_esg.groupby(["Country Name", "Country Code", "Year", "ESG_category"])["value_z"]
        .mean()
        .reset_index()
    )

    # pivot to wide format
    esg_wide = (
        esg_indices.pivot(
            index=["Country Name", "Country Code", "Year"],
            columns="ESG_category",
            values="value_z",
        )
        .reset_index()
    )
    esg_wide.columns.name = None

    esg_wide = esg_wide.rename(columns={"E": "ENV_index", "S": "SOC_index", "G": "GOV_index"})
    esg_wide = esg_wide.sort_values(["Country Code", "Year"]).reset_index(drop=True)
    return esg_wide
    
def build_econ_wide(df_50: pd.DataFrame) -> pd.DataFrame:
    df_econ = (
        df_50[df_50["Indicator"].isin(ECON_INDICATORS)][
            ["Country Name", "Country Code", "Year", "Indicator", "Value", "Region", "Income Group"]
        ]
        .copy()
    )

    econ_wide = (
        df_econ.pivot_table(
            index=["Country Name", "Country Code", "Year", "Region", "Income Group"],
            columns="Indicator",
            values="Value",
            aggfunc="mean",
        )
        .reset_index()
    )
    econ_wide.columns.name = None

    econ_wide = econ_wide.rename(
        columns={
            "GDP growth (annual %)": "gdp_growth",
            "GDP per capita (constant 2015 US$)": "gdp_per_capita",
            "Inflation, consumer prices (annual %)": "inflation",
            "Foreign direct investment, net inflows (% of GDP)": "fdi_inflows",
            "Research and development expenditure (% of GDP)": "R&D_expenditure",
        }
    )
    return econ_wide

def build_regression_dataset(df_50: pd.DataFrame) -> pd.DataFrame:
    esg_wide = build_esg_indices(df_50)
    econ_wide = build_econ_wide(df_50)

    reg_df = econ_wide.merge(
        esg_wide,
        on=["Country Name", "Country Code", "Year"],
        how="left",
    ).sort_values(["Country Code", "Year"]).reset_index(drop=True)

    # rename the country code column (remove the space) so the regression code works
    reg_df = reg_df.rename(columns={"Country Code": "country_code"})
    return reg_df

def save_regression_dataset(reg_df: pd.DataFrame, path: Path = FE_DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reg_df.to_csv(path, index=False)
    print(f"FE regression dataset saved to: {path}")

def run_country_year_fe(reg_df: pd.DataFrame):
    # Keep only vars needed for the base FE model
    fe_df = reg_df[["country_code", "Year", "gdp_growth", "ENV_index", "SOC_index", "GOV_index"]].dropna()

    model = smf.ols(
        formula="""
            gdp_growth ~ ENV_index + SOC_index + GOV_index
            + C(country_code)
            + C(Year)
        """,
        data=fe_df,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": fe_df["country_code"]},
    )

    return model
    
def save_regression_table_tex(model, path: Path = FE_TABLE_TEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latex_table = model.summary().as_latex()
    path.write_text(latex_table, encoding="utf-8")
    print(f"Regression table saved to: {path}")


# =========================
# Run
# =========================
def run_fe_regression() -> None:
    print(f"Reading: {PANEL_50_PATH}")
    df_50 = load_panel_50(PANEL_50_PATH)
    print("Shape:", df_50.shape)

    reg_df = build_regression_dataset(df_50)
    print("Regression dataset shape:", reg_df.shape)
    print("Columns:", list(reg_df.columns))

    save_regression_dataset(reg_df, FE_DATASET_PATH)

    model = run_country_year_fe(reg_df)
    print(model.summary())

    save_regression_table_tex(model, FE_TABLE_TEX_PATH)


if __name__ == "__main__":
    run_fe_regression()