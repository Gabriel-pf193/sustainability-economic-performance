"""
MODULE 1: Data preparation for the project.

This module:
- loads the three raw datasets located under data/raw
- cleans each dataset (reshaping, renaming, etc.)
- merges them into a single dataset
- saves the merged dataset under data/processed

This will provide a country-year panel of roughly 190 countries, from which I will later choose a sample of 50 countries according to data availability and region/income level representation.
"""

from pathlib import Path
import pandas as pd

# --------------------- Paths ---------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Raw files
FILE_ESG = RAW_DIR / "esg-economic-data.csv"
FILE_GDP = RAW_DIR / "gdp-inflation-fdi-data.csv"
FILE_CLASS = RAW_DIR / "country-classification.xlsx"

# --------------------- Loading -------------------

def load_raw_data():
    """Load the three raw datasets from data/raw."""

    df_esg = pd.read_csv(FILE_ESG)
    df_gdp = pd.read_csv(FILE_GDP)
    df_class = pd.read_excel(FILE_CLASS)
    return df_esg, df_gdp, df_class

# ---------------------- Assign Category to ESG and Economic indicators ----------------------

def assign_category(indicator_name) -> str:
    """Assign Environmental / Social / Governance / Economic / Other to the indicators of the first dataset."""
    indicator_name_lower = str(indicator_name).lower()

    # Environmental
    if "co2" in indicator_name_lower:
        return "Environmental"
    if "fossil" in indicator_name_lower:
        return "Environmental"
    if "renewable" in indicator_name_lower:
        return "Environmental"
    if "methane" in indicator_name_lower:
        return "Environmental"
    if "nitrous" in indicator_name_lower:
        return "Environmental"

    # Social
    if "unemployment" in indicator_name_lower:
        return "Social"
    if "gini" in indicator_name_lower:
        return "Social"
    if "rights" in indicator_name_lower:
        return "Social"

    # Governance
    if "corruption" in indicator_name_lower:
        return "Governance"
    if "political" in indicator_name_lower:
        return "Governance"

    # Economic
    if "gdp" in indicator_name_lower:
        return "Economic"
    if "expenditure" in indicator_name_lower:
        return "Economic"

    return "Other"

def assign_economic_category(indicator_name) -> str:
    """Assign categories to GDP/Inflation/FDI indicators."""
    indicator_name_lower = str(indicator_name).lower()

    # Economic
    if "gdp" in indicator_name_lower:
        return "Economic"
    if "inflation" in indicator_name_lower:
        return "Economic"
    if "foreign direct investment" in indicator_name_lower:
        return "Economic"
    if "research" in indicator_name_lower or "r&d" in indicator_name_lower:
        return "Economic"

    return "Other"

# ------------------ Cleaning the datasets ------------------

def clean_esg_dataset(esg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and reshape the ESG dataset from wide to long format,
    and add a Category and Source column.
    """
    # drop series code column (not useful for the project)
    esg_clean = esg_df.drop(columns=["Series Code"])

    # reshape the data from wide to long format
    year_cols = [col for col in esg_clean.columns if "[" in col]
    esg_long = esg_clean.melt(
        id_vars=["Country Name", "Country Code", "Series Name"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value",
    )

    # rename Series Name column to Indicator
    esg_long = esg_long.rename(columns={"Series Name": "Indicator"})

    # clean Year: keep only 4 digits and convert to int
    esg_long["Year"] = esg_long["Year"].str.slice(0, 4)
    esg_long["Year"] = esg_long["Year"].astype(int)

    # assign ESG categories
    esg_long["Category"] = esg_long["Indicator"].apply(assign_category)

    # drop rows where Indicator is NaN
    esg_long = esg_long.dropna(subset=["Indicator"])

    # label source
    esg_long["Source"] = "ESG"

    return esg_long

def clean_gdp_dataset(gdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and reshape the GDP/Inflation/FDI dataset from wide to long format,
    and add Category and Source columns.
    """
    # drop series code column (not useful for the project)
    gdp_clean = gdp_df.drop(columns=["Series Code"])

    # reshape dataset from wide to long format
    year_cols_gdp = [col for col in gdp_clean.columns if "[" in col]
    gdp_long = gdp_clean.melt(
        id_vars=["Country Name", "Country Code", "Series Name"],
        value_vars=year_cols_gdp,
        var_name="Year",
        value_name="Value",
    )

    # rename Series Name column to Indicator
    gdp_long = gdp_long.rename(columns={"Series Name": "Indicator"})

    # clean Year and convert to int
    gdp_long["Year"] = gdp_long["Year"].str.slice(0, 4)
    gdp_long["Year"] = gdp_long["Year"].astype(int)

    # assign Economic-only categories
    gdp_long["Category"] = gdp_long["Indicator"].apply(assign_economic_category)

    # drop rows where Indicator is NaN
    gdp_long = gdp_long.dropna(subset=["Indicator"])

    # label source
    gdp_long["Source"] = "Economic"

    return gdp_long

def prepare_country_classification(class_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the country classification table:
    - rename columns to match the other datasets, so they later merge correctly
    - keep only the columns useful for the project (lending category not useful)
    """
    class_clean = class_df.rename(
        columns={
            "Economy": "Country Name",
            "Code": "Country Code",
            "Income group": "Income Group",
            "Lending category": "Lending Category",
        }
    )[["Country Code", "Country Name", "Region", "Income Group"]]

    return class_clean

# ------------------------- Merging the datasets -------------------------

def merge_to_panel(
    esg_long: pd.DataFrame,
    gdp_long: pd.DataFrame,
    class_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the cleaned ESG + GDP datasets with the country classification table
    into one long panel DataFrame.
    """
    # ensure the column order is the same, before concatenation
    common_cols = [
        "Country Name",
        "Country Code",
        "Indicator",
        "Year",
        "Value",
        "Category",
        "Source",
    ]
    esg_long = esg_long[common_cols]
    gdp_long = gdp_long[common_cols]

    # combine ESG + GDP
    all_long = pd.concat([esg_long, gdp_long], ignore_index=True)

    # sort for neatness
    all_long = all_long.sort_values(
        ["Country Code", "Year", "Category"]
    ).reset_index(drop=True)

    # merge with the country classification dataset
    panel_long = all_long.merge(
        class_clean,
        on="Country Code",
        how="left",
    )

    # clean up duplicate country name columns
    if "Country Name_x" in panel_long.columns:
        panel_long = panel_long.rename(columns={"Country Name_x": "Country Name"})
        panel_long = panel_long.drop(columns=["Country Name_y"])

    # Convert Value column to numeric (turn "." into NaN)
    panel_long["Value"] = pd.to_numeric(panel_long["Value"], errors="coerce")

    return panel_long

# ------------- Main data preparation function -------------

def build_merged_dataset(
    save: bool = True,
    filename: str = "panel_full_unfiltered.csv",
) -> pd.DataFrame:
    """
    High-level function that:
    - loads raw data
    - cleans and reshapes ESG + GDP
    - prepares country classification
    - merges everything into one panel_long DataFrame
    - saves the new dataset to data/processed
    """
    df_esg, df_gdp, df_class = load_raw_data()

    esg_long = clean_esg_dataset(df_esg)
    gdp_long = clean_gdp_dataset(df_gdp)
    class_clean = prepare_country_classification(df_class)

    panel_long = merge_to_panel(esg_long, gdp_long, class_clean)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if save:
        out_path = PROCESSED_DIR / filename
        panel_long.to_csv(out_path, index=False)
        print(f"Saved merged dataset to: {out_path}")

    return panel_long

if __name__ == "__main__":
    df = build_merged_dataset(save=True)
    print(df.head(10))