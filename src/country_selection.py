"""
Module 2: Selection of the sample of 50 countries for the analysis

This module:
- loads the full merged panel created in data_preparation.py
- filters it down to the final sample of 50 countries
- saves the 50-country panel for later modeling

Important:
- Rows with missing values are NOT deleted from the dataset here.
- Missing values will be handled later, separately for:
    * the fixed-effects regression
    * the machine learning models
"""

from pathlib import Path
import pandas as pd


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FILE_FULL_PANEL = PROCESSED_DIR / "panel_full_unfiltered.csv"
FILE_PANEL_50 = PROCESSED_DIR / "panel_50_countries.csv"


# ---------- Loading ----------

def load_full_panel() -> pd.DataFrame:
    """
    Load the full merged panel dataset (all countries, all indicators).
    """
    df = pd.read_csv(FILE_FULL_PANEL)
    return df


# ---------- List of selected countries (by name) ----------

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

# ---------- build the new 50-country dataset ----------

def select_50_countries(df: pd.DataFrame,
                        selected_names: list[str] | None = None) -> pd.DataFrame:
    """
    Filter the full panel to keep only the 50 selected countries (by Country Name).
    """
    if selected_names is None:
        selected_names = SELECTED_COUNTRY_NAMES

    if not selected_names:
        raise ValueError(
            "SELECTED_COUNTRY_NAMES is empty. "
        )

    df_50 = df[df["Country Name"].isin(selected_names)].copy()

    # Sanity check on the number of unique countries
    n_unique = df_50["Country Name"].nunique()
    print(f"Number of unique countries in filtered panel: {n_unique}")

    return df_50


def build_50_country_panel(
    save: bool = True,
    filename: str = "panel_50_countries.csv",
) -> pd.DataFrame:
    """
    Main function for Module 2:

    - load the full merged panel
    - filter it to the selected 50 countries
    - optionally save the filtered dataset to data/processed
    """
    df_full = load_full_panel()
    df_50 = select_50_countries(df_full)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / filename
        df_50.to_csv(out_path, index=False)
        print(f"Saved 50-country panel to: {out_path}")

    return df_50

# ---------- Build tables ----------

def make_region_income_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Region x Income Group crosstab from a panel dataset.

    Can be used for:
    - all 192 countries
    - the 50 selected countries
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
    Return a deduplicated table listing the selected countries with:
    - Country Name
    - Country Code
    - Region
    - Income Group

    """
    meta = (
        df_50[["Country Name", "Country Code", "Region", "Income Group"]]
        .drop_duplicates()
        .sort_values(["Region", "Income Group", "Country Name"])
        .reset_index(drop=True)
    )
    meta.index = meta.index + 1  # start index at 1
    return meta


# ---------- Test execution ----------

if __name__ == "__main__":
    print("Loading full panel...")
    df_full = load_full_panel()
    print("Full panel shape:", df_full.shape)

    print("\nFiltering to selected countries...")
    try:
        df_50 = build_50_country_panel(save=True)
        print("50-country panel shape:", df_50.shape)

        print("\nUnique countries in 50-country panel:")
        print(df_50[["Country Name", "Country Code"]].drop_duplicates().head(50))

        # Optional: Region x Income for the 50-country panel
        region_income_50 = make_region_income_table(df_50)
        print("\nRegion x Income Group table (50 countries):")
        print(region_income_50)

    except ValueError as e:
        print("\nERROR:", e)
        print("Please check SELECTED_COUNTRY_NAMES.")