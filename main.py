"""
Main script for the Sustainability & Economic Performance project.
"""

from src.data_preparation import build_merged_dataset
from src.country_selection import build_50_country_panel
from src.FE_regression import run_fe_regression


def main():
    print("=" * 70)
    print("MODULE 1 — Data preparation (build merged dataset)")
    print("=" * 70)

    df_full = build_merged_dataset(save=True, filename="panel_full_unfiltered.csv")
    print("Merged dataset shape:", df_full.shape)

    print("\n" + "=" * 70)
    print("MODULE 2 — Country selection (build 50-country panel)")
    print("=" * 70)

    df_50 = build_50_country_panel(save=True, filename="panel_50_countries.csv")
    print("50-country panel shape:", df_50.shape)

    print("\n" + "=" * 70)
    print("MODULE 3 — Fixed-effects regression (country + year FE)")
    print("=" * 70)

    run_fe_regression()

if __name__ == "__main__":
    main()