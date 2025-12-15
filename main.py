"""
Main entry point for the Sustainability & Economic Performance project.
Runs the data preparation + country selection pipeline.
"""

from src.data_preparation import build_merged_dataset
from src.country_selection import build_50_country_panel


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


if __name__ == "__main__":
    main()