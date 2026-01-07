"""
Main script for the Sustainability & Economic Performance project.
"""

from src.PART_1_data_preparation import build_merged_dataset
from src.PART_2_country_selection import run_country_selection
from src.PART_3_FE_regression import run_fe_regression
from src.PART_4_ML_models import run_ml_models


def main():

    # ---------- MODULE 1: Data preparation ----------
    
    print("=" * 70)
    print("MODULE 1 — Data preparation (build merged dataset)")
    print("=" * 70)

    print("Merged dataset created")
    df_full = build_merged_dataset(save=True, filename="panel_full_unfiltered.csv")
    print("Saved full merged dataset (PDF)")
    print("Merged dataset shape:", df_full.shape)

    # ---------- MODULE 2: Country selection ----------
    
    print("\n" + "=" * 70)
    print("MODULE 2 — Country selection (build 50-country panel)")
    print("=" * 70)

    print(
    "Saved outputs:\n"
    "- 50-country panel dataset (CSV)\n"
    "- Country distribution tables (HTML)\n"
    "- Percentage of missing values histogram (PDF)\n"
    "- Correlation heatmap (PDF)"
    )
    df_50 = run_country_selection()
    print("50-country panel shape:", df_50.shape)

    # ---------- MODULE 3: Fixed-Effects Regression ----------
    
    print("\n" + "=" * 70)
    print("MODULE 3 — Fixed-effects regression (country + year FE)")
    print("=" * 70)

    print("Regression dataset created")
    run_fe_regression()

    # ---------- MODULE 4: Machine Learning models ----------
    
    print("\n" + "=" * 70)
    print("MODULE 4 — Machine Learning models")
    print("=" * 70)

    run_ml_models()

if __name__ == "__main__":
    main()