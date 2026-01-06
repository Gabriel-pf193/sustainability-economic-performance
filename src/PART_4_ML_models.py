"""
MODULE 4: Machine Learning models

This module:
    1) Loads the FE regression dataset
    2) Drops any row with missing values, so the models run correctly
    3) Trains + evaluates 3 models (Linear Regression, Random Forest, Gradient Boosting)
       - Train/Test metrics (R², RMSE)
       - Cross-validation on train only (mean ± std)
    4) Saves feature importance histograms and learning curves for Random Forest + Gradient Boosting
    5) Saves a model comparison table as a PDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ---------- Define the target and features for the models ----------

Y_COL = "gdp_growth"
X_COLS = [
    "ENV_index",
    "SOC_index",
    "GOV_index",
    "gdp_per_capita",
    "inflation",
    "fdi_inflows",
    "R&D_expenditure",
]

# Set up random state for reproductibility, test split, and cross-validation configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5


def get_project_root() -> Path:
    """
    Returns the repository root (folder that contains 'data' and 'src').
    """
    return Path(__file__).resolve().parents[1]


def ensure_dirs(project_root: Path) -> Dict[str, Path]:
    """
    Creates needed output directories and returns common paths.
    """
    data_processed = project_root / "data" / "processed"
    results_ml = project_root / "results" / "machine_learning"

    data_processed.mkdir(parents=True, exist_ok=True)
    results_ml.mkdir(parents=True, exist_ok=True)

    return {
        "data_processed": data_processed,
        "results_ml": results_ml,
    }


# ---------- Build the dataset for the ML models ----------

# Load the previous dataset
def load_fe_dataset(project_root: Path) -> pd.DataFrame:
    """
    Loads the same dataset used for the FE regression (which is in wide format).
    """
    path = project_root / "data" / "processed" / "panel_FE_regression.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find FE regression dataset at: {path}\n"
        )
    return pd.read_csv(path)


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the ML dataset by keeping X_COLS (features) + Y_COL (target) and dropping rows with any missing value.
    """
    required = X_COLS + [Y_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input dataset: {missing_cols}")

    ml_df = df[required].dropna().copy()
    return ml_df

# Save the dataset
def save_ml_dataset(ml_df: pd.DataFrame, out_path: Path) -> None:
    ml_df.to_csv(out_path, index=False)


# ---------- Set up performance metrics ----------
    """
    This section sets up the performance metrics that will be used later in the code to evaluate the machine 
    learning models. It also creates a standardized print block so the results of the three models are printed 
    in the same format.
    """

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_scores(
    model,
    X_train,
    y_train,
    *,
    scoring_r2: str = "r2",
    scoring_rmse: str = "neg_root_mean_squared_error",
    n_splits: int = N_SPLITS_CV,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = None,
) -> Dict[str, float]:
    """
    Cross-validation on TRAIN only.
    Returns mean/std for R² and RMSE.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_vals = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_r2, n_jobs=n_jobs)
    rmse_neg_vals = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_rmse, n_jobs=n_jobs)
    rmse_vals = -rmse_neg_vals

    return {
        "cv_r2_mean": float(r2_vals.mean()),
        "cv_r2_std": float(r2_vals.std()),
        "cv_rmse_mean": float(rmse_vals.mean()),
        "cv_rmse_std": float(rmse_vals.std()),
    }


def train_test_scores(model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    """
    Fit on train, evaluate on train and test.
    """
    model.fit(X_train, y_train)

    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    return {
        "train_r2": float(r2_score(y_train, yhat_train)),
        "train_rmse": rmse(y_train, yhat_train),
        "test_r2": float(r2_score(y_test, yhat_test)),
        "test_rmse": rmse(y_test, yhat_test),
    }


def print_model_block(model_name: str, cv_dict: Dict[str, float], tt_dict: Dict[str, float]) -> None:
    print(f"\n{model_name.upper()} RESULTS\n")
    print("Cross-Validation (train only):")
    print(f"R² mean = {cv_dict['cv_r2_mean']:.3f} (std = {cv_dict['cv_r2_std']:.3f})")
    print(f"RMSE mean = {cv_dict['cv_rmse_mean']:.3f} (std = {cv_dict['cv_rmse_std']:.3f})\n")
    print("TRAIN:")
    print(f"R² = {tt_dict['train_r2']:.3f}, RMSE = {tt_dict['train_rmse']:.3f}")
    print("\nTEST:")
    print(f"R² = {tt_dict['test_r2']:.3f}, RMSE = {tt_dict['test_rmse']:.3f}")


# ---------- Create and save the feature importance histograms and model comparison table ----------

def save_feature_importance_pdf(model, feature_names, out_path: Path, title: str) -> None:
    """
    Saves a feature importance histogram to a PDF for both Random Forest and Gradient Boosting models.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model has no feature_importances_ attribute.")

    fi = pd.Series(importances * 100, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    fi.plot(kind="bar", ax=ax)

    ax.set_ylabel("Feature importance (%)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_table_pdf(results: pd.DataFrame, out_path: Path, wrap_width: int = 12) -> None:
    """
    Saves the model comparison table as a PDF (matplotlib table).
    """
    import textwrap

    def wrap_label(s: str, width: int) -> str:                   # Wraps columns labels if they're too long
        return "\n".join(textwrap.wrap(str(s), width=width))

    results = results.reset_index(drop=True)

    col_labels = [wrap_label(c, wrap_width) for c in results.columns]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=results.round(3).values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------- Create and save the learning curves for Random Forest and Gradient Boosting models ----------

def save_learning_curve_pdf(
    model,
    X_train,
    y_train,
    out_path: Path,
    title: str,
    n_splits: int = N_SPLITS_CV,
    random_state: int = RANDOM_STATE,
) -> None:
    """
    Saves a learning curve (train vs validation R²) as a PDF.
    Uses CV on the TRAIN set only.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training R²")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation R²")
    ax.set_title(title)
    ax.set_xlabel("Number of training observations")
    ax.set_ylabel("R² score")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nLearning curve saved (PDF)")


# ---------- Machine Learning models ----------

def run_ml_models(save_dataset: bool = True) -> pd.DataFrame:
    """
    This is the function that will be called in main.yp

    This function:
        * Builds + saves ML dataset
        * Trains / evaluates 3 ML models
        * Saves RF + GB feature importance plots
        * Saves model comparison table
        * Prints results blocks to terminal
        * Returns comparison dataframe
    """
    project_root = get_project_root()
    paths = ensure_dirs(project_root)

    # Load FE dataset, build ML dataset, save it
    df = load_fe_dataset(project_root)
    ml_df = build_ml_dataset(df)

    out_dataset_path = paths["data_processed"] / "panel_machine_learning.csv"
    if save_dataset:
        save_ml_dataset(ml_df, out_dataset_path)
        print(f"Final machine learning dataset created and saved (CSV)")

    # Define X, y
    X = ml_df[X_COLS]
    y = ml_df[Y_COL]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ------------------------------------------
    # Model 1: Linear Regression (with scaling)
    # ------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lin = LinearRegression()
    lin_cv = cv_scores(lin, X_train_scaled, y_train)
    lin_tt = train_test_scores(lin, X_train_scaled, X_test_scaled, y_train, y_test)
    print_model_block("Linear Regression", lin_cv, lin_tt)

    # --------------------------------------------------
    # Model 2: Random Forest (regularlized, no scaling)
    # --------------------------------------------------
    """
    This particular settings for the Random Forest model have been chosen to correct an overfitting situation
    that appears if the model is done with no parameters set. With these settings, there is only little to no
    overfitting.
    """
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_cv = cv_scores(rf, X_train, y_train, n_jobs=-1)
    rf_tt = train_test_scores(rf, X_train, X_test, y_train, y_test)
    print_model_block("Random Forest", rf_cv, rf_tt)

    # Save RF feature importances
    rf_imp_path = paths["results_ml"] / "RF_features_importance.pdf"
    save_feature_importance_pdf(
        rf, feature_names=X.columns, out_path=rf_imp_path, title=""
    )

    # Save RF learning curve
    rf_lc_path = paths["results_ml"] / "RF_learning_curve.pdf"
    save_learning_curve_pdf(
    model=rf,
    X_train=X_train,
    y_train=y_train,
    out_path=rf_lc_path,
    title="",
    )
    
    print(f"Features importance histogram saved (PDF)")

    # -----------------------------------------------------
    # Model 3: Gradient Boosting (regularized, no scaling)
    # -----------------------------------------------------
    """
    Same situation as for the previous model, these specific parameters are set to try to avoid overfitting.
    For this model, these settings improved the situation. However, we can still clearly see that the overfitting is 
    still consequent.
    """
    gb = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.7,
        min_samples_leaf=20,
        max_features=0.7,
        random_state=RANDOM_STATE,
    )
    gb_cv = cv_scores(gb, X_train, y_train)
    gb_tt = train_test_scores(gb, X_train, X_test, y_train, y_test)
    print_model_block("Gradient Boosting", gb_cv, gb_tt)

    # Save GB feature importances
    gb_imp_path = paths["results_ml"] / "GB_features_importance.pdf"
    save_feature_importance_pdf(
        gb, feature_names=X.columns, out_path=gb_imp_path, title=""
    )

    # Save GB learning curve
    gb_lc_path = paths["results_ml"] / "GB_learning_curve.pdf"
    save_learning_curve_pdf(
    model=gb,
    X_train=X_train,
    y_train=y_train,
    out_path=gb_lc_path,
    title="",
    )
    
    print(f"Features importance histogram saved (PDF)")

    # -----------------------------
    # Comparison table (save only)
    # -----------------------------
    results = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
            "CV R² (mean)": [lin_cv["cv_r2_mean"], rf_cv["cv_r2_mean"], gb_cv["cv_r2_mean"]],
            "CV R² (std)": [lin_cv["cv_r2_std"], rf_cv["cv_r2_std"], gb_cv["cv_r2_std"]],
            "CV RMSE (mean)": [lin_cv["cv_rmse_mean"], rf_cv["cv_rmse_mean"], gb_cv["cv_rmse_mean"]],
            "CV RMSE (std)": [lin_cv["cv_rmse_std"], rf_cv["cv_rmse_std"], gb_cv["cv_rmse_std"]],
            "Train R²": [lin_tt["train_r2"], rf_tt["train_r2"], gb_tt["train_r2"]],
            "Train RMSE": [lin_tt["train_rmse"], rf_tt["train_rmse"], gb_tt["train_rmse"]],
            "Test R²": [lin_tt["test_r2"], rf_tt["test_r2"], gb_tt["test_r2"]],
            "Test RMSE": [lin_tt["test_rmse"], rf_tt["test_rmse"], gb_tt["test_rmse"]],
        }
    )

    table_path = paths["results_ml"] / "model_comparison.pdf"
    save_model_comparison_table_pdf(results, table_path)
    print(f"\nModel comparison table saved (PDF)")

    return results


if __name__ == "__main__":
    run_ml_models(save_dataset=True)