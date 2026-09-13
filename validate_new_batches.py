#!/usr/bin/env python3
"""
validate_new_batches.py

Independent external validation of the frozen forward ML model using
9 newly fabricated and experimentally tested ceramic tile batches.

IMPORTANT
---------
1. The trained forward_model.joblib is loaded WITHOUT retraining.
2. The 9 new experimental batches are NEVER used for model fitting.
3. Predictions are generated using exactly the same feature names/order
   expected by the trained model.
4. Actual and predicted values are compared in the same units used by
   train_forward_model.py:
       MOR        -> MPa
       WA         -> %
       Shrinkage  -> %

Outputs
-------
data/new_batch_validation_predictions.csv
data/new_batch_validation_metrics.csv

The script also prints:
    - predicted values for all 9 batches
    - R²
    - RMSE
    - MAE
    - MAPE
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOTDIR = Path(__file__).parent

MODELDIR = ROOTDIR / "models"
DATADIR = ROOTDIR / "data"

MODEL_FILE = MODELDIR / "forward_model.joblib"
FEATURE_FILE = MODELDIR / "feature_cols.json"

OUTPUT_PREDICTIONS = DATADIR / "new_batch_validation_predictions.csv"
OUTPUT_METRICS = DATADIR / "new_batch_validation_metrics.csv"


# =============================================================================
# MODEL FEATURES
# =============================================================================

MATERIALS = [
    "AG98",
    "AG22",
    "AG23",
    "SodaF",
    "PotashF",
    "Crushing",
    "ETP",
    "NaSil",
]

FEATURE_COLS = [f"{m}_wtpct" for m in MATERIALS]

TARGET_COLS = [
    "MOR_MPa",
    "WA_pct",
    "Shrinkage_pct",
]

TARGET_LABELS = {
    "MOR_MPa": "MOR",
    "WA_pct": "Water Absorption",
    "Shrinkage_pct": "Firing Shrinkage",
}


# =============================================================================
# NEW EXPERIMENTAL BATCHES
# =============================================================================
#
# Raw MOR is provided in kgf/mm², exactly as in lab_batches_raw.csv.
# It is converted to MPa using:
#
#     1 kgf/mm² = 9.80665 MPa
#
# WA is supplied as a fraction and converted to percentage:
#
#     0.01202 -> 1.202 %
#
# Shrinkage is already in percentage.
#
# =============================================================================

KMM2_TO_MPA = 9.80665


NEW_BATCHES_RAW = [
    # Batch, AG98, AG22, AG23, SodaF, PotashF, Crushing, ETP, NaSil,
    # MOR_kgf_mm2, WA_fraction, Shrinkage_pct

    [1, 15.700, 2.504, 10.117, 42.979, 21.978, 3.264, 2.396, 1.061,
     5.011, 0.01202, 11.30],

    [2, 19.794, 3.924, 14.996, 37.331, 16.334, 3.392, 3.098, 1.130,
     5.633, 0.00176, 11.76],

    [3, 19.854, 2.642, 14.792, 41.975, 15.251, 2.195, 2.383, 0.908,
     4.202, 0.00710, 10.54],

    [4, 19.669, 3.763, 10.543, 42.802, 15.432, 3.321, 3.023, 1.447,
     6.532, 0.00289, 10.92],

    [5, 19.963, 2.571, 14.774, 37.046, 20.059, 2.037, 2.083, 1.465,
     5.932, 0.02303, 11.69],

    [6, 19.298, 2.942, 10.152, 41.383, 21.509, 2.032, 2.107, 0.578,
     5.289, 0.01380, 11.33],

    [7, 15.018, 2.608, 14.009, 41.733, 21.388, 2.549, 2.105, 0.591,
     5.953, 0.00649, 10.14],

    [8, 17.613, 3.608, 10.006, 42.982, 19.071, 2.768, 2.802, 1.150,
     6.316, 0.00985, 11.28],

    [9, 16.737, 2.640, 10.599, 42.990, 21.303, 2.063, 2.304, 1.364,
     4.114, 0.00229, 10.17],
]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_new_batches() -> pd.DataFrame:

    columns = [
        "Batch",
        *FEATURE_COLS,
        "MOR_kgf_mm2",
        "WA_fraction",
        "Shrinkage_pct",
    ]

    df = pd.DataFrame(NEW_BATCHES_RAW, columns=columns)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # The original generate_dataset.py normalises compositions to Σ = 100 wt%.
    #
    # Therefore the same normalisation is applied here before prediction.
    # -------------------------------------------------------------------------

    composition_sum = df[FEATURE_COLS].sum(axis=1)

    print("\nComposition sums BEFORE normalisation:")
    print(composition_sum.round(4).to_string(index=False))

    for col in FEATURE_COLS:
        df[col] = df[col] / composition_sum * 100.0

    composition_sum_after = df[FEATURE_COLS].sum(axis=1)

    print("\nComposition sums AFTER normalisation:")
    print(composition_sum_after.round(6).to_string(index=False))

    # -------------------------------------------------------------------------
    # Convert experimental targets to the units used by the ML model.
    # -------------------------------------------------------------------------

    df["MOR_MPa"] = df["MOR_kgf_mm2"] * KMM2_TO_MPA
    df["WA_pct"] = df["WA_fraction"] * 100.0

    return df


# =============================================================================
# LOAD FROZEN MODEL
# =============================================================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_FILE}\n\n"
            "Place forward_model.joblib inside the models/ directory."
        )

    print("\nLoading frozen forward model:")
    print(f"  {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)

    if FEATURE_FILE.exists():
        with open(FEATURE_FILE) as f:
            saved_features = json.load(f)

        print("\nFeature order stored with model:")
        print(saved_features)

        if saved_features != FEATURE_COLS:
            print("\nWARNING:")
            print("The feature order in feature_cols.json differs from the")
            print("feature order defined in this validation script.")

        features_to_use = saved_features

    else:
        print("\nWARNING: feature_cols.json not found.")
        print("Using feature order defined in this script.")

        features_to_use = FEATURE_COLS

    return model, features_to_use


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metrics(y_true, y_pred):

    results = []

    for i, target in enumerate(TARGET_COLS):

        actual = np.asarray(y_true[:, i], dtype=float)
        predicted = np.asarray(y_pred[:, i], dtype=float)

        r2 = r2_score(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        mae = mean_absolute_error(
            actual, predicted
        )

        # MAPE
        #
        # None of the present values are zero, but the calculation is
        # protected against zero denominators.
        nonzero = np.abs(actual) > np.finfo(float).eps

        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100.0

        results.append({
            "Property": TARGET_LABELS[target],
            "Target_column": target,
            "N": len(actual),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_percent": mape,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def main():

    DATADIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("EXTERNAL VALIDATION OF FROZEN FORWARD MODEL")
    print("9 NEW EXPERIMENTAL CERAMIC TILE BATCHES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load new experimental data
    # -------------------------------------------------------------------------

    df = load_new_batches()

    # -------------------------------------------------------------------------
    # 2. Load frozen model
    # -------------------------------------------------------------------------

    model, features_to_use = load_model()

    # -------------------------------------------------------------------------
    # 3. Prepare model input
    # -------------------------------------------------------------------------

    X_new = df[features_to_use]

    y_true = df[TARGET_COLS].values

    print("\nInput matrix:")
    print(f"  Samples : {len(X_new)}")
    print(f"  Features: {len(features_to_use)}")

    # -------------------------------------------------------------------------
    # 4. PREDICTION
    #
    # No fit(), retraining, optimisation or recalibration occurs here.
    # -------------------------------------------------------------------------

    print("\nGenerating predictions...")
    print("IMPORTANT: model is FROZEN; no retraining is performed.")

    y_pred = model.predict(X_new)

    # -------------------------------------------------------------------------
    # 5. Build detailed prediction table
    # -------------------------------------------------------------------------

    results = df[["Batch"] + FEATURE_COLS].copy()

    # Actual values
    for i, target in enumerate(TARGET_COLS):
        results[f"Actual_{target}"] = y_true[:, i]
        results[f"Predicted_{target}"] = y_pred[:, i]
        results[f"Error_{target}"] = (
            y_pred[:, i] - y_true[:, i]
        )
        results[f"AbsError_{target}"] = np.abs(
            y_pred[:, i] - y_true[:, i]
        )
        results[f"APE_{target}_percent"] = (
            np.abs(
                (y_pred[:, i] - y_true[:, i])
                / y_true[:, i]
            ) * 100.0
        )

    # -------------------------------------------------------------------------
    # 6. Calculate metrics
    # -------------------------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    # -------------------------------------------------------------------------
    # 7. Save outputs
    # -------------------------------------------------------------------------

    results.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False
    )

    # -------------------------------------------------------------------------
    # 8. Print predictions
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("PREDICTED VS. EXPERIMENTAL VALUES")
    print("=" * 80)

    for i in range(len(df)):

        print(f"\nBatch {int(df.iloc[i]['Batch'])}")

        print(
            f"  MOR: "
            f"Actual = {y_true[i, 0]:.3f} MPa | "
            f"Predicted = {y_pred[i, 0]:.3f} MPa | "
            f"Error = {y_pred[i, 0] - y_true[i, 0]:+.3f} MPa"
        )

        print(
            f"  WA:  "
            f"Actual = {y_true[i, 1]:.4f} % | "
            f"Predicted = {y_pred[i, 1]:.4f} % | "
            f"Error = {y_pred[i, 1] - y_true[i, 1]:+.4f} %"
        )

        print(
            f"  Shrinkage: "
            f"Actual = {y_true[i, 2]:.3f} % | "
            f"Predicted = {y_pred[i, 2]:.3f} % | "
            f"Error = {y_pred[i, 2] - y_true[i, 2]:+.3f} %"
        )

    # -------------------------------------------------------------------------
    # 9. Print final metrics
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("9-BATCH EXTERNAL VALIDATION METRICS")
    print("=" * 80)

    print(
        metrics[
            [
                "Property",
                "N",
                "R2",
                "RMSE",
                "MAE",
                "MAPE_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    # -------------------------------------------------------------------------
    # 10. Additional summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for _, row in metrics.iterrows():

        print(
            f"{row['Property']:<20s} | "
            f"R² = {row['R2']:.4f} | "
            f"RMSE = {row['RMSE']:.4f} | "
            f"MAE = {row['MAE']:.4f} | "
            f"MAPE = {row['MAPE_percent']:.2f}%"
        )

    print("\nFiles saved:")
    print(f"  Predictions: {OUTPUT_PREDICTIONS}")
    print(f"  Metrics:     {OUTPUT_METRICS}")

    print("\nValidation completed.")
    print("The model was NOT retrained using the 9 new experimental batches.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()