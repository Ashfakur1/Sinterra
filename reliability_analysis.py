#!/usr/bin/env python3
"""
reliability_analysis.py

Stand-alone companion script for the distance-vs-error reliability plot.
Script 1 (generate_dataset.py) is NOT modified in any way -- this script
only IMPORTS it and reuses its already-defined functions and constants.

Run this from the same folder as your original generate_dataset.py, after
you have already run generate_dataset.py at least once (so that
data/lab_batches.csv exists).

WHAT THIS SCRIPT DOES
  1. Recomputes Leave-One-Out distance and error for the 8 real calibration
     batches. This part needs NO new data -- it reuses generate_dataset.py's
     own Ridge-fitting and physics-prior functions exactly as they are.
  2. Asks you (via keyboard) for the composition, predicted properties, and
     measured properties of your independently fabricated inverse-design
     recommendations (V-1, V-2, V-3, or however many you have). This part
     genuinely cannot be automated -- the measured values only exist in your
     lab records / Table 4, not in any script's output.
  3. Saves what you type to data/validated_recommendations.json so you are
     only asked once; next time it offers to reuse or edit your answers.
  4. Draws the same reliability-vs-distance plot as before and saves it to
     plots/reliability_vs_distance.pdf / .png.

Nothing here changes generate_dataset.py's dataset, coefficients, or any of
its saved files other than reading data/lab_batches.csv.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Import Script 1 as-is -- nothing in it is changed or re-defined here ────
import generate_dataset as gd

ROOTDIR = Path(__file__).parent
OUTDIR = ROOTDIR / "data"
PLOTDIR = ROOTDIR / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
PLOTDIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = OUTDIR / "validated_recommendations.json"


# ═════════════════════════════════════════════════════════════════════════
# PART 1 -- Leave-One-Out distance/error for the 8 calibration batches.
# Reuses generate_dataset.py's own fitting functions (gd._fit_ridge_coefficients,
# gd._apply_physics_priors, gd.INTERACTION_COEFF, gd.AG98_QUAD_COEFF) untouched.
# ═════════════════════════════════════════════════════════════════════════
def compute_loo_reliability() -> pd.DataFrame:
    rows = []
    for i in range(len(gd._LAB_RAW)):
        train_rows = [b for j, b in enumerate(gd._LAB_RAW) if j != i]
        test = gd._LAB_RAW[i]
        train_df = pd.DataFrame(train_rows)

        fold_coeff = gd._apply_physics_priors(
            gd._fit_ridge_coefficients(lab_subset=train_df)
        )
        train_mean_comp = {m: float(train_df[m].mean()) for m in gd.MATS}
        train_mean_props = {t: float(train_df[t].mean()) for t in gd.TGTS}

        cd = {m: test[m] for m in gd.MATS}
        pred = {t: train_mean_props[t] for t in gd.TGTS}

        for m, (cs, cw, cm) in fold_coeff.items():
            d = cd[m] - train_mean_comp[m]
            pred["Shrinkage_pct"] += cs * d
            pred["WA_pct"] += cw * d
            pred["MOR_MPa"] += cm * d

        total_clay = cd["AG98"] + cd["AG22"] + cd["AG23"]
        total_fsp = cd["SodaF"] + cd["PotashF"]
        clay_mean = train_mean_comp["AG98"] + train_mean_comp["AG22"] + train_mean_comp["AG23"]
        fsp_mean = train_mean_comp["SodaF"] + train_mean_comp["PotashF"]
        interaction = (total_clay - clay_mean) * (total_fsp - fsp_mean)
        for t in gd.TGTS:
            pred[t] += gd.INTERACTION_COEFF[t] * interaction

        d_ag98 = cd["AG98"] - train_mean_comp["AG98"]
        for t in gd.TGTS:
            pred[t] += gd.AG98_QUAD_COEFF[t] * (d_ag98 ** 2)

        # Distance of the held-out batch to the remaining 7 only (local
        # computation -- does not call gd._dist, so gd is untouched and its
        # original single-argument signature is never an issue here).
        train_xnorm = train_df[gd.MATS].values / gd._RANGE
        test_comp = np.array([cd[m] for m in gd.MATS])
        dvec = np.linalg.norm(train_xnorm - test_comp / gd._RANGE, axis=1)
        fold_distance = float(np.sort(dvec)[:3].mean())

        for t in gd.TGTS:
            rows.append({
                "batch_id": i + 1,
                "target": t,
                "distance": fold_distance,
                "abs_error": abs(pred[t] - test[t]),
            })

    loo_df = pd.DataFrame(rows)
    ranges = {t: float(max(b[t] for b in gd._LAB_RAW) - min(b[t] for b in gd._LAB_RAW)) for t in gd.TGTS}
    loo_df["pct_error"] = loo_df.apply(lambda r: r.abs_error / ranges[r.target] * 100, axis=1)
    return loo_df


# ═════════════════════════════════════════════════════════════════════════
# PART 2 -- Keyboard input for your fabricated/validated recommendations.
# ═════════════════════════════════════════════════════════════════════════
def _prompt_float(msg: str) -> float:
    while True:
        raw = input(msg).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number, e.g. 16.49")


def _prompt_one_recommendation(default_label: str) -> dict:
    print(f"\n--- Recommendation: {default_label} ---")
    label = input(f"Label [{default_label}]: ").strip() or default_label

    print("Composition (wt%) -- enter all 8 materials, they should sum to ~100:")
    composition = {}
    for m in gd.MATS:
        composition[m] = _prompt_float(f"  {m} [{gd.MAT_LABELS[m]}]: ")
    total = sum(composition.values())
    print(f"  Sum entered = {total:.3f} wt%", "(OK)" if abs(total - 100) < 1.0 else "(!! check this -- far from 100)")

    print("Predicted properties (model output before fabrication):")
    predicted = {t: _prompt_float(f"  Predicted {gd.TGT_LABELS[t]}: ") for t in gd.TGTS}

    print("Measured properties (actual lab result after fabrication):")
    measured = {t: _prompt_float(f"  Measured {gd.TGT_LABELS[t]}: ") for t in gd.TGTS}

    return {"label": label, "composition": composition, "predicted": predicted, "measured": measured}


def get_validated_recommendations() -> list:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cached = json.load(f)
        print(f"\nFound previously entered data for {len(cached)} recommendation(s) in {CACHE_FILE.name}:")
        for r in cached:
            print(f"  - {r['label']}")
        choice = input("Reuse this data? [Y/n], or type 'e' to re-enter from scratch: ").strip().lower()
        if choice in ("", "y", "yes"):
            return cached
        if choice == "e":
            pass  # fall through to re-entry
        else:
            print("Unrecognised input -- reusing cached data.")
            return cached

    n = 0
    while n < 1:
        try:
            n = int(input("\nHow many independently fabricated recommendations do you want to plot? (e.g. 3): ").strip())
        except ValueError:
            print("  Please enter a whole number.")

    default_labels = ["V-1 NN", "V-2 Cost+CO2 NN", "V-3 Bayesian"]
    recs = []
    for i in range(n):
        default_label = default_labels[i] if i < len(default_labels) else f"Recommendation {i+1}"
        recs.append(_prompt_one_recommendation(default_label))

    with open(CACHE_FILE, "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\nSaved your answers to {CACHE_FILE} -- next run will offer to reuse them.")
    return recs


# ═════════════════════════════════════════════════════════════════════════
# PART 3 -- Plot (same design as before, now fed by (1) local LOO computation
# and (2) keyboard-entered recommendations, with Script 1 left untouched).
# ═════════════════════════════════════════════════════════════════════════
def plot_reliability_vs_distance(loo_df: pd.DataFrame, recommendations: list) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"MOR_MPa": "#1f77b4", "WA_pct": "#2ca02c", "Shrinkage_pct": "#d62728"}

    for t in gd.TGTS:
        sub = loo_df[loo_df.target == t]
        ax.scatter(sub.distance, sub.pct_error, color=colors[t], marker="o",
                   s=90, alpha=0.85, edgecolor="white",
                   label=f"{gd.TGT_LABELS[t]} (LOO calibration point)")

    all_x, all_y = list(loo_df.distance), list(loo_df.pct_error)  # PATCH: pooled points for trend line

    ranges = {t: float(max(b[t] for b in gd._LAB_RAW) - min(b[t] for b in gd._LAB_RAW)) for t in gd.TGTS}
    for rec in recommendations:
        comp = np.array([rec["composition"][m] for m in gd.MATS])
        dist = gd._dist(comp)  # unmodified Script 1 function -- distance to all 8 calibration batches
        for t in gd.TGTS:
            pct_err = abs(rec["predicted"][t] - rec["measured"][t]) / ranges[t] * 100
            ax.scatter([dist], [pct_err], color=colors[t], marker="^", s=220,
                       edgecolor="black", linewidth=1.2, zorder=5)
            all_x.append(dist); all_y.append(pct_err)  # PATCH
    if recommendations:
        ax.scatter([], [], color="gray", marker="^", s=220, edgecolor="black",
                   label="Fabricated inverse-design recommendation")

    # PATCH: faint linear trend line across all pooled points (LOO + recommendations).
    # Labelled explicitly as illustrative -- the scatter is genuinely noisy (e.g. two
    # points near distance ~0.80 have very different errors), so this is a rough
    # visual guide to the overall tendency, not a claim of a clean/strong law.
    if len(all_x) >= 2:
        x_arr, y_arr = np.array(all_x), np.array(all_y)
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        x_line = np.array([x_arr.min(), x_arr.max()])
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle=":",
                linewidth=1.8, alpha=0.6, zorder=1,
                label="Linear trend (illustrative, pooled points)")

    ax.axhline(25, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    # PATCH: white background box + high zorder so this label stays readable even
    # when a data point (e.g. a triangle near distance ~0.95) sits right on top of
    # the dashed line. Placed at the left edge, above the line, instead of the
    # right edge, which is the most crowded/high-error region of this plot.
    ax.text(ax.get_xlim()[0] + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), 27,
            "25% pass/warn threshold", ha="left", va="bottom",
            fontsize=12, color="dimgray", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))
    ax.set_xlabel("Normalised distance to nearest calibration data", fontsize=16)
    ax.set_ylabel("Error (% of observed property range)", fontsize=16)
    ax.set_title("Reliability vs. Distance from Calibration Data", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"reliability_vs_distance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {PLOTDIR / 'reliability_vs_distance.pdf'} / .png")


def main():
    print("Computing Leave-One-Out distance/error for the 8 calibration batches ...")
    loo_df = compute_loo_reliability()
    loo_df.to_csv(OUTDIR / "loo_reliability.csv", index=False)
    print(f"Saved: {OUTDIR / 'loo_reliability.csv'}")

    recommendations = get_validated_recommendations()

    plot_reliability_vs_distance(loo_df, recommendations)
    print("\nDone.")


if __name__ == "__main__":
    main()