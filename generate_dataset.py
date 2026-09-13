#!/usr/bin/env python3
"""
generate_dataset.py
Physics-informed synthetic dataset generation for ceramic tile composition
optimisation from 48 laboratory-fabricated calibration batches.

CHANGE LOG (48 batches)
  1. REMOVED magnitude-floor forcing on Ridge coefficients.
  2. Ridge alpha selected PER TARGET via nested Leave-One-Out CV.
  3. validate_physics() reports SKILL SCORE relative to mean-only baseline.

ROBUSTNESS PATCH (N_SYNTHETIC == 0)
  All plotting and summary functions now degrade gracefully when the
  synthetic set is empty: they fall back to the experimental (lab_batch)
  rows, disable KDE when there are too few points, and never abort the
  figure-generation stage. This lets N_SYNTHETIC = 0 produce the full
  figure set from the lab batches alone.

MATHEMATICAL MODEL
  Y = Ȳ_lab + Σ_m β_m·(x_m − x̄_m)
            + γ·(ΣClay − ΣClay_mean)·(ΣFsp − ΣFsp_mean)
            + δ·(x_AG98 − x̄_AG98)²
            + ε(d)
  ε(d) ~ N(0, σ_base·(1 + d/d_ref))
"""

import hashlib, json, logging, math, warnings
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)

ROOTDIR = Path(__file__).parent
OUTDIR  = ROOTDIR / "data"
PLOTDIR = ROOTDIR / "plots"

# ── Master control: change this ONE number to resize the synthetic dataset ───
N_SYNTHETIC = 200

KMM2_TO_MPA = 9.80665

# ── Real Lab Batches ─────────────────────────────────────────────────────────
LAB_BATCHES_FILE = ROOTDIR / "lab_batches_raw.csv"


def _load_lab_batches(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"Lab batches file not found: {path}\n"
            "Create it with columns: AG98,AG22,AG23,SodaF,PotashF,Crushing,"
            "ETP,NaSil,MOR_kgf_mm2,WA_fraction,Shrinkage_pct"
        )
    raw = pd.read_csv(path)
    required = ["AG98", "AG22", "AG23", "SodaF", "PotashF", "Crushing", "ETP",
                "NaSil", "MOR_kgf_mm2", "WA_fraction", "Shrinkage_pct"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if len(raw) < 4:
        raise ValueError(
            f"{path.name} has only {len(raw)} batch(es) - at least 4 are "
            "needed for the Ridge / physics-prior fit to be meaningful."
        )
    batches = []
    for _, r in raw.iterrows():
        batches.append({
            "AG98": float(r.AG98), "AG22": float(r.AG22), "AG23": float(r.AG23),
            "SodaF": float(r.SodaF), "PotashF": float(r.PotashF),
            "Crushing": float(r.Crushing), "ETP": float(r.ETP),
            "NaSil": float(r.NaSil),
            "MOR_MPa": float(r.MOR_kgf_mm2) * KMM2_TO_MPA,
            "WA_pct": float(r.WA_fraction) * 100.0,
            "Shrinkage_pct": float(r.Shrinkage_pct),
        })
    return batches


_LAB_RAW = _load_lab_batches(LAB_BATCHES_FILE)

DATA_HASH = hashlib.md5(LAB_BATCHES_FILE.read_bytes()).hexdigest()[:8]
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

MATS = ["AG98", "AG22", "AG23", "SodaF", "PotashF", "Crushing", "ETP", "NaSil"]
MAT_LABELS = {
    "AG98":     "AG98 (wt%)",
    "AG22":     "AG22 (wt%)",
    "AG23":     "AG23 (wt%)",
    "SodaF":    "Soda Feldspar (wt%)",
    "PotashF":  "Potash Feldspar (wt%)",
    "Crushing": "Crushing (wt%)",
    "ETP":      "ETP Clay (wt%)",
    "NaSil":    "Na-Silicate (wt%)",
}
TGT_LABELS = {
    "MOR_MPa":       "Firing MOR (MPa)",
    "WA_pct":        "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
TGTS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]
TARGET_SIGN_ORDER = ["MOR_MPa", "Shrinkage_pct", "WA_pct"]

for b in _LAB_RAW:
    s = sum(b[m] for m in MATS)
    for m in MATS:
        b[m] = b[m] / s * 100.0

lab_df         = pd.DataFrame(_LAB_RAW)
LAB_MEAN_COMP  = {m: float(lab_df[m].mean()) for m in MATS}
LAB_MEAN_PROPS = {t: float(lab_df[t].mean()) for t in TGTS}

BOUNDS = {
    "AG98":     (15.0, 20.0),
    "AG22":     (2.5,  4.0),
    "AG23":     (10.0, 15.0),
    "SodaF":    (37.0, 43.0),
    "PotashF":  (15.0, 22.0),
    "Crushing": (2.0,  3.5),
    "ETP":      (2.0,  3.1),
    "NaSil":    (0.5,  1.5),
}

COST = {
    "AG98": 6.95,   "AG22": 8.37,   "AG23": 7.024,
    "SodaF": 8.887, "PotashF": 6.241,
    "Crushing": 0.0, "ETP": 0.0,    "NaSil": 23.369,
}
CO2 = {
    "AG98": (0.129, 0.129),
    "AG22": (0.129, 0.129),
    "AG23": (0.129, 0.129),
    "SodaF": (0.053, 0.053),
    "PotashF": (0.0286, 0.0286),
    "Crushing": (0.587, 0.587),
    "ETP": (0.242, 0.242),
    "NaSil": (0.433, 0.433),
}
CO2_MID = {m: float(np.mean(CO2[m])) for m in MATS}

PROC = {
    "press_bar": 100,
    "dryer_time_min": 45,
    "kiln_time_min": 90,
    "kiln_temp_C": 1210,
    "calorific_NG_Kcal_Nm3": 8300,
    "dryer_temp_C": 180.0,
    "gas_Nm3_per_m2": 1.4115,
    "green_length_mm": 109.20,
    "green_width_mm": 54.60,
    "green_thickness_mm": 9.80,
    "green_weight_g": 98.50,
    "fired_length_mm": 98.00,
    "fired_weight_g": 95.05,
}

INTERACTION_COEFF = {
    "MOR_MPa":       -0.08,
    "Shrinkage_pct":  0.012,
    "WA_pct":        -0.003,
}
AG98_QUAD_COEFF = {
    "MOR_MPa":       -0.15,
    "Shrinkage_pct":  0.0,
    "WA_pct":         0.0,
}

SIGN = {"AG98": (+1, +1, -1), "AG22": (+1, +1, -1), "AG23": (+1, +1, -1),
        "SodaF": (-1, -1, +1), "PotashF": (-1, -1, +1),
        "Crushing": (+1, -1, -1), "ETP": (+1, +1, -1), "NaSil": (0, 0, 0)}

ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


# ── Ridge regression helpers ─────────────────────────────────────────────────
def _fit_ridge_one_target(src: pd.DataFrame, alpha: float, target: str) -> np.ndarray:
    X_c = src[MATS].values
    X_c = X_c - X_c.mean(axis=0, keepdims=True)
    A   = X_c.T @ X_c + alpha * np.eye(X_c.shape[1])
    y_c = src[target].values - src[target].mean()
    return np.linalg.solve(A, X_c.T @ y_c)


def _apply_sign_only(coef_vec: np.ndarray, target: str) -> np.ndarray:
    idx = TARGET_SIGN_ORDER.index(target)
    out = coef_vec.copy()
    for i, m in enumerate(MATS):
        s = SIGN[m][idx]
        if s != 0 and out[i] != 0 and np.sign(out[i]) != s:
            out[i] = abs(out[i]) * s
    return out


def _inner_loo_mae(train_df: pd.DataFrame, alpha: float, target: str) -> float:
    errs = []
    n = len(train_df)
    for j in range(n):
        inner_train = train_df.drop(train_df.index[j])
        inner_test  = train_df.iloc[j]
        coef = _apply_sign_only(
            _fit_ridge_one_target(inner_train, alpha, target), target
        )
        mean_comp = {m: float(inner_train[m].mean()) for m in MATS}
        pred = float(inner_train[target].mean())
        for i, m in enumerate(MATS):
            pred += coef[i] * (inner_test[m] - mean_comp[m])
        errs.append(abs(pred - inner_test[target]))
    return float(np.mean(errs))


def _select_alpha_nested(train_df: pd.DataFrame, target: str) -> float:
    best_alpha, best_mae = ALPHA_GRID[0], np.inf
    for a in ALPHA_GRID:
        mae = _inner_loo_mae(train_df, a, target)
        if mae < best_mae:
            best_mae, best_alpha = mae, a
    return best_alpha


def _fit_ridge_coefficients(lab_subset: pd.DataFrame = None,
                             alphas: dict = None) -> dict:
    src = lab_subset if lab_subset is not None else lab_df
    if alphas is None:
        alphas = {t: _select_alpha_nested(src, t) for t in TGTS}
    fitted = {t: _apply_sign_only(
        _fit_ridge_one_target(src, alphas[t], t), t
    ) for t in TGTS}
    coeffs = {m: (float(fitted["Shrinkage_pct"][i]),
                  float(fitted["WA_pct"][i]),
                  float(fitted["MOR_MPa"][i]))
              for i, m in enumerate(MATS)}
    return coeffs, alphas


PHYSICS_COEFF, PHYSICS_ALPHAS = _fit_ridge_coefficients()
logger.info("Production Ridge alphas selected via nested LOO-CV: %s",
            PHYSICS_ALPHAS)

_RANGE     = np.array([BOUNDS[m][1] - BOUNDS[m][0] for m in MATS])
_LAB_XNORM = lab_df[MATS].values / _RANGE
_D_REF     = float(np.median([
    np.linalg.norm(_LAB_XNORM[i] - _LAB_XNORM[j])
    for i in range(len(_LAB_RAW)) for j in range(i + 1, len(_LAB_RAW))
]))


def _dist(comp: np.ndarray, k: int = 3) -> float:
    d = np.linalg.norm(_LAB_XNORM - comp / _RANGE, axis=1)
    return float(np.sort(d)[:k].mean())


def _physics_pred(cd: dict) -> dict:
    p = {t: LAB_MEAN_PROPS[t] for t in TGTS}

    for m, (cs, cw, cm) in PHYSICS_COEFF.items():
        d = cd[m] - LAB_MEAN_COMP[m]
        p["Shrinkage_pct"] += cs * d
        p["WA_pct"]        += cw * d
        p["MOR_MPa"]       += cm * d

    total_clay = cd["AG98"] + cd["AG22"] + cd["AG23"]
    total_fsp  = cd["SodaF"] + cd["PotashF"]
    clay_mean  = (LAB_MEAN_COMP["AG98"] + LAB_MEAN_COMP["AG22"]
                  + LAB_MEAN_COMP["AG23"])
    fsp_mean   = LAB_MEAN_COMP["SodaF"] + LAB_MEAN_COMP["PotashF"]
    interaction = (total_clay - clay_mean) * (total_fsp - fsp_mean)
    for t in TGTS:
        p[t] += INTERACTION_COEFF[t] * interaction

    d_ag98 = cd["AG98"] - LAB_MEAN_COMP["AG98"]
    for t in TGTS:
        p[t] += AG98_QUAD_COEFF[t] * (d_ag98 ** 2)

    return p


def _sample_comps(n: int) -> np.ndarray:
    """Rejection-sample on the simplex. Returns (n, 8) array; (0, 8) when n==0."""
    if n <= 0:
        return np.zeros((0, len(MATS)))
    free   = ["AG98", "AG22", "AG23", "PotashF", "Crushing", "ETP", "NaSil"]
    lo     = np.array([BOUNDS[k][0] for k in free])
    hi     = np.array([BOUNDS[k][1] for k in free])
    lo_s, hi_s = BOUNDS["SodaF"]
    out = []
    while len(out) < n:
        v = rng.uniform(lo, hi)
        s = 100.0 - v.sum()
        if lo_s <= s <= hi_s:
            c = dict(zip(free, v)); c["SodaF"] = s
            out.append([c[m] for m in MATS])
    return np.array(out)


def build_dataset(n_synthetic: int = N_SYNTHETIC) -> pd.DataFrame:
    noise_base = {t: (lab_df[t].max() - lab_df[t].min()) * 0.04 for t in TGTS}
    clip_lo    = {t: lab_df[t].min()           for t in TGTS}
    clip_hi    = {t: lab_df[t].max() * 1.10    for t in TGTS}
    rows = []

    if n_synthetic > 0:
        for comp in _sample_comps(n_synthetic):
            cd  = dict(zip(MATS, comp))
            yp  = _physics_pred(cd)
            d   = _dist(comp)
            row = {f"{m}_wtpct": cd[m] for m in MATS}
            for t in TGTS:
                row[t] = float(np.clip(
                    yp[t] + rng.normal(0, noise_base[t] * (1 + d / _D_REF)),
                    clip_lo[t], clip_hi[t]
                ))
            row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m]    for m in MATS)
            row["CO2_kg_per_kg"]  = sum(cd[m] / 100 * CO2_MID[m] for m in MATS)
            row["source"] = "synthetic"
            row.update(PROC)
            rows.append(row)

    for b in _LAB_RAW:
        cd  = {m: b[m] for m in MATS}
        row = {f"{m}_wtpct": cd[m] for m in MATS}
        for t in TGTS:
            row[t] = b[t]
        row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m]    for m in MATS)
        row["CO2_kg_per_kg"]  = sum(cd[m] / 100 * CO2_MID[m] for m in MATS)
        row["source"] = "lab_batch"
        row.update(PROC)
        rows.append(row)

    for i, r in enumerate(rows, 1):
        r["id"] = i
    cols = (["id", "source"] + [f"{m}_wtpct" for m in MATS] + TGTS
            + list(PROC.keys()) + ["cost_Tk_per_kg", "CO2_kg_per_kg"])
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# ── Surrogate fidelity: nested LOO-CV ────────────────────────────────────────
def validate_physics(df: pd.DataFrame) -> bool:
    loo_errors      = {t: [] for t in TGTS}
    baseline_errors = {t: [] for t in TGTS}
    chosen_alphas   = {t: [] for t in TGTS}

    for i in range(len(_LAB_RAW)):
        train_rows = [b for j, b in enumerate(_LAB_RAW) if j != i]
        test       = _LAB_RAW[i]
        train_df   = pd.DataFrame(train_rows)

        train_mean_comp  = {m: float(train_df[m].mean()) for m in MATS}
        train_mean_props = {t: float(train_df[t].mean()) for t in TGTS}

        fold_alphas = {t: _select_alpha_nested(train_df, t) for t in TGTS}
        fold_coeff, _ = _fit_ridge_coefficients(lab_subset=train_df,
                                                 alphas=fold_alphas)
        for t in TGTS:
            chosen_alphas[t].append(fold_alphas[t])

        cd   = {m: test[m] for m in MATS}
        pred = {t: train_mean_props[t] for t in TGTS}

        for m, (cs, cw, cm) in fold_coeff.items():
            d = cd[m] - train_mean_comp[m]
            pred["Shrinkage_pct"] += cs * d
            pred["WA_pct"]        += cw * d
            pred["MOR_MPa"]       += cm * d

        total_clay = cd["AG98"] + cd["AG22"] + cd["AG23"]
        total_fsp  = cd["SodaF"] + cd["PotashF"]
        clay_mean  = (train_mean_comp["AG98"] + train_mean_comp["AG22"]
                      + train_mean_comp["AG23"])
        fsp_mean   = train_mean_comp["SodaF"] + train_mean_comp["PotashF"]
        interaction = (total_clay - clay_mean) * (total_fsp - fsp_mean)
        for t in TGTS:
            pred[t] += INTERACTION_COEFF[t] * interaction

        d_ag98 = cd["AG98"] - train_mean_comp["AG98"]
        for t in TGTS:
            pred[t] += AG98_QUAD_COEFF[t] * (d_ag98 ** 2)

        for t in TGTS:
            loo_errors[t].append(abs(pred[t] - test[t]))
            baseline_errors[t].append(abs(train_mean_props[t] - test[t]))

    all_pass = True
    for t in TGTS:
        mae      = float(np.mean(loo_errors[t]))
        mae_base = float(np.mean(baseline_errors[t]))
        skill    = 1.0 - mae / mae_base if mae_base > 0 else float("nan")
        r_obs    = float(max(b[t] for b in _LAB_RAW) - min(b[t] for b in _LAB_RAW))
        pct      = mae / r_obs * 100
        status   = "PASS" if pct < 25 else "WARN"
        skill_flag = "genuine skill" if skill > 0.05 else "NO SKILL OVER BASELINE"
        if pct >= 25 or skill <= 0.05:
            all_pass = False
        common_alpha = Counter(chosen_alphas[t]).most_common(3)
        logger.info(
            "LOO-CV  %-18s  MAE=%.4f  base=%.4f  skill=%+.2f (%s)  "
            "range=%.4f  err=%.1f%%  %s  |  alphas: %s",
            t, mae, mae_base, skill, skill_flag, r_obs, pct, status,
            common_alpha
        )
    return all_pass


# ── Save outputs ─────────────────────────────────────────────────────────────
def save(df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lab_df.to_csv(OUTDIR / "lab_batches.csv", index=False)
    df.to_csv(OUTDIR / "dataset.csv", index=False)

    df_s = df[df.source == "synthetic"]
    if len(df_s) == 0:
        # Fallback: report lab-batch ranges when no synthetic rows exist
        df_s = df[df.source == "lab_batch"]
        ranges_source = "lab_batch"
    else:
        ranges_source = "synthetic"

    with open(OUTDIR / "property_ranges.json", "w") as f:
        json.dump(
            {
                "source": ranges_source,
                **{f"{t}_{k}": float(getattr(df_s[t], k)())
                   for t in TGTS for k in ("min", "max")},
            },
            f, indent=2,
        )

    meta = {
        "materials": MATS, "targets": TGTS, "bounds": BOUNDS,
        "cost_tk_per_kg": COST,
        "co2_kg_per_kg": {k: list(v) for k, v in CO2.items()},
        "co2_midpoint": CO2_MID,
        "proc_defaults": PROC,
        "n_lab_batches": len(_LAB_RAW),
        "n_synthetic": int((df.source == "synthetic").sum()),
        "generation_method": (
            "Physics-informed non-linear surrogate. "
            f"Linear terms: Ridge regression on {len(_LAB_RAW)} lab batches, "
            "per-target alpha selected via nested LOO-CV (grid capped at "
            "10.0), sign-only physics correction (no magnitude floor) "
            "(Hoerl & Kennard 1970; Reed 1995). "
            "Non-linear terms: Clay×Feldspar interaction + AG98 quadratic, "
            "fixed constants carried over from the previous calibration "
            "(Carty & Senapati 1998 DOI:10.1111/j.1151-2916.1998.tb02439.x) "
            "-- flagged as a limitation pending nested-CV validation of "
            "their marginal contribution on THIS dataset."
        ),
        "noise_model": "heteroscedastic — σ(d) = σ_base·(1 + d/d_ref); σ_base = 4% of observed range",
        "validation_method": (
            f"Nested Leave-One-Out cross-validation on {len(_LAB_RAW)} "
            "laboratory batches (Ridge coefficients AND alpha refit per "
            "outer fold using only the training fold); reports MAE/range "
            "and skill score relative to a mean-only baseline. A 25%-of-"
            "range PASS is not reported as evidence of a learned "
            "relationship unless skill score is also > 0."
        ),
        "physics_coefficients": {
            m: {"shrinkage": cs, "wa": cw, "mor": cm}
            for m, (cs, cw, cm) in PHYSICS_COEFF.items()
        },
        "ridge_alphas_production_fit": PHYSICS_ALPHAS,
        "interaction_coefficients": INTERACTION_COEFF,
        "ag98_quadratic_coefficients": AG98_QUAD_COEFF,
        "lab_composition_means": LAB_MEAN_COMP,
        "lab_property_means":    LAB_MEAN_PROPS,
        "data_hash":    DATA_HASH,
        "generated_at": GENERATED_AT,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved  %d rows  (synthetic=%d  lab=%d)",
                len(df),
                int((df.source == "synthetic").sum()),
                int((df.source == "lab_batch").sum()))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES — all functions degrade gracefully when n_synthetic == 0
# ══════════════════════════════════════════════════════════════════════════════
_FS_TITLE  = 18
_FS_AX     = 16
_FS_TICK   = 14
_FS_LABEL  = 13
_FS_ANNOT  = 12
_DPI       = 300


def _savefig(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _pick_plot_source(df: pd.DataFrame):
    """
    Choose the rows that figures should be built from.

    Returns (subset_df, source_label, n_rows).
    Preference order: synthetic if non-empty → lab_batch otherwise.
    """
    ds = df[df.source == "synthetic"]
    if len(ds) > 0:
        return ds, "Training Dataset (synthetic)", len(ds)
    ds = df[df.source == "lab_batch"]
    return ds, "Experimental Batches", len(ds)


def plot_distributions(df: pd.DataFrame) -> None:
    comp_cols  = [f"{m}_wtpct" for m in MATS]
    all_cols   = comp_cols + TGTS
    all_labels = [MAT_LABELS[m] for m in MATS] + [TGT_LABELS[t] for t in TGTS]

    ds, source_label, n_actual = _pick_plot_source(df)

    nrows = 3
    ncols = math.ceil(len(all_cols) / nrows)
    pal   = sns.color_palette("husl", len(all_cols) + 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    axes = axes.flatten()

    for i, (col, label) in enumerate(zip(all_cols, all_labels)):
        ax = axes[i]
        vals = ds[col].dropna()

        if len(vals) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=_FS_LABEL, color="gray")
            ax.set_xlabel(label, fontsize=_FS_AX)
            ax.set_ylabel("Frequency", fontsize=_FS_AX)
            ax.tick_params(labelsize=_FS_TICK)
            ax.text(0.5, -0.28, f"({chr(97 + i)})",
                    transform=ax.transAxes, ha="center",
                    fontsize=_FS_LABEL, fontweight="bold")
            continue

        # Robust clipping only when we have enough points
        if len(vals) >= 5:
            data = vals.clip(vals.quantile(0.01), vals.quantile(0.99))
        else:
            data = vals

        # KDE needs ≥ 2 distinct points with non-zero variance
        use_kde = (len(data) >= 2) and (data.nunique() >= 2)
        sns.histplot(data, bins="auto", kde=use_kde, color=pal[i],
                     ax=ax, edgecolor="white")
        for ln in ax.get_lines():
            ln.set_linewidth(2.0); ln.set_alpha(0.85)

        ax.set_xlabel(label, fontsize=_FS_AX)
        ax.set_ylabel("Frequency", fontsize=_FS_AX)
        ax.tick_params(labelsize=_FS_TICK)
        ax.text(0.5, -0.28, f"({chr(97 + i)})",
                transform=ax.transAxes, ha="center",
                fontsize=_FS_LABEL, fontweight="bold")

    for i in range(len(all_cols), len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Feature and Target Distributions — {source_label}"
        f" (n = {n_actual:,})",
        fontsize=24, fontweight="bold", y=1.02,
    )
    plt.tight_layout(h_pad=5.5, w_pad=3.0)
    _savefig(fig, "all_distributions")
    logger.info("Saved: all_distributions.pdf / .png  (source=%s, n=%d)",
                source_label, n_actual)


def plot_source_stripplot(df: pd.DataFrame) -> None:
    colors  = {"synthetic": "#2196F3", "lab_batch": "#E53935"}
    markers = {"synthetic": "o",        "lab_batch": "*"}
    sizes   = {"synthetic": 35,         "lab_batch": 180}
    _rng    = np.random.default_rng(0)

    has_synth = (df.source == "synthetic").sum() > 0
    has_lab   = (df.source == "lab_batch").sum() > 0
    # If synthetic is empty, drop it from the legend/labels — don't draw an
    # empty panel column that would mislead the reader.
    active_sources = [s for s in ("synthetic", "lab_batch")
                      if (df.source == s).sum() > 0]
    if not active_sources:
        logger.warning("plot_source_stripplot: no rows to plot; skipping")
        return

    # One x-tick per active source (1 or 2 columns)
    x_positions = {s: i for i, s in enumerate(active_sources)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, t in zip(axes, TGTS):
        for src in active_sources:
            color = colors[src]
            vals  = df[df.source == src][t].values
            if len(vals) == 0:
                continue
            xpos  = x_positions[src]
            jitter = _rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter, vals,
                color=color,
                alpha=0.50 if src == "synthetic" else 0.95,
                s=sizes[src], marker=markers[src],
                label=("Synthetic" if src == "synthetic"
                       else "Experimental"),
                zorder=3 if src == "lab_batch" else 2,
            )
            ax.hlines(vals.mean(), xpos - 0.30, xpos + 0.30,
                      colors=color, linewidth=2.5, linestyle="--", alpha=0.85)

        # Mann–Whitney U only when both groups actually have data
        if has_synth and has_lab:
            syn_vals = df[df.source == "synthetic"][t].values
            lab_vals = df[df.source == "lab_batch"][t].values
            try:
                _, p_mw = mannwhitneyu(syn_vals, lab_vals,
                                        alternative="two-sided")
                ax.text(0.98, 0.03, f"MWU p = {p_mw:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=_FS_ANNOT, color="gray",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="white", alpha=0.75))
            except Exception:
                pass

        # Build xticklabels using actual counts, in the same order as positions
        xtick_labels = []
        for src in active_sources:
            n_src = int((df.source == src).sum())
            if src == "synthetic":
                xtick_labels.append(f"Synthetic\n(n = {n_src:,})")
            else:
                xtick_labels.append(f"Experimental\n(n = {n_src})")

        ax.set_xticks([x_positions[s] for s in active_sources])
        ax.set_xticklabels(xtick_labels, fontsize=_FS_TICK)
        ax.set_ylabel(TGT_LABELS[t], fontsize=_FS_AX)
        ax.set_title(TGT_LABELS[t], fontsize=_FS_AX, fontweight="bold")
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        if t == TGTS[0]:
            ax.legend(fontsize=_FS_LABEL, loc="upper right")

    subtitle = ("(dashed line = group mean;  MWU = Mann–Whitney U test p-value)"
                if (has_synth and has_lab)
                else "(dashed line = group mean)")
    fig.suptitle(
        "Comparison of Property Distributions: "
        "Synthetic Dataset vs. Experimental Batches\n" + subtitle,
        fontsize=_FS_TITLE, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "source_comparison_stripplot")
    logger.info("Saved: source_comparison_stripplot.pdf / .png")


def plot_composition_correlation(df: pd.DataFrame) -> None:
    comp_cols = [f"{m}_wtpct" for m in MATS]
    labels    = [MAT_LABELS[m].replace(" (wt%)", "") for m in MATS]

    ds, source_label, n_actual = _pick_plot_source(df)

    # Need at least 2 rows to compute correlations
    if len(ds) < 2:
        logger.warning(
            "plot_composition_correlation: only %d row(s); "
            "correlation matrix will be degenerate.", len(ds)
        )

    corr = ds[comp_cols].corr()
    corr.columns = labels
    corr.index   = labels

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={"size": _FS_LABEL}, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(
        "Pearson Correlation Among Composition Variables\n"
        f"({source_label}, n = {n_actual:,}; simplex constraint Σwt% = 100\n"
        "induces structural multicollinearity)",
        fontsize=_FS_TITLE, fontweight="bold", pad=16
    )
    plt.xticks(rotation=45, ha="right", fontsize=_FS_TICK)
    plt.yticks(rotation=0,  fontsize=_FS_TICK)
    plt.tight_layout()
    _savefig(fig, "composition_correlation")
    logger.info("Saved: composition_correlation.pdf / .png  (source=%s, n=%d)",
                source_label, n_actual)


# ── Main ─────────────────────────────────────────────────────────────────────
def _safe_call(fn, *args, **kwargs) -> None:
    """Run a plotting function; log and swallow errors so pipeline never aborts."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.exception("Figure step %s failed: %s", fn.__name__, exc)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PLOTDIR.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset generation — physics-informed non-linear surrogate")
    logger.info("[1/4] Sampling %d synthetic compositions (N_SYNTHETIC=%d) …",
                N_SYNTHETIC, N_SYNTHETIC)
    df = build_dataset(n_synthetic=N_SYNTHETIC)

    logger.info("[2/4] Surrogate fidelity — nested LOO-CV on %d laboratory batches …",
                len(_LAB_RAW))
    validate_physics(df)

    logger.info("[3/4] Saving outputs …")
    save(df)

    logger.info("[4/4] Generating figures …")
    # Ordering note: plot_distributions now degrades to lab batches when the
    # synthetic set is empty, so the two following figures still get produced.
    _safe_call(plot_distributions, df)
    _safe_call(plot_source_stripplot, df)
    _safe_call(plot_composition_correlation, df)

    for t in TGTS:
        logger.info("  %-22s  [%.4f, %.4f]  mean=%.3f  CV=%.2f%%",
                    t, df[t].min(), df[t].max(), df[t].mean(),
                    df[t].std() / df[t].mean() * 100)
    logger.info("Done.")


if __name__ == "__main__":
    main()