#!/usr/bin/env python3
"""
generate_dataset.py
Physics-informed synthetic dataset generation for ceramic tile composition
optimisation from 41 laboratory-fabricated calibration batches.

CHANGE LOG (41 batches)
  This version replaces the earlier 8-batch calibration and fixes two
  methodological issues surfaced by honest nested LOO-CV diagnostics on
  the 41-batch set:

  1. REMOVED magnitude-floor forcing on Ridge coefficients.
     The previous version replaced any Ridge coefficient falling below a
     hand-set floor with the floor value (sign-corrected). At n=41 this
     floor was empirically shown to be actively harmful: for Shrinkage_pct
     specifically, removing the floor (keeping only the physically-motivated
     SIGN correction) cut LOO-CV relative error from ~54% to ~29%, with no
     degradation to MOR or WA. Sign correction is retained -- it encodes
     established sintering physics (Reed 1995) and never contradicts the
     data's own fitted direction unless Ridge itself is noise-dominated.

  2. Ridge alpha is now selected PER TARGET via nested Leave-One-Out
     cross-validation (inner LOO on the 40 training batches only, for each
     outer held-out batch), instead of a single fixed alpha=0.1 justified
     by an n=p=8 near-determined system that no longer holds at n=41.
     The alpha grid is capped at 10.0. This cap is a deliberate choice,
     not an oversight: diagnostics showed that as alpha increases without
     bound, LOO-CV error for every target monotonically converges to the
     error of a trivial "predict the training mean, ignore composition"
     baseline. An unbounded alpha search would therefore "pass" any
     MAE-based threshold by simply giving up on learning a composition-
     property relationship. Capping the grid keeps the fit in a regime
     where it is still doing something other than reproducing the mean.

  3. validate_physics() now reports a SKILL SCORE relative to the
     mean-only baseline for every target, alongside the existing
     MAE/range percentage. A model that does not beat the baseline is
     flagged explicitly rather than allowed to read as a silent PASS.
     This directly addresses the reviewer concern that reported
     performance may not reflect a genuinely learned experimental
     relationship.

RECALIBRATION NOTE (this version)
  Recalibrated against a new 41-batch lab_batches_raw.csv (WA_fraction
  now on a single consistent scale across all rows; one stray corrupted
  cell in the source spreadsheet -- NaSil = "0.907?" -- was read as
  0.907). No logic changes were needed for the linear (Ridge) term: it
  is refit at import time directly from whatever CSV is on disk. The
  interaction and AG98-quadratic terms remain fixed constants carried
  over from the previous calibration (see limitation note in Step 3 of
  COEFFICIENT ESTIMATION below) -- they were NOT refit against this new
  data. Check the validate_physics() skill-score output after loading
  this dataset before relying on them; if skill degrades relative to
  the previous calibration, those two terms are the first place to look.

MATHEMATICAL MODEL
  Y = Ȳ_lab + Σ_m β_m·(x_m − x̄_m)
            + γ·(ΣClay − ΣClay_mean)·(ΣFsp − ΣFsp_mean)   [interaction]
            + δ·(x_AG98 − x̄_AG98)²                         [quadratic]
            + ε(d)
  ε(d) ~ N(0, σ_base·(1 + d/d_ref))   [heteroscedastic]

COEFFICIENT ESTIMATION
  Step 1 — Ridge regression on the 41 lab batches, alpha selected per
            target via nested LOO-CV (see change log above).
            Ref: Hoerl & Kennard (1970) DOI:10.1080/00401706.1970.10488634
  Step 2 — Physics-based SIGN correction only (no magnitude floor) is
            applied to Ridge estimates: if a coefficient's sign
            contradicts the expected sintering-physics direction, its
            sign is flipped and its magnitude preserved. No hand-set
            minimum magnitude is imposed.
            Ref: Reed (1995) Principles of Ceramics Processing, Ch.12.
  Step 3 — A Clay–Feldspar interaction term and an AG98 quadratic term
            (fixed constants, calibrated on the PREVIOUS batch set) capture
            non-linear sintering behaviour. NOTE: these remain fixed
            constants rather than fitted regressors -- flagged here as a
            known limitation for future work, since honest testing of
            their marginal contribution under nested CV has not yet been
            done on this dataset.
            Ref: Carty & Senapati (1998) DOI:10.1111/j.1151-2916.1998.tb02439.x

MATERIAL ROLES — clay-dominant floor tile body, 1210 °C, 100 bar
  AG98      High Plastic Clay   → MOR↑  WA↓  Shrink↑
  AG22      Low Plastic Clay    → MOR↑  WA↓  Shrink↑  (weaker signal)
  AG23      Semi-Plastic Clay   → MOR↑  WA↓  Shrink↑
  SodaF     Soda Feldspar       → MOR↓  WA↑  Shrink↓  (clay diluent)
  PotashF   Potash Feldspar     → MOR↓  WA↑  Shrink↓  (stronger diluent)
  Crushing  Pre-fired filler    → MOR↑  WA↓  Shrink↓
  ETP       ETP sludge (alkali) → MOR↑  WA↓  Shrink↑  (liquid-phase sinter)
  NaSil     Sodium Silicate     → rheology modifier; small effect on WA

CO₂ FACTORS  (kg CO₂ / kg, cradle-to-gate)
  AG98 (High Plastic Clay)  0.129       Zeng et al. 2025 DOI:10.4236/eng.2025.1712035
  AG22 (Low Plastic Clay)   0.129       Proxy: same as AG98 (AP-42 §11.25 scope mismatch)
  AG23 (Semi Plastic Clay)  0.129       Proxy: same as AG98 (AP-42 §11.25 scope mismatch)
  Soda Feldspar (SodaF)     0.053       LB Minerals EPD, Pobežovice site
  Potash Feldspar (PotashF) 0.0286      LB Minerals EPD, Nová Ves site
  Crushing                  0.587       LB Minerals Chamotte EPD
  ETP sludge                0.242       Li et al. 2023, Incineration pathway
  NaSil                     0.433       EPD-IES-0021224, Prochin Italia

VALIDATION
  Leave-One-Out cross-validation on the 41 laboratory batches. For each
  held-out batch, Ridge coefficients (with sign-only physics correction)
  AND the composition/property centroid AND the per-target alpha are all
  re-estimated from the remaining 40 batches only (alpha via an inner
  LOO on those 40), then the held-out batch's properties are predicted.
  This is a true nested LOO-CV: no information from the held-out batch
  leaks into coefficient estimation OR hyperparameter selection.

  IMPORTANT: a "PASS" against the 25%-of-range threshold is NOT on its
  own evidence of a learned relationship -- see skill-score note above.
  Report both numbers.

CENTRALIZED CONFIGURATION (single source of truth)
  N_SYNTHETIC below is the ONLY place the synthetic-sample count needs to
  be changed. It is passed to build_dataset() in main() and the resulting
  actual count is written into data/metadata.json under the key
  "n_synthetic". train_forward_model.py, inverse_design.py, and
  streamlit_app.py all read that key (or derive the count directly from
  dataset.csv) at runtime instead of hardcoding a number, so changing
  N_SYNTHETIC here and re-running the pipeline propagates automatically
  to every downstream script and to every plot title / caption that
  reports the sample size.
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
# Every downstream script reads the resulting count from data/metadata.json
# ("n_synthetic") or directly from dataset.csv, so
# this is the single source of truth for the whole pipeline.
N_SYNTHETIC = 200

KMM2_TO_MPA = 9.80665          # kgf/mm² → MPa  (ISO 13006)

# ── Real Lab Batches — loaded from an external, user-editable CSV ─────────────
LAB_BATCHES_FILE = ROOTDIR / "lab_batches_raw.csv"

def _load_lab_batches(path: Path) -> list:
    """
    Load real laboratory calibration batches from a CSV file.

    Required columns
    -----------------
    AG98, AG22, AG23, SodaF, PotashF, Crushing, ETP, NaSil
        Raw composition inputs (wt%, need not already sum to 100 —
        they are normalised to Sigma = 100 further below).
    MOR_kgf_mm2
        Flexural strength as read directly off the tester, kgf/mm2.
    WA_fraction
        Water absorption as a fraction (e.g. 0.0359, not 3.59).
    Shrinkage_pct
        Fired linear shrinkage, %.
    """
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

# ── Data provenance fingerprint ────────────────────────────────────────────────
DATA_HASH = hashlib.md5(LAB_BATCHES_FILE.read_bytes()).hexdigest()[:8]
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

MATS = ["AG98","AG22","AG23","SodaF","PotashF","Crushing","ETP","NaSil"]
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
TGTS = ["MOR_MPa","WA_pct","Shrinkage_pct"]
# Order matters: matches the SIGN tuple layout below (MOR, Shrinkage, WA)
TARGET_SIGN_ORDER = ["MOR_MPa", "Shrinkage_pct", "WA_pct"]

for b in _LAB_RAW:                             # normalise to Σ = 100 wt%
    s = sum(b[m] for m in MATS)
    for m in MATS:
        b[m] = b[m] / s * 100.0

lab_df            = pd.DataFrame(_LAB_RAW)
LAB_MEAN_COMP     = {m: float(lab_df[m].mean()) for m in MATS}
LAB_MEAN_PROPS    = {t: float(lab_df[t].mean()) for t in TGTS}

# ── Composition feasibility bounds (industrial specification) ─────────────────
BOUNDS = {
    "AG98"    : (15.0, 20.0),
    "AG22"    : (2.5,  4.0),
    "AG23"    : (10.0, 15.0),
    "SodaF"   : (37.0, 43.0),
    "PotashF" : (15.0, 22.0),
    "Crushing": (2.0,  3.5),
    "ETP"     : (2.0,  3.1),
    "NaSil"   : (0.5,  1.5),
}

# ── Cost (BDT/kg) and CO₂ factors ────────────────────────────────────────────
COST = {
    "AG98": 6.95,   "AG22": 8.37,  "AG23": 7.024,
    "SodaF": 8.887, "PotashF": 6.241,
    "Crushing": 0.0, "ETP": 0.0,   "NaSil": 23.369,
}
CO2 = {
    "AG98":(0.129,0.129),
    "AG22":(0.129,0.129),
    "AG23":(0.129,0.129),
    "SodaF":(0.053,0.053),
    "PotashF":(0.0286,0.0286),
    "Crushing":(0.587,0.587),
    "ETP":(0.242,0.242),
    "NaSil":(0.433,0.433),
}
CO2_MID = {m: float(np.mean(CO2[m])) for m in MATS}

# ── Process parameters (fixed single firing cycle) ───────────────────────────
PROC = {
    "press_bar":           100,
    "dryer_time_min":       45,
    "kiln_time_min":        90,
    "kiln_temp_C":        1210,
    "calorific_NG_Kcal_Nm3": 8300,
    "dryer_temp_C":        180.0,
    "gas_Nm3_per_m2":      1.4115,
    "green_length_mm":     109.20,
    "green_width_mm":       54.60,
    "green_thickness_mm":    9.80,
    "green_weight_g":       98.50,
    "fired_length_mm":      98.00,
    "fired_weight_g":       95.05,
}

# ── Non-linear interaction coefficients (calibrated on PREVIOUS batch set) ────
# NOTE: still fixed constants, not fitted regressors, and not yet refit
# against the current lab_batches_raw.csv. See RECALIBRATION NOTE above.
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

# ── Sign priors (physics direction only — no magnitude floor) ────────────────
SIGN = {"AG98":(+1,+1,-1),"AG22":(+1,+1,-1),"AG23":(+1,+1,-1),
        "SodaF":(-1,-1,+1),"PotashF":(-1,-1,+1),
        "Crushing":(+1,-1,-1),"ETP":(+1,+1,-1),"NaSil":(0,0,0)}
# tuple order per material: (MOR, Shrinkage, WA)

ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# ── Ridge regression: single-target fit at a given alpha ─────────────────────
def _fit_ridge_one_target(src: pd.DataFrame, alpha: float, target: str) -> np.ndarray:
    X_c = src[MATS].values
    X_c = X_c - X_c.mean(axis=0, keepdims=True)
    A   = X_c.T @ X_c + alpha * np.eye(X_c.shape[1])
    y_c = src[target].values - src[target].mean()
    return np.linalg.solve(A, X_c.T @ y_c)

def _apply_sign_only(coef_vec: np.ndarray, target: str) -> np.ndarray:
    """Flip sign to match sintering-physics prior; NO magnitude floor."""
    idx = TARGET_SIGN_ORDER.index(target)
    out = coef_vec.copy()
    for i, m in enumerate(MATS):
        s = SIGN[m][idx]
        if s != 0 and out[i] != 0 and np.sign(out[i]) != s:
            out[i] = abs(out[i]) * s
    return out

def _inner_loo_mae(train_df: pd.DataFrame, alpha: float, target: str) -> float:
    """Inner LOO-CV used only to select alpha — never sees the outer test point."""
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
    """Choose alpha via inner LOO on train_df only (grid capped at 10.0 —
    see module docstring for why an unbounded search is invalid here)."""
    best_alpha, best_mae = ALPHA_GRID[0], np.inf
    for a in ALPHA_GRID:
        mae = _inner_loo_mae(train_df, a, target)
        if mae < best_mae:
            best_mae, best_alpha = mae, a
    return best_alpha

def _fit_ridge_coefficients(lab_subset: pd.DataFrame = None,
                             alphas: dict = None) -> dict:
    """
    Fit sign-corrected Ridge coefficients per material, per target.

    Parameters
    ----------
    lab_subset : pd.DataFrame, optional
        Lab batches to fit on. Defaults to the full 41-batch ``lab_df``.
    alphas : dict, optional
        {target: alpha} to use. If None, alpha is selected per target via
        nested LOO-CV on ``lab_subset`` itself (used for the production
        fit on all 41 batches). When this function is called from within
        validate_physics()'s outer LOO loop, the caller passes alphas
        already selected on the training fold only, to avoid leakage.
    """
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
    for i in range(len(_LAB_RAW)) for j in range(i+1, len(_LAB_RAW))
]))

def _dist(comp: np.ndarray, k: int = 3) -> float:
    d = np.linalg.norm(_LAB_XNORM - comp / _RANGE, axis=1)
    return float(np.sort(d)[:k].mean())

def _physics_pred(cd: dict) -> dict:
    """
    Non-linear physics surrogate.

    1. Linear terms: sign-corrected Ridge coefficients (per material).
    2. Clay–Feldspar interaction (fixed constant).
    3. AG98 quadratic (fixed constant).
    """
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
    """Rejection-sample on the simplex: all 8 materials within BOUNDS, Σ = 100."""
    free   = ["AG98","AG22","AG23","PotashF","Crushing","ETP","NaSil"]
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
    clip_lo    = {t: lab_df[t].min()        for t in TGTS}
    clip_hi    = {t: lab_df[t].max() * 1.10 for t in TGTS}
    rows = []

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
        row["cost_Tk_per_kg"] = sum(cd[m] / 100 * COST[m] for m in MATS)
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

# ── Surrogate fidelity: nested Leave-One-Out CV on all lab batches ───────────
def validate_physics(df: pd.DataFrame) -> bool:
    """
    Nested Leave-One-Out cross-validation on the laboratory batches.

    For each held-out batch:
      - alpha per target is selected via an INNER LOO on the remaining
        n-1 batches only (never sees the held-out point),
      - Ridge coefficients (sign-corrected) and the composition/property
        centroid are refit on those same n-1 batches,
      - the held-out batch's properties are predicted.

    Alongside the usual MAE/range percentage, this also reports a SKILL
    SCORE relative to a trivial "predict the training-fold mean, ignore
    composition" baseline:
        skill = 1 - MAE_model / MAE_baseline
    skill > 0 means the model beats the baseline; skill <= 0 means the
    model provides no benefit over ignoring composition entirely. This
    is reported explicitly because the MAE/range threshold alone can be
    satisfied by a model that has effectively degenerated to the
    baseline (see module docstring).
    """
    loo_errors     = {t: [] for t in TGTS}
    baseline_errors = {t: [] for t in TGTS}
    chosen_alphas  = {t: [] for t in TGTS}

    for i in range(len(_LAB_RAW)):
        train_rows = [b for j, b in enumerate(_LAB_RAW) if j != i]
        test       = _LAB_RAW[i]
        train_df   = pd.DataFrame(train_rows)

        train_mean_comp  = {m: float(train_df[m].mean()) for m in MATS}
        train_mean_props = {t: float(train_df[t].mean()) for t in TGTS}

        # Alpha selected on the training fold only (nested — no leakage)
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

# ── Save outputs ──────────────────────────────────────────────────────────────
def save(df: pd.DataFrame) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    lab_df.to_csv(OUTDIR / "lab_batches.csv", index=False)
    df.to_csv(OUTDIR / "dataset.csv", index=False)
    df_s = df[df.source == "synthetic"]
    with open(OUTDIR / "property_ranges.json", "w") as f:
        json.dump({f"{t}_{k}": float(getattr(df_s[t], k)())
                   for t in TGTS for k in ("min", "max")}, f, indent=2)
    meta = {
        "materials": MATS, "targets": TGTS, "bounds": BOUNDS,
        "cost_tk_per_kg": COST,
        "co2_kg_per_kg": {k: list(v) for k, v in CO2.items()},
        "co2_midpoint": CO2_MID,
        "proc_defaults": PROC,
        "n_lab_batches": len(_LAB_RAW),
        # Single source of truth for every downstream script: the ACTUAL
        # number of synthetic rows produced this run (derived from
        # N_SYNTHETIC / whatever value was passed to build_dataset()),
        # never a hardcoded literal.
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
# FIGURES  (unchanged from prior version, except dynamic sample-size labels)
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

def plot_distributions(df: pd.DataFrame) -> None:
    comp_cols  = [f"{m}_wtpct" for m in MATS]
    all_cols   = comp_cols + TGTS
    all_labels = [MAT_LABELS[m] for m in MATS] + [TGT_LABELS[t] for t in TGTS]

    nrows = 3; ncols = math.ceil(len(all_cols) / nrows)
    pal   = sns.color_palette("husl", len(all_cols) + 2)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 4.2))
    axes = axes.flatten()

    ds = df[df.source == "synthetic"]
    n_synth_actual = len(ds)   # actual count, never hardcoded
    for i, (col, label) in enumerate(zip(all_cols, all_labels)):
        ax   = axes[i]
        data = ds[col].clip(ds[col].quantile(0.01), ds[col].quantile(0.99))
        sns.histplot(data, bins="auto", kde=True, color=pal[i],
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
        "Feature and Target Distributions of the Synthetic Training Dataset"
        f" (n = {n_synth_actual:,})",
        fontsize=_FS_TITLE, fontweight="bold", y=1.02
    )
    plt.tight_layout(h_pad=5.5, w_pad=3.0)
    _savefig(fig, "all_distributions")
    logger.info("Saved: all_distributions.pdf / .png")

def plot_source_stripplot(df: pd.DataFrame) -> None:
    colors  = {"synthetic": "#2196F3", "lab_batch": "#E53935"}
    markers = {"synthetic": "o",        "lab_batch": "*"}
    sizes   = {"synthetic": 35,         "lab_batch": 180}
    _rng    = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, t in zip(axes, TGTS):
        for i, (src, color) in enumerate(colors.items()):
            vals   = df[df.source == src][t].values
            jitter = _rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                color=color,
                alpha=0.50 if src == "synthetic" else 0.95,
                s=sizes[src], marker=markers[src],
                label=("Synthetic" if src == "synthetic" else "Experimental"),
                zorder=3 if src == "lab_batch" else 2,
            )
            ax.hlines(vals.mean(), i - 0.30, i + 0.30,
                      colors=color, linewidth=2.5, linestyle="--", alpha=0.85)

        syn_vals = df[df.source == "synthetic"][t].values
        lab_vals = df[df.source == "lab_batch"][t].values
        try:
            _, p_mw = mannwhitneyu(syn_vals, lab_vals, alternative="two-sided")
            ax.text(0.98, 0.03, f"MWU p = {p_mw:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=_FS_ANNOT, color="gray",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.75))
        except Exception:
            pass

        n_synth_actual = len(syn_vals)
        n_lab_actual   = len(lab_vals)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"Synthetic\n(n = {n_synth_actual:,})",
             f"Experimental\n(n = {n_lab_actual})"],
            fontsize=_FS_TICK
        )
        ax.set_ylabel(TGT_LABELS[t], fontsize=_FS_AX)
        ax.set_title(TGT_LABELS[t], fontsize=_FS_AX, fontweight="bold")
        ax.tick_params(axis="y", labelsize=_FS_TICK)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        if t == TGTS[0]:
            ax.legend(fontsize=_FS_LABEL, loc="upper right")

    fig.suptitle(
        "Comparison of Property Distributions: Synthetic Dataset vs."
        " Experimental Batches\n(dashed line = group mean;"
        "  MWU = Mann–Whitney U test p-value)",
        fontsize=_FS_TITLE, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, "source_comparison_stripplot")
    logger.info("Saved: source_comparison_stripplot.pdf / .png")

def plot_composition_correlation(df: pd.DataFrame) -> None:
    comp_cols = [f"{m}_wtpct" for m in MATS]
    labels    = [MAT_LABELS[m].replace(" (wt%)", "") for m in MATS]

    ds   = df[df.source == "synthetic"]
    corr = ds[comp_cols].corr()
    corr.columns = labels
    corr.index   = labels

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, annot_kws={"size": _FS_LABEL}, ax=ax
    )
    ax.set_title(
        "Pearson Correlation Among Composition Variables\n"
        "(synthetic dataset; simplex constraint Σwt% = 100\n"
        "induces structural multicollinearity)",
        fontsize=_FS_TITLE, fontweight="bold", pad=16
    )
    plt.xticks(rotation=45, ha="right", fontsize=_FS_TICK)
    plt.yticks(rotation=0,  fontsize=_FS_TICK)
    plt.tight_layout()
    _savefig(fig, "composition_correlation")
    logger.info("Saved: composition_correlation.pdf / .png")

# ── Main ──────────────────────────────────────────────────────────────────────
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
    plot_distributions(df)
    plot_source_stripplot(df)
    plot_composition_correlation(df)

    for t in TGTS:
        logger.info("  %-22s  [%.4f, %.4f]  mean=%.3f  CV=%.2f%%",
                    t, df[t].min(), df[t].max(), df[t].mean(),
                    df[t].std() / df[t].mean() * 100)
    logger.info("Done.")

if __name__ == "__main__":
    main()