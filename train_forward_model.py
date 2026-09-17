#!/usr/bin/env python3
"""
train_forward_model.py
Trains a multi-output ML forward model mapping ceramic tile composition
variables to mechanical and dimensional properties.

CHANGE LOG (post peer-review)
  Reviewer 1, comment 2: "R² is calculated on a combined test set containing
  synthetic and experimental observations, majority synthetic. The ML model
  is being tested partly on data generated according to the same assumptions
  used to construct the modeling framework."

  This is addressed directly:
    - The original combined-test-set R² block is RETAINED but now clearly
      labelled as a self-consistency sanity check only, not as validation
      evidence. It answers "did the model learn the synthetic generator's
      function", not "does the model work on real ceramics".
    - A NEW experimental-only nested Leave-One-Out cross-validation block
      is added. For each of the n experimental batches, the model is
      retrained from scratch on [full synthetic set] + [all OTHER
      experimental batches], then evaluated on the held-out batch alone.
      No experimental point ever contributes to its own prediction, and
      no synthetic point is present in the held-out evaluation set.
    - A skill score relative to a "predict the mean of the training
      experimental batches, ignore composition" baseline is reported
      alongside MAE/RMSE, mirroring generate_dataset.py's validate_physics().
      This prevents a small MAE number from being mistaken for genuine
      predictive skill if it is really just tracking the sample mean.

  Reviewer 2, comment 6: "All target properties are linear... what
  motivates ML-based optimization? A linear fit would be sufficient."
  A plain LinearRegression candidate is added to the model comparison so
  this question is answered empirically (on the SAME experimental-only
  nested LOO-CV as every other candidate) rather than asserted.

DYNAMIC SAMPLE-SIZE REPORTING
  Every place this script previously printed or plotted the synthetic /
  experimental sample counts used a literal number ("1000 synthetic",
  "~96% synthetic") that silently went stale whenever generate_dataset.py's
  N_SYNTHETIC was changed. All such counts are now computed at runtime
  from the actual loaded dataset.csv / metadata.json (len(synth_idx),
  n_exp, and the percentage derived from them), so changing N_SYNTHETIC
  in generate_dataset.py and re-running the pipeline updates this script's
  console output and plot text automatically — no manual edits needed here.

KEY DECISIONS & JUSTIFICATIONS
  [A] Synthetic-only 5-fold CV is used ONLY for architecture selection
      (which model class fits the physics-informed generator best). This
      is a sanity check on the generator, not evidence of real-world skill.
  [B] Experimental-only nested LOO-CV (new) is the evidence that matters
      for the manuscript's validation claims. See docstring above.
  [C] PDP computed manually on ORIGINAL-SCALE X using the full pipeline.
  [D]/[E]/[F] Composition variables only; process variables held constant
      and confirmed near-zero correlated with targets (unchanged).
"""

import json, math, warnings
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set(style="whitegrid", context="talk", font_scale=1.1)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOTDIR  = Path(__file__).parent
DATADIR  = ROOTDIR / "data"
MODELDIR = ROOTDIR / "models"
PLOTDIR  = ROOTDIR / "plots"
for p in [MODELDIR, PLOTDIR]:
    p.mkdir(exist_ok=True, parents=True)

_FS_TITLE = 18
_FS_AX    = 16
_FS_TICK  = 14
_FS_LABEL = 13
_FS_ANNOT = 12
_DPI      = 300

# ── Load data ─────────────────────────────────────────────────────────────────
df   = pd.read_csv(DATADIR / "dataset.csv")
with open(DATADIR / "metadata.json") as f:
    meta = json.load(f)

DATA_HASH    = meta.get("data_hash", "unknown")
GENERATED_AT = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def savefig(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(PLOTDIR / f"{stem}.{ext}", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

materials   = meta["materials"]
TARGET_COLS = ["MOR_MPa", "WA_pct", "Shrinkage_pct"]

TGT_LABELS = {
    "MOR_MPa":       "Firing MOR (MPa)",
    "WA_pct":        "Water Absorption (%)",
    "Shrinkage_pct": "Fired Shrinkage (%)",
}
MAT_SHORT = {
    "AG98_wtpct":     "AG98",
    "AG22_wtpct":     "AG22",
    "AG23_wtpct":     "AG23",
    "SodaF_wtpct":    "Soda Feldspar",
    "PotashF_wtpct":  "Potash Feldspar",
    "Crushing_wtpct": "Crushing",
    "ETP_wtpct":      "ETP Clay",
    "NaSil_wtpct":    "Na-Silicate",
}

feature_cols = [f"{m}_wtpct" for m in materials if f"{m}_wtpct" in df.columns]
comp_cols    = feature_cols
print(f"Features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols]
y = df[TARGET_COLS]

lab_mask  = df["source"] == "lab_batch"
synth_idx = df[~lab_mask].index
lab_idx   = df[lab_mask].index
n_exp     = len(lab_idx)

# Actual synthetic sample count, read live from the loaded dataset — never
# hardcoded. This is what changes automatically when N_SYNTHETIC changes
# in generate_dataset.py and the pipeline is re-run.
n_synth_total = len(synth_idx)
pct_synth     = n_synth_total / (n_synth_total + n_exp) * 100 if (n_synth_total + n_exp) else 0.0

# HAS_SYNTHETIC gates every downstream block that assumed a non-empty
# synthetic set (the STEP A sanity-check CV, the synthetic train/test
# split, the legacy combined-R² metric). When N_SYNTHETIC=0 in
# generate_dataset.py there is no synthetic data to split, so the
# pipeline falls back to an "experimental-only" mode: architecture
# selection AND evaluation both happen via nested Leave-One-Out CV on
# the n_exp lab batches directly (STEP B's own machinery, which already
# tolerates an empty synthetic set), and the production model is fit on
# ALL experimental batches instead of a synthetic-held-out split.
#
# SCIENTIFIC CAVEAT (read before trusting N_SYNTHETIC=0 results):
# with zero synthetic augmentation, the SAME n_exp batches are used both
# to CHOOSE the model architecture and to REPORT its performance. Nested
# LOO prevents any single fold from leaking into its own prediction, but
# it cannot prevent architecture selection itself from being tuned to
# this particular n_exp-batch sample. Treat STEP B skill scores in this
# mode as optimistic relative to the normal (N_SYNTHETIC>0) pipeline,
# where architecture is chosen on an independent synthetic sanity check.
HAS_SYNTHETIC = n_synth_total > 0

X_synth_full, y_synth_full = X.loc[synth_idx], y.loc[synth_idx].values
X_exp, y_exp = X.loc[lab_idx].reset_index(drop=True), y.loc[lab_idx].reset_index(drop=True)

if HAS_SYNTHETIC:
    synth_train_idx, synth_test_idx = train_test_split(
        synth_idx, test_size=0.20, random_state=7
    )
    train_idx = synth_train_idx
    test_idx  = synth_test_idx.tolist() + lab_idx.tolist()

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx].values, y.loc[test_idx].values

    print(f"Train: {len(X_train)} synthetic | "
          f"Test:  {len(synth_test_idx)} synthetic + {n_exp} experimental")
else:
    X_train, y_train = X_exp, y_exp.values
    X_test, y_test = None, None
    print("N_SYNTHETIC = 0: no synthetic samples were generated.")
    print("Falling back to EXPERIMENTAL-ONLY mode — architecture selection and")
    print("evaluation both run as nested LOO-CV directly on the "
          f"{n_exp} lab batches (see STEP A/B below and the caveat printed there).")

preproc = ColumnTransformer(
    [("num", StandardScaler(), feature_cols)], remainder="drop"
)

# ── Input–output correlation heatmap ─────────────────────────────────────────
comp_only_X = X[comp_cols]
corr_io = pd.concat([comp_only_X, y], axis=1).corr().loc[comp_cols, TARGET_COLS]
corr_io_display = corr_io.copy()
corr_io_display.index   = [MAT_SHORT.get(c, c) for c in corr_io_display.index]
corr_io_display.columns = [TGT_LABELS.get(c, c) for c in corr_io_display.columns]
corr_io.to_csv(DATADIR / "input_output_correlation.csv")

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_io_display, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 16}, ax=ax)
ax.set_title("Pearson Correlation Coefficients Between\n"
             "Composition Variables and Target Properties",
             pad=20, fontsize=20, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(rotation=0,  fontsize=16)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.tight_layout()
savefig(fig, "input_output_correlation_heatmap")
print("Saved: input_output_correlation_heatmap.pdf / .png")
print(f"CAUTION: the heatmap above is computed on the FULL dataset "
      f"({n_synth_total} synthetic + {n_exp} experimental, ~{pct_synth:.0f}% "
      "synthetic). It largely reflects the physics-informed generator's own "
      "assumptions, not an independently-observed experimental relationship "
      "(Reviewer 2, comment 6). See the experimental-only heatmap below.")

# ── Experimental-ONLY correlation heatmap (addresses Reviewer 2, comment 6) ──
# Computed exclusively on the n experimental lab batches. With this few
# points, correlations are noisy (wide confidence intervals) -- that noise
# is reported honestly rather than smoothed over, since it is itself part
# of the answer to "is there a strong linear relationship in the real data".
exp_only = df[df["source"] == "lab_batch"]
corr_io_exp = pd.concat([exp_only[comp_cols], exp_only[TARGET_COLS]], axis=1) \
    .corr().loc[comp_cols, TARGET_COLS]
corr_io_exp_display = corr_io_exp.copy()
corr_io_exp_display.index   = [MAT_SHORT.get(c, c) for c in corr_io_exp_display.index]
corr_io_exp_display.columns = [TGT_LABELS.get(c, c) for c in corr_io_exp_display.columns]
corr_io_exp.to_csv(DATADIR / "input_output_correlation_experimental_only.csv")

# p-values so the reader can see which correlations are not distinguishable
# from zero at this experimental sample size -- essential context that the
# heatmap color alone hides.
from scipy.stats import pearsonr
pval_exp = pd.DataFrame(index=comp_cols, columns=TARGET_COLS, dtype=float)
for c in comp_cols:
    for t in TARGET_COLS:
        _, p = pearsonr(exp_only[c], exp_only[t])
        pval_exp.loc[c, t] = p
pval_exp.to_csv(DATADIR / "input_output_correlation_experimental_only_pvalues.csv")
n_sig = int((pval_exp.values < 0.05).sum())
print(f"Experimental-only (n={n_exp}) correlations: {n_sig} / "
      f"{pval_exp.size} are significant at p<0.05 "
      "(see input_output_correlation_experimental_only_pvalues.csv)")

annot_exp = corr_io_exp_display.round(2).astype(str)
for c in comp_cols:
    for t in TARGET_COLS:
        if pval_exp.loc[c, t] < 0.05:
            # Added a space before the asterisk so it reads as a separate marker
            annot_exp.loc[MAT_SHORT.get(c, c), TGT_LABELS.get(t, t)] += " *"

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_io_exp_display, annot=annot_exp.values, fmt="",
            cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 16}, ax=ax)
ax.set_title(
    f"Pearson Correlation: Composition vs. Target Properties\n"
    f"EXPERIMENTAL BATCHES (n={n_exp})\n"
    f"* indicates p < 0.05",
    pad=20, fontsize=18, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(rotation=0,  fontsize=16)
ax.collections[0].colorbar.ax.tick_params(labelsize=16)
plt.tight_layout()
savefig(fig, "input_output_correlation_heatmap_experimental_only")
print("Saved: input_output_correlation_heatmap_experimental_only.pdf / .png")

# ── Candidate models — LinearRegression added per Reviewer 2 comment 6 ───────
candidate_models: dict = {
    "LinearRegression": MultiOutputRegressor(LinearRegression()),
    "RandomForest_native": RandomForestRegressor(
        n_estimators=500, random_state=7, n_jobs=-1
    ),
    "RandomForest_wrapped": MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=7, n_jobs=-1)
    ),
    "MLP": MultiOutputRegressor(
        MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=3000,
                     random_state=42, early_stopping=True,
                     n_iter_no_change=30)
    ),
}
for _lib, _cls, _name, _kw in [
    ("xgboost",  "XGBRegressor",      "XGB",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,"verbosity":0}),
    ("lightgbm", "LGBMRegressor",     "LGBM",
     {"n_estimators":500,"random_state":7,"n_jobs":-1,
      "verbose":-1,"min_gain_to_split":0.0}),
    ("catboost", "CatBoostRegressor", "CatBoost",
     {"iterations":500,"verbose":0,"random_seed":7}),
]:
    try:
        candidate_models[_name] = MultiOutputRegressor(
            getattr(__import__(_lib), _cls)(**_kw)
        )
    except Exception:
        pass

# ── STEP A: synthetic-only 5-fold CV — architecture sanity check ONLY ────────
# NOTE: this selects which model class best recovers the physics-informed
# synthetic generator's own function. It is NOT evidence of real-world
# predictive skill (Reviewer 1, comment 2). See STEP B below for that.
# Skipped entirely when HAS_SYNTHETIC is False — architecture is instead
# selected inside STEP B, directly from experimental-only nested LOO-CV.
model_scores: dict = {}
best_name = None
final_model = None
y_pred = None
r2_combined = None
lab_mask_test = None

if HAS_SYNTHETIC:
    cv = KFold(n_splits=5, shuffle=True, random_state=7)
    print("\n[STEP A] Synthetic-only 5-fold CV (architecture sanity check, "
          "NOT validation evidence):")
    for name, model in candidate_models.items():
        try:
            pipe   = Pipeline([("preproc", preproc), ("reg", model)])
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=cv, scoring="r2", n_jobs=-1)
            model_scores[name] = float(np.mean(scores))
            print(f"  {name:<28s}  mean={model_scores[name]:.4f}  "
                  f"std={np.std(scores):.4f}")
        except Exception as e:
            print(f"  {name:<28s}  skipped: {e}")

    best_name = max(model_scores, key=model_scores.get)
    print(f"\nBest architecture on synthetic self-consistency: {best_name}  "
          f"(R² = {model_scores[best_name]:.4f})")

    # ── PRODUCTION MODEL OVERRIDE — prefer a bounded-output architecture ──
    # STEP A rewards whichever architecture best reproduces the SYNTHETIC
    # generator's own function. That generator is itself a linear Ridge
    # model (see generate_dataset.py), so LinearRegression will tend to
    # win STEP A almost by construction (it is recovering its own
    # generating function) -- this is a measure of self-consistency, NOT
    # of safety when the composition-space optimiser in inverse_design.py
    # is later evaluated at points the linear coefficients were never
    # built to be robust at (see that module's docstring: "near-cancelling
    # coefficients only cancel ON the manifold").
    #
    # A tree ensemble (RandomForest / XGB / LGBM / CatBoost) cannot have
    # this failure mode: every prediction is an average of leaf values
    # actually observed in training, so it is mathematically bounded
    # within the observed training-target range and can never output a
    # negative MOR or negative WA, regardless of where in the composition
    # box it is queried. We therefore use the best-scoring BOUNDED
    # architecture as the production model whenever one is available,
    # while still reporting every candidate's STEP A/STEP B numbers
    # (including plain LinearRegression, per Reviewer 2 comment 6) for
    # transparency.
    BOUNDED_ARCHITECTURES = {
        "RandomForest_native", "RandomForest_wrapped", "XGB", "LGBM", "CatBoost",
    }
    bounded_scores = {n: s for n, s in model_scores.items() if n in BOUNDED_ARCHITECTURES}
    if not bounded_scores:
        print(
            "WARNING: no tree-ensemble candidate was available/fitted "
            "successfully. Install xgboost/lightgbm/catboost or ensure "
            "RandomForest fits, otherwise the production model may be an "
            "unbounded regressor and the guardrails in inverse_design.py "
            "become the only protection against extrapolated predictions."
        )

    # NOTE: STEP A no longer chooses the production model. It only ranks
    # architectures by how well they recover the synthetic generator, which
    # is a property of the generator, not of real ceramics. Selection moved
    # to STEP B (experimental leave-one-out skill). `best_name` here is
    # provisional and exists only so the legacy combined metric below can be
    # computed for the same architecture the manuscript reports.
    provisional_name = (max(bounded_scores, key=bounded_scores.get)
                        if bounded_scores else best_name)
    print(f"\nProvisional (synthetic-ranked) bounded architecture: "
          f"{provisional_name}. Final selection is made in STEP B on "
          "experimental leave-one-out skill.")
    best_name = provisional_name

    lab_idx_set   = set(lab_idx)
    lab_mask_test = np.array([idx in lab_idx_set for idx in test_idx])
else:
    print("\n[STEP A] Skipped — no synthetic data (N_SYNTHETIC=0), so the "
          "synthetic-only sanity-check CV cannot run. Architecture will be "
          "selected inside STEP B instead, directly from experimental-only "
          "nested LOO-CV (see caveat printed there).")
    print("\n[Legacy combined metric] Skipped — this metric requires a "
          "synthetic hold-out set, which does not exist when N_SYNTHETIC=0.")

# ══════════════════════════════════════════════════════════════════════════
# STEP B: Experimental-only nested Leave-One-Out cross-validation
# ══════════════════════════════════════════════════════════════════════════
# For each experimental batch, retrain the chosen architecture on
# [ALL synthetic rows] + [all OTHER experimental batches], then predict
# the held-out batch. Compare against a "predict the mean of the training
# experimental batches, ignore composition" baseline. This is the direct
# answer to Reviewer 1's comment 2, and — since LinearRegression is one of
# the candidates — also empirically answers Reviewer 2's comment 6.
print(f"\n[STEP B] Experimental-only nested LOO-CV (n={n_exp} batches) — "
      "this is the evidence that matters for the manuscript's validation "
      "claims. Running for every candidate model architecture …")

# ── Per-fold synthetic regeneration (leakage fix) ─────────────────────────────
# THE PROBLEM THIS SOLVES
#   The synthetic rows in dataset.csv come from a surrogate (Eq. 1) that was
#   calibrated on ALL experimental batches. Holding batch i out of the
#   training fold therefore does NOT remove batch i's influence: its
#   composition and measured properties helped set the Ridge coefficients,
#   the per-target lambda, the composition/property means, sigma_base and
#   d_ref, all of which are baked into every synthetic row. Batch i leaks
#   back into its own training fold through the synthetic data, and the
#   resulting skill score is optimistic.
#
# THE FIX
#   For fold i, rebuild the synthetic set from a surrogate calibrated on the
#   OTHER 47 batches only (generate_dataset.build_synthetic_from_batches).
#   Nothing derived from batch i then exists anywhere in the training fold.
#   Folds are cached because the same 48 synthetic sets are reused by every
#   candidate architecture.
import generate_dataset as gd  # noqa: E402  (imported here, next to its use)

_LAB_DF_FULL = gd.lab_df           # 48 rows, same order as the lab rows in X_exp
_N_SYNTH_PER_FOLD = n_synth_total  # match the production synthetic set size
_fold_synth_cache: dict = {}


def _fold_synthetic(i: int):
    """Synthetic (X, y) for fold i, from a surrogate that never saw batch i."""
    if i not in _fold_synth_cache:
        sub = _LAB_DF_FULL.drop(_LAB_DF_FULL.index[i])
        syn = gd.build_synthetic_from_batches(
            sub, _N_SYNTH_PER_FOLD, seed=1000 + i
        )
        _fold_synth_cache[i] = (
            syn[feature_cols].reset_index(drop=True),
            syn[TARGET_COLS].values,
        )
    return _fold_synth_cache[i]


def run_experimental_loo(model_ctor, X_synth, y_synth, X_e, y_e,
                          mode: str = "leakfree"):
    """
    Leave-one-out over the experimental batches.

    mode:
      "leakfree"  — fold i trains on [synthetic regenerated from the other 47
                    batches] + [the other 47 batches]. This is the number to
                    report: no information from the held-out batch reaches
                    the model by any route.
      "leaky"     — fold i trains on [the production synthetic set, calibrated
                    on all 48] + [the other 47 batches]. Retained only so the
                    manuscript can quantify how much the leak was worth.
      "no_synth"  — fold i trains on [the other 47 batches] alone. This is the
                    ablation that answers "does synthetic augmentation help at
                    all?" — the question Reviewer 1 raised when noting that
                    adding synthetic points cannot fix a small experimental
                    set.
    """
    n = len(X_e)
    preds         = np.zeros((n, len(TARGET_COLS)))
    errs          = np.zeros((n, len(TARGET_COLS)))
    baseline_errs = np.zeros((n, len(TARGET_COLS)))
    for i in range(n):
        train_exp_mask = np.ones(n, dtype=bool)
        train_exp_mask[i] = False

        if mode == "no_synth":
            X_tr = X_e.loc[train_exp_mask].reset_index(drop=True)
            y_tr = y_e[train_exp_mask]
        else:
            if mode == "leakfree":
                Xs, ys = _fold_synthetic(i)
            else:  # "leaky"
                Xs, ys = X_synth, y_synth
            X_tr = pd.concat([Xs, X_e.loc[train_exp_mask]], ignore_index=True)
            y_tr = np.vstack([ys, y_e[train_exp_mask]])

        pipe = Pipeline([("preproc", preproc), ("reg", model_ctor())])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_e.loc[[i]])[0]
        preds[i] = pred
        errs[i] = np.abs(pred - y_e[i])
        baseline_pred = y_e[train_exp_mask].mean(axis=0)
        baseline_errs[i] = np.abs(baseline_pred - y_e[i])
    return preds, errs, baseline_errs

# Constructors so each fold gets a fresh, unfitted model instance
from sklearn.base import clone

def _make_ctor(m):
    # Factory avoids the classic late-binding closure bug in a loop.
    return lambda: clone(m)

# EVERY candidate is now run through the experimental-only LOO, not just
# LinearRegression plus whichever architecture won the synthetic sanity
# check. The manuscript states the forward model was "selected by a nested
# leave-one-out cross validation restricted to the 48 experimental batches";
# selecting on synthetic R² and only confirming two architectures on
# experimental data did not match that sentence. It does now.
model_ctors = {name: _make_ctor(model) for name, model in candidate_models.items()}

exp_loo_results = {}
for name, ctor in model_ctors.items():
    preds, errs, baseline_errs = run_experimental_loo(
        ctor, X_synth_full.reset_index(drop=True), y_synth_full,
        X_exp, y_exp.values, mode="leakfree"
    )
    exp_loo_results[name] = (preds, errs, baseline_errs)
    print(f"\n  Architecture: {name}")
    for i, t in enumerate(TARGET_COLS):
        mae      = errs[:, i].mean()
        mae_base = baseline_errs[:, i].mean()
        rmse     = np.sqrt((errs[:, i] ** 2).mean())
        skill    = 1.0 - mae / mae_base if mae_base > 0 else float("nan")
        rng      = y_exp[t].max() - y_exp[t].min()
        pct      = mae / rng * 100
        flag     = "genuine skill" if skill > 0.05 else "NO SKILL OVER BASELINE"
        print(f"    {t:<15s} MAE={mae:.4f}  base={mae_base:.4f}  "
              f"RMSE={rmse:.4f}  skill={skill:+.2f} ({flag})  "
              f"err/range={pct:.1f}%")

def _avg_skill(name):
    _, errs, baseline_errs = exp_loo_results[name]
    skills = [1.0 - errs[:, i].mean() / baseline_errs[:, i].mean()
              for i in range(len(TARGET_COLS))]
    return float(np.mean(skills))


# ── Production architecture: chosen on EXPERIMENTAL skill, not synthetic R² ──
top_name = max(exp_loo_results, key=_avg_skill)

# An unbounded regressor (LinearRegression, MLP) can still top a small-n
# experimental skill comparison while remaining unsafe to query at arbitrary
# points in the composition box, which is what the Bayesian optimiser does.
# The bounded-architecture preference is therefore retained, but it now
# operates on experimental skill rather than on synthetic self-consistency.
BOUNDED_ARCHITECTURES = {
    "RandomForest_native", "RandomForest_wrapped", "XGB", "LGBM", "CatBoost",
}
bounded_names = [n for n in exp_loo_results
                 if n in BOUNDED_ARCHITECTURES and _avg_skill(n) > 0.05]

# OPTIONAL PIN. Set to an architecture name (e.g. "LGBM") to force it as the
# production model; leave as None to let experimental skill decide.
#   Why this exists: the manuscript names LightGBM as the forward model. Once
#   selection is done honestly on experimental leave-one-out skill, a
#   different architecture may win — the margin between the tree ensembles is
#   small and well within what 48 batches can resolve. Pinning is legitimate
#   (they are near-equivalent, and consistency with a submitted manuscript has
#   value), but it must be disclosed rather than hidden, so pinning a model
#   that is not the top scorer prints a warning and records the gap. Do not
#   pin silently and report the pinned model as "selected by cross validation".
PRODUCTION_ARCHITECTURE_PIN = None

if PRODUCTION_ARCHITECTURE_PIN and PRODUCTION_ARCHITECTURE_PIN in exp_loo_results:
    best_name = PRODUCTION_ARCHITECTURE_PIN
    _free_choice = max(bounded_names, key=_avg_skill) if bounded_names else best_name
    if _free_choice != best_name:
        print(
            f"\nDISCLOSURE: production architecture is PINNED to "
            f"'{best_name}' (mean experimental skill {_avg_skill(best_name):+.3f}). "
            f"Unpinned selection would have chosen '{_free_choice}' "
            f"({_avg_skill(_free_choice):+.3f}). The manuscript must not "
            "describe the pinned model as the one cross validation selected; "
            "say it was chosen among near-equivalent tree ensembles and give "
            "both skill scores."
        )
elif bounded_names:
    best_name = max(bounded_names, key=_avg_skill)
    if best_name != top_name:
        print(
            f"\nPRODUCTION MODEL: using '{best_name}' (mean experimental "
            f"skill = {_avg_skill(best_name):+.2f}) instead of the "
            f"top-scoring '{top_name}' ({_avg_skill(top_name):+.2f}). "
            "Reason: the top scorer is an unbounded regressor, which can "
            "emit physically impossible values anywhere in the composition "
            "search space; a tree ensemble cannot."
        )
else:
    best_name = top_name
    print(f"\nWARNING: no bounded architecture cleared skill > 0.05. Falling "
          f"back to '{best_name}'. The guardrails in inverse_design.py are "
          "then the only protection against extrapolated predictions.")

print(f"\n[STEP B] Architecture selected on experimental LOO skill: "
      f"{best_name}  (mean skill = {_avg_skill(best_name):+.2f})")

# Refit the production model now that best_name is known from experimental
# evidence rather than from the synthetic sanity check.
final_model = Pipeline([("preproc", preproc), ("reg", candidate_models[best_name])])
if HAS_SYNTHETIC:
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    mae_combined = np.mean(np.abs(y_test - y_pred), axis=0)
    r2_combined  = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(3)]

    print("\n[Legacy combined metric — synthetic + experimental together]")
    print("CAUTION: majority synthetic; do NOT cite this as validation "
          "evidence (Reviewer 1, comment 2).")
    for t, m, r in zip(TARGET_COLS, mae_combined, r2_combined):
        print(f"  {t:<25s}  MAE={m:.4f}  R²={r:.4f}")
else:
    final_model.fit(X_exp, y_exp.values)

# ══════════════════════════════════════════════════════════════════════════
# STEP C: How much was the leak worth, and does synthetic data help at all?
# ══════════════════════════════════════════════════════════════════════════
# Two comparisons a reviewer will ask for, both run on the production
# architecture and on the linear baseline:
#   leaky     vs leakfree  -> how optimistic were the previously reported
#                             skill scores, given the surrogate had seen
#                             every batch it was later tested against
#   leakfree  vs no_synth  -> does synthetic augmentation earn its place?
#                             If no_synth matches or beats leakfree, the
#                             surrogate step adds nothing and the paper
#                             should say so plainly.
ABLATION_MODELS = [n for n in ("LinearRegression", best_name)
                   if n in candidate_models]
ablation_rows = []
if HAS_SYNTHETIC:
    print("\n[STEP C] Leakage and synthetic-augmentation ablation "
          f"(n={n_exp} experimental batches) …")
    for name in ABLATION_MODELS:
        ctor = _make_ctor(candidate_models[name])
        for mode in ("leaky", "leakfree", "no_synth"):
            if mode == "leakfree":
                _, errs, base = exp_loo_results[name]
            else:
                _, errs, base = run_experimental_loo(
                    ctor, X_synth_full.reset_index(drop=True), y_synth_full,
                    X_exp, y_exp.values, mode=mode
                )
            print(f"\n  {name}  [{mode}]")
            for i, t in enumerate(TARGET_COLS):
                mae   = errs[:, i].mean()
                skill = 1.0 - mae / base[:, i].mean()
                ablation_rows.append({
                    "model": name, "mode": mode, "target": t,
                    "mae": float(mae),
                    "rmse": float(np.sqrt((errs[:, i] ** 2).mean())),
                    "skill": float(skill),
                })
                print(f"    {t:<15s} MAE={mae:.4f}  skill={skill:+.2f}")

    abl = pd.DataFrame(ablation_rows)
    abl.to_csv(DATADIR / "leakage_and_augmentation_ablation.csv", index=False)
    print("\nSaved: data/leakage_and_augmentation_ablation.csv")
    print("REPORT THIS IN THE MANUSCRIPT. 'leakfree' is the honest number; "
          "'leaky' is what the previous pipeline reported; 'no_synth' says "
          "whether the surrogate step was worth including at all.")

# ── Is the non-linear model actually better? Paired test, not an assertion ───
# The manuscript says LGBM and LinearRegression are "statistically
# indistinguishable" for WA. No test was ever run to support the word
# "statistically" — equal MAE to 3 decimals is not a test. A Wilcoxon
# signed-rank test on the 48 paired fold-wise absolute errors is, and it
# also tells us whether the MOR/shrinkage advantages are real or noise.
if "LinearRegression" in exp_loo_results and best_name != "LinearRegression":
    from scipy.stats import wilcoxon
    _, lin_errs, _ = exp_loo_results["LinearRegression"]
    _, bst_errs, _ = exp_loo_results[best_name]
    print(f"\n[Paired Wilcoxon signed-rank: {best_name} vs LinearRegression, "
          f"fold-wise absolute errors, n={n_exp}]")
    wilcox_rows = []
    for i, t in enumerate(TARGET_COLS):
        d = lin_errs[:, i] - bst_errs[:, i]
        if np.allclose(d, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(lin_errs[:, i], bst_errs[:, i])
        rel = (1 - bst_errs[:, i].mean() / lin_errs[:, i].mean()) * 100
        verdict = ("significant" if p < 0.05 else
                   "NOT significant — describe as indistinguishable")
        wilcox_rows.append({"target": t, "mae_linear": float(lin_errs[:, i].mean()),
                            f"mae_{best_name}": float(bst_errs[:, i].mean()),
                            "mae_reduction_pct": float(rel),
                            "wilcoxon_p": float(p), "verdict": verdict})
        print(f"  {t:<15s} MAE reduction = {rel:+.1f}%   p = {p:.4f}   {verdict}")
    pd.DataFrame(wilcox_rows).to_csv(
        DATADIR / "linear_vs_nonlinear_wilcoxon.csv", index=False)
    print("Saved: data/linear_vs_nonlinear_wilcoxon.csv")

print("\n[STEP B SUMMARY] Report the experimental-only LEAKAGE-FREE LOO-CV "
      "numbers above in the manuscript, NOT the combined-test-set R² from "
      "STEP A. A model without skill > 0.05 here should not be described as "
      "having learned a real composition-property relationship, regardless "
      "of what the combined-test-set R² shows.")

# Defined once here (not inside the feature-importance try/except below) so
# it's guaranteed to exist for the PDP section even if that block errors out.
_fit_src = ("fitted on full training data" if HAS_SYNTHETIC
            else "fitted on experimental data only")

# ── Save CSV artefacts ────────────────────────────────────────────────────────
if HAS_SYNTHETIC:
    pd.DataFrame({
        **{f"y_true_{t}": y_test[:, i] for i, t in enumerate(TARGET_COLS)},
        **{f"y_pred_{t}": y_pred[:, i] for i, t in enumerate(TARGET_COLS)},
    }).to_csv(DATADIR / "parity_data.csv", index=False)
else:
    print("Skipped parity_data.csv (no synthetic hold-out set exists when "
          "N_SYNTHETIC=0); use experimental_loo_cv_results.csv for the "
          "out-of-fold experimental predictions instead.")

exp_loo_rows = []
for name, (preds, errs, baseline_errs) in exp_loo_results.items():
    for i, t in enumerate(TARGET_COLS):
        exp_loo_rows.append({
            "model": name, "target": t,
            "mae": errs[:, i].mean(),
            "mae_baseline": baseline_errs[:, i].mean(),
            "rmse": float(np.sqrt((errs[:, i] ** 2).mean())),
            "skill": 1.0 - errs[:, i].mean() / baseline_errs[:, i].mean(),
        })
pd.DataFrame(exp_loo_rows).to_csv(DATADIR / "experimental_loo_cv_results.csv", index=False)
print("Saved: data/experimental_loo_cv_results.csv")

# ── Feature importance ────────────────────────────────────────────────────────
try:
    reg = final_model.named_steps["reg"]
    if hasattr(reg, "feature_importances_"):
        imp = reg.feature_importances_
    elif hasattr(reg, "estimators_") and hasattr(reg.estimators_[0], "feature_importances_"):
        imp = np.mean([e.feature_importances_ for e in reg.estimators_], axis=0)
    elif hasattr(reg, "estimators_") and hasattr(reg.estimators_[0], "coef_"):
        # Linear model: use mean absolute standardized coefficient across
        # the three outputs as an importance proxy (features are already
        # standardized by the preprocessing pipeline).
        imp = np.mean([np.abs(e.coef_) for e in reg.estimators_], axis=0)
    elif hasattr(reg, "coef_"):
        imp = np.abs(reg.coef_).mean(axis=0) if reg.coef_.ndim > 1 else np.abs(reg.coef_)
    else:
        raise AttributeError("No feature_importances_ or coef_ found on regressor")
    imp = imp / imp.sum()
    fi = (pd.DataFrame({"feature": feature_cols, "importance": imp})
          .sort_values("importance", ascending=False))
    fi["label"] = fi["feature"].map(lambda c: MAT_SHORT.get(c, c))
    fi.to_csv(DATADIR / "feature_importances.csv", index=False)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ["#E53935" if "_wtpct" in f else "#90A4AE" for f in fi["feature"]]
    sns.barplot(x="importance", y="label", data=fi,
                palette=palette, ax=ax, edgecolor="white")
    ax.legend(handles=[Patch(color="#E53935", label="Composition variable")],
              fontsize=_FS_LABEL, loc="lower right")
    ax.set_title(
        "Mean Feature Importance Across All Target Properties",
        pad=14, fontsize=_FS_TITLE, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=_FS_AX)
    ax.set_ylabel("", fontsize=_FS_AX)
    ax.tick_params(labelsize=_FS_TICK)
    plt.tight_layout()
    savefig(fig, "feature_importances")
    print("Saved: feature_importances.pdf / .png")
except Exception as e:
    print(f"Feature importance: {e}")

# ── Parity plots — now annotated with STEP B skill score, not just combined R² ─
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
units = {"MOR_MPa": "MPa", "WA_pct": "%", "Shrinkage_pct": "%"}
best_preds, best_errs, best_baseline = exp_loo_results[best_name]

if HAS_SYNTHETIC:
    for i, t in enumerate(TARGET_COLS):
        ax   = axes[i]
        smsk = ~lab_mask_test

        skill_i = 1.0 - best_errs[:, i].mean() / best_baseline[:, i].mean()
        rmse_exp = float(np.sqrt((best_errs[:, i] ** 2).mean()))
        exp_label = f"LOO RMSE (exp, n={n_exp}) = {rmse_exp:.3f} {units[t]}  |  skill={skill_i:+.2f}"

        ax.scatter(y_test[smsk, i], y_pred[smsk, i],
                   alpha=0.55, s=40, color="teal")
        if lab_mask_test.sum() > 0:
            ax.scatter(y_test[lab_mask_test, i], y_pred[lab_mask_test, i],
                       alpha=0.9, s=130, color="crimson", marker="*",
                       label="Experimental", zorder=5)
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel("Measured", fontsize=_FS_AX)
        ax.set_ylabel("Predicted", fontsize=_FS_AX)
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title(
            f"{TGT_LABELS[t]}\n"
            f"Combined R² (caution, see docstring) = {r2_combined[i]:.4f}\n"
            f"{exp_label}",
            fontsize=_FS_AX
        )
        ax.legend(fontsize=_FS_LABEL)
else:
    # No synthetic hold-out set exists (N_SYNTHETIC=0), so the parity plot
    # uses the experimental-only nested LOO out-of-fold predictions
    # directly: each point was predicted by a model that never saw that
    # point during fitting, so this is a legitimate (if small-n) parity
    # check, not a training-set-echo plot.
    y_exp_vals = y_exp.values
    for i, t in enumerate(TARGET_COLS):
        ax = axes[i]
        skill_i = 1.0 - best_errs[:, i].mean() / best_baseline[:, i].mean()
        rmse_exp = float(np.sqrt((best_errs[:, i] ** 2).mean()))
        exp_label = f"LOO RMSE (exp, n={n_exp}) = {rmse_exp:.3f} {units[t]}  |  skill={skill_i:+.2f}"

        ax.scatter(y_exp_vals[:, i], best_preds[:, i],
                   alpha=0.9, s=130, color="crimson", marker="*",
                   label="Experimental (LOO out-of-fold)", zorder=5)
        lo = min(y_exp_vals[:, i].min(), best_preds[:, i].min())
        hi = max(y_exp_vals[:, i].max(), best_preds[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel("Measured", fontsize=_FS_AX)
        ax.set_ylabel("Predicted", fontsize=_FS_AX)
        ax.tick_params(labelsize=_FS_TICK)
        ax.set_title(
            f"{TGT_LABELS[t]}\n"
            f"{exp_label}",
            fontsize=_FS_AX
        )
        ax.legend(fontsize=_FS_LABEL)

plt.suptitle("Predicted vs. Measured Values: Forward Model Parity Plots",
             fontsize=_FS_TITLE, fontweight="bold", y=0.84)
plt.tight_layout(rect=[0, 0, 1, 0.90])
savefig(fig, "parity_plots")
print("Saved: parity_plots.pdf / .png")

# ── PDP Section ────────────────────────────────────────────────────────────────
# X_train is the synthetic training split when HAS_SYNTHETIC, or the full
# experimental set (X_exp) when N_SYNTHETIC=0 — set earlier in either case.
_train_set_label = "synthetic" if HAS_SYNTHETIC else "experimental (N_SYNTHETIC=0)"
print(f"\nComputing PDPs on original-scale composition variables "
      f"(marginalised over {_train_set_label} training set, n = {len(X_train)}) …")

def compute_pdp(pipeline, X_orig: pd.DataFrame, feature: str, n_grid: int = 60):
    grid = np.linspace(X_orig[feature].min(), X_orig[feature].max(), n_grid)
    means = []
    for val in grid:
        Xt = X_orig.copy()
        Xt[feature] = val
        means.append(pipeline.predict(Xt).mean(axis=0))
    return grid, np.array(means)

_FS_PDP_SUPTITLE = 26
_FS_PDP_TITLE    = 24
_FS_PDP_LABEL    = 20
_FS_PDP_TICK     = 18
colors = ["#1565C0", "#D32F2F", "#388E3C"]

# PDPs are produced over TWO reference sets:
#   "synthetic"    — marginalised over the synthetic training set (the
#                    original behaviour, kept for continuity)
#   "experimental" — marginalised over the 48 real batches
# The distinction matters for the manuscript's argument. Section 3.2 uses
# step-like, non-monotonic partial dependence as evidence that a non-linear
# model is warranted. But the synthetic set was generated by a LINEAR Ridge
# surrogate, so a tree ensemble fitted to it will produce axis-aligned steps
# whether or not real ceramics behave that way — the steps are partly an
# artefact of the generator plus the learner. Marginalising over the real
# compositions instead removes that circularity. If the non-monotonicity
# survives on the experimental reference set, the argument stands; if it
# only appears on the synthetic one, the claim should be dropped and the
# case rested on the Wilcoxon result from STEP C instead.
_PDP_REFERENCE_SETS = [("synthetic", X_train)] if HAS_SYNTHETIC else []
_PDP_REFERENCE_SETS.append(("experimental", X_exp))

for _pdp_tag, _pdp_X in _PDP_REFERENCE_SETS:
  for t_idx, tname in enumerate(TARGET_COLS):
    n_comp = len(comp_cols)
    ncols  = 4
    nrows  = math.ceil(n_comp / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.2, nrows * 6.4), dpi=250)
    axes = axes.flatten()
    for ax_idx, feat in enumerate(comp_cols):
        ax = axes[ax_idx]
        grid, means = compute_pdp(final_model, _pdp_X, feat)
        ax.plot(grid, means[:, t_idx], color=colors[t_idx], linewidth=3.5)
        short = MAT_SHORT.get(feat, feat.replace("_wtpct", ""))
        ax.set_xlabel(f"{short} (wt%)", fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_ylabel(TGT_LABELS[tname],  fontsize=_FS_PDP_LABEL, labelpad=10)
        ax.set_title(short, fontsize=_FS_PDP_TITLE, fontweight="bold", pad=15)
        ax.tick_params(labelsize=_FS_PDP_TICK, width=1.5, length=8)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for ax in axes[n_comp:]:
        ax.set_visible(False)
    _ref_label = ("synthetic training set" if _pdp_tag == "synthetic"
                  else f"{n_exp} experimental batches")
    fig.suptitle(
        f"Partial Dependence of {TGT_LABELS[tname]}\non Composition Variables (wt%)\n"
        f"(marginalised over {_ref_label}, n = {len(_pdp_X)}; {_fit_src})",
        fontsize=_FS_PDP_SUPTITLE, fontweight="bold", y=0.86
    )
    plt.tight_layout(rect=[0, 0, 1, 0.82])
    _stem = f"PDP_{tname}" if _pdp_tag == "synthetic" else f"PDP_{tname}_experimental"
    savefig(fig, _stem)
    print(f"  Saved: {_stem}.pdf / .png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODELDIR / "forward_model.joblib")
with open(MODELDIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nAll figures → plots/")
print("Model      → models/forward_model.joblib")
print("Experimental-only nested LOO-CV results → data/experimental_loo_cv_results.csv")
print("Done.")