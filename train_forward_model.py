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

DYNAMIC SAMPLE-SIZE REPORTING (this version)
  Every place this script previously printed or plotted the synthetic /
  experimental sample counts used a literal number ("1000 synthetic",
  "~96% synthetic") that silently went stale whenever generate_dataset.py's
  N_SYNTHETIC was changed. All such counts are now computed at runtime
  from the actual loaded dataset.csv / metadata.json (len(synth_idx),
  n_exp, and the percentage derived from them), so changing N_SYNTHETIC
  in generate_dataset.py and re-running the pipeline updates this script's
  console output and plot text automatically — no manual edits needed here.

KEY DECISIONS & JUSTIFICATIONS (retained from prior version, updated)
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
            annot_exp.loc[MAT_SHORT.get(c, c), TGT_LABELS.get(t, t)] += "*"

fig, ax = plt.subplots(figsize=(11, 10))
sns.heatmap(corr_io_exp_display, annot=annot_exp.values, fmt="",
            cmap="coolwarm", center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 16}, ax=ax)
ax.set_title(
    f"Pearson Correlation: Composition vs. Target Properties\n"
    f"EXPERIMENTAL BATCHES ONLY (n={n_exp}) — * = p<0.05\n"
    "Noisy at this sample size; this is the honest real-world picture,\n"
    "not the synthetic-dominated heatmap above",
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

    final_model = Pipeline([("preproc", preproc), ("reg", candidate_models[best_name])])
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    mae_combined = np.mean(np.abs(y_test - y_pred), axis=0)
    r2_combined  = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(3)]

    print("\n[Legacy combined metric — synthetic + experimental together]")
    print("CAUTION: majority synthetic; do NOT cite this as validation evidence")
    print("(Reviewer 1, comment 2). See STEP B below for the real evidence.")
    for t, m, r in zip(TARGET_COLS, mae_combined, r2_combined):
        print(f"  {t:<25s}  MAE={m:.4f}  R²={r:.4f}")

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

def run_experimental_loo(model_ctor, X_synth, y_synth, X_e, y_e):
    n = len(X_e)
    preds         = np.zeros((n, len(TARGET_COLS)))
    errs          = np.zeros((n, len(TARGET_COLS)))
    baseline_errs = np.zeros((n, len(TARGET_COLS)))
    for i in range(n):
        train_exp_mask = np.ones(n, dtype=bool)
        train_exp_mask[i] = False
        X_tr = pd.concat([X_synth, X_e.loc[train_exp_mask]], ignore_index=True)
        y_tr = np.vstack([y_synth, y_e[train_exp_mask]])
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

if HAS_SYNTHETIC:
    model_ctors = {"LinearRegression": _make_ctor(candidate_models["LinearRegression"])}
    if best_name != "LinearRegression":
        model_ctors[best_name] = _make_ctor(candidate_models[best_name])
else:
    # No synthetic sanity check was possible in STEP A, so EVERY candidate
    # is run through experimental-only nested LOO-CV here, and the best
    # one (by mean skill score across targets) is selected as best_name.
    model_ctors = {name: _make_ctor(model) for name, model in candidate_models.items()}

exp_loo_results = {}
for name, ctor in model_ctors.items():
    preds, errs, baseline_errs = run_experimental_loo(
        ctor, X_synth_full.reset_index(drop=True), y_synth_full, X_exp, y_exp.values
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

if not HAS_SYNTHETIC:
    def _avg_skill(name):
        _, errs, baseline_errs = exp_loo_results[name]
        skills = [1.0 - errs[:, i].mean() / baseline_errs[:, i].mean()
                  for i in range(len(TARGET_COLS))]
        return float(np.mean(skills))

    best_name = max(exp_loo_results, key=_avg_skill)
    print(f"\n[STEP A fallback — architecture chosen here, inside STEP B] "
          f"Best architecture: {best_name}  (mean skill = {_avg_skill(best_name):+.2f}). "
          "CAUTION: with N_SYNTHETIC=0 the same "
          f"{n_exp} experimental batches were used both to CHOOSE this "
          "architecture and to REPORT its performance below — treat these "
          "skill scores as optimistic relative to the normal pipeline, "
          "where architecture is chosen on an independent synthetic check.")

    # Fit the production model on ALL experimental batches — there is no
    # synthetic hold-out to reserve any of them for a separate test set.
    final_model = Pipeline([("preproc", preproc), ("reg", candidate_models[best_name])])
    final_model.fit(X_exp, y_exp.values)

print("\n[STEP B SUMMARY] Report the experimental-only nested LOO-CV numbers "
      "above in the manuscript, NOT the combined-test-set R² from STEP A. "
      "A model without skill > 0.05 here should not be described as having "
      "learned a real composition-property relationship, regardless of "
      "what the combined-test-set R² shows.")

# Defined once here (not inside the feature-importance try/except below) so
# it's guaranteed to exist for the PDP section even if that block errors out.
_fit_src = ("fitted on synthetic data" if HAS_SYNTHETIC
            else "fitted on experimental data only (N_SYNTHETIC=0)")

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
        f"Mean Feature Importance Across All Target Properties\n"
        f"({best_name}; {_fit_src} — see STEP B for real-world validity)",
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
                   alpha=0.55, s=40, color="teal", label="Synthetic")
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
            "No synthetic data (N_SYNTHETIC=0)\n"
            f"{exp_label}",
            fontsize=_FS_AX
        )
        ax.legend(fontsize=_FS_LABEL)

plt.suptitle("Predicted vs. Measured Values: Forward Model Parity Plots",
             fontsize=_FS_TITLE, fontweight="bold")
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

for t_idx, tname in enumerate(TARGET_COLS):
    n_comp = len(comp_cols)
    ncols  = 4
    nrows  = math.ceil(n_comp / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7.2, nrows * 6.4), dpi=250)
    axes = axes.flatten()
    for ax_idx, feat in enumerate(comp_cols):
        ax = axes[ax_idx]
        grid, means = compute_pdp(final_model, X_train, feat)
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
    fig.suptitle(
        f"Partial Dependence of {TGT_LABELS[tname]}\non Composition Variables (wt%)\n"
        f"(marginalised over {_train_set_label} training set, n = {len(X_train)}; "
        f"{_fit_src} — see STEP B for real-world validity)",
        fontsize=_FS_PDP_SUPTITLE, fontweight="bold", y=0.96
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    savefig(fig, f"PDP_{tname}")
    print(f"  Saved: PDP_{tname}.pdf / .png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(final_model, MODELDIR / "forward_model.joblib")
with open(MODELDIR / "feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\nAll figures → plots/")
print("Model      → models/forward_model.joblib")
print("Experimental-only nested LOO-CV results → data/experimental_loo_cv_results.csv")
print("Done.")