# %% [markdown]
# # Project 1: Preventable Readmissions Risk Stratifier
# ## Notebook 04: Predictive Modeling, Hyperparameter Optimization, and Model Selection
# 
# **Objective:** Develop, optimize, and cross-validate state-of-the-art gradient boosted trees (XGBoost) against classical baselines.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    classification_report
)

from scripts.utils import load_config, setup_logger, set_all_seeds
from scripts.model_training import ReadmissionModelTrainer

config = load_config()
set_all_seeds(42)

# %% [markdown]
# ### 1. Dataset Loading and Stratified Partitioning

# %%
processed_dir = Path(config["paths"]["processed_data_dir"])
feat_file = processed_dir / "features_matrix.parquet"

if not feat_file.is_file():
    print("Features matrix missing. Running feature engineering...")
    from scripts.feature_engineering import main as fe_main
    fe_main()

df = pd.read_parquet(feat_file)
trainer = ReadmissionModelTrainer(config)
X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df)

print(f"Features: {len(feature_cols)}")
print(f"Training split: {X_train.shape[0]:,} records (Readmit: {y_train.mean():.2%})")
print(f"Test split:     {X_test.shape[0]:,} records (Readmit: {y_test.mean():.2%})")

# %% [markdown]
# ### 2. Benchmark Model Comparison: Logistic Regression vs Random Forest vs XGBoost
# We compare baseline models across discrimination (AUC-ROC), precision-recall (PR-AUC), and calibration (Brier Score).

# %%
models = {
    "Logistic Regression (L2)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    ),
    "XGBoost (Baseline)": xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, scale_pos_weight=3.5, eval_metric="auc"
    )
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc_val = roc_auc_score(y_test, probs)
    prauc_val = average_precision_score(y_test, probs)
    brier_val = brier_score_loss(y_test, probs)
    
    results.append({
        "Model": name,
        "ROC-AUC": round(auc_val, 4),
        "PR-AUC": round(prauc_val, 4),
        "Brier Score": round(brier_val, 4)
    })

df_benchmarks = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("Model Benchmark Comparison on Test Partition:")
print(df_benchmarks.to_string(index=False))

# %% [markdown]
# ### 3. Bayesian Hyperparameter Tuning via Optuna
# Search over max_depth, learning_rate, subsample, regularization, and class weighting to maximize 5-fold cross-validated AUC.

# %%
print("Executing Bayesian search with Optuna...")
best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=15, timeout=300)
print("\nTop Hyperparameter Configuration Identified:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ### 4. Stratified 5-Fold Cross-Validation Ensemble
# Train across 5 stratified folds to verify generalization consistency and generate out-of-fold probability estimates.

# %%
fold_models, oof_preds, cv_summary = trainer.train_cross_validated_ensemble(X_train, y_train, best_params)

print(f"\nOOF Cross-Validation Summary:")
print(f"  Mean Fold AUC:   {cv_summary['mean_fold_auc']:.4f} (+/- {cv_summary['std_fold_auc']:.4f})")
print(f"  Overall OOF AUC: {cv_summary['oof_auc']:.4f}")
print(f"  OOF PR-AUC:      {cv_summary['oof_prauc']:.4f}")
print(f"  OOF Brier Score: {cv_summary['oof_brier']:.4f}")

# %% [markdown]
# ### 5. Final Model Fitting & Decision Threshold Calibration
# A default threshold of 0.50 is suboptimal for imbalanced clinical readmission data. We evaluate F1 score and clinical utility across varying thresholds.

# %%
final_model = trainer.fit_final_model(X_train, y_train, X_test, y_test, best_params)
test_probs = final_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.1, 0.9, 17)
threshold_metrics = []

for t in thresholds:
    preds = (test_probs >= t).astype(int)
    f1 = f1_score(y_test, preds, zero_division=0)
    pos_flagged = preds.mean() * 100
    threshold_metrics.append({"threshold": round(t, 2), "f1_score": round(f1, 4), "patients_flagged_pct": round(pos_flagged, 1)})

df_thresh = pd.DataFrame(threshold_metrics)

plt.figure(figsize=(9, 4))
plt.plot(df_thresh["threshold"], df_thresh["f1_score"], marker="o", color="#2980b9", lw=2, label="F1 Score")
plt.axvline(df_thresh.loc[df_thresh["f1_score"].idxmax()]["threshold"], color="red", linestyle="--",
            label=f"Optimal F1 Threshold ({df_thresh.loc[df_thresh['f1_score'].idxmax()]['threshold']})")
plt.xlabel("Readmission Decision Threshold")
plt.ylabel("F1 Score")
plt.title("Classification Performance Across Clinical Decision Thresholds", fontsize=12, weight="bold")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Model Serialization
# Save the tuned model, feature names, and metadata to disk.

# %%
test_metrics = trainer.evaluate_test_set(final_model, X_test, y_test)
saved_path = trainer.serialize_pipeline(final_model, feature_cols, best_params, {**cv_summary, **test_metrics})
print(f"Model successfully saved to {saved_path}")
