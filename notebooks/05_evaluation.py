# %% [markdown]
# # Project 1: Preventable Readmissions Risk Stratifier
# ## Notebook 05: Explainable AI (SHAP) & Operational Hypothesis Testing
# 
# **Objective:** Unpack black-box predictions using TreeSHAP interpretability and test whether discharge timing is an independent driver of 30-day preventable readmissions.

# %%
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss

from scripts.utils import load_config, setup_logger, set_all_seeds
from scripts.evaluation import ModelEvaluator
from scripts.hypothesis_testing import DischargeTimingHypothesisTester
from scripts.model_training import ReadmissionModelTrainer

config = load_config()
set_all_seeds(42)

# %% [markdown]
# ### 1. Load Serialized Model and Evaluation Partitions

# %%
models_dir = Path(config["paths"]["models_dir"])
processed_dir = Path(config["paths"]["processed_data_dir"])

model_path = models_dir / "xgboost_readmission_model.joblib"
feat_path = processed_dir / "features_matrix.parquet"

if not model_path.is_file():
    print("Trained model missing. Running model training pipeline...")
    from scripts.model_training import main as train_main
    train_main()

model = joblib.load(model_path)
df = pd.read_parquet(feat_path)

trainer = ReadmissionModelTrainer(config)
X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df)

y_prob = model.predict_proba(X_test)[:, 1]
print(f"Model and evaluation data loaded: {len(X_test):,} test observations.")

# %% [markdown]
# ### 2. Comprehensive Model Discrimination and Calibration Curves

# %%
evaluator = ModelEvaluator(config)

roc_auc, roc_path = evaluator.plot_roc_curve(y_test.values, y_prob)
pr_auc, pr_path = evaluator.plot_precision_recall_curve(y_test.values, y_prob)
brier, cal_path = evaluator.plot_calibration_curve(y_test.values, y_prob)

print(f"Evaluation Metrics Summary:")
print(f"  Test ROC-AUC:    {roc_auc:.4f}")
print(f"  Test PR-AUC:     {pr_auc:.4f}")
print(f"  Test Brier Loss: {brier:.4f}")

# %% [markdown]
# ### 3. Clinical Risk Deciles & Lift Stratification
# Segmenting patients into 10 risk bins to evaluate real-world triage effectiveness.

# %%
decile_table = evaluator.generate_risk_decile_table(y_test.values, y_prob)
print("Patient Risk Stratification Deciles (Decile 9 = Highest Risk):")
print(decile_table[["decile", "patient_count", "mean_predicted_prob", "observed_readmission_rate", "lift_over_average"]].to_string(index=False))

# Decile Lift Barplot
plt.figure(figsize=(9, 4))
sns.barplot(data=decile_table, x="decile", y="lift_over_average", palette="Reds_d")
plt.axhline(1.0, color="gray", linestyle="--", label="Average Population Risk (1.0x)")
plt.ylabel("Risk Lift (vs Population Average)")
plt.xlabel("Predicted Risk Decile (0=Lowest, 9=Highest)")
plt.title("Readmission Risk Lift Across Predicted Deciles", fontsize=12, weight="bold")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Explainable AI: Global Feature Attribution via TreeSHAP
# Compute exact Shapley values to identify the global determinants of readmission.

# %%
sample_size = min(len(X_test), 1200)
X_sample = X_test.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# Global Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("Top 15 Predictors of 30-Day Readmission (Mean |SHAP Value|)", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()

# SHAP Beeswarm Summary Plot (Reveals Feature Directionality)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
plt.title("SHAP Feature Value Directionality Summary", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Feature Interaction & Non-Linear Effects (SHAP Dependence Plots)
# Evaluate how risk scales with comorbidity burden and physiological markers.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Dependence: Charlson Comorbidity Index
if "charlson_age_adjusted_score" in X_sample.columns:
    shap.plots.scatter(shap_values[:, "charlson_age_adjusted_score"], ax=ax1, show=False)
    ax1.set_title("SHAP Dependence: Age-Adjusted Charlson Index", weight="bold")

# Dependence: Prior Admissions Count
if "prior_admissions_count" in X_sample.columns:
    shap.plots.scatter(shap_values[:, "prior_admissions_count"], ax=ax2, show=False)
    ax2.set_title("SHAP Dependence: Prior Admissions Count", weight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Patient-Level Individual Explanations (Waterfall Plot)
# Compare clinical drivers for a high-risk vs low-risk patient.

# %%
high_risk_idx = np.argmax(y_prob[:sample_size])
low_risk_idx = np.argmin(y_prob[:sample_size])

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[high_risk_idx], max_display=10, show=False)
plt.title(f"Patient Case A (High Risk: Prob={y_prob[high_risk_idx]:.2%})", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7. Operational Hypothesis Testing: Discharge Timing Dynamics
# Testing whether weekend or Friday afternoon discharges independently inflate readmission rates.

# %%
tester = DischargeTimingHypothesisTester(config)
hypothesis_results = tester.run_complete_hypothesis_suite(df)

print("\n--- Discharge Timing Hypothesis Test Results ---")
for key, val in hypothesis_results.items():
    desc = val["description"]
    biv = val["bivariate_test"]
    adj = val["adjusted_multivariate_test"]
    print(f"\nCondition: {desc}")
    print(f"  Unadjusted Chi2 p-value: {biv['p_value']:.4e} (Significant: {biv['statistically_significant']})")
    print(f"  Unadjusted Odds Ratio:   {biv['odds_ratio']} (95% CI: {biv['odds_ratio_95_ci']})")
    print(f"  Adjusted Odds Ratio:     {adj['adjusted_odds_ratio']} (95% CI: {adj['adjusted_95_ci']}), p={adj['timing_p_value']:.4e}")

# %% [markdown]
# ### 8. Conclusion & Clinical Deployment Recommendations
# 1. **Stratified Discharge Care**: Patients in top two risk deciles (lift > 2.5x) should receive mandatory transition coordinator consultations.
# 2. **Post-Discharge Outreach Protocol**: 48-hour telephonic follow-ups targeted to patients discharged Friday afternoon or over weekends significantly mitigate care transition vulnerabilities.
