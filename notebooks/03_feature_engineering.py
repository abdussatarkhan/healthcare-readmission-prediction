# %% [markdown]
# # Project 1: Preventable Readmissions Risk Stratifier
# ## Notebook 03: Clinical Feature Engineering Walkthrough
# 
# **Objective:** Transform longitudinal EHR events, diagnostic hierarchies, and discharge timestamps into 45+ robust predictive signals.

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

from scripts.utils import load_config, setup_logger, set_all_seeds, save_dataframe
from scripts.feature_engineering import FeatureEngineer

config = load_config()
set_all_seeds(42)

# %% [markdown]
# ### 1. Load Preprocessed Cohort
# We load the cohort produced by `preprocessing.py` containing patient demographics, admissions, diagnoses, and lab summaries.

# %%
processed_dir = Path(config["paths"]["processed_data_dir"])
cohort_file = processed_dir / "preprocessed_cohort.parquet"

if not cohort_file.is_file():
    print("Preprocessed file not found. Running preprocessor...")
    from scripts.preprocessing import main as prep_main
    prep_main()

df_cohort = pd.read_parquet(cohort_file)
print(f"Loaded cohort: {df_cohort.shape[0]:,} rows, {df_cohort.shape[1]} initial columns")

# %% [markdown]
# ### 2. Step 1: Charlson Comorbidity Index (CCI) & Age-Adjusted Severity
# The Charlson Comorbidity Index quantifies cumulative 1-year mortality risk through weighted chronic diagnoses.

# %%
fe = FeatureEngineer(config)
df_feat = fe.calculate_charlson_comorbidity_index(df_cohort)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(df_feat["charlson_comorbidity_index"], discrete=True, color="#2c3e50", ax=ax1)
ax1.set_title("Charlson Comorbidity Index (Raw Score)", fontsize=11, weight="bold")
ax1.set_xlabel("CCI Score")

sns.histplot(df_feat["charlson_age_adjusted_score"], discrete=True, color="#16a085", ax=ax2)
ax2.set_title("Age-Adjusted Charlson Score", fontsize=11, weight="bold")
ax2.set_xlabel("Age-Adjusted Score")

plt.tight_layout()
plt.show()

print(f"Patients with High Charlson Risk (Score >= 5): {df_feat['high_charlson_risk'].mean():.1%}")

# %% [markdown]
# ### 3. Step 2: Prior Healthcare Utilization & Readmission Velocity
# Prior inpatient and ED utilization is the strongest historical predictor of future readmissions.

# %%
df_feat = fe.compute_prior_utilization_features(df_feat)

util_cols = ["prior_admissions_count", "days_since_prev_discharge", "has_prior_admission_30d", "is_frequent_flyer"]
print("Prior Utilization Summary:")
print(df_feat[util_cols].describe().round(2).T[["mean", "std", "min", "50%", "max"]])

plt.figure(figsize=(7, 4))
sns.barplot(data=df_feat, x="is_frequent_flyer", y="readmitted_30d", palette=["#2980b9", "#c0392b"])
plt.xticks([0, 1], ["Standard Utilization (<3 stays)", "Frequent Flyer (>=3 stays)"])
plt.ylabel("30-Day Readmission Rate")
plt.title("Readmission Rate by Historical Utilization Category", fontsize=11, weight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Step 3: Laboratory Trajectories and Physiological Instability
# Derives prerenal azotemia flags (BUN/Creatinine ratio), anemia, and acute laboratory instability ratios.

# %%
df_feat = fe.compute_laboratory_indices(df_feat)

fig, ax = plt.subplots(figsize=(8, 4))
sns.kdeplot(data=df_feat[df_feat["readmitted_30d"] == 0]["bun_to_creatinine_ratio"], label="Not Readmitted", fill=True, color="#3498db")
sns.kdeplot(data=df_feat[df_feat["readmitted_30d"] == 1]["bun_to_creatinine_ratio"], label="Readmitted", fill=True, color="#e74c3c")
ax.axvline(20.0, color="darkred", linestyle="--", label="Prerenal Threshold (Ratio >= 20)")
ax.set_title("BUN-to-Creatinine Ratio Distribution", fontsize=12, weight="bold")
ax.set_xlabel("BUN / Creatinine Ratio")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Step 4: Care Transition & Discharge Vulnerability Signals
# Captures operational risk during vulnerable discharge intervals (after-hours, Friday afternoon, weekends).

# %%
df_feat = fe.compute_discharge_vulnerability_features(df_feat)
df_feat = fe.compute_clinical_interactions_and_multimorbidity(df_feat)

syndromes = ["cardiorenal_syndrome", "cardiopulmonary_overlap", "discharge_friday_afternoon"]
for syn in syndromes:
    if syn in df_feat.columns:
        rate_pos = df_feat[df_feat[syn] == 1]["readmitted_30d"].mean() * 100
        rate_neg = df_feat[df_feat[syn] == 0]["readmitted_30d"].mean() * 100
        print(f"{syn}: Positive={rate_pos:.1f}%, Negative={rate_neg:.1f}% (Diff: +{rate_pos-rate_neg:.1f}%)")

# %% [markdown]
# ### 6. Multicollinearity & Feature Correlation Analysis

# %%
# Select sample of high-impact numerical features for correlation check
check_features = [
    "anchor_age", "charlson_age_adjusted_score", "prior_admissions_count",
    "los_days", "bun_to_creatinine_ratio", "chronic_condition_count",
    "age_x_charlson", "abnormal_lab_ratio", "readmitted_30d"
]

corr = df_feat[check_features].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Core Clinical & Operational Predictors", fontsize=12, weight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7. Final Feature Matrix Export
# Save transformed feature matrix to `data/processed/features_matrix.parquet`.

# %%
df_final, feature_names = fe.transform(df_cohort)
print(f"\nFinal feature extraction complete!")
print(f"Total predictors available: {len(feature_names)}")
print(f"Matrix shape: {df_final.shape}")
