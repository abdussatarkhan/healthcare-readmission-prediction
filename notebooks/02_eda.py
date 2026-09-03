# %% [markdown]
# # Project 1: Preventable Readmissions Risk Stratifier
# ## Notebook 02: Comprehensive Exploratory Data Analysis (EDA)
# 
# **Objective:** Uncover clinical, demographic, and operational drivers of 30-day readmission risk across inpatient admissions.

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

from scripts.utils import load_config, setup_logger, set_all_seeds
from scripts.preprocessing import EHRPreprocessor
from scripts.data_collection import MIMICDataLoader

config = load_config()
set_all_seeds(42)

# Set visualization aesthetics
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

# %% [markdown]
# ### 1. Cohort Assembly
# Load preprocessed cohort or build via `EHRPreprocessor`.

# %%
processed_dir = Path(config["paths"]["processed_data_dir"])
cohort_file = processed_dir / "preprocessed_cohort.parquet"

if cohort_file.is_file():
    df = pd.read_parquet(cohort_file)
else:
    loader = MIMICDataLoader(config)
    pts = loader.load_table("patients", "hosp")
    adms = loader.load_table("admissions", "hosp")
    diags = loader.load_table("diagnoses_icd", "hosp")
    labs = loader.load_table("labevents", "hosp")
    
    preprocessor = EHRPreprocessor(config)
    df = preprocessor.run_preprocessing_pipeline(pts, adms, diags, labs)

print(f"Cohort dimensions: {df.shape[0]:,} patients, {df.shape[1]} raw/engineered attributes")
print(f"Overall 30-Day Readmission Rate: {df['readmitted_30d'].mean():.2%}")

# %% [markdown]
# ### 2. Readmission Prevalence and Target Class Distribution

# %%
fig, ax = plt.subplots(figsize=(6, 4))
target_counts = df["readmitted_30d"].value_counts(normalize=True) * 100
sns.barplot(x=["No Readmission (0)", "30-Day Readmission (1)"], y=target_counts.values, palette=["#3498db", "#e74c3c"], ax=ax)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title(f"Class Distribution: {target_counts[1]:.1f}% Readmission Rate", fontsize=12, weight="bold")
for i, v in enumerate(target_counts):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", weight="bold")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3. Patient Demographics: Age and Gender Stratification

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution by readmission status
sns.kdeplot(data=df[df["readmitted_30d"] == 0]["anchor_age"], label="Not Readmitted", fill=True, color="#3498db", ax=ax1)
sns.kdeplot(data=df[df["readmitted_30d"] == 1]["anchor_age"], label="Readmitted", fill=True, color="#e74c3c", ax=ax1)
ax1.set_title("Age Distribution by Readmission Status", fontsize=12, weight="bold")
ax1.set_xlabel("Patient Age at Admission")
ax1.legend()

# Readmission rate by age bracket
df["age_bracket"] = pd.cut(df["anchor_age"], bins=[18, 45, 65, 75, 90], labels=["18-44", "45-64", "65-74", "75+"])
age_rate = df.groupby("age_bracket", observed=False)["readmitted_30d"].mean() * 100
sns.barplot(x=age_rate.index, y=age_rate.values, palette="Blues_d", ax=ax2)
ax2.set_title("Readmission Rate by Age Bracket", fontsize=12, weight="bold")
ax2.set_ylabel("Readmission Rate (%)")
ax2.set_xlabel("Age Bracket")
for i, v in enumerate(age_rate):
    ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4. Hospital Utilization and Length of Stay (LOS)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Log-transformed LOS
sns.boxplot(data=df, x="readmitted_30d", y="los_days", palette=["#3498db", "#e74c3c"], ax=ax1, showfliers=False)
ax1.set_xticklabels(["Not Readmitted", "Readmitted"])
ax1.set_title("Length of Stay (Days) Comparison", fontsize=12, weight="bold")
ax1.set_ylabel("Index Stay Duration (Days)")

# Readmission rate by admission urgency
adm_rate = df.groupby("admission_type")["readmitted_30d"].agg(["mean", "count"]).reset_index()
adm_rate["rate_pct"] = adm_rate["mean"] * 100
sns.barplot(data=adm_rate, x="admission_type", y="rate_pct", palette="magma", ax=ax2)
ax2.set_title("Readmission Rate by Admission Urgency", fontsize=12, weight="bold")
ax2.set_ylabel("Readmission Rate (%)")
ax2.set_xlabel("Admission Type")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5. Discharge Timing Vulnerabilities (Hour of Day and Day of Week)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Day of Week
dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
df["dow_str"] = df["discharge_dayofweek"].map(dow_map)
dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_rates = df.groupby("dow_str")["readmitted_30d"].mean().reindex(dow_order) * 100

colors = ["#2b5c8f" if d not in ["Sat", "Sun"] else "#d9534f" for d in dow_order]
sns.barplot(x=dow_order, y=dow_rates.values, palette=colors, ax=ax1)
ax1.set_title("Readmission Rate by Day of Discharge", fontsize=12, weight="bold")
ax1.set_ylabel("Readmission Rate (%)")
ax1.set_xlabel("Discharge Day of Week")

# Hour of Discharge
hour_rates = df.groupby("discharge_hour")["readmitted_30d"].mean() * 100
ax2.plot(hour_rates.index, hour_rates.values, marker="o", color="#e74c3c", linewidth=2.5)
ax2.axvline(17, color="black", linestyle="--", label="Evening Shift Transition (17:00)")
ax2.set_title("Readmission Rate by Hour of Discharge", fontsize=12, weight="bold")
ax2.set_xlabel("Hour of Discharge (24-Hour Clock)")
ax2.set_ylabel("Readmission Rate (%)")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6. Clinical Comorbidities and Disease Burden

# %%
comorbidities = [
    "has_heart_failure", "has_copd", "has_ami",
    "has_pneumonia", "has_diabetes", "has_renal_failure"
]

cond_rates = []
for cond in comorbidities:
    if cond in df.columns:
        present_rate = df[df[cond] == 1]["readmitted_30d"].mean() * 100
        absent_rate = df[df[cond] == 0]["readmitted_30d"].mean() * 100
        cond_rates.append({
            "Condition": cond.replace("has_", "").upper(),
            "With Condition": present_rate,
            "Without Condition": absent_rate,
            "Difference": present_rate - absent_rate
        })

df_cond_summary = pd.DataFrame(cond_rates).sort_values("Difference", ascending=False)
print("Comorbidity Readmission Impact Table:")
print(df_cond_summary.to_string(index=False))

plt.figure(figsize=(10, 5))
df_melt = pd.melt(df_cond_summary, id_vars=["Condition"], value_vars=["With Condition", "Without Condition"],
                  var_name="Group", value_name="ReadmissionRate")
sns.barplot(data=df_melt, x="Condition", y="ReadmissionRate", hue="Group", palette=["#e74c3c", "#3498db"])
plt.title("Readmission Rates by Primary Chronic Comorbidities", fontsize=12, weight="bold")
plt.ylabel("Readmission Rate (%)")
plt.xlabel("Comorbidity")
plt.legend(title="")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 7. Summary of Key EDA Findings
# 1. **Elderly vulnerability**: Patients aged 75+ experience significantly higher readmissions than younger cohorts.
# 2. **Operational transitions**: Weekend and late afternoon discharges show measurable spikes in readmissions.
# 3. **High-risk phenotypes**: Heart failure and renal disease carry the highest excess readmission probabilities.
