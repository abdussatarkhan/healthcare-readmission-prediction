# %% [markdown]
# # Project 1: Preventable Readmissions Risk Stratifier
# ## Notebook 01: Data Ingestion and Source Harmonization
# 
# **Author:** Clinical Data Science Team  
# **Dataset:** Centers for Medicare & Medicaid Services (CMS) HRRP + MIMIC-IV v2.2 (Beth Israel Deaconess Medical Center)  
# **Objective:** Extract, validate, and profile multi-source electronic health record (EHR) data and federal benchmark registries.

# %%
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import load_config, setup_logger, compute_missingness_report, set_all_seeds
from scripts.data_collection import CMSDataCollector, MIMICDataLoader

logger = setup_logger("notebook_01_ingestion")
set_all_seeds(42)
config = load_config()

print(f"Project: {config['project']['title']}")
print(f"Root directory: {PROJECT_ROOT}")

# %% [markdown]
# ### 1. CMS Hospital Readmissions Reduction Program (HRRP) Ingestion
# The CMS HRRP records 30-day readmission performance and excess readmission ratios (ERR) across US acute care hospitals.
# We query the open federal endpoint or load local benchmark caches.

# %%
cms_collector = CMSDataCollector(config)
cms_df = cms_collector.fetch_hrrp_metrics(limit=5000)

print(f"CMS Data Retrieved: {cms_df.shape[0]:,} records across {cms_df.shape[1]} columns")
cms_df.head()

# %% [markdown]
# ### 2. CMS Readmission Metric Profiling
# Inspect the distribution of Excess Readmission Ratios (ERR) across target conditions (AMI, HF, COPD, PNA).
# An ERR > 1.00 indicates readmission performance worse than national risk-standardized expectations, triggering Medicare reimbursement penalties.

# %%
if "excess_readmission_ratio" in cms_df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=cms_df, x="excess_readmission_ratio", kde=True, bins=40, color="#1f77b4")
    plt.axvline(1.0, color="red", linestyle="--", linewidth=2, label="CMS Penalty Threshold (ERR = 1.00)")
    plt.title("Distribution of Hospital Excess Readmission Ratios (HRRP)", fontsize=13, weight="bold")
    plt.xlabel("Excess Readmission Ratio (ERR)")
    plt.ylabel("Hospital Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    penalty_share = (cms_df["excess_readmission_ratio"] > 1.0).mean() * 100
    print(f"Percentage of hospitals penalized (ERR > 1.0): {penalty_share:.1f}%")

# %% [markdown]
# ### 3. MIMIC-IV Core Hospital Tables Ingestion
# We now load patient records, hospital admissions, ICD diagnoses, and laboratory events from PhysioNet MIMIC-IV schema.

# %%
mimic_loader = MIMICDataLoader(config)

df_patients = mimic_loader.load_table("patients", module="hosp")
df_admissions = mimic_loader.load_table("admissions", module="hosp")
df_diagnoses = mimic_loader.load_table("diagnoses_icd", module="hosp")
df_labs = mimic_loader.load_table("labevents", module="hosp")

print(f"Patients Table:   {df_patients.shape[0]:,} rows, {df_patients.shape[1]} columns")
print(f"Admissions Table: {df_admissions.shape[0]:,} rows, {df_admissions.shape[1]} columns")
print(f"Diagnoses Table:  {df_diagnoses.shape[0]:,} rows, {df_diagnoses.shape[1]} columns")
print(f"Lab Events Table: {df_labs.shape[0]:,} rows, {df_labs.shape[1]} columns")

# %% [markdown]
# ### 4. Table Schemas and Missingness Profiling
# Assess missing value percentages and data types across the primary clinical tables.

# %%
print("--- Admissions Missing Value Report ---")
adm_missing = compute_missingness_report(df_admissions)
print(adm_missing.head(10))

print("\n--- Patients Missing Value Report ---")
pts_missing = compute_missingness_report(df_patients)
print(pts_missing)

# %% [markdown]
# ### 5. Patient Cohort Linkage and Unique Identifier Verification
# Ensure all admissions map to valid patient records, and assess distribution of admissions per patient.

# %%
unique_subjects_adm = df_admissions["subject_id"].nunique()
unique_subjects_pts = df_patients["subject_id"].nunique()
overlap = set(df_admissions["subject_id"]).intersection(set(df_patients["subject_id"]))

print(f"Unique patients in Admissions: {unique_subjects_adm:,}")
print(f"Unique patients in Patients:   {unique_subjects_pts:,}")
print(f"Exact intersection count:      {len(overlap):,} ({(len(overlap)/unique_subjects_adm)*100:.1f}%)")

admissions_per_pt = df_admissions.groupby("subject_id")["hadm_id"].count()
print(f"Mean admissions per patient:   {admissions_per_pt.mean():.2f}")
print(f"Max admissions for one pt:     {admissions_per_pt.max():,}")
print(f"Patients with multiple stays:  {(admissions_per_pt > 1).sum():,} ({(admissions_per_pt > 1).mean()*100:.1f}%)")

# %% [markdown]
# ### 6. Temporal Integrity Validation
# Check for timestamp anomalies: admissions where discharge occurs before admission (invalid clinical timeline).

# %%
df_admissions["admittime"] = pd.to_datetime(df_admissions["admittime"])
df_admissions["dischtime"] = pd.to_datetime(df_admissions["dischtime"])

duration_check = (df_admissions["dischtime"] < df_admissions["admittime"]).sum()
print(f"Anomalous records (dischtime < admittime): {duration_check}")
assert duration_check == 0, "Temporal anomaly detected in admissions timeline!"

# %% [markdown]
# ### 7. Staging and Conclusion
# Raw data ingestion and initial health checks are complete. Data is ready for exploratory clinical analysis in `02_eda.py`.
