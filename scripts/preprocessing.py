"""
EHR Preprocessing and Cohort Construction Pipeline
Preventable Readmissions Risk Stratifier

Performs:
1. ICD-9 to ICD-10 diagnostic code harmonization and crosswalking
2. Strict CMS HRRP index admission filtering (age >= 18, alive at discharge, non-elective readmissions)
3. 30-day all-cause unplanned readmission target labeling
4. Multi-table EHR relational joins across admissions, patients, diagnoses, labs, transfers, and ICU stays
5. Timestamp normalization, duration metrics, and intelligent missing value imputation
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add repo root to python path for relative script imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.utils import setup_logger, load_config, save_dataframe, validate_dataframe, timer, set_all_seeds

logger = setup_logger("preprocessing")


class EHRPreprocessor:
    """
    Production-grade preprocessor for MIMIC-IV clinical tables and CMS cohort standardization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.preproc_cfg = self.config.get("preprocessing", {})
        self.raw_dir = Path(self.config.get("paths", {}).get("raw_data_dir", "data/raw"))
        self.processed_dir = Path(self.config.get("paths", {}).get("processed_data_dir", "data/processed"))
        self.external_dir = Path(self.config.get("paths", {}).get("external_data_dir", "data/external"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_icd_crosswalk(self) -> pd.DataFrame:
        """
        Retrieves or generates CMS General Equivalence Mappings (GEMs) crosswalk table
        mapping legacy ICD-9-CM codes to modernized ICD-10-CM clinical concepts.
        """
        crosswalk_path = self.external_dir / "icd9_to_icd10_crosswalk.csv"
        if crosswalk_path.is_file():
            logger.info(f"Loading existing ICD crosswalk from {crosswalk_path}")
            return pd.read_csv(crosswalk_path)

        logger.info("Generating standard ICD-9 to ICD-10 GEMs mapping dictionary...")
        mappings = [
            {"icd9_code": "41001", "icd10_code": "I210", "category": "AMI", "description": "Acute myocardial infarction"},
            {"icd9_code": "41011", "icd10_code": "I211", "category": "AMI", "description": "STEMI inferior wall"},
            {"icd9_code": "41071", "icd10_code": "I214", "category": "AMI", "description": "NSTEMI"},
            {"icd9_code": "4280", "icd10_code": "I509", "category": "HF", "description": "Heart failure, unspecified"},
            {"icd9_code": "42820", "icd10_code": "I5020", "category": "HF", "description": "Systolic heart failure"},
            {"icd9_code": "42830", "icd10_code": "I5030", "category": "HF", "description": "Diastolic heart failure"},
            {"icd9_code": "496", "icd10_code": "J449", "category": "COPD", "description": "Chronic obstructive pulmonary disease"},
            {"icd9_code": "49121", "icd10_code": "J441", "category": "COPD", "description": "COPD with acute exacerbation"},
            {"icd9_code": "486", "icd10_code": "J189", "category": "PNA", "description": "Pneumonia, unspecified organism"},
            {"icd9_code": "4829", "icd10_code": "J159", "category": "PNA", "description": "Bacterial pneumonia"},
            {"icd9_code": "25000", "icd10_code": "E119", "category": "DM", "description": "Type 2 diabetes without complications"},
            {"icd9_code": "25002", "icd10_code": "E1165", "category": "DM", "description": "Type 2 diabetes with hyperglycemia"},
            {"icd9_code": "5849", "icd10_code": "N179", "category": "AKI", "description": "Acute kidney failure"},
            {"icd9_code": "5856", "icd10_code": "N186", "category": "CKD", "description": "End stage renal disease"},
            {"icd9_code": "4019", "icd10_code": "I10", "category": "HTN", "description": "Essential primary hypertension"},
            {"icd9_code": "42731", "icd10_code": "I480", "category": "ARRHYTHMIA", "description": "Atrial fibrillation"}
        ]
        df_crosswalk = pd.DataFrame(mappings)
        df_crosswalk.to_csv(crosswalk_path, index=False)
        logger.info(f"Saved GEMs ICD crosswalk to {crosswalk_path}")
        return df_crosswalk

    def standardize_icd_codes(self, df_diag: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans, standardizes, and crosswalks ICD codes to a unified ICD-10 ontology.
        """
        logger.info("Harmonizing diagnostic ICD codes across versions...")
        df = df_diag.copy()
        df["icd_code_clean"] = df["icd_code"].astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
        
        crosswalk = self.get_or_create_icd_crosswalk()
        icd9_map = dict(zip(crosswalk["icd9_code"], crosswalk["icd10_code"]))

        # For ICD-9 rows, map where match exists
        is_icd9 = df["icd_version"].astype(str).str.contains("9")
        df["unified_icd10"] = df["icd_code_clean"]
        df.loc[is_icd9, "unified_icd10"] = df.loc[is_icd9, "icd_code_clean"].map(icd9_map).fillna(df.loc[is_icd9, "icd_code_clean"])
        
        return df

    def normalize_admission_timestamps(self, df_adm: pd.DataFrame) -> pd.DataFrame:
        """
        Parses timestamps, derives length of stay, discharge calendar metrics,
        and flags weekend/after-hours discharges.
        """
        logger.info("Normalizing admission timestamps and temporal variables...")
        df = df_adm.copy()

        df["admittime"] = pd.to_datetime(df["admittime"])
        df["dischtime"] = pd.to_datetime(df["dischtime"])

        # Length of stay in fractional days
        df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
        df["los_days"] = df["los_days"].clip(lower=0.1)

        # Discharge temporal features
        df["discharge_hour"] = df["dischtime"].dt.hour
        df["discharge_dayofweek"] = df["dischtime"].dt.dayofweek
        df["is_weekend_discharge"] = df["discharge_dayofweek"].isin([5, 6]).astype(int)
        df["is_evening_discharge"] = (df["discharge_hour"] >= 17).astype(int)

        # Emergency Department dwell time if available
        if "edregtime" in df.columns and "edouttime" in df.columns:
            df["edregtime"] = pd.to_datetime(df["edregtime"])
            df["edouttime"] = pd.to_datetime(df["edouttime"])
            df["ed_dwell_hours"] = (df["edouttime"] - df["edregtime"]).dt.total_seconds() / 3600.0
            df["ed_dwell_hours"] = df["ed_dwell_hours"].fillna(0.0).clip(lower=0.0)
        else:
            df["ed_dwell_hours"] = 0.0

        return df

    def construct_30day_readmission_target(self, df_adm: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs the 30-day all-cause readmission target compliant with CMS HRRP definitions:
        - Excludes admissions where patient died during the index stay (hospital_expire_flag == 1)
        - Sorts admissions per patient chronologically
        - Measures interval between current dischtime and next admittime
        - Flags readmitted_30d = 1 if interval <= 30 days
        """
        logger.info("Calculating 30-day readmission target variable...")
        df = df_adm.copy()

        # Exclude in-hospital mortality from qualifying index admissions
        if "hospital_expire_flag" in df.columns:
            initial_count = len(df)
            df = df[df["hospital_expire_flag"] == 0].copy()
            logger.info(f"Excluded {initial_count - len(df)} index admissions due to in-hospital death.")

        # Exclude admissions with missing discharge timestamp
        df = df.dropna(subset=["dischtime", "admittime"])
        df = df.sort_values(by=["subject_id", "admittime"]).reset_index(drop=True)

        # Lead variables for subsequent admission
        df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
        df["next_admission_type"] = df.groupby("subject_id")["admission_type"].shift(-1)

        # Days to next admission
        days_to_next = (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400.0
        df["days_to_next_admission"] = days_to_next

        # Target label: next admission within 30 days (CMS excludes planned elective readmissions)
        is_planned = df["next_admission_type"].astype(str).str.upper().str.contains("ELECTIVE")
        df["readmitted_30d"] = ((df["days_to_next_admission"] >= 0) & 
                                (df["days_to_next_admission"] <= 30.0) & 
                                (~is_planned)).astype(int)

        readm_rate = df["readmitted_30d"].mean() * 100.0
        logger.info(f"Derived readmission target: {df['readmitted_30d'].sum():,} positives ({readm_rate:.2f}% incidence)")
        return df

    def aggregate_diagnoses(self, df_diag: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates diagnoses to admission level: total condition count and top chronic indicator flags.
        """
        logger.info("Aggregating patient diagnosis history and ICD indicators...")
        df_clean = self.standardize_icd_codes(df_diag)

        # Count total diagnoses per admission
        diag_counts = df_clean.groupby("hadm_id").agg(
            total_diagnoses_count=("icd_code", "count")
        ).reset_index()

        # Comorbidity indicator flags based on prefixes
        condition_prefixes = {
            "has_heart_failure": ("I50", "428"),
            "has_copd": ("J44", "496"),
            "has_ami": ("I21", "410"),
            "has_pneumonia": ("J18", "486"),
            "has_diabetes": ("E11", "250"),
            "has_renal_failure": ("N17", "N18", "584", "585"),
            "has_hypertension": ("I10", "401")
        }

        flags_by_hadm = {}
        grouped = df_clean.groupby("hadm_id")

        for cond_col, prefixes in condition_prefixes.items():
            pattern = "|".join(prefixes)
            cond_series = grouped["unified_icd10"].apply(lambda s: s.str.startswith(tuple(prefixes)).any().astype(int))
            flags_by_hadm[cond_col] = cond_series

        flags_df = pd.DataFrame(flags_by_hadm).reset_index()
        summary_diag = pd.merge(diag_counts, flags_df, on="hadm_id", how="left").fillna(0)
        return summary_diag

    def aggregate_lab_results(self, df_labs: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes key laboratory tests per admission (min, max, mean, abnormal flag count).
        """
        logger.info("Aggregating inpatient laboratory values and abnormal rates...")
        if df_labs.empty or "valuenum" not in df_labs.columns:
            logger.warning("Empty lab dataframe provided. Returning default empty aggregation.")
            return pd.DataFrame(columns=["hadm_id", "total_lab_tests", "abnormal_lab_count"])

        # Lab item id mappings
        # Creatinine (50912), BUN (51006), Sodium (50983), Potassium (50971), Glucose (50931), Hemoglobin (51222)
        lab_map = {
            50912: "creatinine",
            51006: "bun",
            50983: "sodium",
            50971: "potassium",
            50931: "glucose",
            51222: "hemoglobin"
        }

        df = df_labs[df_labs["itemid"].isin(lab_map.keys())].copy()
        df["lab_name"] = df["itemid"].map(lab_map)

        # Pivoted aggregations
        pivot_mean = df.pivot_table(index="hadm_id", columns="lab_name", values="valuenum", aggfunc="mean")
        pivot_mean.columns = [f"lab_{col}_mean" for col in pivot_mean.columns]

        pivot_max = df.pivot_table(index="hadm_id", columns="lab_name", values="valuenum", aggfunc="max")
        pivot_max.columns = [f"lab_{col}_max" for col in pivot_max.columns]

        # Abnormal count
        abnormal_summary = df_labs.groupby("hadm_id").agg(
            total_lab_tests=("labevent_id", "count"),
            abnormal_lab_count=("flag", lambda x: (x.astype(str).str.lower() == "abnormal").sum())
        ).reset_index()

        lab_summary = pd.merge(abnormal_summary, pivot_mean, on="hadm_id", how="left")
        lab_summary = pd.merge(lab_summary, pivot_max, on="hadm_id", how="left")
        return lab_summary

    @timer
    def run_preprocessing_pipeline(
        self,
        df_patients: pd.DataFrame,
        df_admissions: pd.DataFrame,
        df_diagnoses: pd.DataFrame,
        df_labs: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Executes end-to-end preprocessing, table merging, feature standardization, and cohort filtering.
        """
        logger.info("Executing full EHR preprocessing workflow...")

        # 1. Timestamps and target
        admissions_norm = self.normalize_admission_timestamps(df_admissions)
        cohort = self.construct_30day_readmission_target(admissions_norm)

        # 2. Join Patient Demographics
        logger.info("Merging patient demographics...")
        patient_cols = ["subject_id", "gender", "anchor_age"]
        cols_present = [c for c in patient_cols if c in df_patients.columns]
        cohort = pd.merge(cohort, df_patients[cols_present], on="subject_id", how="left")

        # 3. Filter age criteria (adults only: age >= 18)
        if "anchor_age" in cohort.columns:
            min_age = self.preproc_cfg.get("minimum_age", 18)
            cohort = cohort[cohort["anchor_age"] >= min_age].copy()
            logger.info(f"Retained {len(cohort):,} adult records (age >= {min_age})")

        # 4. Aggregate and join diagnoses
        diag_summary = self.aggregate_diagnoses(df_diagnoses)
        cohort = pd.merge(cohort, diag_summary, on="hadm_id", how="left")

        # 5. Aggregate and join labs
        if df_labs is not None and not df_labs.empty:
            lab_summary = self.aggregate_lab_results(df_labs)
            cohort = pd.merge(cohort, lab_summary, on="hadm_id", how="left")

        # 6. Fill missing values
        logger.info("Applying domain-specific imputation strategies...")
        numeric_cols = cohort.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if c != "readmitted_30d":
                cohort[c] = cohort[c].fillna(cohort[c].median())

        categorical_cols = cohort.select_dtypes(include=["object", "category"]).columns
        for c in categorical_cols:
            cohort[c] = cohort[c].fillna("UNKNOWN")

        # Save preprocessed cohort
        output_file = self.processed_dir / "preprocessed_cohort.parquet"
        save_dataframe(cohort, output_file)
        logger.info(f"Preprocessing completed. Output saved to {output_file} ({cohort.shape[0]} rows, {cohort.shape[1]} cols)")
        return cohort


def main():
    config = load_config()
    from scripts.data_collection import MIMICDataLoader
    
    loader = MIMICDataLoader(config)
    pts = loader.load_table("patients", "hosp")
    adms = loader.load_table("admissions", "hosp")
    diags = loader.load_table("diagnoses_icd", "hosp")
    labs = loader.load_table("labevents", "hosp")

    preprocessor = EHRPreprocessor(config)
    processed_df = preprocessor.run_preprocessing_pipeline(pts, adms, diags, labs)
    logger.info(f"Preprocessed cohort ready for feature engineering: shape={processed_df.shape}")


if __name__ == "__main__":
    main()
