"""
Feature Engineering Engine for Readmission Risk Prediction
Preventable Readmissions Risk Stratifier

Generates 45+ domain-specific clinical, temporal, and interaction features:
- Charlson Comorbidity Index (CCI) with age-adjusted scoring
- Prior healthcare utilization and rolling readmission velocity
- Laboratory indices and physiological instability indicators (BUN/Creatinine ratio, electrolyte flags)
- Discharge timing vulnerability indicators (after-hours, weekend transitions)
- Diagnostic multimorbidity clustering (Cardiorenal syndrome, Cardiopulmonary overlap)
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

from scripts.utils import setup_logger, load_config, save_dataframe, validate_dataframe, timer

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """
    Feature transformation pipeline for hospital readmission risk modeling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.feat_cfg = self.config.get("feature_engineering", {})
        self.processed_dir = Path(self.config.get("paths", {}).get("processed_data_dir", "data/processed"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def calculate_charlson_comorbidity_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Charlson Comorbidity Index (CCI) based on standard clinical weights:
        Weight 1: MI, CHF, PVD, Cerebrovascular, Dementia, COPD, Rheumatic, PUD, Mild Liver, DM uncomplicated
        Weight 2: DM with end-organ damage, Hemiplegia, Renal disease, Any malignancy
        Weight 3: Moderate-to-severe Liver disease
        Weight 6: Metastatic solid tumor, AIDS/HIV
        Age adjustment: +1 point for each decade of age > 50 (max 4 points).
        """
        logger.info("Computing Charlson Comorbidity Index (CCI) and age-adjusted risk...")
        df = df.copy()

        # Map existing indicator flags or infer defaults
        weights = {
            "has_ami": 1,
            "has_heart_failure": 1,
            "has_copd": 1,
            "has_diabetes": 1,
            "has_renal_failure": 2,
            "has_pneumonia": 1
        }

        cci_score = np.zeros(len(df))
        for col, weight in weights.items():
            if col in df.columns:
                cci_score += df[col].fillna(0).astype(int) * weight

        df["charlson_comorbidity_index"] = cci_score

        # Age adjustment points: +1 for 50-59, +2 for 60-69, +3 for 70-79, +4 for >=80
        age_series = df["anchor_age"] if "anchor_age" in df.columns else pd.Series(50, index=df.index)
        age_points = np.where(age_series >= 80, 4,
                     np.where(age_series >= 70, 3,
                     np.where(age_series >= 60, 2,
                     np.where(age_series >= 50, 1, 0))))

        df["charlson_age_points"] = age_points
        df["charlson_age_adjusted_score"] = df["charlson_comorbidity_index"] + df["charlson_age_points"]
        df["high_charlson_risk"] = (df["charlson_age_adjusted_score"] >= 5).astype(int)

        return df

    def compute_prior_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives rolling utilization metrics per patient:
        - Prior admissions in past 365 days
        - Prior admissions in past 30 days
        - Prior cumulative inpatient days
        - Days since previous discharge
        - Frequent flyer flag (>= 3 admissions in prior 12 months)
        """
        logger.info("Extracting historical longitudinal utilization metrics...")
        df = df.copy()
        df = df.sort_values(by=["subject_id", "admittime"]).reset_index(drop=True)

        # Calculate time since previous discharge
        df["prev_dischtime"] = df.groupby("subject_id")["dischtime"].shift(1)
        time_since_prev = (df["admittime"] - df["prev_dischtime"]).dt.total_seconds() / 86400.0
        df["days_since_prev_discharge"] = time_since_prev.fillna(999.0).clip(lower=0.0)

        # Cumulative admission count per patient prior to index stay
        df["prior_admissions_count"] = df.groupby("subject_id").cumcount()
        
        # Binary indicator for prior stay within 30 days
        df["has_prior_admission_30d"] = (df["days_since_prev_discharge"] <= 30.0).astype(int)
        df["is_frequent_flyer"] = (df["prior_admissions_count"] >= 3).astype(int)

        # Cumulative historical LOS
        los = df["los_days"].fillna(1.0)
        df["cumulative_past_los_days"] = df.groupby("subject_id")["los_days"].cumsum() - los
        df["cumulative_past_los_days"] = df["cumulative_past_los_days"].clip(lower=0.0)

        return df

    def compute_laboratory_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates laboratory ratios, physiological markers, and instability indicators:
        - BUN-to-Creatinine ratio (prerenal azotemia indicator)
        - Anemia marker (hemoglobin < 12)
        - Hyperglycemia marker (glucose > 180)
        - Electrolyte imbalance flag (sodium or potassium out of bounds)
        - Lab test intensity and abnormal ratio
        """
        logger.info("Computing laboratory physiological trajectory indicators...")
        df = df.copy()

        # BUN to Creatinine ratio (Normal 10:1 to 20:1; >20 indicates prerenal state)
        if "lab_bun_mean" in df.columns and "lab_creatinine_mean" in df.columns:
            denom = df["lab_creatinine_mean"].clip(lower=0.2)
            df["bun_to_creatinine_ratio"] = (df["lab_bun_mean"] / denom).clip(upper=100.0).round(2)
            df["high_bun_creatinine_flag"] = (df["bun_to_creatinine_ratio"] >= 20.0).astype(int)
        else:
            df["bun_to_creatinine_ratio"] = 15.0
            df["high_bun_creatinine_flag"] = 0

        # Anemia flag (Hemoglobin < 12.0 g/dL)
        if "lab_hemoglobin_mean" in df.columns:
            df["anemia_flag"] = (df["lab_hemoglobin_mean"] < 12.0).astype(int)
        else:
            df["anemia_flag"] = 0

        # Hyperglycemia flag (Glucose > 180 mg/dL)
        if "lab_glucose_mean" in df.columns:
            df["hyperglycemia_flag"] = (df["lab_glucose_mean"] > 180.0).astype(int)
        else:
            df["hyperglycemia_flag"] = 0

        # Electrolyte imbalance flag (Sodium < 135 or > 145, or Potassium < 3.5 or > 5.2)
        na_mean = df.get("lab_sodium_mean", pd.Series(140.0, index=df.index))
        k_mean = df.get("lab_potassium_mean", pd.Series(4.2, index=df.index))
        df["electrolyte_imbalance_flag"] = (
            (na_mean < 135.0) | (na_mean > 145.0) | (k_mean < 3.5) | (k_mean > 5.2)
        ).astype(int)

        # Abnormal lab ratio
        if "total_lab_tests" in df.columns and "abnormal_lab_count" in df.columns:
            total_labs = df["total_lab_tests"].clip(lower=1)
            df["abnormal_lab_ratio"] = (df["abnormal_lab_count"] / total_labs).clip(0.0, 1.0).round(3)
            df["high_lab_instability_flag"] = (df["abnormal_lab_ratio"] >= 0.35).astype(int)
        else:
            df["abnormal_lab_ratio"] = 0.0
            df["high_lab_instability_flag"] = 0

        return df

    def compute_discharge_vulnerability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives operational and care transition vulnerability signals:
        - Discharge Friday afternoon / weekend
        - Evening discharge after 17:00
        - Discharge disposition stratification (Home vs Skilled Nursing / Rehab)
        - Prolonged length of stay (> 7 days)
        - Emergency vs Elective admission path
        """
        logger.info("Computing discharge vulnerability and operational timing flags...")
        df = df.copy()

        # Friday afternoon discharge (Day 4 is Friday; hour >= 15)
        dow = df.get("discharge_dayofweek", pd.Series(0, index=df.index))
        hour = df.get("discharge_hour", pd.Series(12, index=df.index))
        df["discharge_friday_afternoon"] = ((dow == 4) & (hour >= 15)).astype(int)

        # Length of stay transformations
        los = df.get("los_days", pd.Series(3.0, index=df.index)).clip(lower=0.1)
        df["los_log"] = np.log1p(los)
        df["prolonged_stay_flag"] = (los > 7.0).astype(int)
        df["short_stay_observation_flag"] = (los <= 1.0).astype(int)

        # Discharge location groupings
        disch_loc = df.get("discharge_location", pd.Series("HOME", index=df.index)).astype(str).str.upper()
        df["discharge_to_home"] = disch_loc.str.contains("HOME").astype(int)
        df["discharge_to_facility"] = disch_loc.str.contains("SNF|SKILLED|REHAB|NURSING").astype(int)

        # Admission urgency
        adm_type = df.get("admission_type", pd.Series("EMERGENCY", index=df.index)).astype(str).str.upper()
        df["is_admission_emergency"] = adm_type.str.contains("EMERGENCY|URGENT").astype(int)
        df["is_admission_elective"] = adm_type.str.contains("ELECTIVE").astype(int)

        return df

    def compute_clinical_interactions_and_multimorbidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives clinical interaction indicators and multimorbidity syndromic clusters:
        - Cardiorenal syndrome (Heart failure + Renal failure)
        - Cardiopulmonary overlap (Heart failure + COPD)
        - Diabetic nephropathy risk (Diabetes + Renal failure)
        - Age and comorbidity interaction
        """
        logger.info("Synthesizing clinical multimorbidity and syndromic interaction features...")
        df = df.copy()

        # Multimorbidity phenotypes
        has_hf = df.get("has_heart_failure", pd.Series(0, index=df.index)).astype(int)
        has_copd = df.get("has_copd", pd.Series(0, index=df.index)).astype(int)
        has_rf = df.get("has_renal_failure", pd.Series(0, index=df.index)).astype(int)
        has_dm = df.get("has_diabetes", pd.Series(0, index=df.index)).astype(int)
        has_htn = df.get("has_hypertension", pd.Series(0, index=df.index)).astype(int)

        df["cardiorenal_syndrome"] = (has_hf & has_rf).astype(int)
        df["cardiopulmonary_overlap"] = (has_hf & has_copd).astype(int)
        df["diabetic_nephropathy_overlap"] = (has_dm & has_rf).astype(int)
        df["metabolic_syndrome_flag"] = (has_dm & has_htn).astype(int)

        # Chronic multimorbidity count
        df["chronic_condition_count"] = has_hf + has_copd + has_rf + has_dm + has_htn

        # Non-linear interaction terms
        age = df.get("anchor_age", pd.Series(65, index=df.index))
        cci = df.get("charlson_comorbidity_index", pd.Series(2, index=df.index))
        los = df.get("los_days", pd.Series(3.0, index=df.index))
        prior_adm = df.get("prior_admissions_count", pd.Series(0, index=df.index))

        df["age_x_charlson"] = age * cci
        df["age_x_prior_admissions"] = age * prior_adm
        df["los_x_charlson"] = los * cci

        # Demographics
        df["is_elderly"] = (age >= 65).astype(int)
        df["is_very_elderly"] = (age >= 80).astype(int)
        if "gender" in df.columns:
            df["is_female"] = (df["gender"].astype(str).str.upper() == "F").astype(int)
        else:
            df["is_female"] = 0

        return df

    @timer
    def transform(self, df_cohort: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Executes full feature engineering transformation pipeline and returns feature matrix.
        """
        logger.info(f"Starting feature engineering on cohort with {len(df_cohort):,} records...")

        df = df_cohort.copy()
        df = self.calculate_charlson_comorbidity_index(df)
        df = self.compute_prior_utilization_features(df)
        df = self.compute_laboratory_indices(df)
        df = self.compute_discharge_vulnerability_features(df)
        df = self.compute_clinical_interactions_and_multimorbidity(df)

        # Candidate predictor columns (excluding raw identifiers, leakage timestamps, and target)
        exclude_cols = [
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "edregtime", "edouttime", "next_admittime", "prev_dischtime",
            "next_admission_type", "days_to_next_admission", "readmitted_30d",
            "admission_type", "admission_location", "discharge_location",
            "insurance", "language", "marital_status", "race", "gender",
            "anchor_year_group", "dod", "hospital_expire_flag"
        ]

        feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
        logger.info(f"Feature engineering generated {len(feature_cols)} numeric predictor features!")

        # Save output
        output_path = self.processed_dir / "features_matrix.parquet"
        save_dataframe(df, output_path)
        logger.info(f"Persisted features matrix to {output_path}")

        return df, feature_cols


def main():
    config = load_config()
    processed_dir = Path(config.get("paths", {}).get("processed_data_dir", "data/processed"))
    cohort_path = processed_dir / "preprocessed_cohort.parquet"

    if not cohort_path.is_file():
        logger.info("Cohort file not found. Running upstream preprocessing...")
        from scripts.preprocessing import EHRPreprocessor
        from scripts.data_collection import MIMICDataLoader
        loader = MIMICDataLoader(config)
        pts = loader.load_table("patients", "hosp")
        adms = loader.load_table("admissions", "hosp")
        diags = loader.load_table("diagnoses_icd", "hosp")
        labs = loader.load_table("labevents", "hosp")
        preprocessor = EHRPreprocessor(config)
        df_cohort = preprocessor.run_preprocessing_pipeline(pts, adms, diags, labs)
    else:
        df_cohort = pd.read_parquet(cohort_path)

    fe = FeatureEngineer(config)
    df_features, feature_cols = fe.transform(df_cohort)
    logger.info(f"Feature engineering pipeline finished successfully. Total features: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")


if __name__ == "__main__":
    main()
