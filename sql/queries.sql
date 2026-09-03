-- ==============================================================================
-- Preventable Readmissions Risk Stratifier: Clinical Data Modeling Queries
-- Target Database: PostgreSQL 14+ / MIMIC-IV v2.2 (hosp & icu schemas)
-- Description: Standardized cohort extraction, 30-day readmission labeling,
--              Charlson Comorbidity Index calculation, and laboratory aggregations.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. ADULT COHORT & CMS HRRP QUALIFYING INDEX ADMISSIONS
-- Filters out pediatric patients (<18yo) and in-hospital deaths
-- ------------------------------------------------------------------------------
WITH cohort_base AS (
    SELECT
        adm.subject_id,
        adm.hadm_id,
        adm.admittime,
        adm.dischtime,
        adm.deathtime,
        adm.admission_type,
        adm.admission_location,
        adm.discharge_location,
        adm.insurance,
        adm.race,
        pts.gender,
        pts.anchor_age,
        pts.anchor_year,
        ROUND(EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime)) / 86400.0, 2) AS los_days,
        EXTRACT(HOUR FROM adm.dischtime) AS discharge_hour,
        EXTRACT(DOW FROM adm.dischtime) AS discharge_dow, -- 0=Sunday, 6=Saturday
        CASE 
            WHEN EXTRACT(DOW FROM adm.dischtime) IN (0, 6) THEN 1 
            ELSE 0 
        END AS is_weekend_discharge,
        CASE 
            WHEN EXTRACT(DOW FROM adm.dischtime) = 5 AND EXTRACT(HOUR FROM adm.dischtime) >= 15 THEN 1 
            ELSE 0 
        END AS discharge_friday_afternoon,
        CASE 
            WHEN EXTRACT(HOUR FROM adm.dischtime) >= 17 OR EXTRACT(HOUR FROM adm.dischtime) < 7 THEN 1 
            ELSE 0 
        END AS is_after_hours_discharge,
        CASE 
            WHEN adm.edouttime IS NOT NULL AND adm.edregtime IS NOT NULL 
            THEN ROUND(EXTRACT(EPOCH FROM (adm.edouttime - adm.edregtime)) / 3600.0, 2)
            ELSE 0.0 
        END AS ed_dwell_hours
    FROM mimiciv_hosp.admissions adm
    INNER JOIN mimiciv_hosp.patients pts
        ON adm.subject_id = pts.subject_id
    WHERE pts.anchor_age >= 18
      AND adm.hospital_expire_flag = 0  -- Patient survived index stay
      AND adm.dischtime IS NOT NULL
),

-- ------------------------------------------------------------------------------
-- 2. 30-DAY ALL-CAUSE UNPLANNED READMISSION TARGET
-- Calculates interval to next admission; flags readmission within 30 days
-- excluding planned/elective procedures compliant with CMS HRRP definitions
-- ------------------------------------------------------------------------------
readmission_target_cte AS (
    SELECT
        cb.*,
        LEAD(cb.admittime) OVER (
            PARTITION BY cb.subject_id 
            ORDER BY cb.admittime
        ) AS next_admittime,
        LEAD(cb.admission_type) OVER (
            PARTITION BY cb.subject_id 
            ORDER BY cb.admittime
        ) AS next_admission_type,
        ROUND(
            EXTRACT(EPOCH FROM (
                LEAD(cb.admittime) OVER (PARTITION BY cb.subject_id ORDER BY cb.admittime) - cb.dischtime
            )) / 86400.0, 
            2
        ) AS days_to_next_admission,
        ROW_NUMBER() OVER (
            PARTITION BY cb.subject_id 
            ORDER BY cb.admittime
        ) - 1 AS prior_admissions_count
    FROM cohort_base cb
),

labeled_cohort AS (
    SELECT
        rt.*,
        CASE
            WHEN rt.days_to_next_admission BETWEEN 0.0 AND 30.0 
                 AND UPPER(COALESCE(rt.next_admission_type, '')) NOT LIKE '%ELECTIVE%'
            THEN 1
            ELSE 0
        END AS readmitted_30d,
        CASE 
            WHEN rt.prior_admissions_count >= 3 THEN 1 
            ELSE 0 
        END AS is_frequent_flyer
    FROM readmission_target_cte rt
),

-- ------------------------------------------------------------------------------
-- 3. CHARLSON COMORBIDITY INDEX (CCI) SCORING (ICD-9 & ICD-10)
-- Maps diagnostic records to validated Charlson comorbidity categories
-- ------------------------------------------------------------------------------
diagnoses_mapped AS (
    SELECT
        diag.hadm_id,
        diag.icd_code,
        diag.icd_version,
        -- Myocardial Infarction
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code LIKE '410%') 
                   OR (diag.icd_version = 10 AND diag.icd_code LIKE 'I21%') THEN 1 ELSE 0 END) AS cci_mi,
        -- Congestive Heart Failure
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code LIKE '428%') 
                   OR (diag.icd_version = 10 AND diag.icd_code LIKE 'I50%') THEN 1 ELSE 0 END) AS cci_chf,
        -- Peripheral Vascular Disease
        MAX(CASE WHEN (diag.icd_version = 9 AND (diag.icd_code LIKE '440%' OR diag.icd_code LIKE '441%')) 
                   OR (diag.icd_version = 10 AND (diag.icd_code LIKE 'I70%' OR diag.icd_code LIKE 'I71%')) THEN 1 ELSE 0 END) AS cci_pvd,
        -- Cerebrovascular Disease
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '43[0-8]%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'I6[0-9]%') THEN 1 ELSE 0 END) AS cci_cvd,
        -- Dementia
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code LIKE '290%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'F0[1-3]%') THEN 1 ELSE 0 END) AS cci_dementia,
        -- Chronic Pulmonary Disease (COPD)
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '49[0-6]%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'J4[0-7]%') THEN 1 ELSE 0 END) AS cci_copd,
        -- Diabetes without chronic complication
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '250[0-3]%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'E1[0-4]9%') THEN 1 ELSE 0 END) AS cci_dm_uncomp,
        -- Diabetes with chronic complication (Renal, Ophthalmic, Neurological)
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '250[4-9]%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'E1[0-4][2-6]%') THEN 1 ELSE 0 END) AS cci_dm_comp,
        -- Renal Disease
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '58[2-6]%') 
                   OR (diag.icd_version = 10 AND (diag.icd_code LIKE 'N18%' OR diag.icd_code LIKE 'N19%')) THEN 1 ELSE 0 END) AS cci_renal,
        -- Severe Liver Disease
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code LIKE '572%') 
                   OR (diag.icd_version = 10 AND diag.icd_code LIKE 'K72%') THEN 1 ELSE 0 END) AS cci_severe_liver,
        -- Metastatic Carcinoma
        MAX(CASE WHEN (diag.icd_version = 9 AND diag.icd_code SIMILAR TO '19[6-9]%') 
                   OR (diag.icd_version = 10 AND diag.icd_code SIMILAR TO 'C7[7-9]%') THEN 1 ELSE 0 END) AS cci_metastatic,
        COUNT(DISTINCT diag.icd_code) AS total_diagnoses_count
    FROM mimiciv_hosp.diagnoses_icd diag
    GROUP BY diag.hadm_id
),

charlson_computed AS (
    SELECT
        dm.hadm_id,
        dm.total_diagnoses_count,
        dm.cci_mi,
        dm.cci_chf,
        dm.cci_copd,
        dm.cci_renal,
        (dm.cci_mi * 1 +
         dm.cci_chf * 1 +
         dm.cci_pvd * 1 +
         dm.cci_cvd * 1 +
         dm.cci_dementia * 1 +
         dm.cci_copd * 1 +
         dm.cci_dm_uncomp * 1 +
         dm.cci_dm_comp * 2 +
         dm.cci_renal * 2 +
         dm.cci_severe_liver * 3 +
         dm.cci_metastatic * 6
        ) AS charlson_comorbidity_index
    FROM diagnoses_mapped dm
),

-- ------------------------------------------------------------------------------
-- 4. LABORATORY TRAJECTORIES & PHYSIOLOGICAL INSTABILITY
-- Aggregates inpatient labs (Creatinine, BUN, Glucose, Hemoglobin, Potassium, Sodium)
-- ------------------------------------------------------------------------------
labs_aggregated AS (
    SELECT
        lab.hadm_id,
        COUNT(lab.labevent_id) AS total_lab_tests,
        SUM(CASE WHEN LOWER(lab.flag) = 'abnormal' THEN 1 ELSE 0 END) AS abnormal_lab_count,
        ROUND(AVG(CASE WHEN lab.itemid = 50912 THEN lab.valuenum END), 2) AS lab_creatinine_mean,
        ROUND(MAX(CASE WHEN lab.itemid = 50912 THEN lab.valuenum END), 2) AS lab_creatinine_max,
        ROUND(AVG(CASE WHEN lab.itemid = 51006 THEN lab.valuenum END), 2) AS lab_bun_mean,
        ROUND(MAX(CASE WHEN lab.itemid = 51006 THEN lab.valuenum END), 2) AS lab_bun_max,
        ROUND(AVG(CASE WHEN lab.itemid = 50983 THEN lab.valuenum END), 2) AS lab_sodium_mean,
        ROUND(AVG(CASE WHEN lab.itemid = 50971 THEN lab.valuenum END), 2) AS lab_potassium_mean,
        ROUND(AVG(CASE WHEN lab.itemid = 50931 THEN lab.valuenum END), 2) AS lab_glucose_mean,
        ROUND(AVG(CASE WHEN lab.itemid = 51222 THEN lab.valuenum END), 2) AS lab_hemoglobin_mean
    FROM mimiciv_hosp.labevents lab
    WHERE lab.hadm_id IS NOT NULL
      AND lab.itemid IN (50912, 51006, 50983, 50971, 50931, 51222)
    GROUP BY lab.hadm_id
),

-- ------------------------------------------------------------------------------
-- 5. ICU STAY DYNAMICS
-- Evaluates ICU admission status, total ICU length of stay
-- ------------------------------------------------------------------------------
icu_aggregated AS (
    SELECT
        icu.hadm_id,
        COUNT(icu.stay_id) AS total_icu_stays,
        ROUND(SUM(icu.los), 2) AS total_icu_los_days,
        1 AS had_icu_stay
    FROM mimiciv_icu.icustays icu
    GROUP BY icu.hadm_id
)

-- ------------------------------------------------------------------------------
-- 6. MASTER ANALYTICAL DATASET (FINAL EXPORT)
-- Joins cohort base, readmission label, comorbidities, labs, and ICU stays
-- ------------------------------------------------------------------------------
SELECT
    lc.subject_id,
    lc.hadm_id,
    lc.admittime,
    lc.dischtime,
    lc.anchor_age,
    lc.gender,
    lc.admission_type,
    lc.discharge_location,
    lc.los_days,
    lc.discharge_hour,
    lc.is_weekend_discharge,
    lc.discharge_friday_afternoon,
    lc.is_after_hours_discharge,
    lc.ed_dwell_hours,
    lc.prior_admissions_count,
    lc.is_frequent_flyer,
    COALESCE(cc.total_diagnoses_count, 0) AS total_diagnoses_count,
    COALESCE(cc.charlson_comorbidity_index, 0) AS charlson_comorbidity_index,
    COALESCE(cc.cci_mi, 0) AS has_ami,
    COALESCE(cc.cci_chf, 0) AS has_heart_failure,
    COALESCE(cc.cci_copd, 0) AS has_copd,
    COALESCE(cc.cci_renal, 0) AS has_renal_failure,
    -- Cardiorenal syndrome overlap
    CASE WHEN COALESCE(cc.cci_chf, 0) = 1 AND COALESCE(cc.cci_renal, 0) = 1 THEN 1 ELSE 0 END AS cardiorenal_syndrome,
    COALESCE(la.total_lab_tests, 0) AS total_lab_tests,
    COALESCE(la.abnormal_lab_count, 0) AS abnormal_lab_count,
    COALESCE(la.lab_creatinine_mean, 1.0) AS lab_creatinine_mean,
    COALESCE(la.lab_bun_mean, 15.0) AS lab_bun_mean,
    ROUND(COALESCE(la.lab_bun_mean, 15.0) / GREATEST(COALESCE(la.lab_creatinine_mean, 1.0), 0.2), 2) AS bun_to_creatinine_ratio,
    COALESCE(la.lab_sodium_mean, 140.0) AS lab_sodium_mean,
    COALESCE(la.lab_potassium_mean, 4.0) AS lab_potassium_mean,
    COALESCE(la.lab_glucose_mean, 110.0) AS lab_glucose_mean,
    COALESCE(la.lab_hemoglobin_mean, 13.5) AS lab_hemoglobin_mean,
    COALESCE(icu.had_icu_stay, 0) AS had_icu_stay,
    COALESCE(icu.total_icu_los_days, 0.0) AS total_icu_los_days,
    lc.readmitted_30d
FROM labeled_cohort lc
LEFT JOIN charlson_computed cc
    ON lc.hadm_id = cc.hadm_id
LEFT JOIN labs_aggregated la
    ON lc.hadm_id = la.hadm_id
LEFT JOIN icu_aggregated icu
    ON lc.hadm_id = icu.hadm_id
ORDER BY lc.subject_id, lc.admittime;
