{{ config(
    materialized='table',
    indexes=[
      {'columns': ['subject_id', 'hadm_id'], 'unique': True},
      {'columns': ['readmitted_30d']}
    ]
) }}

WITH base_admissions AS (
    SELECT
        adm.subject_id,
        adm.hadm_id,
        adm.admittime,
        adm.dischtime,
        adm.admission_type,
        adm.discharge_location,
        pts.anchor_age,
        pts.gender,
        ROUND(EXTRACT(EPOCH FROM (adm.dischtime - adm.admittime)) / 86400.0, 2) AS los_days,
        EXTRACT(DOW FROM adm.dischtime) AS discharge_dow,
        EXTRACT(HOUR FROM adm.dischtime) AS discharge_hour,
        CASE WHEN EXTRACT(DOW FROM adm.dischtime) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend_discharge,
        LEAD(adm.admittime) OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS next_admittime,
        LEAD(adm.admission_type) OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS next_admission_type
    FROM {{ source('mimiciv_hosp', 'admissions') }} adm
    INNER JOIN {{ source('mimiciv_hosp', 'patients') }} pts
        ON adm.subject_id = pts.subject_id
    WHERE pts.anchor_age >= 18
      AND adm.hospital_expire_flag = 0
)

SELECT
    b.subject_id,
    b.hadm_id,
    b.admittime,
    b.dischtime,
    b.anchor_age,
    b.gender,
    b.admission_type,
    b.discharge_location,
    b.los_days,
    b.discharge_hour,
    b.is_weekend_discharge,
    ROUND(EXTRACT(EPOCH FROM (b.next_admittime - b.dischtime)) / 86400.0, 2) AS days_to_next_admission,
    CASE 
        WHEN EXTRACT(EPOCH FROM (b.next_admittime - b.dischtime)) / 86400.0 BETWEEN 0.0 AND 30.0
             AND UPPER(COALESCE(b.next_admission_type, '')) NOT LIKE '%ELECTIVE%'
        THEN 1
        ELSE 0 
    END AS readmitted_30d
FROM base_admissions b
