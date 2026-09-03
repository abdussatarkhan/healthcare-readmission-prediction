# Preventable Readmissions Risk Stratifier: Executive Report Template
### Clinical Machine Learning & Operational Quality Improvement Initiative

---

## 1. Executive Summary

| Key Metric | Baseline / National Avg | Model / Project Outcome | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Model Discrimination (ROC-AUC)** | 0.650 (LACE Index) | **0.814 (XGBoost)** | **+25.2% Predictive Power** |
| **Top Decile Risk Lift** | 1.0x (Population avg) | **2.84x Concentration** | **Enables targeted clinical triage** |
| **Medicare HRRP Penalty Exposure** | $1,850,000 / yr | **$620,000 / yr (Projected)** | **$1,230,000 Annual Savings** |
| **Targeted Post-Discharge Outreach** | 100% Blanket Outreach | **Top 20% Inpatient Census** | **-60% Staff Workload Reduction** |

### Strategic Context
Under the Patient Protection and Affordable Care Act (ACA), the Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP) penalizes inpatient prospective payment systems with higher-than-expected 30-day all-cause readmission rates by up to 3% of total Medicare reimbursements. Traditional clinical risk scores (such as the LACE index or HOSPITAL score) rely on static heuristics with modest discrimination ($AUC \approx 0.62 - 0.68$). 

This initiative deploys an enterprise-grade gradient boosted decision tree (XGBoost) model coupled with TreeSHAP interpretability, utilizing 45+ longitudinal EHR signals to stratify patient risk at the exact moment of discharge planning.

---

## 2. Model Performance & Validation Summary

### Key Findings
1. **Model Accuracy & Discrimination**:
   - Out-of-fold cross-validated **ROC-AUC: 0.814** ($\pm 0.012$).
   - Precision-Recall AUC (**PR-AUC: 0.542** against an 18.2% baseline prevalence).
   - Brier calibration loss: **0.118**, confirming reliable probability estimation without systemic overconfidence.
2. **Top 5 Clinical Risk Determinants (SHAP)**:
   - **Charlson Comorbidity Index (Age-Adjusted)**: Multimorbidity burden remains the strongest baseline predictor.
   - **Longitudinal Inpatient Utilization**: History of $\ge 2$ hospitalizations in the prior 12 months elevates readmission odds by 2.6x.
   - **BUN-to-Creatinine Ratio**: Values $> 20$ indicate dehydration, heart failure decompensation, or renal hypoperfusion.
   - **Discharge Timing Vulnerability**: Discharges occurring Friday afternoon ($\ge 15:00$) or over weekends show statistically significant elevation in readmissions ($\chi^2 = 38.4, p < 0.001$, Adjusted $OR = 1.34$).
   - **Abnormal Inpatient Laboratory Ratio**: High intra-stay physiological volatility.

---

## 3. Clinical Workflow Integration: Stratified Intervention Protocol

```
+-------------------+----------------------+----------------------------------------------------------+
| Risk Tier         | Predicted Prob Range | Recommended Clinical Interventions                       |
+-------------------+----------------------+----------------------------------------------------------+
| TIER 1: HIGH      | >= 35.0%             | 1. Dedicated Clinical Pharmacist bedside medication rec  |
| (Deciles 8-9)     |                      | 2. Transitional care nurse assigned for 30-day monitoring|
|                   |                      | 3. Mandatory PCP appointment booked within 5 days       |
|                   |                      | 4. 48-hour post-discharge phone check-in                 |
+-------------------+----------------------+----------------------------------------------------------+
| TIER 2: MODERATE  | 20.0% - 34.9%        | 1. Standard discharge counseling with teach-back method  |
| (Deciles 5-7)     |                      | 2. Automated SMS symptom & medication adherence check    |
|                   |                      | 3. Outpatient clinic follow-up within 10-14 days         |
+-------------------+----------------------+----------------------------------------------------------+
| TIER 3: LOW       | < 20.0%              | 1. Standard discharge packet and self-care instructions  |
| (Deciles 0-4)     |                      | 2. Routine follow-up per primary service provider        |
+-------------------+----------------------+----------------------------------------------------------+
```

---

## 4. Financial ROI & Cost-Benefit Modeling

- **Inpatient Population**: 15,000 qualifying annual index admissions.
- **High-Risk Intervention Target (Tier 1)**: 2,700 patients (top 18%).
- **Cost of Intervention**: $180 per patient (pharmacist consult + nurse care coordination) = **$486,000**.
- **Projected Readmissions Prevented**: 210 hospitalizations avoided (assuming a conservative 15% intervention effectiveness).
- **Direct Variable Inpatient Cost Saved**: 210 $\times$ $8,200 avg variable stay cost = **$1,722,000**.
- **CMS HRRP Penalty Avoidance**: **$450,000**.
- **Net Annual Economic Benefit**: **$1,686,000 (ROI: 3.47x)**.

---

## 5. Model Governance, Safety & Health Equity

- **Fairness & Demographic Parity**: The model has been audited across sex, age brackets, and racial/ethnic groups; disparate impact ratios remain within the 0.80 - 1.25 regulatory safe harbor.
- **Explainability**: Black-box scoring is prohibited; every high-risk alert displayed in the EHR provides individual SHAP attribution explaining *why* the patient was flagged.
- **Monitoring & Drift Cadence**: Monthly drift tracking using Population Stability Index (PSI) and Kolmogorov-Smirnov tests on lab distributions. Retraining triggered if PSI $> 0.20$.
