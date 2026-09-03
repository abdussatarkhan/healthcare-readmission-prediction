# Clinical Decision Support & Leadership Dashboards
### Tableau & PowerBI BI Architecture: Preventable Readmissions Risk Stratifier

---

## 1. Executive Overview

This dashboard suite bridges predictive machine learning intelligence with bedside clinical triage and hospital executive operations. Built to integrate with Electronic Health Records (Epic / Cerner / MEDITECH) and clinical data warehouses, the system surfaces risk-stratified cohorts and root-cause drivers to reduce CMS HRRP financial penalties and improve post-acute patient outcomes.

```
+-----------------------------------------------------------------------------------------------+
|                       HOSPITAL READMISSION RISK STRATIFICATION HUB                            |
+------------------------------------+-----------------------------------+----------------------+
|  Active Inpatients: 842            | High Risk Census (Deciles 8-9): 94| Est. Penalty: $1.42M |
|  Mean Predicted Risk: 18.4%        | Post-Acute Referral Rate: 78.2%   | Avoided Readm: 142/yr |
+------------------------------------+-----------------------------------+----------------------+
|                                                                                               |
| [Tab 1: Executive HRRP Performance]  [Tab 2: Inpatient Unit Triage]  [Tab 3: Discharge Timing]|
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Dashboard Views and Wireframes

### View 1: Executive Leadership & CMS HRRP Penalty Avoidance
*Target Audience: Chief Medical Officer (CMO), Chief Quality Officer (CQO), Chief Financial Officer (CFO)*

- **KPI Metric Ribbon**:
  - **Excess Readmission Ratio (ERR)**: Real-time trailing 12-month ERR vs CMS national threshold (1.000).
  - **Estimated Penalty Exposure**: Projected Medicare Inpatient Prospective Payment System (IPPS) reduction ($0 - $3.2M).
  - **30-Day Readmission Rate**: Hospital baseline vs target threshold (17.5% vs 21.2%).
  - **Discharge Bottleneck Velocity**: Share of discharges occurring outside standard weekday hours.
- **Condition-Specific Scorecards**:
  - Breakdown by CMS target disease categories: Acute Myocardial Infarction (AMI), Heart Failure (HF), Chronic Obstructive Pulmonary Disease (COPD), Pneumonia (PNA), Coronary Artery Bypass Graft (CABG), Elective Total Hip/Knee Arthroplasty (THA/TKA).
- **Interactive Visualizations**:
  - *Monthly ERR Trend vs Penalty Threshold*: Area chart displaying seasonality and rolling impact of quality improvement interventions.
  - *Discharge Disposition Migration*: Sankey diagram tracking transitions to Home, Home Health, SNF, and Long-Term Acute Care.

---

### View 2: Floor Nurse & Case Manager Triage Board
*Target Audience: Charge Nurses, Transition Coordinators, Clinical Pharmacists, Social Workers*

- **Real-Time Patient Risk Census**:
  - Filterable by nursing floor, service line (Cardiology, Pulmonology, General Internal Medicine), and attending physician.
  - Dynamic sorting by **Predicted Readmission Probability (%)**.
- **Bedside Patient Profile Card**:
  - **Risk Tier**: High (Probability > 35%), Moderate (20% - 35%), Low (< 20%).
  - **Top SHAP Clinical Drivers**: Personalized bar chart highlighting the top 3 drivers of that individual patient's risk (e.g., `Charlson Index = 6`, `BUN/Creatinine = 24.1`, `Discharge Friday PM`).
  - **Actionable Intervention Checklist**:
    - [ ] Bedside medication reconciliation completed by Clinical Pharmacist.
    - [ ] Post-discharge primary care appointment scheduled within 7 days.
    - [ ] Outpatient telephone check-in scheduled for 48 hours post-discharge.
    - [ ] DME / Oxygen equipment delivery confirmed prior to departure.

---

### View 3: Operational Discharge Dynamics Heatmap
*Target Audience: Bed Management, Patient Flow Logistics, Hospitalists*

- **Heatmap Matrix: Hour of Day vs Day of Week**:
  - Cell intensity colored by 30-day readmission rate.
  - Highlights high-vulnerability operational periods:
    - *Friday 15:00 - 19:00*: 26.4% readmission rate (+6.2% excess).
    - *Sunday 16:00 - 20:00*: 24.8% readmission rate (+4.6% excess).
- **Staffing vs Discharge Volume Cross-Correlation**:
  - Visualizing the discrepancy between nurse staffing ratios and afternoon discharge surges.

---

## 3. Data Dictionary & Tableau Calculated Fields

| Field Name | Formula / Logic | Description |
| :--- | :--- | :--- |
| `Risk Tier` | `IF [Predicted Prob] >= 0.35 THEN "High" ELSEIF [Predicted Prob] >= 0.20 THEN "Moderate" ELSE "Low" END` | Operational risk grouping |
| `High Vulnerability Discharge` | `IF [Discharge DOW] IN (0, 6) OR ([Discharge DOW]=5 AND [Discharge Hour]>=15) THEN 1 ELSE 0 END` | Care transition flag |
| `Penalty Exposure ($)` | `[Medicare Revenue Base] * MIN([Max Penalty Pct], MAX(0.0, ([Excess Readmission Ratio] - 1.0) * 0.03))` | Financial penalty estimator |
| `BUN/Cr Ratio Category` | `IF [BUN_to_Cr_Ratio] > 20 THEN "Prerenal Azotemia" ELSE "Normal / Intrinsic" END` | Clinical physiological index |

---

## 4. Tableau Workbook Deployment

1. **Tableau Desktop / Server**: Open `dashboards/healthcare_readmissions.twbx` (or connect to `data/processed/features_matrix.parquet` via Hyper API).
2. **Database Direct Connect**: Use PostgreSQL connector pointing to `mimiciv_derived.master_readmission_cohort`.
3. **Automated Refresh**: Schedule hourly extracts during inpatient shift changes (07:00, 15:00, 23:00).

---

## 5. UI Mockup Placeholders

![Tableau Executive Dashboard Screenshot Placeholder](../images/dashboard_preview.png)
*(Run `scripts/evaluation.py` to generate associated visual artifacts stored in `images/`)*
