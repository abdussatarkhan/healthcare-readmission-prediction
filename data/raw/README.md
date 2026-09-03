# Data Acquisition and Raw Storage Guide

This directory holds the raw, uncurated source datasets used by the **Preventable Readmissions Risk Stratifier**. Due to Health Insurance Portability and Accountability Act (HIPAA) privacy regulations and PhysioNet Data Use Agreements (DUA), patient-level MIMIC-IV files are **never committed to version control**.

---

## 1. MIMIC-IV v2.2 (Medical Information Mart for Intensive Care)

### Overview
MIMIC-IV is a de-identified, comprehensive electronic health record (EHR) database comprising hospital admissions to Beth Israel Deaconess Medical Center (Boston, MA) between 2008 and 2019.

### Step-by-Step Credentialing & Download
1. **PhysioNet Account**: Register an account on [PhysioNet](https://physionet.org/).
2. **Ethics Training**: Complete the CITI program course *"Data or Specimens Only Research"* or *"Conflicts of Interest"*. Submit your completion certificate to PhysioNet.
3. **DUA Verification**: Sign the Data Use Agreement pledging never to re-identify patients or share raw data.
4. **Download via `physionet-build` or `wget`**:
   Once access is granted, download the `hosp` and `icu` modules to `data/raw/mimiciv/`:
   ```bash
   # Using wget with PhysioNet user credentials
   wget -r -N -c -np --user <your_username> --ask-password \
     https://physionet.org/files/mimiciv/2.2/
   ```
5. **Expected Directory Layout**:
   ```
   data/raw/mimiciv/
   ├── hosp/
   │   ├── admissions.csv.gz
   │   ├── d_hcpcs.csv.gz
   │   ├── d_icd_diagnoses.csv.gz
   │   ├── d_icd_procedures.csv.gz
   │   ├── d_labitems.csv.gz
   │   ├── diagnoses_icd.csv.gz
   │   ├── drgcodes.csv.gz
   │   ├── emar.csv.gz
   │   ├── emar_detail.csv.gz
   │   ├── hcpcsevents.csv.gz
   │   ├── labevents.csv.gz
   │   ├── microbiologyevents.csv.gz
   │   ├── omr.csv.gz
   │   ├── patients.csv.gz
   │   ├── pharmacy.csv.gz
   │   ├── poe.csv.gz
   │   ├── poe_detail.csv.gz
   │   ├── prescriptions.csv.gz
   │   ├── procedures_icd.csv.gz
   │   ├── provider.csv.gz
   │   ├── services.csv.gz
   │   └── transfers.csv.gz
   └── icu/
       ├── chartevents.csv.gz
       ├── datetimeevents.csv.gz
       ├── icustays.csv.gz
       ├── inputevents.csv.gz
       ├── outputevents.csv.gz
       └── procedureevents.csv.gz
   ```

---

## 2. CMS Hospital Readmissions Reduction Program (HRRP)

### Overview
The Centers for Medicare & Medicaid Services (CMS) HRRP penalizes hospitals with excess readmissions for specific conditions:
- Acute Myocardial Infarction (AMI)
- Chronic Obstructive Pulmonary Disease (COPD)
- Heart Failure (HF)
- Pneumonia (PNA)
- Coronary Artery Bypass Graft (CABG) surgery
- Elective Primary Total Hip/Knee Arthroplasty (THA/TKA)

### Download Methods
#### Method A: Automated Download via Python Script
Run the automated ingestion script:
```bash
python scripts/data_collection.py --source cms --output-dir data/raw/cms/
```

#### Method B: Manual Download from Data.CMS.gov
1. Navigate to [CMS Hospital Readmissions Reduction Program dataset](https://data.cms.gov/provider-data/dataset/9n3s-kdb3).
2. Click **Export** > **CSV**.
3. Save the resulting file as `data/raw/cms/FY_Hospital_Readmissions_Reduction_Program_Hospital.csv`.

---

## 3. ICD-9 to ICD-10 Crosswalk

The National Bureau of Economic Research (NBER) / CMS General Equivalence Mappings (GEMs) provide mapping between ICD-9-CM and ICD-10-CM diagnosis codes:
```bash
python scripts/data_collection.py --source crosswalk --output-dir data/external/
```
Expected output: `data/external/icd9_to_icd10_crosswalk.csv`.

---

## 4. Integrity and Security Checklist
- [ ] Ensure `.gitignore` ignores `data/raw/` and all `*.csv`, `*.parquet`, and `*.gz` extensions.
- [ ] Verify MD5 / SHA-256 checksums after downloading PhysioNet files.
- [ ] Restrict file permissions locally (`chmod 600` or Windows NTFS ACLs).
- [ ] Never post patient identifiers or de-identified raw timestamps on public forums.
