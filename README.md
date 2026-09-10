# Clinical 30-Day Hospital Readmission Risk Stratifier

[![CI](https://github.com/abdussatarkhan/healthcare-readmission-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/healthcare-readmission-prediction/actions)
[![XGBoost](https://img.shields.io/badge/XGBoost-Clinical_ML-EB5424?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/) [![SHAP](https://img.shields.io/badge/Explainability-SHAP-00C896?style=for-the-badge)](https://shap.readthedocs.io/) [![Python](https://img.shields.io/badge/Python-EHR_Analytics-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **An explainable machine learning risk stratifier trained on CMS HRRP and MIMIC-IV electronic health records (EHR) utilizing XGBoost and SHAP interpretability to prevent 30-day all-cause hospital readmissions.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    EHR[MIMIC-IV & CMS Electronic Health Records] --> Preprocess[ICD-10 Crosswalk & Lab Feature Extraction]
    Preprocess --> Model[Calibrated XGBoost Classifier]
    Model --> SHAP[TreeSHAP Feature Attribution Engine]
    SHAP --> Dashboard[Clinical Bedside Risk Summary]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `XGBoost` `SHAP` `Scikit-Learn` `Pandas` `EHR Analytics`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![healthcare-readmission-prediction preview](images/roc_curve.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/healthcare-readmission-prediction.git
cd healthcare-readmission-prediction
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 🗺️ Roadmap & Upcoming Features

- [x] 30-day all-cause readmission prediction on MIMIC-IV and CMS HRRP
- [x] TreeSHAP feature attribution & explainability plots
- [ ] Interactive Streamlit bedside clinical risk calculator
- [ ] Demographic fairness and algorithmic bias auditing
- [ ] FHIR / HL7 clinical data ingestion interface

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.