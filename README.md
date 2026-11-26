# SentinelAI 
### AI-Driven Anomaly Detection & Predictive Screening System

**Problem Statement ID:** 26170  
**Organization:** Indian Space Research Organisation (ISRO)  
**Theme:** Smart Automation  
**Category:** Software

---

##  Overview

SentinelAI is a machine learning-based component reliability screening system designed to detect latent defects and predict electrical parameter drift in electronic components undergoing Burn-In testing.

Traditional quality screening methods rely on fixed parametric thresholds, which can overlook components that technically pass initial specifications but exhibit abnormal behaviour over time.

SentinelAI addresses this challenge through two integrated machine learning modules:

- **Dynamic Anomaly Detection:** Identifies abnormal component behaviour by comparing electrical measurements across a batch.
- **Time-Series Drift Prediction:** Forecasts future electrical parameters to identify components at risk of degradation.

The platform also provides explainable predictions and an interactive dashboard to support quality assurance decisions.

##  Objectives

- Detect anomalous electronic components that pass conventional static limits.
- Identify subtle deviations in electrical parameters.
- Predict future component behaviour using early-stage measurements.
- Calculate component drift rates and compare them against configurable safety thresholds.
- Provide interpretable model predictions for quality inspection.
- Enable batch-level component screening and automated reporting.

##  Key Features

### 1. Dynamic Anomaly Detection

Uses the Isolation Forest algorithm to identify unusual component behaviour.

Analyses parameters such as:
- Standby current (Iddq)
- Leakage current
- Propagation delay
- Temperature
- Early-stage electrical measurements

Components are classified into three categories:

| Classification | Description |
|---|---|
| NORMAL | Expected electrical behaviour |
| WARNING | Potential abnormality requiring attention |
| ANOMALOUS | Significant deviation from expected behaviour |

### 2. Time-Series Drift Prediction

Uses a Random Forest Regression model to predict the electrical parameter value at 168 hours.

**Input parameters:**
- Value_0h
- Value_24h

**Prediction:**
- Value_168h

The system calculates predicted drift and drift rate, then compares the result against a configurable safety slope to identify potentially unreliable components.

### 3. Explainable AI

Provides interpretable explanations for flagged components, including:

- Parameters contributing to anomaly detection.
- Deviations from batch-level behaviour.
- Predicted electrical drift.
- Safety threshold comparisons.
- Component risk classification.

### 4. Interactive Dashboard

The web application provides:

- Component screening overview.
- Batch-level analytics.
- Interactive data visualizations.
- Component-level analysis.
- Searchable and filterable component records.
- CSV dataset upload.
- CSV and PDF report exports.
- Screening history and database storage.

##  Machine Learning Approach

| Module | Algorithm | Purpose |
|---|---|---|
| Anomaly Detection | Isolation Forest | Detect abnormal components |
| Drift Prediction | Random Forest Regressor | Forecast Value_168h |
| Preprocessing | RobustScaler | Scale electrical measurements |
| Explainability | Feature attribution and statistical analysis | Explain model decisions |

### Model Evaluation

The drift prediction model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The anomaly detection module prioritizes identifying defective components while minimizing false negatives.

##  System Architecture

```text
       Component Dataset
               |
               v
      Data Preprocessing
               |
        +------+------+
        |             |
        v             v
   Anomaly        Drift
   Detection     Prediction
        |             |
        v             v
   Risk Score    Future Value
        |             |
        +------+------+
               |
               v
        Explainable AI
               |
               v
      Quality Dashboard
               |
        +------+------+
        |             |
        v             v
    Analytics      Reports
                  CSV / PDF
```

##  Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | Scikit-learn, XGBoost (where available) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly.js |
| Database | SQLite |
| Model Persistence | Joblib |
| PDF Generation | FPDF2 |

##  Project Structure

```text
SentinelAI/
│
├── app.py
├── requirements.txt
├── README.md
├── database.db
│
├── models/
│   ├── anomaly_model.py
│   ├── drift_model.py
│   ├── model_training.py
│   └── saved/
│
├── data/
│   ├── generate_dataset.py
│   └── sample_components.csv
│
├── utils/
│   ├── preprocessing.py
│   ├── prediction.py
│   └── explanation.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── components.html
│   ├── analysis.html
│   └── reports.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── uploads/
```

##  Installation & Setup

### Prerequisites

- Python 3.11 or later
- Git
- Web browser

### 1. Clone the repository

```bash
git clone https://github.com/Priya1706/SkillLab.git
```

### 2. Navigate to the project directory

```bash
cd SkillLab/SentinelAI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000
```

##  Dataset

The project includes a synthetic Burn-In testing dataset containing 1,200 component records distributed across 12 batches.

It represents different component behaviour patterns:

- Normal components.
- High leakage components.
- Gradual drift components.
- Latent defect components.
- Sudden electrical shifts.

Measurements are represented at 0h, 24h, 96h and 168h.

The application also supports uploading compatible CSV datasets for component screening.

**Note:** The included dataset is synthetically generated for demonstration and model development. It is not actual ISRO test data or proprietary aerospace component data.

##  Evaluation & Validation

The implemented drift prediction model was evaluated using a batch-grouped train/test split.

| Metric | Observed Result |
|---|---:|
| MAE | 0.3349 |
| RMSE | 0.5642 |
| R² Score | 0.9791 |

These results are from the synthetic demonstration dataset and should not be interpreted as validated performance on real-world aerospace components.

##  Future Scope

- Integration with real Burn-In testing equipment.
- Support for additional electrical parameters.
- Advanced time-series forecasting models.
- Improved explainability using SHAP.
- Real-time component monitoring.
- Integration with industrial quality management systems.

##  Developer

**Priyamvadhaa Hoskere**

GitHub: [@Priya1706](https://github.com/Priya1706)

---

*SentinelAI – Making component reliability screening smarter through machine learning.*
