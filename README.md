# 🏭 AI Maintenance & Optimization Dashboard

An end-to-end machine learning application for predictive industrial maintenance — detecting anomalies, predicting failure risk, and providing an interactive dashboard for maintenance teams.

## 📌 Overview

This project uses real industrial sensor data to help maintenance teams identify machines that are behaving abnormally or at risk of failure, before a breakdown happens. It combines unsupervised anomaly detection with supervised failure prediction, all wrapped in a role-based, interactive web dashboard.

## ✨ Features

- 🔐 **Role-based Login** — Admin and Employee views with different access levels
- 📊 **Interactive Dashboard** — Real-time metrics, pie charts, and distribution plots
- ⚠️ **Anomaly Detection** — Isolation Forest model flags abnormal sensor readings
- 🔮 **Failure Prediction** — Random Forest model predicts failure probability and risk level (Low/Medium/High)
- 🔍 **Machine Lookup** — Search any machine by ID to see its full health status and a maintenance recommendation
- 📈 **Sensor Trends** — Visualize sensor readings over time

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn (Isolation Forest, Random Forest) |
| Dashboard/UI | Streamlit |
| Visualization | Plotly |
| Authentication | streamlit-authenticator |
| Data Source | [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) |

## 📂 Project Structure

```
├── app.py                          # Main Streamlit application
├── train_failure_model.py          # Script to train the Random Forest failure model
├── config.yaml                     # Login credentials configuration
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── ai4i2020_cleaned.csv            # Cleaned dataset
├── isolation_forest_model.pkl      # Trained anomaly detection model
├── failure_prediction_model.pkl    # Trained failure prediction model
├── Code_to_Clean_DataSet.ipynb     # Data cleaning notebook (Google Colab)
└── README.md
```

## 🚀 How to Run Locally

1. Clone this repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Install dependencies
```bash
pip install streamlit streamlit-authenticator pandas scikit-learn joblib plotly pyyaml
```

3. Run the app
```bash
streamlit run app.py
```

4. Open `http://localhost:8501` in your browser

**Demo Credentials:**
- Admin: `admin` / `admin123`
- Employee: `employee` / `employee123`

## 🤖 Machine Learning Approach

- **Anomaly Detection:** Isolation Forest (unsupervised) — flags statistically unusual sensor readings without needing labeled failure data
- **Failure Prediction:** Random Forest Classifier (supervised) — trained on the labeled `machine_failure` column, using `class_weight='balanced'` to handle the ~3% class imbalance. Achieves ~96% accuracy and ~81% recall on the failure class.

## 🔮 Future Improvements

- Replace static CSV with a live database (PostgreSQL) for real-time sensor ingestion
- Add a FastAPI backend layer for proper API-based architecture
- Add automated model retraining pipeline to handle model drift
- Add RAG + LLM layer for natural-language failure explanations and maintenance recommendations
- Real-time alerting (email/SMS) for high-risk machines

## 📊 Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, a synthetic dataset reflecting real industrial predictive maintenance data, containing air/process temperature, rotational speed, torque, tool wear, and failure labels.

## 👤 Author

Aayush Srivastava
