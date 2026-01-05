
# 🏃‍♀️ FitPulse – Health & Fitness Analytics System

FitPulse is an end-to-end **health data analytics and intelligence system** designed to simulate wearable fitness data, process it, generate insights, and apply predictive and analytical intelligence.

The project is developed incrementally across **four milestones**, with each milestone adding meaningful and functional capabilities that together form a complete health intelligence pipeline.

---

## 🚩 Problem Statement

Wearable fitness devices generate large volumes of health data such as heart rate, steps, and sleep patterns.  
However, raw data alone is not useful without:

- Proper data cleaning and structuring  
- Trend analysis and forecasting  
- Intelligence to detect anomalies and behavioral patterns  

**FitPulse** addresses this challenge by building a complete pipeline that transforms raw fitness data into actionable insights.

---

## 🧩 Milestone 1 – Data Collection & Preprocessing

### Features Implemented
- Synthetic generation of fitness data:
  - Heart Rate (CSV)
  - Steps Count (CSV)
  - Sleep Cycles (JSON)
- Data cleaning and preprocessing:
  - Handling missing values
  - Timestamp alignment
  - Merging multiple sources into a unified dataset
- Export of cleaned dataset for further analysis

### Output
- Unified cleaned dataset (`cleaned_fitness_data.csv`)
- Raw and cleaned data preview via Streamlit

---

## 📈 Milestone 2 – Forecasting & Trend Analysis

### Features Implemented
- Time-series forecasting for:
  - Heart rate trends
  - Sleep duration patterns
  - Step count behavior
- Visualization of:
  - Forecasted trends
  - Seasonal and trend components
- Interactive forecast selection through the UI

### Outcome
- Enables short-term prediction of fitness metrics  
- Helps analyze future behavior patterns

---

## 🧠 Milestone 3 – Intelligence & Insights Layer

### Comparative Analytics
- Daily aggregation of fitness data
- Comparison of:
  - Average heart rate
  - Total steps
  - Average sleep duration
- Visual trend comparison for easy interpretation

### Anomaly Detection
- Automatic detection of abnormal heart rate values
- Useful for identifying unusual or irregular health events

### Behaviour Analysis
- Computes:
  - Average daily steps
  - Average sleep duration
- Classifies lifestyle as:
  - Sedentary
  - Moderately Active
  - Active

### Outcome
- Converts raw metrics into meaningful health intelligence  
- Adds analytical depth beyond basic visualization

---

## 🧩 Milestone 4 – Unified Dashboard & Productivity

### Features Implemented
- Integrated all milestones into a single **Streamlit dashboard**
- Centralized access to:
  - Data generation
  - Preprocessing
  - Forecasting
  - Intelligence & insights
- Automated health summary generation
- Export options for:
  - Cleaned dataset (CSV)
  - Health summary report (CSV)

### Outcome
Milestone 4 transforms FitPulse into a **unified, user-friendly health analytics application** with productivity and reporting features.

---

## 🛠️ Technology Stack

- **Frontend / UI:** Streamlit  
- **Backend:** Python  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Streamlit Charts, Matplotlib  
- **Forecasting:** Time-series models  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure

```

FitPulse_Health_Detection_Project/
│
├── app.py                     # Main Streamlit application
├── src/
│   ├── generate_data.py        # Synthetic data generation
│   ├── preprocess.py           # Data cleaning & merging
│   ├── forecasting.py          # Time-series forecasting
│   ├── anomaly_detection.py    # Anomaly detection logic
│   ├── visualization.py        # Chart rendering
│   ├── milestone3/
│   │   ├── anomaly_detection.py
│   │   ├── behavior_analysis.py
│   │   └── comparison.py
│   ├── milestone4/
│   │   └── report_utils.py
│   └── utility modules
│
├── data/
│   ├── raw/
│   │   ├── heart_rate.csv
│   │   ├── steps.csv
│   │   └── sleep.json
│   └── cleaned_fitness_data.csv
│
├── notebooks/
├── README.md

````

---

## ▶️ How to Run the Project

### Clone the repository
```bash
git clone https://github.com/yuvarajkaleru/Fitpulse-Health-Detection.git
````

### Navigate to the project folder

```bash
cd FitPulse_Health_Detection_Project
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## ✅ Project Status

All features across all milestones are fully implemented and functional.

Just tell me 👍
```
