<h1><b>🚀 FitPulse – Wearable Fitness Data Intelligence</b></h1>

A full pipeline for preprocessing, feature extraction, clustering, and forecasting physiological sensor data.

⸻

<b>📌 Overview</b>

FitPulse is an end-to-end system built to transform wearable fitness sensor data into clean, meaningful, and predictive insights.
The project handles raw timestamped data, extracts behavioural features, clusters activity patterns, and forecasts future physiology trends.

This project simulates what a real HealthTech or Sports Analytics workflow looks like.

⸻

<b>✅ Milestone 1</b>

A full preprocessing engine that:
	•	Cleans raw sensor data
	•	Fixes timestamps and numeric issues
	•	Removes outliers and domain errors
	•	Handles missing values with interpolation
	•	Smooths heart-rate using rolling filters
	•	Produces a final tidy CSV with the columns:

timestamp, heart_rate, hrv, stress_score, skin_temp



⸻

<b>✅ Milestone 2</b>

Milestone 2 adds analytics and machine learning components:

📐 Feature Extraction (TSFresh)
	•	Extracts statistical & temporal features
	•	Rolling windows across heart rate / HRV / stress / temperature
	•	Computes metrics such as mean, variance, entropy, autocorrelation, trend slope

🧩 Clustering (KMeans & DBSCAN)
	•	Groups behavioural segments based on extracted features
	•	Detects unusual or anomalous windows
	•	PCA visualisation to understand cluster separation

📈 Forecasting (Prophet)
	•	Heart rate prediction (60 days → next 14 days)
	•	Sleep duration modelling with weekly seasonality
	•	Steps forecasting with holiday/event impacts
	•	Trend, seasonality, and confidence-interval interpretation

🎛 Interactive Streamlit Dashboard

Users can:
	•	Upload CSV/JSON
	•	Select which signal to analyse
	•	Choose clustering algorithm
	•	Run TSFresh, KMeans, DBSCAN, or Prophet forecasting
	•	View diagnostic plots and results instantly

This milestone turns FitPulse into a real analytical platform, not just a preprocessing tool.

⸻

<b>🔜 Milestone 3 — Coming Next</b>

The next milestone will focus on expanding FitPulse into a complete intelligence system.
Planned additions include:

🔧 Advanced ML / DL Models
	•	LSTM-based forecasting
	•	Deep anomaly detection
	•	Behaviour recognition

📊 Comparative analytics module
	•	Compare days, weeks, or events
	•	Personal wellness scoring

☁ Deployment
	•	Hosting the entire application
	•	Optional real-time ingestion endpoints

Milestone 3 moves the project closer to a production-ready fitness analytics engine.

⸻

<b>🤖 AI-Assisted Development (Honest & Professional)</b>

This project was built with the help of AI tools like ChatGPT for:
	•	Code generation for repetitive sections
	•	UI styling ideas
	•	Debugging
	•	Faster experimentation

But:
	•	All architecture decisions
	•	Data logic
	•	Model selection
	•	Integration between modules
	•	Validation & testing

…were done by me.

Using AI responsibly allowed me to work faster, learn better, and focus on meaningful design, just like modern developers do with tools such as Copilot.


<b><h1>🧪 How to Run</h1></b>

Install dependencies:

pip install streamlit tsfresh prophet scikit-learn pandas numpy matplotlib

Run the dashboard:

streamlit run app.py


⸻

<h2>⭐ If this project helps or inspires you, please consider giving it a star — it really motivates me!</h2>
