# Gait Anomaly Detection for Neurodegenerative Diseases

A machine learning system that detects signs of neurodegenerative diseases from walking sensor data. Trained on the PhysioNet Gait in Neurodegenerative Disease Database, it classifies gait patterns as healthy or impaired across ALS, Parkinson's, and Huntington's disease.

This project mirrors the AI layer behind smart insole technology — the same gait signals captured by embedded foot sensors are used here to drive clinical predictions.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.918 |
| Recall | 0.920 |
| Cross-validation | 5-fold stratified |

Recall is the primary metric: in medical AI, missing a sick patient is worse than a false alarm.

---

## How It Works

Each patient's gait file contains hundreds of 10-second walking windows. For each window, 12 biomechanical signals are recorded — stride intervals, swing/stance percentages, double support time, and step timing — from both feet.

From these time-series, 5 statistical features are extracted per signal: mean, standard deviation, coefficient of variation, median, and IQR. The coefficient of variation is the most discriminative feature: healthy walkers are highly consistent, while patients with neurodegenerative conditions show measurably higher variability.

A Random Forest classifier with balanced class weights is trained on these 60 features across 64 subjects.

---

## Dataset

**Source:** [PhysioNet — Gait in Neurodegenerative Disease Database](https://physionet.org/content/gaitndd/1.0.0/)

| Group | Subjects | Label |
|---|---|---|
| Healthy controls | 16 | 0 |
| ALS patients | 13 | 1 |
| Parkinson's patients | 15 | 1 |
| Huntington's patients | 20 | 1 |

---

## Project Structure

```
gait-anomaly-detection/
├── esteps.ipynb       # full walkthrough: loading, features, training, evaluation
├── app.py             # Streamlit demo app
├── requirements.txt
└── gait-in-neurodegenerative-disease-database-1.0.0/
    ├── als*.ts
    ├── park*.ts
    ├── hunt*.ts
    └── control*.ts
```

---

## Setup

```bash
pip install -r requirements.txt
```

Train the model by running all cells in `esteps.ipynb`. This saves `model.pkl` to the project folder.

Then launch the demo:

```bash
streamlit run app.py
```

Upload any `.ts` file from the dataset to get a prediction.

---

## Demo

The Streamlit app accepts a `.ts` gait file and returns:
- A healthy / impaired prediction with confidence score
- A live chart of left vs. right stride intervals
- Key extracted features including stride variability and double support time

---

## Tech Stack

Python, scikit-learn, pandas, NumPy, Streamlit, Matplotlib, joblib

---

## Note on Architecture

Streamlit is used here for demo purposes. A production system would expose a FastAPI `/predict` endpoint consumed by a mobile application, with the model served behind a REST API.
