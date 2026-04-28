# CardioSense — Heart Disease Predictor

CardioSense is a Streamlit web application that predicts heart disease risk using a Logistic Regression model trained on the Cleveland Heart Disease dataset.

## Live Link :-
https://cardiosense-e.streamlit.app/

## Overview

The app collects common clinical parameters such as age, sex, chest pain type, blood pressure, cholesterol, heart rate, and other diagnostic features, then returns a binary prediction:

- **0** → No Heart Disease
- **1** → Heart Disease Detected

It also shows a risk probability and allows the result to be downloaded as a CSV report.

## Features

- Clean Streamlit UI with tabs for prediction, about, and dataset information
- Logistic Regression-based prediction model
- Probability-based risk display
- Downloadable patient summary report
- Educational overview of heart disease and prevention tips
- Built using Python, pandas, NumPy, and scikit-learn

## Project Structure

```text
.
├── DeployModel_improved.py   # Streamlit app
├── retrain.py                # Model training script
├── trained_model.sav         # Saved trained model
├── heart_disease_data.csv    # Dataset used for training
├── requirements.txt          # Python dependencies
└── .python-version           # Python version for deployment
```

## Dataset

This project uses the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.


### Features Used

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

## Model Details

- **Algorithm:** Logistic Regression
- **Library:** scikit-learn
- **Training Script:** `retrain.py`
- **Saved Model:** `trained_model.sav`

The deployment script loads the trained model and uses `predict()` and `predict_proba()` for inference.

## Requirements

Install the dependencies listed in `requirements.txt`:

```txt
streamlit==1.32.0
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.2
```

## How to Run Locally

### 1. Clone or download the project

Make sure all files are in the same folder.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run DeployModel_improved.py
```

## How to Retrain the Model

If you want to retrain the model from scratch:

```bash
python retrain.py
```

This will:

1. Load `heart_disease_data.csv`
2. Split the data into train and test sets
3. Train a Logistic Regression model
4. Save the model as `trained_model.sav`

## Input Format

The model expects the features in this order:

```text
[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
```

## Limitations

- This project is for **educational/demo purposes only**
- It is **not** a substitute for medical diagnosis


## Disclaimer

This application is intended for learning and demonstration. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment.


---

**Built with:** Python, Streamlit, pandas, NumPy, scikit-learn
