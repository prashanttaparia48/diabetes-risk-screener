#  Diabetes Risk Screener

A machine learning web app that estimates your risk of diabetes based on a few basic health measurements. Built with Scikit-learn and Streamlit as part of my BYOP (Bring Your Own Project) submission.

## Why I built this

Diabetes is everywhere in India — and most people don't find out they have it until something goes wrong. Lab tests cost money, clinics aren't always nearby, and awareness is low. I wanted to build something that anyone could open on their phone, fill in a few numbers, and at least get a heads-up that they should see a doctor.

It's not a diagnostic tool. It's more like a friendly nudge.

## What it does

- Takes 8 basic health inputs (glucose, BMI, age, etc.)
- Runs them through a trained Random Forest model
- Returns a **Low / Moderate / High** risk rating with a probability score
- Shows which factors influenced the result most
- Works entirely in your browser — nothing is stored or sent anywhere

## Project structure

```
diabetes-risk-screener/
├── app.py                  # The Streamlit web app
├── requirements.txt        # Everything you need to install
├── models/                 # Where the trained model lives (created after training)
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
├── src/
│   └── train_model.py      # Training pipeline — run this first
└── notebooks/
    └── eda.py              # Exploratory data analysis
```

## How to run it

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Train the model**
```bash
python src/train_model.py
```
This downloads the dataset and saves the trained model to `/models`. Takes about 10–20 seconds.

**Step 3 — Launch the app**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> **Mac users:** If Step 2 gives an SSL error, run this first:
> `/Applications/Python 3.x/Install Certificates.command`

## The dataset

**PIMA Indians Diabetes Dataset** — originally from the National Institute of Diabetes and Digestive and Kidney Diseases, widely available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

768 records, 8 features, binary outcome (diabetic / not diabetic).

## Model details

I went with Random Forest after comparing it against Logistic Regression and a basic Decision Tree. It had the best ROC-AUC and gives feature importances for free, which made the app more explainable.

A few things I paid attention to:
- Zeros in columns like Glucose and BMI are biologically impossible — I treated them as missing values and replaced them with column medians
- The dataset is imbalanced (~65% non-diabetic), so I used `class_weight='balanced'` to stop the model from just predicting "no diabetes" all the time
- Added two engineered features: BMI×Age (risk compounds with age) and Glucose/Insulin ratio (proxy for insulin resistance)

**Results on test set:**
- Accuracy: ~79%
- ROC-AUC: ~0.85
- CV AUC (5-fold): ~0.83 ± 0.02

## Honest disclaimer

This is a student project. It hasn't been clinically validated and should not be used as a substitute for medical advice. If you're worried about your health, please see a doctor.

## AUTHOR
- NAME : PRASHANT TAPARIA
- REG.NO : 25BCE11043
- BRANCH : BTECH CSE
- VIT BHOPAL UNIVERSITY
