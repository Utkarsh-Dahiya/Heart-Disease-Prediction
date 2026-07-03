# ❤️ CardioPredict — Heart Disease Risk Assessment Dashboard

A production-ready **Streamlit** web application for predicting heart disease
risk, built on top of an existing scikit-learn machine-learning pipeline.
The ML logic (preprocessing, encoding, scaling, and model selection) is
**unchanged** from the original notebook — this project only adds a polished,
medical-themed UI and a clean application structure around it.

![status](https://img.shields.io/badge/status-ready-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-ff4b4b)

---

## ✨ Features

- 🎨 Animated gradient dark background with glassmorphism cards
- ❤️ Beating heart icon / hero header
- 🩺 Sidebar clinical intake form, grouped by category
- 📊 Plotly gauge chart for risk confidence
- 📈 Probability bar chart (Healthy vs. Disease)
- 🧾 Attractive metric cards (prediction, confidence, probabilities)
- 📖 Feature glossary explaining every clinical input
- 🤖 "Model Information" panel: selected model, accuracy, and a comparison
  chart of all 5 candidate algorithms from the notebook
- 📱 Responsive layout that adapts to mobile screens
- 🌙 Dark-themed by default (configured via `.streamlit/config.toml`)

---

## 🗂️ Project Structure

```
heart-disease-app/
├── app.py                  # Streamlit UI (this is what you run)
├── train_model.py          # Reproduces the notebook's ML pipeline -> model.pkl
├── model.pkl                # Model artifact (model + scaler + schema)
├── sample_data_demo.csv      # ⚠️ Synthetic placeholder data (demo/testing only)
├── requirements.txt         # Python dependencies
├── README.md                 # You are here
└── .streamlit/
    └── config.toml           # Dark medical dashboard theme
```

> ⚠️ **Important:** The `model.pkl` and `sample_data_demo.csv` shipped in
> this template were generated from **randomly-generated synthetic data**
> (not the real UCI/Kaggle heart disease dataset), purely so the app runs
> out of the box for UI testing. Its accuracy numbers are **not
> representative** of the notebook's real ~87% result. **Before using this
> for real predictions, regenerate `model.pkl` from your actual `heart.csv`**
> using the steps below.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate the model artifact

The trained model isn't stored in source control as raw pickled binary in
this template — you generate it once from your dataset using the exact
notebook logic:

```bash
python train_model.py --data heart.csv
```

This will:
1. Load `heart.csv` (the same dataset used in the notebook)
2. Impute zero-values in `Cholesterol` and `RestingBP` with the mean of
   non-zero values (exactly as the notebook does)
3. One-hot encode categorical features
4. Standardize numeric features
5. Train & compare 5 classifiers (Logistic Regression, KNN, Naive Bayes,
   Decision Tree, SVM)
6. Save the **best-performing model** together with the fitted scaler and
   feature schema into `model.pkl`

> 💡 Logistic Regression was the top performer in the original notebook run
> (~87% test accuracy) and will typically be auto-selected again.

### 3. Launch the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`) in your
browser.

---

## 🧠 How Predictions Work

1. You fill in clinical values in the sidebar (age, sex, chest pain type,
   blood pressure, cholesterol, etc.).
2. `app.py`'s `preprocess_input()` function converts your raw inputs into
   **exactly** the same feature representation the model was trained on:
   zero-imputation → one-hot encoding → scaling → column alignment.
3. The saved model predicts the class (`0` = No Heart Disease,
   `1` = Heart Disease) and outputs class probabilities.
4. The UI renders the result as a banner, metric cards, a gauge chart, and
   a probability bar chart.

---

## 🩺 Clinical Features Used

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Sex | Male / Female |
| ChestPainType | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (Yes/No) |
| RestingECG | Normal, ST-T abnormality, Left Ventricular Hypertrophy |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Yes/No) |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment (Up/Flat/Down) |

---

## ⚠️ Disclaimer

This application is built for **educational and demonstrative purposes
only**. It is **not** a certified medical device and must **not** be used
as a substitute for professional medical diagnosis or advice.

---

## 👨‍💻 Developer

Built with ❤️ using **Streamlit**, **scikit-learn**, and **Plotly**.
