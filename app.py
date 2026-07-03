"""
app.py
------
Streamlit front-end for the Heart Disease Prediction model.

IMPORTANT: The machine-learning logic (imputation, one-hot encoding,
scaling, model selection) is untouched from the original notebook.
This file is purely responsible for UI/UX, input handling, and
translating raw user input into the exact feature vector the trained
model expects (see `preprocess_input`).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict | Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# --------------------------------------------------------------------------
# Custom CSS — gradient background, glass cards, animations, responsiveness
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* ---------- App-wide gradient background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 45%, #24243e 100%);
    background-attachment: fixed;
}

/* subtle animated glow blobs */
.stApp::before {
    content: "";
    position: fixed;
    top: -10%;
    left: -10%;
    width: 40%;
    height: 40%;
    background: radial-gradient(circle, rgba(255,75,110,0.25) 0%, rgba(255,75,110,0) 70%);
    z-index: 0;
    animation: float1 12s ease-in-out infinite;
}
.stApp::after {
    content: "";
    position: fixed;
    bottom: -10%;
    right: -10%;
    width: 45%;
    height: 45%;
    background: radial-gradient(circle, rgba(56,189,248,0.20) 0%, rgba(56,189,248,0) 70%);
    z-index: 0;
    animation: float2 14s ease-in-out infinite;
}
@keyframes float1 {
    0%,100% { transform: translate(0,0); }
    50% { transform: translate(30px, 40px); }
}
@keyframes float2 {
    0%,100% { transform: translate(0,0); }
    50% { transform: translate(-30px, -30px); }
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #e8e8f0 !important;
}

/* ---------- Glass card ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    color: #f0f0f5;
    margin-bottom: 1rem;
}

/* ---------- Hero header ---------- */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff4b6e, #ff8fa3, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.1rem;
    text-align: center;
}
.hero-subtitle {
    text-align: center;
    color: #b8b8cc;
    font-size: 1.05rem;
    margin-bottom: 1.6rem;
}

/* ---------- Beating heart ---------- */
.heartbeat {
    display: inline-block;
    font-size: 2.4rem;
    animation: beat 1.15s infinite;
    filter: drop-shadow(0 0 12px rgba(255,75,110,0.7));
}
@keyframes beat {
    0%   { transform: scale(1); }
    14%  { transform: scale(1.25); }
    28%  { transform: scale(1); }
    42%  { transform: scale(1.2); }
    70%  { transform: scale(1); }
}

/* ---------- Metric cards ---------- */
.metric-card {
    border-radius: 16px;
    padding: 1.1rem 1rem;
    text-align: center;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    height: 100%;
}
.metric-card h3 {
    font-size: 0.85rem;
    font-weight: 500;
    opacity: 0.9;
    margin: 0 0 0.3rem 0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0;
}
.metric-teal   { background: linear-gradient(135deg, #0d9488, #14b8a6); }
.metric-purple { background: linear-gradient(135deg, #6d28d9, #8b5cf6); }
.metric-red    { background: linear-gradient(135deg, #dc2626, #f87171); }
.metric-green  { background: linear-gradient(135deg, #16a34a, #4ade80); }
.metric-blue   { background: linear-gradient(135deg, #1d4ed8, #60a5fa); }

/* ---------- Result banners ---------- */
.result-banner {
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.result-high {
    background: linear-gradient(135deg, rgba(220,38,38,0.9), rgba(248,113,113,0.85));
}
.result-low {
    background: linear-gradient(135deg, rgba(22,163,74,0.9), rgba(74,222,128,0.85));
}
.result-banner h2 {
    color: white;
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0.2rem 0;
}
.result-banner p {
    color: rgba(255,255,255,0.92);
    margin: 0;
}

/* ---------- Section title ---------- */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f0f0f5;
    margin: 1.4rem 0 0.7rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---------- Footer ---------- */
.app-footer {
    text-align: center;
    padding: 1.6rem 0 0.6rem 0;
    color: #9797ad;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin-top: 2rem;
}
.app-footer a {
    color: #ff8fa3;
    text-decoration: none;
    font-weight: 600;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    background: linear-gradient(135deg, #ff4b6e, #ff8fa3);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
    font-size: 1.02rem;
    width: 100%;
    box-shadow: 0 6px 18px rgba(255,75,110,0.35);
    transition: transform 0.15s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(255,75,110,0.45);
    color: white;
}

/* ---------- Misc text on dark bg ---------- */
h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {
    color: #eaeaf2;
}
.streamlit-expanderHeader {
    color: #eaeaf2 !important;
    font-weight: 600 !important;
}

/* ---------- Mobile responsiveness ---------- */
@media (max-width: 768px) {
    .hero-title { font-size: 1.9rem; }
    .metric-card .value { font-size: 1.4rem; }
    .glass-card { padding: 1rem 1.1rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_model_bundle()

# Known category levels (mirrors the notebook's pd.get_dummies output)
CATEGORY_LEVELS = {
    "Sex": ["F", "M"],
    "ChestPainType": ["ASY", "ATA", "NAP", "TA"],
    "RestingECG": ["LVH", "Normal", "ST"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Down", "Flat", "Up"],
}
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def preprocess_input(raw: dict, bundle: dict) -> pd.DataFrame:
    """
    Convert raw user-entered values into the exact feature vector the
    trained model expects. Mirrors the notebook pipeline:
      1. zero-value imputation (Cholesterol / RestingBP)
      2. one-hot encoding
      3. StandardScaler transform (using the FITTED scaler, not refit)
      4. column alignment to `feature_columns`
    """
    row = dict(raw)

    # 1) Zero-value imputation using means captured at training time
    if row["Cholesterol"] == 0:
        row["Cholesterol"] = bundle["cholesterol_mean"]
    if row["RestingBP"] == 0:
        row["RestingBP"] = bundle["resting_bp_mean"]

    # 2) Manual one-hot encoding (safe for a single-row input)
    encoded = {
        "Age": row["Age"],
        "RestingBP": row["RestingBP"],
        "Cholesterol": row["Cholesterol"],
        "FastingBS": row["FastingBS"],
        "MaxHR": row["MaxHR"],
        "Oldpeak": row["Oldpeak"],
    }
    for col, levels in CATEGORY_LEVELS.items():
        for level in levels:
            encoded[f"{col}_{level}"] = 1 if row[col] == level else 0

    df_row = pd.DataFrame([encoded])

    # 3) Scale numeric columns with the scaler FIT DURING TRAINING
    df_row[NUMERIC_COLS] = bundle["scaler"].transform(df_row[NUMERIC_COLS])

    # 4) Align to training column order (fills any missing dummy cols with 0)
    df_row = df_row.reindex(columns=bundle["feature_columns"], fill_value=0)
    return df_row


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-top: -0.5rem;">
        <span class="heartbeat">❤️</span>
    </div>
    <div class="hero-title">CardioPredict</div>
    <div class="hero-subtitle">🩺 AI-Powered Heart Disease Risk Assessment Dashboard</div>
    """,
    unsafe_allow_html=True,
)

if bundle is None:
    st.error(
        "⚠️ **model.pkl not found.** Please generate it first by running:\n\n"
        "```bash\npython train_model.py --data heart.csv\n```\n"
        "This trains the model using the exact notebook logic and saves the "
        "bundle this app needs to make predictions."
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar — patient input form
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧾 Patient Information")
    st.caption("Enter the patient's clinical details below.")

    st.markdown("#### 👤 Demographics")
    age = st.slider("Age (years)", 18, 100, 50, help="Patient's age in years")
    sex_label = st.radio("Sex", ["Male", "Female"], horizontal=True)
    sex = "M" if sex_label == "Male" else "F"

    st.markdown("#### 💓 Cardiac Symptoms")
    cp_map = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP",
        "Asymptomatic": "ASY",
    }
    cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()), index=1)
    chest_pain = cp_map[cp_label]

    exercise_angina_label = st.radio(
        "Exercise-Induced Angina", ["No", "Yes"], horizontal=True
    )
    exercise_angina = "Y" if exercise_angina_label == "Yes" else "N"

    st.markdown("#### 🩸 Vitals & Labs")
    resting_bp = st.slider(
        "Resting Blood Pressure (mm Hg)", 80, 220, 120,
        help="Resting systolic blood pressure on admission"
    )
    cholesterol = st.slider(
        "Serum Cholesterol (mg/dl)", 0, 603, 200,
        help="0 will be treated as missing and imputed with the training mean"
    )
    fasting_bs_label = st.radio(
        "Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True
    )
    fasting_bs = 1 if fasting_bs_label == "Yes" else 0

    st.markdown("#### 📈 ECG & Stress Test")
    ecg_map = {
        "Normal": "Normal",
        "ST-T Wave Abnormality": "ST",
        "Left Ventricular Hypertrophy": "LVH",
    }
    ecg_label = st.selectbox("Resting ECG", list(ecg_map.keys()))
    resting_ecg = ecg_map[ecg_label]

    max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider(
        "Oldpeak (ST Depression)", -2.6, 6.2, 0.0, 0.1,
        help="ST depression induced by exercise relative to rest"
    )

    slope_map = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}
    slope_label = st.selectbox("ST Slope", list(slope_map.keys()))
    st_slope = slope_map[slope_label]

    st.markdown("---")
    predict_clicked = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)

raw_input = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "RestingECG": resting_ecg,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak,
    "ST_Slope": st_slope,
}

# --------------------------------------------------------------------------
# Prediction + results
# --------------------------------------------------------------------------
if "has_predicted" not in st.session_state:
    st.session_state.has_predicted = False

if predict_clicked:
    st.session_state.has_predicted = True
    st.session_state.raw_input = raw_input

if st.session_state.has_predicted:
    X_input = preprocess_input(st.session_state.raw_input, bundle)
    model = bundle["model"]

    pred = int(model.predict(X_input)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0]
    else:  # fallback if a model without predict_proba is ever selected
        proba = np.array([1 - pred, pred], dtype=float)

    prob_no_disease, prob_disease = float(proba[0]), float(proba[1])
    confidence = max(prob_no_disease, prob_disease) * 100

    # ---------------- Result banner ----------------
    if pred == 1:
        st.markdown(
            f"""
            <div class="result-banner result-high">
                <div style="font-size:2.2rem;">⚠️</div>
                <h2>Heart Disease Risk Detected</h2>
                <p>The model predicts a <b>high likelihood</b> of heart disease based on the provided data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-banner result-low">
                <div style="font-size:2.2rem;">✅</div>
                <h2>No Heart Disease Detected</h2>
                <p>The model predicts a <b>low likelihood</b> of heart disease based on the provided data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- Metric cards ----------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="metric-card metric-purple">
                    <h3>🎯 Prediction</h3>
                    <p class="value">{"Positive" if pred == 1 else "Negative"}</p>
                </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card metric-teal">
                    <h3>📊 Confidence</h3>
                    <p class="value">{confidence:.1f}%</p>
                </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card metric-red">
                    <h3>❤️ Disease Prob.</h3>
                    <p class="value">{prob_disease*100:.1f}%</p>
                </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-card metric-green">
                    <h3>💚 Healthy Prob.</h3>
                    <p class="value">{prob_no_disease*100:.1f}%</p>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>📈 Risk Visualization</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    # ---------------- Gauge chart ----------------
    with c1:
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prob_disease * 100,
                number={"suffix": "%", "font": {"size": 42, "color": "#f0f0f5"}},
                delta={"reference": 50, "increasing": {"color": "#f87171"}, "decreasing": {"color": "#4ade80"}},
                title={"text": "Heart Disease Risk Score", "font": {"size": 16, "color": "#eaeaf2"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#eaeaf2", "tickfont": {"color": "#eaeaf2"}},
                    "bar": {"color": "#ff4b6e"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 33], "color": "rgba(74,222,128,0.55)"},
                        {"range": [33, 66], "color": "rgba(250,204,21,0.55)"},
                        {"range": [66, 100], "color": "rgba(248,113,113,0.65)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.85,
                        "value": prob_disease * 100,
                    },
                },
            )
        )
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eaeaf2"},
            height=320,
            margin=dict(l=20, r=20, t=60, b=10),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ---------------- Probability bar chart ----------------
    with c2:
        bar_fig = go.Figure(
            go.Bar(
                x=[prob_no_disease * 100, prob_disease * 100],
                y=["No Heart Disease", "Heart Disease"],
                orientation="h",
                marker=dict(
                    color=["#4ade80", "#ff4b6e"],
                    line=dict(color="rgba(255,255,255,0.2)", width=1),
                ),
                text=[f"{prob_no_disease*100:.1f}%", f"{prob_disease*100:.1f}%"],
                textposition="outside",
                textfont=dict(color="#eaeaf2", size=14),
            )
        )
        bar_fig.update_layout(
            title={"text": "Class Probability Breakdown", "font": {"size": 16, "color": "#eaeaf2"}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#eaeaf2"},
            xaxis=dict(range=[0, 115], showgrid=False, title="Probability (%)"),
            yaxis=dict(showgrid=False),
            height=320,
            margin=dict(l=10, r=30, t=60, b=30),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown(
        """
        <div class="glass-card">
        ⚠️ <b>Medical Disclaimer:</b> This tool is for educational and
        demonstrative purposes only. It is <b>not</b> a substitute for
        professional medical advice, diagnosis, or treatment. Always seek
        the guidance of a qualified healthcare provider.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <h3>👋 Welcome!</h3>
            <p>Fill in the patient's clinical details in the sidebar and click
            <b>"Predict Heart Disease Risk"</b> to get an instant AI-powered assessment,
            complete with confidence scores and visual risk breakdown.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Feature descriptions
# --------------------------------------------------------------------------
st.markdown("<div class='section-title'>📖 Feature Descriptions</div>", unsafe_allow_html=True)
with st.expander("Click to view what each clinical feature means"):
    feature_desc = pd.DataFrame(
        [
            ("Age", "Age of the patient in years."),
            ("Sex", "Biological sex of the patient (Male / Female)."),
            ("ChestPainType", "TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic."),
            ("RestingBP", "Resting blood pressure in mm Hg on admission to the hospital."),
            ("Cholesterol", "Serum cholesterol level in mg/dl."),
            ("FastingBS", "1 if fasting blood sugar > 120 mg/dl, otherwise 0."),
            ("RestingECG", "Normal / ST: ST-T wave abnormality / LVH: probable left ventricular hypertrophy."),
            ("MaxHR", "Maximum heart rate achieved during the stress test (numeric value between 60–220)."),
            ("ExerciseAngina", "Exercise-induced angina (Yes / No)."),
            ("Oldpeak", "ST depression induced by exercise relative to rest (numeric value)."),
            ("ST_Slope", "The slope of the peak exercise ST segment (Up / Flat / Down)."),
        ],
        columns=["Feature", "Description"],
    )
    st.table(feature_desc)

# --------------------------------------------------------------------------
# Model information
# --------------------------------------------------------------------------
st.markdown("<div class='section-title'>🤖 Model Information</div>", unsafe_allow_html=True)
with st.expander("Click to view model details & performance", expanded=False):
    info_col1, info_col2 = st.columns([1, 1])

    with info_col1:
        st.markdown(
            f"""
            <div class="glass-card">
            <b>🏆 Selected Model:</b> {bundle.get('model_name', 'N/A')}<br>
            <b>🎯 Test Accuracy:</b> {bundle.get('test_accuracy', 0)*100:.2f}%<br>
            <b>🧪 Test Samples:</b> {bundle.get('n_test_samples', 'N/A')}<br>
            <b>📚 Training Samples:</b> {bundle.get('n_train_samples', 'N/A')}<br>
            <b>🧮 Total Features:</b> {len(bundle.get('feature_columns', []))}<br>
            <b>✂️ Split:</b> 80% train / 20% test (random_state=42)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        results = bundle.get("all_model_results", [])
        if results:
            comp_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
            comp_df["Accuracy"] = (comp_df["Accuracy"] * 100).round(2)
            comp_fig = go.Figure(
                go.Bar(
                    x=comp_df["Accuracy"],
                    y=comp_df["Model"],
                    orientation="h",
                    marker=dict(color="#8b5cf6"),
                    text=comp_df["Accuracy"].astype(str) + "%",
                    textposition="outside",
                    textfont=dict(color="#eaeaf2"),
                )
            )
            comp_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#eaeaf2"},
                xaxis=dict(range=[0, 105], title="Accuracy (%)", showgrid=False),
                yaxis=dict(showgrid=False, autorange="reversed"),
                height=300,
                margin=dict(l=10, r=30, t=20, b=30),
            )
            st.plotly_chart(comp_fig, use_container_width=True)

    st.markdown(
        """
        <div class="glass-card">
        <b>🔬 Preprocessing pipeline:</b>
        <ol>
        <li>Zero-value imputation for <code>Cholesterol</code> and <code>RestingBP</code>
        (replaced with the mean of non-zero training values).</li>
        <li>One-hot encoding of categorical clinical features.</li>
        <li>Standardization (z-score) of numeric features:
        Age, RestingBP, Cholesterol, MaxHR, Oldpeak.</li>
        <li>80/20 stratified train-test split (random_state = 42).</li>
        <li>5 candidate classifiers benchmarked — best model selected by
        held-out accuracy.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built with ❤️ using <b>Streamlit</b> &amp; <b>scikit-learn</b><br>
        Developed by <a href="#">Your Name</a> · 🩺 CardioPredict v1.0 ·
        For educational purposes only, not a medical device.
    </div>
    """,
    unsafe_allow_html=True,
)
