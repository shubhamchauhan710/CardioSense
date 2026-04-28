import numpy as np
import pickle
import streamlit as st
import time
import pandas as pd
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense — Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Root Variables ── */
:root {
    --red:       #E5343A;
    --red-light: #FF6B6B;
    --red-dark:  #8B1A1E;
    --green:     #1DB954;
    --bg:        #0D0F14;
    --surface:   #161A22;
    --surface2:  #1E2330;
    --border:    rgba(255,255,255,0.07);
    --text:      #E8EAF0;
    --muted:     #8892A4;
    --font:      'DM Sans', sans-serif;
    --serif:     'DM Serif Display', serif;
}

/* ── Global Reset ── */
html, body, [class*="css"]  {
    font-family: var(--font) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1100px; }

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1a0a0b 0%, #0D0F14 60%, #0a1020 100%);
    border: 1px solid rgba(229,52,58,0.25);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🫀';
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 7rem;
    opacity: 0.08;
    filter: blur(2px);
}
.hero h1 {
    font-family: var(--serif) !important;
    font-size: 2.8rem !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    margin: 0 0 0.5rem !important;
    line-height: 1.1 !important;
}
.hero h1 span { color: var(--red-light); }
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    max-width: 550px;
}
.hero-tag {
    display: inline-block;
    background: rgba(229,52,58,0.15);
    border: 1px solid rgba(229,52,58,0.3);
    color: var(--red-light);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Section Cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 1.2rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Input Labels ── */
label, .stSelectbox label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Inputs ── */
input[type="number"], .stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}
input[type="number"]:focus, .stSelectbox > div > div:focus-within {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 3px rgba(229,52,58,0.15) !important;
}

/* ── Primary Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--red), var(--red-dark)) !important;
    color: white !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(229,52,58,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(229,52,58,0.45) !important;
}

/* ── Result Boxes ── */
.result-positive {
    background: linear-gradient(135deg, rgba(229,52,58,0.18), rgba(139,26,30,0.12));
    border: 1.5px solid var(--red);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: slideUp 0.4s ease;
}
.result-negative {
    background: linear-gradient(135deg, rgba(29,185,84,0.15), rgba(29,185,84,0.05));
    border: 1.5px solid var(--green);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: slideUp 0.4s ease;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-title {
    font-family: var(--serif) !important;
    font-size: 1.6rem !important;
    font-weight: 400 !important;
    margin: 0.3rem 0 !important;
}
.result-title.danger  { color: var(--red-light) !important; }
.result-title.safe    { color: var(--green) !important; }
.result-sub { color: var(--muted); font-size: 0.88rem; margin: 0; }

@keyframes slideUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0);    }
}

/* ── Risk Pill Badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-high   { background: rgba(229,52,58,0.2); color: var(--red-light); }
.badge-medium { background: rgba(255,165,0,0.2); color: #FFA500; }
.badge-low    { background: rgba(29,185,84,0.2);  color: var(--green); }

/* ── Metric tiles ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-tile {
    flex: 1 1 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-tile .label { color: var(--muted); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-tile .value { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: var(--red) !important;
    color: white !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info box ── */
.info-box {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid var(--red);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    color: var(--muted);
}

/* ── About cards ── */
.about-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.about-card h4 { margin: 0 0 0.4rem; font-size: 0.95rem; color: var(--text); }
.about-card p  { margin: 0; font-size: 0.84rem; color: var(--muted); line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try relative path first (for deployment), then absolute fallback
    paths = [
        'trained_model.sav',
        os.path.join(os.path.dirname(__file__), 'trained_model.sav'),
    ]
    for p in paths:
        if os.path.exists(p):
            return pickle.load(open(p, 'rb'))
    st.error("⚠️ Model file not found. Place `trained_model.sav` in the same folder as this script.")
    return None

loaded_model = load_model()

# ── Prediction Logic ──────────────────────────────────────────────────────────
def predict(inputs: list):
    arr = np.asarray(inputs).reshape(1, -1)
    pred = loaded_model.predict(arr)[0]
    proba = loaded_model.predict_proba(arr)[0]
    return pred, proba

# ── Risk Level Helper ─────────────────────────────────────────────────────────
def risk_badge(proba_positive: float) -> str:
    p = proba_positive * 100
    if p >= 70:
        return '<span class="badge badge-high">High Risk</span>'
    elif p >= 40:
        return '<span class="badge badge-medium">Moderate Risk</span>'
    else:
        return '<span class="badge badge-low">Low Risk</span>'

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI-Powered Diagnostic Tool</div>
    <h1>Cardio<span>Sense</span></h1>
    <p>Enter patient clinical parameters to assess cardiovascular risk using a trained Logistic Regression model on the Cleveland Heart Disease dataset.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_about, tab_data = st.tabs(["🩺 Prediction", "📖 About", "📊 Dataset Info"])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab_pred:

    with st.form("patient_form"):

        # ── Section 1: Demographics ──
        st.markdown('<p class="section-title">👤 Patient Demographics</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        with c2:
            sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        with c3:
            cp = st.selectbox(
                "Chest Pain Type",
                options=[(0,"Typical Angina"), (1,"Atypical Angina"), (2,"Non-anginal Pain"), (3,"Asymptomatic")],
                format_func=lambda x: x[1]
            )[0]

        st.markdown("---")

        # ── Section 2: Vitals ──
        st.markdown('<p class="section-title">💓 Cardiovascular Vitals</p>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            trestbps = st.number_input("Resting BP (mm Hg)", min_value=60, max_value=250, value=120, step=1)
        with c5:
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
        with c6:
            thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150, step=1)

        c7, c8 = st.columns(2)
        with c7:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [(0,"No"), (1,"Yes")], format_func=lambda x: x[1])[0]
        with c8:
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        st.markdown("---")

        # ── Section 3: Diagnostics ──
        st.markdown('<p class="section-title">🔬 Diagnostic Results</p>', unsafe_allow_html=True)
        c9, c10 = st.columns(2)
        with c9:
            restecg = st.selectbox(
                "Resting ECG",
                [(0,"Normal"), (1,"ST-T Abnormality"), (2,"Left Ventricular Hypertrophy")],
                format_func=lambda x: x[1]
            )[0]
        with c10:
            exang = st.selectbox("Exercise-Induced Angina", [(0,"No"), (1,"Yes")], format_func=lambda x: x[1])[0]

        c11, c12, c13 = st.columns(3)
        with c11:
            slope = st.selectbox(
                "ST Slope",
                [(0,"Upsloping"), (1,"Flat"), (2,"Downsloping")],
                format_func=lambda x: x[1]
            )[0]
        with c12:
            ca = st.number_input("Major Vessels (fluoroscopy)", min_value=0, max_value=4, value=0, step=1)
        with c13:
            thal = st.selectbox(
                "Thalassemia",
                [(1,"Normal"), (2,"Fixed Defect"), (3,"Reversible Defect")],
                format_func=lambda x: x[1]
            )[0]

        submitted = st.form_submit_button("🔍 Analyse Risk")

    # ── Result ──────────────────────────────────────────────
    if submitted and loaded_model:
        inputs = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        with st.spinner("Analysing cardiac parameters…"):
            time.sleep(1.2)
            pred, proba = predict(inputs)

        prob_positive = proba[1]
        prob_pct      = round(prob_positive * 100, 1)

        if pred == 1:
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-emoji">🚨</div>
                <p class="result-title danger">Heart Disease Detected</p>
                <p class="result-sub">Model confidence: <strong>{prob_pct}%</strong> &nbsp;|&nbsp; {risk_badge(prob_positive)}</p>
                <p class="result-sub" style="margin-top:0.8rem;">Please consult a cardiologist for a comprehensive evaluation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-emoji">✅</div>
                <p class="result-title safe">No Heart Disease Detected</p>
                <p class="result-sub">Model confidence: <strong>{round(proba[0]*100,1)}%</strong> &nbsp;|&nbsp; {risk_badge(prob_positive)}</p>
                <p class="result-sub" style="margin-top:0.8rem;">Maintain a healthy lifestyle and schedule regular check-ups.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="info-box">⚠️ This tool is for educational purposes only. It is not a substitute for professional medical advice.</div>', unsafe_allow_html=True)

        # ── Sidebar: summary + download ──────────────────────
        input_dict = {
            "Age": age, "Sex": "Male" if sex else "Female",
            "Chest Pain Type": cp, "Resting BP": trestbps,
            "Cholesterol": chol, "Fasting BS > 120": "Yes" if fbs else "No",
            "Resting ECG": restecg, "Max HR": thalach,
            "Exercise Angina": "Yes" if exang else "No",
            "ST Depression": oldpeak, "ST Slope": slope,
            "Major Vessels": ca, "Thalassemia": thal,
            "Prediction": "Heart Disease" if pred == 1 else "No Heart Disease",
            "Risk Probability (%)": prob_pct,
        }
        df_out = pd.DataFrame([input_dict])

        with st.sidebar:
            st.markdown("### 📋 Patient Summary")
            st.markdown("---")
            st.metric("Age", f"{age} yrs")
            st.metric("Sex", "Male" if sex else "Female")
            st.metric("Risk Score", f"{prob_pct}%")
            st.metric("Result", "⚠️ Positive" if pred == 1 else "✅ Negative")
            st.markdown("---")
            st.download_button(
                label="⬇️ Download Report (CSV)",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="CardioSense_Report.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### About Heart Disease")
    st.markdown("""
    Heart disease is a broad term covering conditions that affect the heart's structure and function.
    It remains the **leading cause of death worldwide**, claiming over 17 million lives annually.
    """)

    conditions = [
        ("🩸 Coronary Artery Disease (CAD)", "Narrowed or blocked coronary arteries reduce blood flow to the heart, often causing chest pain (angina) or heart attacks."),
        ("⚡ Arrhythmias", "Irregular heartbeats — too fast (tachycardia), too slow (bradycardia), or erratic — that affect pumping efficiency."),
        ("🔧 Heart Valve Disease", "Dysfunction of one or more heart valves causing improper blood flow between chambers."),
        ("🏗️ Congenital Defects", "Structural heart issues present from birth, ranging from minor to life-threatening."),
        ("💧 Heart Failure", "The heart fails to pump blood effectively, causing fluid buildup and fatigue."),
        ("🧬 Cardiomyopathy", "Disease of the heart muscle — thickened, enlarged, or rigid walls — impairing function."),
    ]
    for title, desc in conditions:
        st.markdown(f"""
        <div class="about-card">
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🛡️ Prevention")
    col1, col2 = st.columns(2)
    tips = [
        ("🥗 Healthy Diet", "Fruits, vegetables, whole grains, lean proteins. Limit salt and processed foods."),
        ("🏃 Regular Exercise", "At least 30 minutes of moderate activity most days."),
        ("🚭 No Smoking", "Quit smoking and avoid second-hand smoke exposure."),
        ("🧘 Manage Stress", "Meditation, yoga, and adequate sleep reduce cardiac risk."),
        ("📊 Monitor Health", "Check BP, cholesterol, and glucose regularly."),
        ("💊 Medication", "Adhere to prescribed medications for hypertension and diabetes."),
    ]
    for i, (t, d) in enumerate(tips):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""<div class="about-card"><h4>{t}</h4><p>{d}</p></div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — DATASET INFO
# ════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### 📊 Cleveland Heart Disease Dataset")
    st.markdown("""
    This model was trained on the **Cleveland Heart Disease dataset** from the UCI Machine Learning Repository.
    It contains **303 patient records** with **13 clinical features** and a binary target variable.
    """)

    features = {
        "Feature": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"],
        "Description": [
            "Age in years",
            "1 = Male, 0 = Female",
            "Chest pain type (0–3)",
            "Resting blood pressure (mm Hg)",
            "Serum cholesterol (mg/dL)",
            "Fasting blood sugar > 120 mg/dL",
            "Resting ECG results (0–2)",
            "Maximum heart rate achieved",
            "Exercise-induced angina",
            "ST depression induced by exercise",
            "Slope of peak exercise ST segment",
            "Number of major vessels (0–4, fluoroscopy)",
            "Thalassemia type (1–3)",
        ],
        "Type": ["Numeric","Binary","Categorical","Numeric","Numeric","Binary","Categorical","Numeric","Binary","Numeric","Categorical","Numeric","Categorical"],
    }
    st.dataframe(pd.DataFrame(features), use_container_width=True, hide_index=True)

    st.markdown("### 🤖 Model Details")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="about-card">
            <h4>Algorithm</h4>
            <p>Logistic Regression (sklearn) with L2 regularisation, trained with stratified 80/20 split.</p>
        </div>
        <div class="about-card">
            <h4>Reported Accuracy</h4>
            <p>~82% on test set (Cleveland dataset). Performance may vary on unseen populations.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="about-card">
            <h4>Output</h4>
            <p>Binary classification: <strong>0</strong> = No Disease, <strong>1</strong> = Disease present, with class probabilities.</p>
        </div>
        <div class="about-card">
            <h4>Limitations</h4>
            <p>Small dataset size, single algorithm, no cross-validation ensemble — ideal for learning, not clinical use.</p>
        </div>
        """, unsafe_allow_html=True)
