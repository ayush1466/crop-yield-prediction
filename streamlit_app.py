"""
Crop Yield Prediction — Streamlit Web App
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ── Styles ───────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        color: #1B5E20; text-align: center; margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border-left: 5px solid #2E7D32;
        padding: 1.5rem; border-radius: 10px; margin-top: 1rem;
    }
    .metric-label { font-size: 0.85rem; color: #555; font-weight: 600; }
    .metric-value { font-size: 2rem; color: #1B5E20; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Synthetic Training Data (replace with real CSV) ──
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 3000
    CROPS   = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Soybean', 'Groundnut']
    STATES  = ['Punjab', 'Haryana', 'UP', 'Bihar', 'Maharashtra', 'Karnataka', 'AP', 'MP']
    SEASONS = ['Kharif', 'Rabi', 'Zaid']
    SOIL    = ['Alluvial', 'Black', 'Red', 'Laterite', 'Sandy']

    data = {
        'crop':        np.random.choice(CROPS, n),
        'state':       np.random.choice(STATES, n),
        'season':      np.random.choice(SEASONS, n),
        'soil_type':   np.random.choice(SOIL, n),
        'rainfall_mm': np.random.uniform(300, 2500, n),
        'temperature': np.random.uniform(15, 42, n),
        'humidity':    np.random.uniform(30, 95, n),
        'fertilizer':  np.random.uniform(50, 400, n),
        'area_ha':     np.random.uniform(0.5, 100, n),
    }
    df = pd.DataFrame(data)
    df['yield'] = (
        2000 + df['rainfall_mm'] * 0.4
        - (df['temperature'] - 28) ** 2 * 5
        + df['humidity'] * 3
        + df['fertilizer'] * 2.5
        + np.random.normal(0, 200, n)
    ).clip(500, 8000)

    le_dict = {}
    for col in ['crop', 'state', 'season', 'soil_type']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        le_dict[col] = {label: idx for idx, label in enumerate(le.classes_)}

    feats = ['crop_enc','state_enc','season_enc','soil_type_enc',
             'rainfall_mm','temperature','humidity','fertilizer','area_ha']
    X = df[feats]
    y = df['yield']
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    fi = pd.Series(model.feature_importances_,
                   index=['Crop','State','Season','Soil','Rainfall','Temp','Humidity','Fertilizer','Area'])
    return model, le_dict, fi

model, le_dict, feat_imp = train_model()

CROPS   = sorted(le_dict['crop'].keys())
STATES  = sorted(le_dict['state'].keys())
SEASONS = sorted(le_dict['season'].keys())
SOILS   = sorted(le_dict['soil_type'].keys())

# ── Header ───────────────────────────────────
st.markdown('<div class="main-header">🌾 Crop Yield Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered prediction for Indian agriculture</div>', unsafe_allow_html=True)

# ── Layout ───────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📋 Enter Field Details")

    crop    = st.selectbox("Crop Type", CROPS)
    state   = st.selectbox("State", STATES)
    season  = st.selectbox("Season", SEASONS)
    soil    = st.selectbox("Soil Type", SOILS)

    st.markdown("---")
    rainfall    = st.slider("🌧 Rainfall (mm)", 200, 3000, 900, step=50)
    temperature = st.slider("🌡 Temperature (°C)", 10, 45, 28)
    humidity    = st.slider("💧 Humidity (%)", 20, 100, 65)
    fertilizer  = st.slider("🧪 Fertilizer (kg/ha)", 0, 500, 150, step=10)
    area        = st.number_input("🌱 Field Area (hectares)", 0.1, 500.0, 5.0, step=0.5)

    predict_btn = st.button("🔍 Predict Yield", use_container_width=True)

with col_result:
    st.subheader("📊 Prediction & Insights")

    if predict_btn:
        features = np.array([[
            le_dict['crop'].get(crop, 0),
            le_dict['state'].get(state, 0),
            le_dict['season'].get(season, 0),
            le_dict['soil_type'].get(soil, 0),
            rainfall, temperature, humidity, fertilizer, area
        ]])
        pred_yield = model.predict(features)[0]
        total_prod  = pred_yield * area

        st.markdown(f"""
        <div class="result-box">
            <div class="metric-label">PREDICTED YIELD</div>
            <div class="metric-value">{pred_yield:,.0f} kg/ha</div>
            <hr style="border:none; border-top:1px solid #A5D6A7; margin:0.8rem 0;">
            <div class="metric-label">TOTAL PRODUCTION (for {area:.1f} ha)</div>
            <div class="metric-value">{total_prod:,.0f} kg</div>
        </div>
        """, unsafe_allow_html=True)

        rating = "Excellent 🌟" if pred_yield > 4000 else "Good ✅" if pred_yield > 2500 else "Average ⚠️"
        st.info(f"**Yield Rating**: {rating}")

    # Feature Importance Chart
    st.markdown("#### 🔑 What Drives Yield?")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ['#1B5E20' if v == feat_imp.max() else '#81C784' for v in feat_imp.values]
    feat_imp.sort_values().plot(kind='barh', ax=ax, color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

# ── Footer ───────────────────────────────────
st.markdown("---")
st.caption("Model: Random Forest Regressor | Training data: synthetic (replace with Kaggle crop production dataset)")