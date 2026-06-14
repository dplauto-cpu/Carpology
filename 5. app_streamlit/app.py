# ============================================
# BLOCK 1: IMPORT LIBRARIES
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import folium
from streamlit_folium import folium_static

# ============================================
# BLOCK 2: PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Archaeological Cereal Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# BLOCK 2B: CUSTOM STYLING
# ============================================

st.markdown("""
<style>
    /* Plant-inspired typography */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Lora:ital,wght@0,400;0,500;1,400&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Lora', Georgia, serif;
    }

    h1, h2, h3,
    .main-header h1,
    .section-title,
    .subsection-title,
    .result-card h2 {
        font-family: 'Cormorant Garamond', Georgia, serif;
    }

    /* General app background */
    .stApp {
        background-color: #F7ECD8;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #6F4E37 0%, #C19A6B 100%);
        padding: 2.2rem 2rem;
        border-radius: 14px;
        color: #FFFDF8;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 14px rgba(111, 78, 55, 0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .main-header p {
        margin: 0.7rem auto 0;
        max-width: 720px;
        font-size: 1.05rem;
        line-height: 1.55;
        opacity: 0.95;
    }
    .main-header .accuracy-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 1rem;
        background: rgba(255, 255, 255, 0.18);
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }

    /* Section titles with icon */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: #5C4033;
        margin: 0.6rem 0 1.1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E8DCC8;
    }
    .section-title .icon-circle {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.1rem;
        height: 2.1rem;
        min-width: 2.1rem;
        border-radius: 50%;
        background: #F3E5D8;
        font-size: 1.1rem;
    }

    /* Column subtitles */
    .subsection-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #8A6D4E;
        letter-spacing: 0.4px;
        margin-bottom: 0.6rem;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #FFFDF8;
        border: 1px solid #E8DCC8;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #6F4E37;
    }

    /* Classification result card */
    .result-card {
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        color: #FFFDF8;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .result-card .result-icon {
        font-size: 1.8rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .result-card p {
        margin: 0.35rem 0 0;
        font-size: 0.92rem;
        opacity: 0.92;
    }
    .result-wheat  { background: linear-gradient(135deg, #CD853F 0%, #E0B687 100%); }
    .result-barley { background: linear-gradient(135deg, #8B4513 0%, #B5764A 100%); }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E8DCC8;
        border-radius: 10px;
        background-color: #FFFDF8;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #C19A6B;
        color: #5C4033;
    }
    .stButton > button:hover {
        border-color: #8B4513;
        color: #8B4513;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #9C9088;
        font-size: 0.85rem;
        padding: 1.2rem 0 0.4rem;
        line-height: 1.7;
    }
    .app-footer .footer-icon {
        opacity: 0.7;
    }
    .app-footer a {
        color: #6F4E37;
        text-decoration: none;
        font-weight: 600;
    }
    .app-footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# BLOCK 3: TITLE AND DESCRIPTION
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🌾 Archaeological Cereal Classifier</h1>
    <p>
        Identify whether a charred seed is <strong>Barley</strong> or <strong>Wheat</strong>
        from stable isotopes, its geographic location and its chronological context.
    </p>
    <span class="accuracy-badge">🌿 Balanced Random Forest &nbsp;·&nbsp; Accuracy 74.78% &nbsp;·&nbsp; AUC 0.824</span>
</div>
""", unsafe_allow_html=True)

# ============================================
# BLOCK 4: LOAD THE MODEL (FUNCTION)
# ============================================

@st.cache_resource
def cargar_modelo():
    """Loads the model and preprocessors from the models/ folder"""
    modelo = joblib.load('../models/random_forest_balanceado.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    le_periodo = joblib.load('../models/le_periodo.pkl')
    le_cuenca = joblib.load('../models/le_cuenca.pkl')
    return modelo, scaler, le_periodo, le_cuenca

# ============================================
# BLOCK 5: RUN MODEL LOADING
# ============================================

try:
    modelo, scaler, le_periodo, le_cuenca = cargar_modelo()
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# ============================================
# BLOCK 6: USER INTERFACE (CONTROLS)
# ============================================

st.markdown("""
<div class="section-title"><span class="icon-circle">📊</span> Classify your own sample</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# COLUMN 1: Isotopic data
with col1:
    st.markdown('<div class="subsection-title">🔬 Isotopic data</div>', unsafe_allow_html=True)

    icon_col1, slider_col1 = st.columns([1, 6])
    with icon_col1:
        st.image("icon_d13.png", width=55)
    with slider_col1:
        d13C = st.slider(
            "δ13C (‰)",
            min_value=-28.0,
            max_value=-18.0,
            value=-23.0,
            step=0.1,
            help="More negative values = better water access"
        )

    icon_col2, slider_col2 = st.columns([1, 6])
    with icon_col2:
        st.image("icon_d15.png", width=55)
    with slider_col2:
        d15N = st.slider(
            "δ15N (‰)",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1,
            help="Higher values = greater aridity or manuring"
        )

    # Position of the sample on the δ13C / δ15N isotopic plane
    st.markdown('<div class="subsection-title" style="margin-top:0.8rem;">📐 Isotopic position of the sample</div>', unsafe_allow_html=True)

    fig_iso = px.scatter(
        x=[d13C], y=[d15N]
    )
    fig_iso.update_traces(marker=dict(size=14, color='#8B4513', line=dict(width=1, color='#FFFDF8')))
    fig_iso.update_layout(
        xaxis=dict(range=[-28, -18], title='δ13C (‰)', gridcolor='#E8DCC8', zeroline=False),
        yaxis=dict(range=[0, 14], title='δ15N (‰)', gridcolor='#E8DCC8', zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
    )
    st.plotly_chart(fig_iso, use_container_width=True)

# COLUMN 2: Geographic and chronological data
with col2:
    st.markdown('<div class="subsection-title">📍 Geographic and chronological data</div>', unsafe_allow_html=True)

    latitud = st.slider(
        "Latitude (°N)",
        min_value=25.0,
        max_value=50.0,
        value=40.0,
        step=0.5
    )

    longitud = st.slider(
        "Longitude (°E)",
        min_value=-10.0,
        max_value=40.0,
        value=15.0,
        step=0.5
    )

    periodo_opciones = [
        'Neolithic', 'Late_Neolithic', 'Chalcolithic',
        'Early_Bronze_Age', 'Middle_Late_Bronze_Age',
        'Early_Iron_Age', 'Late_Iron_Age', 'Other'
    ]
    periodo = st.selectbox("Archaeological period", periodo_opciones)

    cuenca_opciones = ['Western_Med', 'Central_Med', 'Eastern_Med', 'other']
    cuenca = st.selectbox("Mediterranean basin", cuenca_opciones)

# ============================================
# BLOCK 7: ENCODE CATEGORICAL VARIABLES
# ============================================

# Period
try:
    periodo_encoded = le_periodo.transform([periodo])[0]
except ValueError:
    st.warning(f"⚠️ Period '{periodo}' not recognised. Using 'Other'.")
    periodo_encoded = le_periodo.transform(['Other'])[0]

# Basin
try:
    cuenca_encoded = le_cuenca.transform([cuenca])[0]
except ValueError:
    st.warning(f"⚠️ Basin '{cuenca}' not recognised. Using 'other'.")
    cuenca_encoded = le_cuenca.transform(['other'])[0]

# ============================================
# BLOCK 8: PREDICTION
# ============================================

# Build array with the 6 features
input_features = np.array([[d13C, d15N, latitud, longitud, periodo_encoded, cuenca_encoded]])

# Scale
input_scaled = scaler.transform(input_features)

# Predict
prediccion = modelo.predict(input_scaled)[0]
probabilidades = modelo.predict_proba(input_scaled)[0]

# ============================================
# BLOCK 9: SHOW RESULTS
# ============================================

st.markdown("---")
st.markdown("""
<div class="section-title"><span class="icon-circle">🎯</span> Classification result</div>
""", unsafe_allow_html=True)

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    if prediccion == 1:
        st.markdown(f"""
        <div class="result-card result-barley">
            <span class="result-icon">🌿</span>
            <h2>Barley</h2>
            <p>Probability: {probabilidades[1]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-wheat">
            <span class="result-icon">🌾</span>
            <h2>Wheat</h2>
            <p>Probability: {probabilidades[0]:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

with col_res2:
    st.metric("📊 Model confidence", f"{max(probabilidades):.1%}")

with col_res3:
    st.metric("🎯 Overall accuracy", "74.78%")

# Bar chart
st.markdown('<div class="subsection-title">📈 Classification probability</div>', unsafe_allow_html=True)

prob_df = pd.DataFrame({
    'Cereal': ['Wheat', 'Barley'],
    'Probability': [probabilidades[0], probabilidades[1]]
})

fig = px.bar(
    prob_df,
    x='Cereal',
    y='Probability',
    color='Cereal',
    range_y=[0, 1],
    color_discrete_map={'Wheat': '#CD853F', 'Barley': '#8B4513'}
)
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    margin=dict(t=10, b=10, l=10, r=10)
)
st.plotly_chart(fig, use_container_width=True)

# ============================================
# BLOCK 10: INTERPRETING THE ISOTOPES
# ============================================

with st.expander("📖  Interpreting the isotopic values"):
    st.markdown("""
    **δ13C (Carbon 13) - Water stress:**
    - **More negative (< -24‰):** Good water access
    - **Intermediate (-24‰ to -22‰):** Normal rainfed conditions
    - **More positive (> -22‰):** Significant water stress

    **δ15N (Nitrogen 15) - Aridity / Manuring:**
    - **Low (< 4‰):** Wet soils, low aridity
    - **Medium (4‰ - 8‰):** Normal conditions
    - **High (> 8‰):** High aridity or manure fertilisation
    """)

# ============================================
# BLOCK 11: MODEL INFORMATION
# ============================================

with st.expander("ℹ️  Technical model information"):
    st.markdown("""
    **Model:** Random Forest with class balancing (`class_weight='balanced'`)

    **Feature importance:**
    | Feature | Importance |
    |:---|:---|
    | δ13C | 41% |
    | δ15N | 28% |
    | Latitude | 13% |
    | Longitude | 10% |
    | Period | 5% |
    | Basin | 3% |

    **Metrics:**
    - Overall accuracy: 74.78%
    - AUC-ROC: 0.824
    """)

# ============================================
# BLOCK 12: SAMPLE LOCATION MAP
# ============================================

st.markdown("---")
st.markdown("""
<div class="section-title"><span class="icon-circle">🗺️</span> Sample location</div>
""", unsafe_allow_html=True)

if st.button("📍 Show on map", type="primary"):
    mapa_muestra = folium.Map(
        location=[latitud, longitud],
        zoom_start=6,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; National Geographic'
    )

    texto_cereal = "Barley" if prediccion == 1 else "Wheat"
    color_marcador = '#8B4513' if prediccion == 1 else '#CD853F'

    folium.Marker(
        location=[latitud, longitud],
        popup=f"<b>{texto_cereal}</b><br>δ13C: {d13C}‰<br>δ15N: {d15N}‰<br>{periodo}<br>{cuenca}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(mapa_muestra)

    folium_static(mapa_muestra, width=700, height=400)
    st.caption(f"📍 Red marker: sample classified as **{texto_cereal}** at {latitud}°N, {longitud}°E")

# ============================================
# BLOCK 13: FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div class="app-footer">
    <span class="footer-icon">🌾</span> <strong>Project:</strong> Carpology and Ancient Crops ML
    &nbsp;·&nbsp; <strong>Author:</strong> David Larreina-García<br>
    <strong>Model:</strong> Balanced Random Forest (74.78% accuracy)<br>
    <strong>Repository:</strong>
    <a href="https://github.com/dplauto-cpu/Carpology" target="_blank">github.com/dplauto-cpu/Carpology</a>
</div>
""", unsafe_allow_html=True)
