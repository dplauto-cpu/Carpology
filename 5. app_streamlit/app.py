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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# BLOCK 2B: CUSTOM STYLE (FONTS, COLOURS, TEXTURE)
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');

.stApp {
    background-color: #fdf6e3;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.15'/%3E%3C/svg%3E");
    background-blend-mode: multiply;
    font-family: 'Cormorant Garamond', serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Cinzel', serif, "Segoe UI Emoji", "Noto Color Emoji" !important;
    color: #5a3e1b !important;
    letter-spacing: 0.5px;
}

p, label, li, td, th, caption {
    font-family: 'Cormorant Garamond', serif, "Segoe UI Emoji", "Noto Color Emoji" !important;
}

[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
}

[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'Cinzel', serif, "Segoe UI Emoji", "Noto Color Emoji" !important;
    color: #5a3e1b !important;
}

.stButton button {
    background-color: #c8a96e !important;
    color: #5a3e1b !important;
    border: 1px solid #8B4513 !important;
    font-family: 'Cinzel', serif, "Segoe UI Emoji", "Noto Color Emoji" !important;
}

.stButton button:hover {
    background-color: #b8965a !important;
    color: #fdf6e3 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# BLOCK 3: TITLE AND DESCRIPTION
# ============================================

st.title("🌿 Archaeological Cereal Classifier")

st.markdown("""
**Identify whether a charred seed is Wheat or Barley using stable isotopes**

This application uses a **balanced Random Forest model** trained on 1,147 archaeological samples from the Mediterranean.
*Model accuracy: 74.78% | AUC: 0.824*
""")

st.markdown("---")

# ============================================
# BLOCK 4: LOAD THE MODEL (FUNCTION)
# ============================================

@st.cache_resource
def load_model():
    """Loads the model and preprocessors from the models/ folder"""
    model = joblib.load('../models/random_forest_balanceado.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    le_period = joblib.load('../models/le_periodo.pkl')
    le_basin = joblib.load('../models/le_cuenca.pkl')
    return model, scaler, le_period, le_basin

# ============================================
# BLOCK 5: RUN MODEL LOADING
# ============================================

try:
    modelo, scaler, le_periodo, le_cuenca = load_model()
    st.success("🍃 Model loaded successfully")
except Exception as e:
    st.error(f"🥀 Error loading the model: {e}")
    st.stop()

# ============================================
# BLOCK 6: USER INTERFACE (CONTROLS)
# ============================================

st.subheader("🏺 Classify your own sample")

col1, col2 = st.columns(2)

# COLUMN 1: Isotopic data
with col1:
    st.markdown("### 🧪 Isotopic data")

    d13C = st.slider(
        "δ13C (‰)",
        min_value=-28.0,
        max_value=-18.0,
        value=-23.0,
        step=0.1,
        help="More negative values indicate better access to water"
    )

    d15N = st.slider(
        "δ15N (‰)",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.1,
        help="More positive values indicate greater aridity or manuring"
    )

    # Small chart locating the sample within the isotopic range
    fig_locator = px.scatter(
        x=[d13C],
        y=[d15N],
        labels={'x': 'δ13C (‰)', 'y': 'δ15N (‰)'}
    )
    fig_locator.update_traces(marker=dict(size=14, color='#5a3e1b', symbol='circle'))
    fig_locator.update_xaxes(range=[-28.0, -18.0])
    fig_locator.update_yaxes(range=[0.0, 14.0])
    fig_locator.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_locator, use_container_width=True)

# COLUMN 2: Geographical and chronological data
with col2:
    st.markdown("### 🧭 Geographical and chronological data")

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
st.subheader("🏺 Classification result")

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    if prediccion == 1:
        st.metric("🌿 **Barley**", f"{probabilidades[1]:.1%}")
        st.success("🍃 Classified as **Barley**")
    else:
        st.metric("🌿 **Wheat**", f"{probabilidades[0]:.1%}")
        st.success("🍃 Classified as **Wheat**")

with col_res2:
    st.metric("🏺 Model confidence", f"{max(probabilidades):.1%}")

with col_res3:
    st.caption("🔑 **Overall accuracy:** 74.78%")

# Bar chart
st.markdown("### 📜 Classification probability")

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
st.plotly_chart(fig, use_container_width=True)

# ============================================
# BLOCK 10: INTERPRETING THE ISOTOPIC VALUES
# ============================================

with st.expander("📖 Interpreting the isotopic values"):
    st.markdown("""
    **δ13C (Carbon 13) – Water stress:**
    - **More negative (< -24‰):** Good access to water
    - **Intermediate (-24‰ to -22‰):** Normal rainfed conditions
    - **More positive (> -22‰):** Significant water stress

    **δ15N (Nitrogen 15) – Aridity / Manuring:**
    - **Low (< 4‰):** Wet soils, low aridity
    - **Medium (4‰ – 8‰):** Normal conditions
    - **High (> 8‰):** High aridity or manure fertilisation
    """)

# ============================================
# BLOCK 11: MODEL INFORMATION
# ============================================

with st.expander("🔑 Technical model information"):
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
st.subheader("🧭 Sample location")

if st.button("🧭 Show on map", type="primary"):
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
        icon=folium.Icon(color='red', icon='leaf')
    ).add_to(mapa_muestra)

    folium_static(mapa_muestra, width=700, height=400)
    st.caption(f"🧭 Red marker: sample classified as **{texto_cereal}** at {latitud}°N, {longitud}°E")

# ============================================
# BLOCK 13: FOOTER
# ============================================

st.markdown("---")
st.caption("""
**Project:** Carpology and Ancient Crops ML | **Author:** David Larreina-García
**Model:** Balanced Random Forest (74.78% accuracy)
**Repository:** [github.com/dplauto-cpu/Carpology](https://github.com/dplauto-cpu/Carpology)
""")
