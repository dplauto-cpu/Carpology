# Carpology and Ancient Crops ML

**Stable isotope analysis of archaeological cereals using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## 📋 Project description

This project applies **Machine Learning** techniques to isotopic data (δ13C and δ15N) from archaeological cereals (barley and wheat) gathered from sites across Europe and the Mediterranean.

The main objective is to **classify charred seeds as barley or wheat** using isotopic, geographical, and chronological features.

Secondary objectives achieved:
- **Unsupervised clustering** (K-means) to identify isotopic groups
- **Feature importance analysis** to understand which variables drive classification
- **Interactive mapping** (Folium) with vintage National Geographic style

---

## 🗂️ Repository structure
Carpology/  
│  
├── data/  
│ └── Neo_Met_plants.csv # Main dataset (1341 × 20)  
│  
├── notebooks/  
│ ├── 01_EDA.ipynb # Exploratory analysis  
│ ├── 02_preprocessing.ipynb # Cleaning and preparation  
│ ├── 03_clustering.ipynb # K-means clustering  
│ ├── 04_RandomForest_XGBoost.ipynb # Supervised models  
│ └── 05_final_model.ipynb # Balanced Random Forest  
│
├── src/  
│ └── utils.py # Helper functions  
│
├── models/  
│ ├── random_forest_balanceado.pkl # Final model  
│ ├── scaler.pkl # StandardScaler  
│ ├── le_periodo.pkl # LabelEncoder (period)  
│ └── le_cuenca.pkl # LabelEncoder (basin)  
│
├── app_streamlit/  
│ └── app.py # Streamlit demo app  
│  
├── docs/  
│ ├── fig1_relación_isótopos.png  
│ ├── fig2_tendencias_cuenca.png  
│ ├── fig3_matriz_correlación.png  
│ ├── fig4_roc.png  
│ ├── fig5_matriz_confusión.png  
│ ├── fig6_países_cuenca.png  
│ └── mapa_final_con_yacimientos.html  
│  
├── README.md # This file  
└── memoria.md # Technical report (English)  

text

---

## 📊 Dataset

### Data source
The data comes from the **MAIA** database (Mediterranean Archive of Isotopic dAta), merged with another dataset from the IsoTOPIKLab (University of Burgos).

### Final dataset (`Neo_Met_plants.csv`)

| Feature | Value |
|:---|:---|
| **Total samples** | 1,341 |
| **Columns** | 20 |
| **Target classes** | Barley (586), Wheat (561) |
| **Geographical distribution** | Europe, Mediterranean, Near East |
| **Chronological periods** | Neolithic, Bronze Age, Iron Age |

### Key variables

| Variable | Description |
|:---|:---|
| `IRMS_d13C_Collagen` | Carbon isotope (water stress) |
| `d15N_Collagen` | Nitrogen isotope (aridity / manuring) |
| `Latitude_N` / `Longitude_E` | Geographic coordinates |
| `Chronological_Period_clean` | Archaeological period |
| `Mediterranean_Basin` | Western, Central, Eastern, or undefined |
| `Cereal` | Target: 'barley' or 'wheat' |

---

## 🎯 Models implemented

| # | Problem | Type | Target | Best Model | Accuracy | AUC |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | Species identification | Binary classification | Barley vs Wheat | Random Forest (balanced) | **74.78%** | 0.824 |
| 2 | Isotopic grouping | Unsupervised clustering | 4 clusters | K-means | Silhouette: 0.42 | - |

### Model comparison

Another model was trained, but RF was finally selected. The metrics can be optimized. It is intended a follow up once the DS formation is concluded.

| Model | Accuracy | AUC | Top feature |
|:---|:---|:---|:---|
| Random Forest (base) | 74.35% | 0.825 | δ13C (42%) |
| XGBoost | 74.35% | 0.813 | Latitude (28%) |
| **Random Forest (balanced)** | **74.78%** | **0.824** | δ13C (41%) |

---

## 🛠️ Technologies used

- **Python 3.10+**
- **Pandas** / **NumPy** - Data manipulation
- **Matplotlib** / **Seaborn** - Visualisation
- **Scikit-learn** - Random Forest, K-means, preprocessing
- **XGBoost** - Gradient boosting
- **Folium** - Interactive maps
- **Streamlit** - Demo application
- **Jupyter Notebooks** - Development environment

---

## 📈 Key findings

1. **Isotopes alone are not sufficient** for perfect classification (≈75% accuracy)
2. **Random Forest outperforms XGBoost** for this problem due to non-linear isotopic relationships
3. **Class balancing** improved accuracy by +0.43%
4. **Clustering reveals 4 distinct isotopic groups** corresponding to different water stress and aridity levels
5. **Geographical patterns** show clear regional differences in agricultural practices

---

## 🚀 How to run the project

### 1. Clone the repository
```bash
git clone https://github.com/dplauto-cpu/Carpology.git
cd Carpology
2. Install dependencies
bash
pip install -r requirements.txt
3. Run the Streamlit app
bash
cd app_streamlit
streamlit run app.py
4. Explore the notebooks
bash
jupyter notebook notebooks/

👨‍🔬 Author
David Larreina-García
Archaeologist & Data Science student
GitHub: dplauto-cpu
---
## 📝 How to run the project

### 1. Clone the repository
```bash
git clone https://github.com/your_username/ancient-crops-ml.git
cd ancient-crops-ml
