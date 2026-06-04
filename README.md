# Carpology and Ancient Crops ML

**Stable isotope analysis of archaeological cereals using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20development-yellow.svg)]()

---

## 📋 Project description

This project applies **Machine Learning** techniques to isotopic data (δ13C and δ15N) from archaeological cereals (wheat and barley) originating from sites across Europe and the Mediterranean.

The aim is to explore whether isotopic patterns can help to:
- **Date** archaeological samples without carbon-14
- **Identify species** (wheat vs barley) in charred grains
- **Estimate growing conditions** (water stress and manuring)
- **Trace geographical origin** of cereals
- **Discover regions** with similar agricultural practices

---

## 🗂️ Repository structure

ancient-crops-ml/  
│  
├── data/  
│ ├── carpo_nuevo.csv # Main dataset (2,323 samples)  
│ ├── trigo_cebada_clean_ML.csv # Secondary dataset (3,344 samples)  
│ └── neo_plants.csv # Merged dataset (2,323 × 24)  
│  
├── notebooks/  
│ ├── 01_EDA.ipynb # Exploratory analysis  
│ ├── 02_preprocessing.ipynb # Cleaning and preparation  
│ └── 03_models.ipynb # Training and evaluation  
│
├── models/ # Trained models (pickle)  
├── images/ # Graphs for presentation  
├── presentation.md # Presentation script  
├── README.md # This file  
└── requirements.txt # Project dependencies  

---

## 📊 Dataset

### Data source
The data comes from the **MAIA** database (Mediterranean Archive of Isotopic dAta), which compiles isotopic information from the early Neolithic to the end of the Iron Age (approx. 6000-600 BCE), merged with a smaller dataset gathered by Carlota Pintado at the UBU (University of Burgos). For the sake of this exercise uniquely the entries and variables of interest habe been merged into a new dataset: `neo_plants`.

### Merged dataset (`neo_plants.csv`)

| Feature | Value |
|:---|:---|
| **Total samples** | 2,323 |
| **Columns** | 24 |
| **Main species** | Wheat (*Triticum*), Barley (*Hordeum*) |
| **Geographical distribution** | Europe, Mediterranean, Near East |
| **Cultural periods** | Neolithic, Bronze Age, Iron Age |

### Key variables

| Variable | Description |
|:---|:---|
| `IRMS_d13C_Collagen` | Carbon isotope (water stress) |
| `d15N_Collagen` | Nitrogen isotope (manuring intensity) |
| `Latitude_N` / `Longitude_E` | Geographic coordinates |
| `Genus` | Genus (Triticum / Hordeum) |
| `Cultural_Horizon` | Archaeological period |
| `Modern_Country` | Modern country of the site |

---

## 🎯 Proposed models (note that is a preliminary state)

| # | Problem | Type | Target | Algorithms |
|:---|:---|:---|:---|:---|
| 1 | Dating by isotopes | Classification | Cultural period | RF, XGBoost, SVM |
| 2 | Species identification | Classification | Wheat vs Barley | RF, XGBoost, Logistic Regression |
| 3 | Water stress | Regression | δ13C | RF Reg, XGBoost Reg, Linear Reg |
| 4 | Manuring intensity | Regression | δ15N | RF Reg, XGBoost Reg, Linear Reg |
| 5 | Geographical origin | Classification | Country | RF, XGBoost, SVM |
| 6 | Regional patterns | Unsupervised | Clusters | K-means, DBSCAN |

**Total: 6 models (5 supervised + 1 unsupervised)**

---

## 🛠️ Technologies used

- **Python 3.10+**
- **Pandas** / **NumPy** - Data manipulation
- **Matplotlib** / **Seaborn** - Visualisation
- **Scikit-learn** - ML models (Random Forest, SVM, Logistic Regression, K-means, DBSCAN)
- **XGBoost** - High-performance model
- **Jupyter Notebooks** - Development environment

---

## 📈 Project status

| Phase | Status |
|:---|:---|
| Data collection | ✅ Completed |
| Data cleaning and merging | ✅ Completed |
| Exploratory analysis (EDA) | 🔄 In progress |
| Preprocessing | ⏳ Pending |
| Model training | ⏳ Pending |
| Evaluation and validation | ⏳ Pending |
| Results interpretation | ⏳ Pending |
| Final presentation | ⏳ Pending |

---

## 🚀 Next steps

1. **Complete EDA**: visualisation of distributions, correlations, maps
2. **Preprocessing**: handling null values, encoding categorical variables, scaling
3. **Training**: cross-validation, hyperparameter tuning
4. **Evaluation**: accuracy, F1-score, R², RMSE
5. **Interpretation**: feature importance, cluster maps
6. **Presentation**: PPT slides with results

---

## 📝 How to run the project

### 1. Clone the repository
```bash
git clone https://github.com/your_username/ancient-crops-ml.git
cd ancient-crops-ml
