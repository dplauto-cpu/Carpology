# Carpology and Ancient Crops ML

**Stable isotope analysis of archaeological cereals using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

---

## 📋 Project description

This project applies **Machine Learning** techniques to isotopic data (δ13C and δ15N) from archaeological cereals (barley and wheat) gathered from archaeological sites across the Mediterranean. Specifically, from eight modern countries: Cyprus, France, Greece, Italy, Morocco, Spain, Syria and Turkey.

The main objective is to **discern charred seeds as barley or wheat** using isotopic, geographical, and chronological features.

Secondary objectives:
- **Feature importance analysis** to understand which variables drive classification
- **Interactive mapping** (Folium) with vintage National Geographic style

## 📋 Quick guide

The fundamentals of this project can be checked out in two documents: `Technical_Report` (concerning to DS) and `Archaeological_Report` (concerning to academic conclusions), both of them in folder `6. docs`

---

## 🗂️ Repository structure
Carpology, discerning cereals by isotopes/  
│  
├── data/  
│ ├── raw/  
│ │ ├── maia_plant.csv  
│ │ └── trigo_cebada_clean.csv *This dataset is not included, is a private working dataset from IsoTOPIK  
│ │  
│ ├── processed/  
│ │ ├── Med_Plants.csv  
│ │  
│ ├── train/  
│ │ ├── X_train.csv  
│ │ └── y_train.csv  
│ │  
│ └── test/  
│ ├── X_test.csv  
│ └── y_test.csv  
│  
├── notebooks/  
│ ├── 01_Sources.ipynb  
│ ├── 02_Transform_and_Fet_Eng.ipynb  
│ ├── 03_EDA.ipynb  
│ └── 04.Models/  
│ ├── Model1_clustering.ipynb  
│ ├── Model2_RFC_Hy.ipynb  
│ ├── Model3_SVM.ipynb  
│ ├── Model4_NB.ipynb  
│ └── Model5_XGBo_Hy.ipynb  
│  
├── src/  
│ ├── data_processing.py  
│ ├── training.py  
│ └── evaluation.py  
│  
├── models/  
│ ├── final_model_random.pkl  
│ ├── trained_XGBoost_model.pkl  
│ ├── trained_svm_model.pkl  
│ ├── trained_kmeans_model.pkl  
│ ├── trained_naive_bayes.pkl  
│ └── model_config.yaml  
│  
├── app_streamlit/  
│ ├── app.py  
│ └── requirements.txt  
│  
├── docs/  
│ ├── Technical_Report.ipynb  
│ ├── Archaeological_Report.ipynb  
│ ├── fig1_Relacion_isotopos.png  
│ ├── fig2_Tendencias_cuenca.png  
│ ├── fig3_matriz_correlación.png  
│ ├── fig4_ROC_RF.png  
│ ├── fig5_matriz confusión.png   
│ ├── fig5_matriz confusión.png   
│ ├── fig6_países_cuenca.png  
│ └──mapa_arqueologico_final.html  
│  
├── README.md  

---

## 📊 Dataset

### Data source
The data comes from  merging two datasets: the **MAIA** database (Mediterranean Archive of Isotopic dAta), merged with another dataset gathered by a member of the Research Group at the IsoTOPIKLab (University of Burgos, UBU).

### Final dataset (`Med_plants.csv`)

|:---|:---|
| **Total samples** | 1,341 |
| **Columns** | 20 |
| **Target classes** | Barley (586), Wheat (561) |
| **Geographical distribution** | Europe, Mediterranean, Near East |
| **Chronological periods** | Neolithic to Iron_Roman (see Archaeological Report |

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
Archaeologist
GitHub: dplauto-cpu
---
