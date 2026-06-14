"""
Módulo para entrenar y guardar el modelo
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_processed_data(ruta):
    """Carga los datos procesados"""
    return pd.read_csv(ruta)

def preprocess_for_training(df):
    """Prepara los datos para el entrenamiento"""
    # Filtrar solo barley y wheat
    df = df[df['Cereal'].isin(['barley', 'wheat'])].copy()
    
    # Crear target
    df['target'] = (df['Cereal'] == 'barley').astype(int)
    
    # Características
    features = ['IRMS_d13C_Collagen', 'd15N_Collagen', 'Latitude_N', 'Longitude_E',
                'Chronological_Period_clean', 'Mediterranean_Basin']
    
    X = df[features]
    y = df['target']
    
    # Codificar variables categóricas
    le_periodo = LabelEncoder()
    le_cuenca = LabelEncoder()
    
    X['periodo_encoded'] = le_periodo.fit_transform(X['Chronological_Period_clean'])
    X['cuenca_encoded'] = le_cuenca.fit_transform(X['Mediterranean_Basin'])
    
    X = X[['IRMS_d13C_Collagen', 'd15N_Collagen', 'Latitude_N', 'Longitude_E',
           'periodo_encoded', 'cuenca_encoded']]
    
    return X, y, le_periodo, le_cuenca

def train_model(X_train, y_train):
    """Entrena el modelo Random Forest balanceado"""
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model_and_preprocessors(model, scaler, le_periodo, le_cuenca, ruta_modelo='models/'):
    """Guarda el modelo y los preprocesadores"""
    os.makedirs(ruta_modelo, exist_ok=True)
    joblib.dump(model, f'{ruta_modelo}/random_forest_balanceado.pkl')
    joblib.dump(scaler, f'{ruta_modelo}/scaler.pkl')
    joblib.dump(le_periodo, f'{ruta_modelo}/le_periodo.pkl')
    joblib.dump(le_cuenca, f'{ruta_modelo}/le_cuenca.pkl')

def run_training():
    """Ejecuta todo el pipeline de entrenamiento"""
    df = load_processed_data('data/processed/Med_Plants.csv')
    X, y, le_periodo, le_cuenca = preprocess_for_training(df)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar
    model = train_model(X_train, y_train)
    
    # Guardar
    save_model_and_preprocessors(model, scaler, le_periodo, le_cuenca)
    
    print(f"✅ Modelo entrenado con {len(X_train)} muestras")
    print(f"   Precisión en train: {model.score(X_train, y_train):.2%}")

if __name__ == "__main__":
    run_training()