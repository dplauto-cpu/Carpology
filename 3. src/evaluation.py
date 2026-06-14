"""
Módulo para evaluar el modelo guardado
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def load_model_and_data():
    """Carga el modelo guardado y los datos de test"""
    model = joblib.load('models/random_forest_balanceado.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le_periodo = joblib.load('models/le_periodo.pkl')
    le_cuenca = joblib.load('models/le_cuenca.pkl')
    
    X_test = pd.read_csv('data/test/X_test.csv')
    y_test = pd.read_csv('data/test/y_test.csv').values.ravel()
    
    return model, scaler, le_periodo, le_cuenca, X_test, y_test

def preprocess_test_data(X_test, le_periodo, le_cuenca, scaler):
    """Preprocesa los datos de test (mismo que en entrenamiento)"""
    X = X_test.copy()
    
    # Codificar
    X['periodo_encoded'] = le_periodo.transform(X['Chronological_Period_clean'])
    X['cuenca_encoded'] = le_cuenca.transform(X['Mediterranean_Basin'])
    
    X = X[['IRMS_d13C_Collagen', 'd15N_Collagen', 'Latitude_N', 'Longitude_E',
           'periodo_encoded', 'cuenca_encoded']]
    
    return scaler.transform(X)

def evaluate(y_test, y_pred, y_proba):
    """Calcula y muestra las métricas"""
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n📊 Métricas de evaluación:")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   AUC: {auc:.3f}")
    print(f"\n   Matriz de confusión:")
    print(f"   Trigo reales: {cm[0,0]} bien | {cm[0,1]} mal")
    print(f"   Cebada reales: {cm[1,0]} mal | {cm[1,1]} bien")
    
    return acc, auc, cm

def run_evaluation():
    """Ejecuta la evaluación completa"""
    model, scaler, le_periodo, le_cuenca, X_test, y_test = load_model_and_data()
    X_test_scaled = preprocess_test_data(X_test, le_periodo, le_cuenca, scaler)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    evaluate(y_test, y_pred, y_proba)

if __name__ == "__main__":
    run_evaluation()