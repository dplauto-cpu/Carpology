"""
Módulo para procesar datos desde raw hasta processed
"""

import pandas as pd
import os

def load_raw_data(ruta_maia, ruta_trigo):
    """Carga los datasets originales"""
    maia = pd.read_csv(ruta_maia, sep=';')
    trigo = pd.read_csv(ruta_trigo)
    return maia, trigo

def clean_and_merge(maia, trigo):
    """Limpia y fusiona los datasets"""
    # Aquí va todo tu proceso de limpieza y merge
    # (similar a lo que hiciste en merge.ipynb)
    pass

def save_processed_data(df, ruta_salida):
    """Guarda los datos procesados"""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)

def run_pipeline():
    """Ejecuta todo el pipeline de procesamiento"""
    maia, trigo = load_raw_data('data/raw/maia_plant.csv', 
                                 'data/raw/trigo_cebada_clean_ML.csv')
    df_procesado = clean_and_merge(maia, trigo)
    save_processed_data(df_procesado, 'data/processed/Med_Plants.csv')
    print("✅ Pipeline completado")

if __name__ == "__main__":
    run_pipeline()