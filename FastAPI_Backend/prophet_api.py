# FastAPI pour le déploiement des modèles Prophet (page 8 de L'API Streamlit)

# Importation des bibliothèques nécessaires
from fastapi import FastAPI, HTTPException
from prophet.serialize import model_from_json
import json
import pandas as pd

# Création de l'instance FastAPI
app = FastAPI()

# Dictionnaire global pour stocker les modèles chargés
models = {}

# Fonction pour charger un modèle Prophet depuis un fichier JSON
def load_model(file_name):
    with open(file_name, 'r') as f:
        model = model_from_json(json.load(f))
    return model

# Fonction pour charger les régresseurs depuis le répertoire "data"
def load_regressors(variable):
    regressors_path = f"data/{variable}_regressors.csv"
    regressors = pd.read_csv(regressors_path).drop(columns=['Unnamed: 0'], axis=1)
    regressors['ds'] = pd.to_datetime(regressors['ds'])  # Convertir la colonne 'ds' en datetime
    return regressors

# Cette fonction sera exécutée au démarrage de l'API pour charger tous les modèles
@app.on_event("startup")
async def load_models_on_startup():
    global models
    models = {
        'total_accidents': load_model('models/Prophet_model_tot_acc.json'),
        'gravite_accident_tué': load_model('models/Prophet_model_acc_tués.json'),
        'gravite_accident_blessé_léger': load_model('models/Prophet_model_acc_legers.json'),
        'gravite_accident_blessé_hospitalisé': load_model('models/Prophet_model_acc_hosp.json'),
        'gravite_accident_indemne': load_model('models/Prophet_model_acc_indemnes.json')
    }
# Route pour la page d'accueil
@app.get("/")
def read_root():
    return {
        "message": "Bienvenue sur l'API Prophet !",
        "docs": "Pour voir la documentation, visitez /docs",
    }


# Route pour effectuer des prédictions en utilisant un modèle spécifié
@app.post("/predict/{model_name}/{days}")
async def predict(model_name: str, days: int):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Modèle {model_name} introuvable")
    
    # Chargement des régresseurs
    regressors = load_regressors(model_name)

    # Création du DataFrame future
    future = models[model_name].make_future_dataframe(periods=days, freq='D', include_history=True)

    # Fusion de future avec les régresseurs sur la colonne 'ds'
    future = pd.merge(future, regressors, on='ds', how='left')

    # Effectuer la prédiction
    forecast = models[model_name].predict(future)

    # Retourner le résultat sous forme de dictionnaire
    return forecast.to_dict(orient='records')




