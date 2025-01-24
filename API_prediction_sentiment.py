#!/usr/bin/env python
# coding: utf-8

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from pathlib import Path

# Configuration des chemins pour les ressources NLTK
if "CI" in os.environ:  # Si en environnement CI (GitHub Actions)
    NLTK_DATA_PATH = str(Path.home() / "nltk_data")
else:  # Si en local
    VIRTUAL_ENV_PATH = r'C:\Users\attia\P7_Réalisez_une_analyse_de_sentiments_grâce_au_Deep_Learning\environnement\myenv'
    NLTK_DATA_PATH = os.path.join(VIRTUAL_ENV_PATH, 'nltk_data')

# Créer le dossier pour les données NLTK s'il n'existe pas
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Télécharger les ressources NLTK si nécessaire
RESOURCES = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
for resource in RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else f"corpora/{resource}")
    except LookupError:
        print(f"Téléchargement du package NLTK : {resource}")
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Initialisation du lemmatiseur et des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Fonction pour nettoyer le texte
def clean_text(text):
    """
    Nettoie le texte en supprimant la ponctuation, les chiffres et en mettant en minuscule.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    return text




# Fonction pour prétraiter le texte
def preprocess_text(text):
    """
    Prétraite le texte en le nettoyant, le tokenisant et en appliquant la lemmatisation.
    """
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Initialisation de l'application FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    """
    Endpoint racine pour vérifier que l'API est opérationnelle.
    """
    return {"message": "Bienvenue sur l'API d'Analyse de Sentiments!"}

# Chargement du pipeline (TF-IDF + Modèle) sauvegardé
try:
    pipeline = joblib.load('best_model.pkl')
except Exception as e:
    raise RuntimeError(f"Échec du chargement du pipeline : {str(e)}")

# Modèle de données pour les prédictions
class InputData(BaseModel):
    text: str

    @validator('text')
    def validate_text(cls, v):
        """
        Valide que le texte fourni n'est pas vide.
        """
        if not v or not isinstance(v, str):
            raise ValueError("Le texte doit être une chaîne non vide.")
        return v

@app.post('/predict/')
async def predict(input_data: InputData):
    """
    Endpoint pour effectuer une prédiction basée sur un texte fourni.
    """
    try:
        # Prétraitement du texte
        text_data = input_data.text
        cleaned_text = preprocess_text(text_data)

        # Prédiction avec le modèle chargé
        predictions = pipeline.predict([cleaned_text])
        prediction_label = int(predictions[0])
        sentiment = "Positive" if prediction_label == 1 else "Negative"

        return {
            "prediction": prediction_label,
            "sentiment": sentiment
        }

    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Point d'entrée principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
