#!/usr/bin/env python
# coding: utf-8

import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

# Configuration des chemins pour les ressources NLTK
NLTK_DATA_PATH = str(Path.home() / "nltk_data")  # Utilisation du dossier "nltk_data" dans le répertoire utilisateur

# Créer le dossier pour les données NLTK s'il n'existe pas
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Télécharger les ressources NLTK nécessaires si elles sont manquantes
RESOURCES = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
for resource in RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else f"corpora/{resource}")
    except LookupError:
        print(f"Téléchargement du package NLTK : {resource}")
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Initialisation du lemmatiseur et des stopwords
try:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
except Exception as e:
    raise RuntimeError(f"Erreur d'initialisation NLTK : {str(e)}")

# Fonction pour nettoyer le texte
def clean_text(text: str) -> str:
    """
    Nettoie le texte en supprimant la ponctuation, les chiffres et en convertissant tout en minuscule.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprime la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprime les chiffres
    return text

# Fonction pour prétraiter le texte : nettoyage, tokenisation, lemmatisation
def preprocess_text(text: str) -> str:
    """
    Prétraite le texte : nettoyage, tokenisation, lemmatisation.
    """
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Initialisation de l'application FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    """Endpoint racine pour vérifier que l'API fonctionne."""
    return {"message": "Bienvenue sur l'API d'Analyse de Sentiments!"}

# Chargement du modèle
try:
    pipeline = joblib.load('best_model.pkl')  # Charger le modèle enregistré
except Exception as e:
    raise RuntimeError(f"Échec du chargement du pipeline : {str(e)}")

# Modèle de données pour les prédictions
class InputData(BaseModel):
    text: str

    @validator('text')
    def validate_text(cls, v):
        """
        Valide que le texte n'est pas vide.
        """
        if not v or not isinstance(v, str):
            raise ValueError("Le texte doit être une chaîne non vide.")
        return v

@app.post('/predict/')
async def predict(input_data: InputData):
    """
    Endpoint pour effectuer une prédiction de sentiment.
    """
    try:
        # Prétraitement du texte
        cleaned_text = preprocess_text(input_data.text)

        # Prédiction avec le modèle
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

# Code pour démarrer l'API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
