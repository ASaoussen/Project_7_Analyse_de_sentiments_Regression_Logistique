#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import uvicorn
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Télécharger les ressources nécessaires de NLTK si ce n’est pas déjà fait
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')  # Pour tokenisation

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Liste des mots vides
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage générique (minuscule, suppression ponctuation et chiffres)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    return text

# Fonction de prétraitement avec lemmatisation
def preprocess_text(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Initialisation de l’application FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API d'Analyse de Sentiments! Veuillez envoyer un texte pour prédire son sentiment."}

# Charger le pipeline complet (TF-IDF + Modèle) sauvegardé
try:
    pipeline = joblib.load('best_model.pkl')  # Pipeline avec TF-IDF et le modèle
except Exception as e:
    raise RuntimeError(f"Échec du chargement du pipeline : {str(e)}")

# Définir le modèle de données d’entrée pour FastAPI
class InputData(BaseModel):
    text: str

    # Validation pour s'assurer que 'text' est une chaîne de caractères non vide
    @validator('text')
    def check_text_non_empty(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Le texte doit être une chaîne non vide.")
        return v

# Endpoint de prédiction

        
@app.post('/predict/')
async def predict(input_data: InputData):
    try:
        # Extraire et prétraiter le texte
        text_data = input_data.text
        
        # Prétraiter le texte d'entrée
        cleaned_text = preprocess_text(text_data)

        # Faire la prédiction directement avec le pipeline
        predictions = pipeline.predict([cleaned_text])

        # Retourner les prédictions sous forme de JSON
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        print(f"Erreur lors de la prédiction : {str(e)}")  # Ajouter un log pour l'erreur
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
# Lancer le serveur avec la commande suivante :
# uvicorn script_name:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
