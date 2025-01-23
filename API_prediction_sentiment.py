#!/usr/bin/env python
# coding: utf-8

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import joblib
import uvicorn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Définir le chemin vers nltk_data dans l'environnement virtuel
virtual_env_path = r'C:\Users\attia\P7_Réalisez_une_analyse_de_sentiments_grâce_au_Deep_Learning\environnement\myenv'
nltk_data_path = os.path.join(virtual_env_path, 'nltk_data')

# Créer le dossier s'il n'existe pas
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Ajouter ce chemin aux chemins de recherche de NLTK
nltk.data.path.append(nltk_data_path)

# Télécharger les ressources nécessaires si elles ne sont pas déjà disponibles
resources = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
for resource in resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()

# Charger la liste des mots vides (stopwords)
stop_words = set(stopwords.words('english'))

# Fonction de nettoyage de texte
def clean_text(text):
    """
    Nettoie le texte en supprimant la ponctuation, les chiffres et en mettant en minuscule.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer la ponctuation
    text = re.sub(r'\d+', '', text)  # Supprimer les chiffres
    return text

# Fonction de prétraitement avec lemmatisation
def preprocess_text(text):
    """
    Prétraite le texte en nettoyant, en tokenisant et en appliquant la lemmatisation.
    """
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Initialisation de l’application FastAPI
app = FastAPI()

@app.get("/")
async def read_root():
    """
    Endpoint racine pour vérifier que l'API est opérationnelle.
    """
    return {"message": "Bienvenue sur l'API d'Analyse de Sentiments!"}

# Charger le pipeline complet (TF-IDF + Modèle) sauvegardé
try:
    pipeline = joblib.load('best_model.pkl')  # Charger le fichier contenant le pipeline
except Exception as e:
    raise RuntimeError(f"Échec du chargement du pipeline : {str(e)}")

# Définir le modèle de données d’entrée pour FastAPI
class InputData(BaseModel):
    text: str

    @validator('text')
    def check_text_non_empty(cls, v):
        """
        Valide que le texte d'entrée n'est pas vide.
        """
        if not v or not isinstance(v, str):
            raise ValueError("Le texte doit être une chaîne non vide.")
        return v

# Endpoint de prédiction
@app.post('/predict/')
async def predict(input_data: InputData):
    """
    Endpoint pour effectuer une prédiction à partir du texte fourni.
    """
    try:
        # Prétraiter le texte d'entrée
        text_data = input_data.text
        cleaned_text = preprocess_text(text_data)
        
        # Faire une prédiction avec le pipeline chargé
        predictions = pipeline.predict([cleaned_text])
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        # Ajouter un log en cas d'erreur
        print(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Point d'entrée pour lancer l'application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
