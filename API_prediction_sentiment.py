#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# Endpoint de prédiction
@app.post('/predict/')
async def predict(input_data: InputData):
    try:
        # Extraire et prétraiter le texte
        text_data = input_data.text
        if isinstance(text_data, str):
            text_data = [text_data]

        df = pd.DataFrame({'text': text_data})

        # Appliquer le nettoyage et la lemmatisation
        df['cleaned_text'] = df['text'].apply(preprocess_text)

        # Faire la prédiction directement avec le pipeline
        predictions = pipeline.predict(df['cleaned_text'])

        # Retourner les prédictions sous forme de JSON
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Lancer le serveur avec la commande suivante :
# uvicorn script_name:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)







