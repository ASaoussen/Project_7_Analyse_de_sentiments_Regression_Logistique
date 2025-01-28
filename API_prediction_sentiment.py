import os
import re
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuration des journaux
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration de NLTK
NLTK_DATA_PATH = str(Path.home() / "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

RESOURCES = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
for resource in RESOURCES:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == 'punkt' else f"corpora/{resource}")
    except LookupError:
        logging.info(f"Téléchargement du package NLTK : {resource}")
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Initialisation de l'analyseur lexical et des stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Nettoie le texte en supprimant les caractères spéciaux et les chiffres."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def preprocess_text(text: str) -> str:
    """Prépare le texte en le nettoyant, le tokenisant et en le lemmatisant."""
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Création de l'application FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API d'Analyse de Sentiments !"}

# Chargement du modèle
MODEL_PATH = "best_model.pkl"
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le fichier {MODEL_PATH} est introuvable.")
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle : {e}")
    raise RuntimeError(f"Échec du chargement du modèle : {e}")

# Modèle de données pour l'entrée de l'utilisateur
class InputData(BaseModel):
    text: str

    @field_validator('text')
    def validate_text(cls, v):
        """Valide que le texte n'est pas vide."""
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide.")
        return v

@app.post("/predict/")
async def predict(input_data: InputData):
    """Prédit le sentiment d'un texte."""
    try:
        # Prétraitement du texte
        cleaned_text = preprocess_text(input_data.text)
        
        # Prédiction avec le modèle
        predictions = pipeline.predict([cleaned_text])
        sentiment = "Positive" if predictions[0] == 1 else "Negative"
        
        # Retour de la réponse
        return {"prediction": int(predictions[0]), "sentiment": sentiment}
    except ValueError as ve:
        logging.error(f"Erreur lors de la prédiction : {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

# Code pour démarrer l'API via Uvicorn (si nécessaire)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
