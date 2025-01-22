from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import mlflow
import uvicorn
import mlflow.pyfunc
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
    return {"message": "Bienvenue sur l'API Analyse de Sentiments!"}

# Chargement du modèle et du vectoriseur
try:
    # Définir l’URI de suivi MLflow
    mlflow.set_tracking_uri('http://localhost:5001')

    # Spécifier le chemin de l’artéfact
    artifact_uri = 'mlflow-artifacts:/135230274734192630/312e0c08a9c9423783e82a3818656cb3/artifacts/LogisticRegression_lemmatized_model'

    # Télécharger le modèle à un emplacement local
    local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path='path_to_save_model')

    # Charger le modèle depuis l’emplacement local
    model = mlflow.pyfunc.load_model(local_model_path)

    # Charger le vectoriseur
    #vectorizer_path = r'C:\Users\attia\P7_Réalisez_une_analyse_de_sentiments_grâce_au_Deep_Learning\Project_7_Analyse_de_sentiments\tfidf_vectorizer_lemmatized.pkl'
    vectorizer = joblib.load('tfidf_vectorizer_lemmatized.pkl')

except Exception as e:
    raise RuntimeError(f"Échec du chargement du modèle ou du vectoriseur : {str(e)}")

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

        # Prétraiter le texte avec nettoyage et lemmatisation
        df['cleaned_text'] = df['text'].apply(preprocess_text)

        # Transformer les textes nettoyés en utilisant le vectorizer
        transformed_text = vectorizer.transform(df['cleaned_text'])

        # Faire la prédiction
        predictions = model.predict(transformed_text)

        # Retourner les prédictions sous forme de JSON
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

# Lancer le serveur avec la commande suivante :
# uvicorn script_name:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
