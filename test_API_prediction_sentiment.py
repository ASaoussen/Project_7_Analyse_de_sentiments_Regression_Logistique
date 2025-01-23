#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi.testclient import TestClient
from API_prediction_sentiment import app  #  nom de votre fichier Python contenant l'application FastAPI

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue sur l'API d'Analyse de Sentiments! Veuillez envoyer un texte pour prédire son sentiment."}

def test_predict_positive_sentiment():
    response = client.post(
        "/predict/",
        json={"text": "I am very happy with the service!"}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_predict_negative_sentiment():
    response = client.post(
        "/predict/",
        json={"text": "I am very disappointed with the service."}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)

def test_predict_invalid_input():
    response = client.post(
        "/predict/",
        json={"text": 1234}  # Envoyer une valeur non valide
    )
    assert response.status_code == 422  # Erreur de validation FastAPI


# In[ ]:



