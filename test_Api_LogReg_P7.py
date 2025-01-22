import pytest
from fastapi.testclient import TestClient
from API_p7_Reglog import app  # Remplacez par le chemin d'importation correct pour votre fichier API

# Créer un client de test
client = TestClient(app)

def test_predict_endpoint_valid_input():
    # Préparer les données d'entrée
    input_data = {
        "text": "I loved the experience with this airline, the service was excellent!"
    }

    # Envoyer une requête POST au endpoint de prédiction
    response = client.post("/predict/", json=input_data)

    # Vérifier que la requête a réussi
    assert response.status_code == 200

    # Extraire la réponse JSON
    response_data = response.json()

    # Vérifier que la réponse contient les prédictions
    assert "predictions" in response_data
    assert isinstance(response_data["predictions"], list)

def test_predict_endpoint_invalid_input():
    # Préparer des données d'entrée invalides
    input_data = {
        "text": 12345  # Ce n'est pas une chaîne de caractères
    }

    # Envoyer une requête POST au endpoint de prédiction
    response = client.post("/predict/", json=input_data)

    # Vérifier que la requête retourne un statut d'erreur
    assert response.status_code == 422  # Unprocessable Entity pour une mauvaise entrée de données

def test_predict_endpoint_empty_input():
    # Préparer des données d'entrée vides
    input_data = {
        "text": ""
    }

    # Envoyer une requête POST au endpoint de prédiction
    response = client.post("/predict/", json=input_data)

    # Vérifier que la requête a réussi
    assert response.status_code == 200

    # Extraire la réponse JSON
    response_data = response.json()

    # Vérifier que la réponse contient les prédictions
    assert "predictions" in response_data
    assert isinstance(response_data["predictions"], list)
    assert len(response_data["predictions"]) == 1  # Une seule prédiction attendue pour une seule entrée

def test_predict_endpoint_internal_error():
    # Simuler une situation où le modèle ou le vectoriseur est indisponible
    # Vous pouvez temporairement simuler cette erreur en modifiant les chemins d'accès ou en utilisant un mock pour le modèle.

    input_data = {
        "text": "This is a test text."
    }

    response = client.post("/predict/", json=input_data)

    # Si le modèle ou le vectoriseur est manquant, le serveur doit retourner une erreur 500
    assert response.status_code == 500
    assert "Erreur" in response.json()["detail"]
