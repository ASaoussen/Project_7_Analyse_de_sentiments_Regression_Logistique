from fastapi.testclient import TestClient
from API_prediction_sentiment import app  # nom de votre fichier Python contenant l'application FastAPI

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
    # Vérification du code de réponse
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    # Vérification de la présence de la clé "predictions"
    response_json = response.json()
    assert "predictions" in response_json, f"Expected 'predictions' key, got {response_json}"
    
    # Vérification du type de la réponse
    assert isinstance(response_json["predictions"], list), f"Expected list, got {type(response_json['predictions'])}"

def test_predict_negative_sentiment():
    response = client.post(
        "/predict/",
        json={"text": "I am very disappointed with the service."}
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    response_json = response.json()
    assert "predictions" in response_json, f"Expected 'predictions' key, got {response_json}"
    assert isinstance(response_json["predictions"], list), f"Expected list, got {type(response_json['predictions'])}"

def test_predict_invalid_input():
    response = client.post(
        "/predict/",
        json={"text": 1234}  # Envoi d'une valeur non valide
    )
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"  # Erreur de validation FastAPI

def test_predict_empty_input():
    response = client.post(
        "/predict/",
        json={"text": ""}  # Test pour une chaîne vide
    )
    assert response.status_code == 422, f"Expected 422, got {response.status_code}"  # Vérification de la gestion d'entrée vide
