#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

# URL de base de l'API
BASE_URL = "http://localhost:8001"

def test_root_endpoint():
    """
    Tester l'endpoint racine (/)
    """
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, f"Erreur : Code {response.status_code}"
    print("Réponse de l'endpoint racine :", response.json())

def test_predict_endpoint():
    """
    Tester l'endpoint de prédiction (/predict/)
    """
    payload = {"text": "I love the service, everything was perfect!"}
    
    # Faire une requête POST à l'endpoint de prédiction
    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    assert response.status_code == 200, f"Erreur : Code {response.status_code}"
    
    print("Réponse de l'endpoint /predict/ :", response.json())

def test_predict_invalid_payload():
    """
    Tester la gestion des erreurs avec un payload invalide
    """
    payload = {"invalid_key": "This will cause an error"}  # Clé invalide
    
    # Faire une requête POST à l'endpoint de prédiction
    response = requests.post(f"{BASE_URL}/predict/", json=payload)
    assert response.status_code == 422, f"Erreur attendue mais obtenue : Code {response.status_code}"
    
    print("Gestion des erreurs - Réponse :", response.json())

if __name__ == "__main__":
    print("Test de l'endpoint racine...")
    test_root_endpoint()
    
    print("\nTest de l'endpoint /predict/ avec une entrée valide...")
    test_predict_endpoint()
    
    print("\nTest de l'endpoint /predict/ avec une entrée invalide...")
    test_predict_invalid_payload()






