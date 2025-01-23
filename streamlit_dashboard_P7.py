#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import requests

st.title('Test de l\'API de Sentiment')

# Saisie du texte utilisateur
user_input = st.text_area("Entrez votre texte ici")

if st.button('Prédire'):
    # Appel de l'API
    try:
        response = requests.post("http://localhost:8001/predict/", json={"text": user_input})
        response.raise_for_status()  # Vérifie les erreurs HTTP
        prediction = response.json()
        
        # Affichage du résultat
        st.write(f"Prédiction : {prediction['predictions']}")
        
        # Demande de validation
        is_valid = st.radio("La prédiction est-elle correcte ?", ("Oui", "Non"))
        
        # Message selon la validation
        if is_valid == "Non":
            st.write("Merci pour votre retour, nous tiendrons compte de votre feedback.")
        else:
            st.write("Merci de valider la prédiction.")
    
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")

