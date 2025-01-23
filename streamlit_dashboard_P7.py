#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests

# Titre de l'application
st.title('Bienvenue sur notre application d\'analyse de sentiments')

# Demande du nom et prénom
prenom = st.text_input("Entrez votre prénom", "")
nom = st.text_input("Entrez votre nom", "")

if prenom and nom:
    st.write(f"Bonjour {prenom} {nom}, bienvenue sur notre application d'analyse de sentiments!")

# Saisie du texte utilisateur pour l'analyse de sentiment
user_input = st.text_area("Entrez le texte que vous souhaitez analyser ici")

if st.button('Prédire'):
    # Vérifie si un texte a été saisi
    if not user_input:
        st.error("Veuillez entrer un texte avant de cliquer sur 'Prédire'.")
    else:
        # Appel de l'API
        try:
            response = requests.post("http://localhost:8080/predict/", json={"text": user_input})
            response.raise_for_status()  # Vérifie les erreurs HTTP
            prediction = response.json()

            # Affichage de la prédiction
            st.write(f"Prédiction : {prediction['predictions']}")

            # Confirmation de la validité de la prédiction
            is_valid = st.radio("La prédiction est-elle correcte ?", ("Oui", "Non"))

            # Affichage du feedback utilisateur
            if is_valid == "Non":
                st.write("Merci pour votre retour, nous tiendrons compte de votre feedback.")
            else:
                st.write("Merci d'avoir validé la prédiction.")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}")


