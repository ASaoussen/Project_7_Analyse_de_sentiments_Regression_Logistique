import streamlit as st
import requests
import logging
from applicationinsights import TelemetryClient
from applicationinsights.logging import LoggingHandler


# Fonction d'envoi de trace à Azure Application Insights avec logger
def setup_logger():
    connection_string = "InstrumentationKey=aef8c367-658f-4580-92e6-05a082b5cd6c;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/;ApplicationId=fb603895-82d4-423c-855a-9e3b2e18dc72"  # Connection String d'Application Insights
    handler = LoggingHandler(connection_string)  # Handler pour envoyer les logs à Application Insights
    logger = logging.getLogger("AzureLogger")
    logger.setLevel(logging.INFO)  # Niveau de log INFO, à ajuster si nécessaire
    logger.addHandler(handler)
    return logger

# Application de styles CSS personnalisés
st.markdown("""
    <style>
        body {
            background-color: #003366;  /* Bleu marine */
            color: white;
        }
        .stTextArea>div>textarea {
            background-color: #99ccff;  /* Bleu ciel pour les zones de texte */
            color: black;
        }
        .stTextInput>div>input {
            background-color: #99ccff;
            color: black;
        }
        .stButton>button {
            background-color: #003366;
            color: white;
        }
        .stTitle {
            text-align: center;
            color: #003366;
        }
        .stRadio>div>label {
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application centré
st.markdown("<h1 style='text-align: center; color: #003366;'>Analyse de Sentiments - Air Paradis</h1>", unsafe_allow_html=True)

# Saisie du prénom et du nom
prenom = st.text_input("Entrez votre prénom", "").strip()
nom = st.text_input("Entrez votre nom", "").strip()

if prenom and nom:
    st.write(f"Bonjour {prenom} {nom}, bienvenue sur notre application d'analyse de sentiments !")

# Saisie du texte utilisateur
user_input = st.text_area("Entrez le texte que vous souhaitez analyser")

# Vérification et prédiction
if st.button("Prédire"):
    if not user_input.strip():
        st.error("Veuillez entrer un texte valide avant de cliquer sur 'Prédire'.")
    else:
        try:
            # Configurer le logger pour envoyer des logs à Application Insights
            logger = setup_logger()

            # Log une trace personnalisée
            logger.info("Texte soumis pour analyse", extra={"custom_properties": {"texte_utilisateur": user_input}})

            # Appel de l'API pour la prédiction
            response = requests.post("http://localhost:8000/predict/", json={"text": user_input})
            response.raise_for_status()
            prediction = response.json()

            # Affichage de la prédiction
            st.write(f"Prédiction : {prediction['prediction']} (Sentiment : {prediction['sentiment']})")

            # Initialiser feedback si nécessaire
            if "feedback" not in st.session_state:
                st.session_state.feedback = None

            # Afficher le choix de feedback uniquement après avoir vu la prédiction
            if st.session_state.feedback is None:  # Vérifier si le feedback est déjà donné
                is_valid = st.radio(
                    "La prédiction est-elle correcte ?", 
                    ("Oui", "Non"), 
                    index=0,
                    key="feedback"
                )

                # Vérifier la réponse et afficher le message
                if is_valid == "Non":
                    st.session_state.feedback = "Non"  # Mémoriser la réponse
                    st.write("Merci pour votre retour, nous améliorerons notre modèle.")
                    # Log à Application Insights avec le feedback
                    logger.info("Prédiction non valide", extra={"custom_properties": {"is_valid": False}})
                elif is_valid == "Oui":
                    st.session_state.feedback = "Oui"  # Mémoriser la réponse
                    st.write("Merci d'avoir validé la prédiction.")
                    # Log à Application Insights avec le feedback
                    logger.info("Prédiction validée", extra={"custom_properties": {"is_valid": True}})

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur de connexion à l'API : {e}")
            # Log l'erreur d'API dans Application Insights
            logger.error(f"Erreur de connexion à l'API : {e}", extra={"custom_properties": {"is_valid": False}})
