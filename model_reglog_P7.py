#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# Charger les données sauvegardées depuis un fichier CSV
data = pd.read_csv('data_cleaned_text.csv')

import mlflow
import mlflow.sklearn
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Définir l'URI du serveur MLFlow
mlflow.set_tracking_uri("http://localhost:5001")

# Spécifier l'expérience sous un nom personnalisé
mlflow.set_experiment("New Logistic Regression Experiment")

# Fonction de tracking des expériences dans MLFlow
def log_mlflow_experiment(model, X_train, y_train, X_valid, y_valid, X_test, y_test, model_name, params=None):
    with mlflow.start_run(run_name=model_name) as run:
        # Générer un nom de run explicite basé sur les paramètres
        run_name = f"{model_name}_C_{params['C']}_solver_{params['solver']}"
        mlflow.set_tag("run_name", run_name)
        
        # Enregistrement des hyperparamètres 
        if params:
            for param, value in params.items():
                mlflow.log_param(param, value)
        
        # Mesure du temps d'exécution du modèle
        start_time = time.time()

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions et calcul des métriques
        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calcul AUC
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        valid_auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Calcul du temps d'entraînement
        end_time = time.time()
        training_time = end_time - start_time

        # Enregistrement des résultats dans MLFlow
        mlflow.log_metric("Train Accuracy", train_accuracy)
        mlflow.log_metric("Validation Accuracy", valid_accuracy)
        mlflow.log_metric("Test Accuracy", test_accuracy)
        mlflow.log_metric("Train AUC", train_auc)
        mlflow.log_metric("Validation AUC", valid_auc)
        mlflow.log_metric("Test AUC", test_auc)
        mlflow.log_metric("Training Time", training_time)

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")

        # Sauvegarde du modèle avec un exemple d'entrée pour la signature
        input_example = X_train[0:1]  # Utiliser un exemple des données d'entrée (1er exemple)
        mlflow.sklearn.log_model(model, f"{model_name}_model", input_example=input_example)
        
        # Affichage du temps d'exécution
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Récupérer l'ID du run actif
        run_id = run.info.run_id
        
    return {
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
        "train_auc": train_auc,
        "valid_auc": valid_auc,
        "test_auc": test_auc,
        "training_time": training_time,
        "run_id": run_id
    }

# Sélectionner un sous-ensemble de 16000 tweets
subset_data = data.sample(n=16000, random_state=42)

# Diviser en Train, Validation, et Test
X_train, X_temp, y_train, y_temp = train_test_split(subset_data[['cleaned_text_lemmatized', 'cleaned_text_stemmed']], subset_data['target'], test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Nettoyage des NaN dans les colonnes de texte
X_train['cleaned_text_lemmatized'] = X_train['cleaned_text_lemmatized'].fillna('')
X_train['cleaned_text_stemmed'] = X_train['cleaned_text_stemmed'].fillna('')
X_valid['cleaned_text_lemmatized'] = X_valid['cleaned_text_lemmatized'].fillna('')
X_valid['cleaned_text_stemmed'] = X_valid['cleaned_text_stemmed'].fillna('')
X_test['cleaned_text_lemmatized'] = X_test['cleaned_text_lemmatized'].fillna('')
X_test['cleaned_text_stemmed'] = X_test['cleaned_text_stemmed'].fillna('')

# Transformation des textes lemmatisés en vecteurs TF-IDF
vectorizer_lemmatized = TfidfVectorizer(max_features=5000)
X_train_tfidf_lemmatized = vectorizer_lemmatized.fit_transform(X_train['cleaned_text_lemmatized'])
X_valid_tfidf_lemmatized = vectorizer_lemmatized.transform(X_valid['cleaned_text_lemmatized'])
X_test_tfidf_lemmatized = vectorizer_lemmatized.transform(X_test['cleaned_text_lemmatized'])

# Sauvegarder le vectorizer lemmatisé
joblib.dump(vectorizer_lemmatized, 'tfidf_vectorizer_lemmatized.pkl')

# Transformation des textes stemmés en vecteurs TF-IDF
vectorizer_stemmed = TfidfVectorizer(max_features=5000)
X_train_tfidf_stemmed = vectorizer_stemmed.fit_transform(X_train['cleaned_text_stemmed'])
X_valid_tfidf_stemmed = vectorizer_stemmed.transform(X_valid['cleaned_text_stemmed'])
X_test_tfidf_stemmed = vectorizer_stemmed.transform(X_test['cleaned_text_stemmed'])

# Sauvegarder le vectorizer stemmé
joblib.dump(vectorizer_stemmed, 'tfidf_vectorizer_stemmed.pkl')

# Initialiser le modèle de régression logistique
model = LogisticRegression(max_iter=1000)

# Définir la grille des hyperparamètres pour la recherche de paramètres via GridSearchCV
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'saga']
}

# Appliquer GridSearchCV pour le lemmatisation
grid_search_lemmatized = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search_lemmatized.fit(X_train_tfidf_lemmatized, y_train)

# Afficher les meilleurs hyperparamètres trouvés pour le lemmatisation
print(f"Best hyperparameters for Lemmatization: {grid_search_lemmatized.best_params_}")

# Utiliser le modèle avec les meilleurs hyperparamètres trouvés pour le lemmatisation
best_model_lemmatized = grid_search_lemmatized.best_estimator_

# Enregistrer les résultats de l'expérience dans MLFlow pour le lemmatisation avec un nom explicite
params_lemmatized = grid_search_lemmatized.best_params_
result_lemmatized = log_mlflow_experiment(best_model_lemmatized, X_train_tfidf_lemmatized, y_train, X_valid_tfidf_lemmatized, y_valid, X_test_tfidf_lemmatized, y_test, "LogisticRegression_lemmatized", params=params_lemmatized)

# Appliquer GridSearchCV pour le stemming
grid_search_stemmed = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
grid_search_stemmed.fit(X_train_tfidf_stemmed, y_train)

# Afficher les meilleurs hyperparamètres trouvés pour le stemming
print(f"Best hyperparameters for Stemming: {grid_search_stemmed.best_params_}")

# Utiliser le modèle avec les meilleurs hyperparamètres trouvés pour le stemming
best_model_stemmed = grid_search_stemmed.best_estimator_

# Enregistrer les résultats de l'expérience dans MLFlow pour le stemming avec un nom explicite
params_stemmed = grid_search_stemmed.best_params_
result_stemmed = log_mlflow_experiment(best_model_stemmed, X_train_tfidf_stemmed, y_train, X_valid_tfidf_stemmed, y_valid, X_test_tfidf_stemmed, y_test, "LogisticRegression_stemmed", params=params_stemmed)

# Comparer les résultats des deux modèles
results_df = pd.DataFrame([result_lemmatized, result_stemmed])

# Trier les modèles par accuracy du test pour déterminer le meilleur modèle
best_model_result = results_df.sort_values(by="test_accuracy", ascending=False).iloc[0]

# Afficher le meilleur modèle
print(f"Best Model: {best_model_result['model_name']}")
print(f"Test Accuracy: {best_model_result['test_accuracy']:.4f}")
print(f"Test AUC: {best_model_result['test_auc']:.4f}")
print(f"Training Time: {best_model_result['training_time']:.2f} seconds")

# Charger le meilleur modèle en fonction du nom
best_model_uri = f"runs:/{best_model_result['run_id']}/{best_model_result['model_name']}_model"
best_model = mlflow.sklearn.load_model(best_model_uri)

# Vérifier si le vectorizer est bien chargé
if best_model_result['model_name'] == "LogisticRegression_lemmatized":
    vectorizer = joblib.load('tfidf_vectorizer_lemmatized.pkl')
    X_test_best_model = X_test_tfidf_lemmatized
else:
    vectorizer = joblib.load('tfidf_vectorizer_stemmed.pkl')
    X_test_best_model = X_test_tfidf_stemmed

# Transformer le texte d'exemple
df = pd.DataFrame({'text': X_test['cleaned_text_lemmatized'][:1] if best_model_result['model_name'] == "LogisticRegression_lemmatized" else X_test['cleaned_text_stemmed'][:1]})

# Appliquer la transformation du vectorizer et prédire
vectorized_text = vectorizer.transform(df['text'])
prediction = best_model.predict(vectorized_text)

# Afficher la prédiction
print(f"Predicted Class: {prediction[0]}")


# In[ ]:




