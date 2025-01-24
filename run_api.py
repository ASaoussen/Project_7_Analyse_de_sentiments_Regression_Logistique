#!/usr/bin/env python
# coding: utf-8

import subprocess
import time

# Lancer l'API FastAPI en arrière-plan
subprocess.Popen(["uvicorn", "API_prediction_sentiment:app", "--host", "127.0.0.1", "--port", "8080", "--log-level", "debug"])

# Attendre un peu pour vous assurer que l'API a démarré
time.sleep(5)  # attendre 5 secondes pour donner à l'API le temps de démarrer



#




