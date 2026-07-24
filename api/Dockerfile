# Image autonome : diffusée aux serveurs partenaires, mise à jour uniquement
# via `docker pull` (pas de port d'admin, pas d'écriture entrante).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Votre code existant, copié tel quel : aucune réécriture de la logique
# d'analyse ni du client de synchro.
COPY xray_detector ./xray_detector
COPY gateway_client.py .

# Modèle(s) entraîné(s), bakés dans l'image (un par minerai cible).
COPY models ./models

COPY main.py .

ENV PYTHONPATH=/app
EXPOSE 8000

# 0.0.0.0 dans le container : c'est `-p 127.0.0.1:8000:8000` au `docker run`
# (voir docker-compose.yml) qui restreint réellement l'exposition au host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
