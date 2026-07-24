# Image autonome : diffusée aux serveurs partenaires, mise à jour uniquement
# via `docker pull` (pas de port d'admin, pas d'écriture entrante).
#
# IMPORTANT : ce Dockerfile se build depuis la RACINE du repo, pas depuis api/ :
#     docker build -f api/Dockerfile -t xrayindexer-api:0.2.0 .
# Ainsi les chemins COPY ci-dessous pointent vers les sources UNIQUES du repo
# (src/xray_detector, data/models) au lieu d'en dupliquer une copie dans api/.
# Un .dockerignore à la racine restreint le contexte de build (sinon data/raw
# ~102 Go partirait dans le contexte).
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le pipeline d'analyse ET gateway_client.py (qui vit dans ce package) : source
# unique du repo, copiée telle quelle, aucune réécriture ni duplication.
COPY src/xray_detector ./xray_detector

# Modèle(s) entraîné(s), bakés dans l'image (un par minerai cible), depuis leur
# source unique data/models/ — on ne copie que les .joblib de prod (pas les .bak).
COPY data/models/anomaly_iforest_ancient_debris.joblib \
     data/models/anomaly_iforest_diamond.joblib \
     ./models/

COPY api/main.py .

ENV PYTHONPATH=/app
EXPOSE 8000

# 0.0.0.0 dans le container : c'est `-p 127.0.0.1:8000:8000` au `docker run`
# (voir docker-compose.yml) qui restreint réellement l'exposition au host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
