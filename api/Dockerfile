# Image autonome : diffusée aux serveurs partenaires, mise à jour uniquement
# via `docker pull` (pas de port d'admin, pas d'écriture entrante).
#
# IMPORTANT : ce Dockerfile se build depuis la RACINE du repo, pas depuis api/ :
#     docker build -f api/Dockerfile -t xrayindexer-api:0.2.0 .
# Ainsi les chemins COPY ci-dessous pointent vers les sources UNIQUES du repo
# (src/xray_detector, data/models) au lieu d'en dupliquer une copie dans api/.
# Un .dockerignore à la racine restreint le contexte de build (sinon data/raw
# ~102 Go partirait dans le contexte).
#
# Version de patch épinglée (et non le tag flottant 3.11-slim) : reproductibilité
# de l'image livrée aux partenaires. Encore plus strict possible via un digest
# (@sha256:...), au prix d'un pull réseau pour le récupérer.
FROM python:3.11.15-slim

# Utilisateur non-privilégié dédié : rien ne tourne en root dans le container.
RUN useradd --system --uid 10001 --user-group \
    --home-dir /app --shell /usr/sbin/nologin appuser

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

# Le renderer reste une commande explicite du service Compose dedie : il ne
# s'expose pas par HTTP et lit le miroir en lecture seule.
COPY scripts/render_mining_3d.py ./scripts/render_mining_3d.py

# /data = point de montage du volume persistant (mirror.db). On le crée et on le
# donne à appuser DANS l'image : un volume nommé hérite de ce propriétaire à sa
# première création, donc appuser pourra y écrire le miroir sans tourner en root.
RUN mkdir -p /data /app/data \
    && ln -s /app/models /app/data/models \
    && chown -R appuser:appuser /app /data

ENV PYTHONPATH=/app
USER appuser
EXPOSE 8000

# 0.0.0.0 dans le container : c'est `-p 127.0.0.1:8000:8000` au `docker run`
# (voir docker-compose.yml) qui restreint réellement l'exposition au host.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
