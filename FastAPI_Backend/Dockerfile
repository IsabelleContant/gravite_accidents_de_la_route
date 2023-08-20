# Utilisez l'image officielle de Python 3.10
FROM python:3.10-slim

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les dépendances de l'application et les fichiers de l'application dans le répertoire de travail
COPY requirements.txt app/requirements.txt
COPY . /app

# Installez les dépendances de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port 8000
EXPOSE 8000

# Exécutez l'application
CMD ["uvicorn", "prophet_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]