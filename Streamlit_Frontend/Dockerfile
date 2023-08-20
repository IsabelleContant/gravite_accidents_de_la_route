# Utilisez l'image officielle de Python 3.10
FROM python:3.10-slim

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez les fichiers de l'application dans le répertoire de travail
COPY requirements.txt app/requirements.txt
COPY . /app

# Installez les dépendances de l'application
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port 8501
EXPOSE 8501

# Exécutez l'application
CMD ["streamlit", "run", "01_🏡_Homepage.py"]


