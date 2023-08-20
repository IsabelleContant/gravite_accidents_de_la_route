# Démarrer les conteneurs Docker
docker-compose up -d

# Attendre quelques secondes pour permettre aux conteneurs de démarrer
Start-Sleep -Seconds 5

# Ouvrir les pages web dans le navigateur par défaut
Start-Process "http://localhost:8000"
Start-Process "http://localhost:8501"
