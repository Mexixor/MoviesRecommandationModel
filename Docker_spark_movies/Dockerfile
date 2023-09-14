# Utilisation d'une image de base avec Python et Spark préinstallés
FROM mrsipan/py-spark:latest

# Exposer le port sur lequel Flask va écouter (le port par défaut de Flask est 5000)
EXPOSE 5000

# Définir le répertoire de travail dans l'image Docker
WORKDIR /app

# Copiez le fichier requirements.txt dans l'image
COPY requirements.txt .

# Installez les dépendances Python requises
RUN pip install -r requirements.txt

# Copiez le reste de votre application Flask dans l'image
COPY . .

# Commande par défaut pour lancer Flask (ajustez cela en fonction de la structure de votre application)
CMD ["python", "app.py"]
