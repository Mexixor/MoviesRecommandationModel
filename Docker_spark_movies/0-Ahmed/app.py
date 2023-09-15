from flask import Flask, Blueprint, render_template, jsonify, request
import json
import findspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from engine_v1 import RecommendationEngine



# Création d'un Blueprint "main"
main = Blueprint("main", __name__)
# Initialisation de SparkContext
findspark.init()

@main.route("/", methods=["GET", "POST", "PUT"])
def home():
    return render_template("index.html")

# Route pour obtenir les détails d'un film par son ID
@main.route("/movies/<int:movie_id>", methods=["GET"])
def get_movie(movie_id):
    # Récupérer les détails du film avec l'ID spécifié
    movie_details = recommender.get_movie(movie_id)
    # Vérifier si le film existe
    if movie_details is not None:
        # Convertir le DataFrame en un format JSON compatible
        movie_details_json = movie_details.toJSON().collect()
        return json.dumps(movie_details_json), 200, {'Content-Type': 'application/json'}
    else:
        # Si le film n'existe pas, retourner une réponse 404
        return json.dumps({"message": "Film non trouvé"}), 404, {'Content-Type': 'application/json'}

# Route pour ajouter de nouvelles évaluations pour les films
@main.route("/newratings/<int:user_id>", methods=["POST"])
def new_ratings(user_id):
    # Vérifier si l'utilisateur existe déjà
    if not recommender.is_user_known(user_id):
        # Si l'utilisateur n'existe pas, créez-le
        user_id = recommender.create_user(user_id)
    # Récupérer les évaluations depuis la requête
    ratings = request.get_json()
    # Ajouter les évaluations au moteur de recommandation
    recommender.add_ratings(user_id, ratings)
    # Renvoyer l'identifiant de l'utilisateur s'il est nouveau, sinon renvoyer une chaîne vide
    if not recommender.is_user_known(user_id):
        return str(user_id)
    else:
        return ""

# Route pour ajouter des évaluations à partir d'un fichier
@main.route("/<int:user_id>/ratings", methods=["POST"])
def add_ratings(user_id):
    # Code pour récupérer le fichier téléchargé depuis la requête
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Lire les données du fichier et ajouter-les au moteur de recommandation
        ratings = []

        with open(uploaded_file.filename, 'r') as file:
            for line in file:
                movie_id, rating = line.strip().split(',')
                ratings.append((int(movie_id), float(rating)))

        recommender.add_ratings(user_id, ratings)

        # Renvoyer un message indiquant que le modèle de prédiction a été recalculé
        return "Le modèle de prédiction a été recalculé."
    else:
        return "Aucun fichier téléchargé."

# Route pour obtenir la note prédite d'un utilisateur pour un film
@main.route("/<int:user_id>/ratings/<int:movie_id>", methods=["GET"])
def movie_ratings(user_id, movie_id):
    # Prédire la note de l'utilisateur pour le film spécifié
    predicted_rating = recommender.predict_rating(user_id, movie_id)
    # Vérifier si une prédiction a pu être faite
    if predicted_rating != -1:
        # Si oui, retourner la note prédite au format texte
        return f"La note prédite pour le film {movie_id} est : {predicted_rating}"
    else:
        # Sinon, retourner un message indiquant qu'aucune prédiction n'a pu être faite
        return "Impossible de faire une prédiction pour cette combinaison utilisateur/film."

# Route pour obtenir les meilleures évaluations recommandées pour un utilisateur
@main.route("/<int:user_id>/recommendations/<int:nb_movies>", methods=["GET"])
def get_recommendations(user_id, nb_movies):
    # Obtenir les meilleures recommandations pour l'utilisateur donné
    recommended_movies = recommender.recommend_for_user(user_id, nb_movies)
    # Convertir les recommandations en format JSON
    recommendations_json = recommended_movies.toJSON().collect()
    return json.dumps(recommendations_json), 200, {'Content-Type': 'application/json'}

# Route pour obtenir les évaluations d'un utilisateur
@main.route("/ratings/<int:user_id>", methods=["GET"])
def get_ratings_for_user(user_id):
    # Obtenir les évaluations de l'utilisateur spécifié
    user_ratings = recommender.get_ratings_for_user(user_id)

    # Convertir les évaluations en format JSON
    ratings_json = user_ratings.toJSON().collect()

    return json.dumps(ratings_json), 200, {'Content-Type': 'application/json'}


# Fonction pour créer l'application Flask
def create_app(sc, movies_set_path, ratings_set_path):
    global recommender
    # Initialiser le moteur de recommandation avec le contexte Spark et les jeux de données
    recommender = RecommendationEngine(sc, movies_set_path, ratings_set_path)
    # Créer une instance de l'application Flask
    app = Flask(__name__)
    # Enregistrez le Blueprint "main" dans l'application
    app.register_blueprint(main)
    # Configurer les options de l'application Flask
    # app.config['SECRET_KEY'] = 'ahmed'
    # Configurez les options de l'application Flask (si nécessaire)
    # app.config['OPTION_NAME'] = 'OPTION_VALUE'
    # Renvoyez l'application Flask créée
    return app