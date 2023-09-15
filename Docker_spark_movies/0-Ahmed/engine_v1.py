from pyspark.sql.types import *
from pyspark.sql.functions import explode, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkContext

class RecommendationEngine:

    def create_user(self, user_id=None):
        if user_id is None:
            self.max_user_identifier += 1
            return self.max_user_identifier
        else:
            if user_id > self.max_user_identifier:
                self.max_user_identifier = user_id
            return user_id

    def is_user_known(self, user_id):
        return user_id is not None and user_id <= self.max_user_identifier

    def get_movie(self, movie_id=None):
        if movie_id is None:
            random_movie = self.movies_df.sample(n=1)
            return random_movie[["movieId", "title"]]
        else:
            movie_info = self.movies_df.filter(col("movieId") == movie_id)[["movieId", "title"]]
            return movie_info

    def get_ratings_for_user(self, user_id):
        user_ratings = self.ratings_df.filter(col("userId") == user_id)[["movieId", "userId", "rating"]]
        return user_ratings

    def add_ratings(self, user_id, ratings):
        new_ratings = [(user_id, movie_id, rating) for movie_id, rating in ratings]
        new_ratings_df = self.spark.createDataFrame(new_ratings, ["userId", "movieId", "rating"])
        updated_ratings_df = self.ratings_df.union(new_ratings_df)
        train_data, test_data = updated_ratings_df.randomSplit([0.8, 0.2], seed=42)
        self.__train_model(train_data)

    def predict_rating(self, user_id, movie_id):
        prediction_data = [(user_id, movie_id)]
        prediction_df = self.spark.createDataFrame(prediction_data, ["userId", "movieId"])
        predictions = self.model.transform(prediction_df)
        if predictions.isEmpty():
            return -1
        else:
            predicted_rating = predictions.select("prediction").collect()[0][0]
            return predicted_rating

    def recommend_for_user(self, user_id, nb_movies):
        user_df = self.spark.createDataFrame([(user_id,)], ["userId"])
        recommendations = self.model.recommendForUserSubset(user_df, nb_movies)
        recommended_movies = recommendations.select("recommendations.movieId")
        recommended_movies_details = recommended_movies.join(self.movies_df, recommended_movies.movieId == self.movies_df.movieId, "inner").select(self.movies_df["title"])
        return recommended_movies_details

    def __train_model(self, train_data):
        als = ALS(rank=10, maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = als.fit(train_data)

    def __evaluate(self):
        predictions = self.model.transform(self.ratings_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print(f"Root Mean Squared Error (RMSE) = {rmse}")

    def __init__(self, sc, movies_set_path, ratings_set_path):
        self.spark = SparkSession.builder.appName("RecommendationEngine").getOrCreate()
        self.sc = sc
    
        self.movies_set_path = movies_set_path
        self.ratings_set_path = ratings_set_path        

        # Charger les ensembles de données
        self.movies_df = self.spark.read.csv(movies_set_path, header=True, inferSchema=True)
        self.ratings_df = self.spark.read.csv(ratings_set_path, header=True, inferSchema=True)
        # Initialiser l'identifiant utilisateur maximum
        self.max_user_identifier = 0

        als = ALS(rank=10, maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating")
        self.model = als.fit(self.ratings_df)
        