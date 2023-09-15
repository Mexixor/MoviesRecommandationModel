from DataLoader import loadParquet
from DataExplorer import DataExplorer
from MovieRecommendationModel import MovieRecommendationModel

from pyspark.sql import SparkSession


def printInfo(links_df, movies_df, ratings_df, tags_df):
    """ print info """

    print("=================")
    print("LINKS :")
    print("=================\n")
    links_df.printSchema()
    print("\n=================")
    print("MOVIES :")
    print("=================\n")
    movies_df.printSchema()
    print("\n=================")
    print("RATINGS :")
    print("=================\n")
    ratings_df.printSchema()
    print("\n=================")
    print("TAGS :")
    print("=================\n")
    tags_df.printSchema()

if __name__ == '__main__':
    """main program"""
    
    spark = SparkSession \
        .builder \
        .appName("Movies recommandations") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()


    links_df, movies_df, ratings_df, tags_df = loadParquet(spark)

    # printInfo(links_df, movies_df, ratings_df, tags_df)

    dataExplo = DataExplorer(links_df, movies_df, ratings_df, tags_df)

    dataExplo.getNumberOfRatings().show()

    dataExplo.getAllGenres()

    dataExplo.get_movies_().show(5)

    dataExplo.getTopRatedMovies("Comedy").show(5)

    dataExplo.getMostRatedMovies("Comedy").show(10)


    model = MovieRecommendationModel(spark)

    movie_ratings = ratings_df.select(["userID","movieId","rating"])

    model.trainModel(movie_ratings, 5, 30, 0.05,"toto")
    model.runCrossValidation(movie_ratings)
