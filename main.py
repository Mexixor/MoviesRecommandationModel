from loadData import loadParquet
from dataExploration import DataExplorer

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