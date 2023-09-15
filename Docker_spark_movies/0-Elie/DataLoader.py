import pandas as pd
from pyspark.sql import SparkSession

def loadDataToParquet():
    """ transform csv to parquet files"""
    links_df = pd.read_csv("Data/links.csv")
    movies_df = pd.read_csv("Data/movies.csv")
    ratings_df = pd.read_csv("Data/ratings.csv")
    tags_df = pd.read_csv("Data/tags.csv")

    links_df.to_parquet("Data/links.parquet")
    movies_df.to_parquet("Data/movies.parquet")
    ratings_df.to_parquet("Data/ratings.parquet")
    tags_df.to_parquet("Data/tags.parquet")


def loadParquet(spark):
    """read parquet files """
    links_df = spark.read.parquet("Data/links.parquet")
    movies_df = spark.read.parquet("Data/movies.parquet")
    ratings_df = spark.read.parquet("Data/ratings.parquet")
    tags_df = spark.read.parquet("Data/tags.parquet")
    
    '''links_df.cache()
    movies_df.cache()
    ratings_df.cache()
    tags_df.cache()'''

    return links_df, movies_df, ratings_df, tags_df

if __name__ == '__main__':
    spark = SparkSession \
        .builder \
        .appName("Movies recommandations") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    # loadDataToParquet()

    links_df, movies_df, ratings_df, tags_df = loadParquet(spark)

    links_df.printSchema()
    movies_df.printSchema()
    ratings_df.printSchema()
    tags_df.printSchema()