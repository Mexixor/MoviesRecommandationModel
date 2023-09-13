from pyspark.sql.functions import avg, count, desc, when, col

class DataExplorer():
    """ """
    def __init__(self, links_df, movies_df, ratings_df, tags_df):
        """ """
        self.links_ = links_df
        self.movies_ = movies_df
        self.ratings_ = ratings_df
        self.tags_ = tags_df

        self.ratingsAndNames_ = self.ratings_.join(self.movies_,on="movieId", how="inner")

        self.set_genres_ = self.__extractListOfGenres()
        self.__buildColumnsGenres()

    def getNumberOfRatings(self):
        """Number of ratings"""
        # #with pandas dataframe: 
        # return self.ratings_.value_counts("userId")

        #with spark dataframe
        return self.ratings_.groupBy("userId").count().sort("count",ascending=False)

    def getAllGenres(self):
        """return set of genres """
        return self.set_genres_

    def get_links_(self):
        """getter links_"""
        return self.links_

    def get_movies_(self):
        """getter movies_"""
        return self.movies_

    def get_ratings_(self):
        """getter ratings_"""
        return self.ratings_

    def get_tags_(self):
        """getter tags_"""
        return self.tags_

    def getTopRatedMovies(self, genre):
        """ get the TOP rated selected genre Movies (not aggregated)"""

        return self.ratingsAndNames_.groupBy(["movieId","genres","title"]).mean(
            "rating").where(self.ratingsAndNames_.genres.contains(genre)).sort(
                "avg(rating)", ascending=False)


    def __extractListOfGenres(self):
        """extract list of genres from tags data"""

        # # with pandas dataframe : 
        # uniques = self.movies_.genres.unique()
        # set_genre = set()

        # for val in uniques:
        #     for elem in val.split("|"):
        #         set_genre.add(elem)

        # #with spark dataframe
        val_distincs = self.movies_.select("genres").distinct().collect()
        set_genre = set()

        for row in val_distincs:
            for elem in row["genres"].split("|"):
                set_genre.add(elem)

        return set_genre

    def __buildColumnsGenres(self):
        """ explode columns tags in multiples columns with 1 or 0 if film is tagged"""

        ##with pandas dataframe
        # for genre_col in self.set_genres_:
        #     self.movies_[genre_col] = self.movies_["genres"].apply(lambda x: 1 if genre_col in x else 0)

        #with spark dataframe :
        for genre_col in self.set_genres_:
            self.movies_ = self.movies_.withColumn(genre_col, when(col("genres").contains(genre_col),1)
                .otherwise(0))