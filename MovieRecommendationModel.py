from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

class MovieRecommendationModel():
    def __init__(self, spark):
        """ init """
        self.spark_ = spark

    def trainModel(self, movie_ratings, maxIter, rank, regParam, seed = 42):
        """Build the recommendation model using ALS on the training data"""

        (training, test) = movie_ratings.randomSplit([.8, .2], seed)

        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(maxIter=maxIter, rank=rank, regParam=regParam, userCol='userID', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')

        # fit the ALS model to the training set
        self.model_=als.fit(training)

        predictions = self.model_.transform(test)
        evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
        rmse = evaluator.evaluate(predictions)
        print(rmse)

    def runCrossValidation(self, movie_ratings):
        """ cross validation """

        # initialize the ALS model
        als_model = ALS(userCol='userID', itemCol='movieId', ratingCol='rating', coldStartStrategy='drop')

        # create the parameter grid
        params = ParamGridBuilder().addGrid(als_model.regParam, [.01, .05, .1, .15]).addGrid(als_model.rank, [10, 50, 100, 150]).build()

        evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

        #instantiating crossvalidator estimator
        cv = CrossValidator(estimator=als_model, estimatorParamMaps=params, evaluator=evaluator, parallelism=4)
        best_model = cv.fit(movie_ratings)
        model = best_model.bestModel

        model.save("Model")