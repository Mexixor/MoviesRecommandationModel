import time
import sys
import cherrypy
import os
from cheroot.wsgi import Server as WSGIServer
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from app import create_app

def init_spark_context():
    # load spark context
    conf = SparkConf().setAppName("movie_recommendation-server")
    # IMPORTANT: pass aditional Python modules to each worker
    sc = SparkContext(conf=conf, pyFiles=['engine.py', 'app.py'])
    return sc

def run_server(app):
    # Set the configuration of the web server
    cherrypy.tree.graft(app.wsgi_app, '/')
    cherrypy.config.update({'engine.autoreload.on': True,
                            'log.screen': True,
                            'server.socket_port': 5432,
                            'server.socket_host': '127.0.0.1'})
    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    # Init spark context and load libraries
    sc = init_spark_context()
    movies_set_path = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/DELL/SPARK-MOVIE/ml-latest/movies.csv"
    ratings_set_path = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/DELL/SPARK-MOVIE/ml-latest/ratings.csv"

    app = create_app(sc, movies_set_path, ratings_set_path)
 
    # start web server
    run_server(app)