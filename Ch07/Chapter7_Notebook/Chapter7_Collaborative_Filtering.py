# Databricks notebook source
# MAGIC %md
# MAGIC # Collaborative FIltering

# COMMAND ----------

# The below code was executed on DataBricks Community Edition. 
# The version information is provided for your reference below:
# 14.3 LTS (includes Apache Spark 3.5.0, Scala 2.12)
# Driver Type: Community Optimized - 15.3 GM Memory, 2 Cores, 1 DBU

# COMMAND ----------

# Import Sparksession
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("CF").getOrCreate()

# COMMAND ----------

# Print PySpark and Python versions
import sys
print('Python version: '+sys.version)
print('Spark version: '+spark.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# Read data
file_location = "/FileStore/tables/cf_data.csv"
file_type = "csv"
infer_schema = "false"
first_row_is_header = "true"


df = spark.read.format(file_type)\
.option("inferSchema", infer_schema)\
.option("header", first_row_is_header)\
.load(file_location)


# COMMAND ----------

# Print Metadata
df.printSchema()

# COMMAND ----------

#  Count data
df.count()
print('The total number of records in the credit card dataset are '+str(df.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC # Import appropriate libraries
# MAGIC

# COMMAND ----------

# Import appropriate libraries
from pyspark.sql.types import *
import pyspark.sql.functions as sql_fun
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model building

# COMMAND ----------

# Casting variables
int_vars=['userId','movieId']
for column in int_vars:
	df=df.withColumn(column,df[column].cast(IntegerType()))
float_vars=['rating']
for column in float_vars:
	df=df.withColumn(column,df[column].cast(FloatType()))

(training, test) = df.randomSplit([0.8, 0.2])

als = ALS(rank=15,maxIter=2, regParam=0.01, 
          userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=False) 
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# COMMAND ----------

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs.count()
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.count()

# COMMAND ----------

userRecs_df = userRecs.toPandas()
print(userRecs_df.shape)

movieRecs_df = movieRecs.toPandas()
print(movieRecs_df.shape)

# COMMAND ----------

display(userRecs_df.head())

# COMMAND ----------

