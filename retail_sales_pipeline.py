# -*- coding: utf-8 -*-
"""
Retail Sales — PySpark Machine Learning Pipeline

This script:
1. Installs & configures Java and PySpark (for Colab environment)
2. Loads a retail sales dataset into Spark DataFrame
3. Performs data preprocessing (drop columns, handle date fields, create new features)
4. Encodes categorical features and scales numeric ones
5. Applies Gradient Boosted Tree Regressor to predict Total Amount
6. Evaluates model using RMSE on both train and test data

Original file generated from:
https://colab.research.google.com/drive/15sN8-EEpe8QLSud5wL_mgGbEZlxNdhcI
"""

# ----------------------------------------------------
# Install Java (required for PySpark) — for Google Colab
# ----------------------------------------------------
!apt-get install openjdk-11-jdk-headless -qq > /dev/null

# ----------------------------------------------------
# Set JAVA_HOME environment variable (required by PySpark)
# ----------------------------------------------------
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# ----------------------------------------------------
# Install PySpark
# ----------------------------------------------------
!pip install -q pyspark

# ----------------------------------------------------
# Start Spark session
# ----------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

spark = SparkSession.builder \
    .appName("MySparkProject") \
    .getOrCreate()

# ----------------------------------------------------
# Load dataset
# ----------------------------------------------------
df = spark.read.csv('/content/retail_sales_dataset.csv', header=True, inferSchema=True)

# Explore data structure
df.show()
df.printSchema()
df.count()
df.describe().show()

# Count missing values per column
df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

# Drop ID columns, not useful for prediction
df = df.drop('Customer ID', 'Transaction ID')

# ----------------------------------------------------
# FEATURE ENGINEERING: create new columns from Date
# ----------------------------------------------------
from pyspark.sql.functions import col, sum, avg, count, month, year, dayofweek, when, lit

# Convert Date column to timestamp
df = df.withColumn("Date", col("Date").cast("timestamp"))

# Extract date parts
df = df.withColumn("Year", year(col("Date"))) \
       .withColumn("Month", month(col("Date"))) \
       .withColumn("DayOfWeek", dayofweek(col("Date")))  # (1=Sunday, 7=Saturday)

# Categorize ages into age groups
df = df.withColumn(
    "AgeGroup",
    when(col("Age") < 18, "<18")
    .when((col("Age") >= 18) & (col("Age") < 30), "18-29")
    .when((col("Age") >= 30) & (col("Age") < 50), "30-49")
    .otherwise("50+")
)

# Flag for unusually large quantity purchases
df = df.withColumn("HighQuantityFlag", when(col("Quantity") > 10, 1).otherwise(0))

# Drop Date column because we extracted features from it
df = df.drop('Date')

# ----------------------------------------------------
# START BUILDING MACHINE LEARNING PIPELINE
# ----------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import log1p, expm1, col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ----------------------------------------------------
# Log-transform target column to reduce skewness
# ----------------------------------------------------
df = df.withColumn("LogTotalAmount", log1p(col("Total Amount")))

# ----------------------------------------------------
# Encode categorical columns using StringIndexer + OneHotEncoder
# ----------------------------------------------------
categorical_cols = ["Gender", "Product Category", "AgeGroup"]

# Convert categorical strings to numeric indexes
indexers = [StringIndexer(inputCol=c, outputCol=c + "Index") for c in categorical_cols]

# One-hot encode indexed columns
encoder = OneHotEncoder(
    inputCols=[c + "Index" for c in categorical_cols],
    outputCols=[c + "Vec" for c in categorical_cols]
)

# ----------------------------------------------------
# Assemble feature vector (numeric + encoded categorical)
# ----------------------------------------------------
numeric_cols = ["Age", "Quantity", "Price per Unit", "Year", "Month", "DayOfWeek", "HighQuantityFlag"]
encoded_cols = [c + "Vec" for c in categorical_cols]

feature_cols = numeric_cols + encoded_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Standardize numeric features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# ----------------------------------------------------
# Gradient Boosted Trees Regression model
# ----------------------------------------------------
gbt = GBTRegressor(featuresCol="scaledFeatures", labelCol="LogTotalAmount", maxIter=200, maxDepth=8)

# ----------------------------------------------------
# Pipeline: index → encode → assemble → scale → train model
# ----------------------------------------------------
pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, gbt])

# ----------------------------------------------------
# Split dataset into training and testing
# ----------------------------------------------------
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# ----------------------------------------------------
# Train the model
# ----------------------------------------------------
model = pipeline.fit(train_df)

# ----------------------------------------------------
# Predict on test set
# ----------------------------------------------------
predictions = model.transform(test_df)

# Convert prediction back from log scale
predictions = predictions.withColumn("prediction_orig", expm1(col("prediction")))

predictions.select("Total Amount", "prediction_orig").show(10)

# ----------------------------------------------------
# Model Evaluation: RMSE (Root Mean Squared Error)
# ----------------------------------------------------
evaluator = RegressionEvaluator(labelCol="Total Amount", predictionCol="prediction_orig", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# ----------------------------------------------------
# Compare Train vs. Test performance (to detect overfitting)
# ----------------------------------------------------
train_predictions = model.transform(train_df)
train_predictions = train_predictions.withColumn("prediction_orig", expm1(col("prediction")))

test_predictions = model.transform(test_df)
test_predictions = test_predictions.withColumn("prediction_orig", expm1(col("prediction")))

train_rmse = evaluator.evaluate(train_predictions)
test_rmse = evaluator.evaluate(test_predictions)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
