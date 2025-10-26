from pyspark.sql import SparkSession
from src.data_extraction import load_taxi_data

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC Taxi Data Extraction") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")

    file_path = "/app/data/yellow_tripdata_2015-01.csv"
    df = load_taxi_data(spark, file_path)

    print(f"Кількість рядків у DataFrame: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")

    spark.stop()
