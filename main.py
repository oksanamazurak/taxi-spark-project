from pyspark.sql import SparkSession
from src.data_extraction import load_taxi_data
from src.transform import *
from src.questions import *
from pyspark.sql import Row
import sys

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC Taxi Data Extraction") \
        .master("spark://spark-master:7077") \
        .config("spark.executorEnv.PYSPARK_PYTHON", "/usr/bin/python3") \
        .config("spark.pyspark.python", "/usr/bin/python3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")
    spark.sparkContext.setLogLevel("WARN")

    file_path = "/app/data/yellow_tripdata_2015-01.csv"

    df = load_taxi_data(spark, file_path)

    print(f"Кількість рядків у DataFrame: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")

    # Загальна інформація
    get_dataset_info(df)

    # Аналіз пропущених значень
    check_missing_values(df)

    # Видаляємо пропуски у колонці 'improvement_surcharge'
    df = remove_missing_in_column(df, 'improvement_surcharge')

    # Очищення комбіноване
    df = clean_taxi_data(df)

    # Далі статистика, викиди і решта аналізу
    get_numeric_stats(df)

    # Статистика для числових стовпців
    numeric_cols = list_numeric_columns(df)
    print("\nЧислові колонки:", numeric_cols)

    stats = compute_column_stats(df, numeric_cols)
    print("\n--- Mean / Stddev / Count non-null для числових колонок ---")
    for col_name, (mean_v, std_v, cnt) in stats.items():
        print(f"{col_name}: mean={mean_v}, stddev={std_v}, non_null_count={cnt}")

    # Звіт по викидам (z-score)
    out_summary = show_outliers_summary(df, cols=list_numeric_columns(df), z_thresh=3.0)
    print("\n--- Підсумок по викидам (z-score > 3.0) ---")
    out_summary.show(truncate=False)

    # --- Прапорці викидів для основних колонок ---
    cols_to_flag = [c for c in ['trip_distance', 'fare_amount'] if c in df.columns]
    if cols_to_flag:
        df = add_outlier_flags(df, cols=cols_to_flag, z_thresh=3.0)
        print("\nПоказати кілька рядків з прапорцями викидів:")
        df.select(cols_to_flag + [f"{c}_is_outlier" for c in cols_to_flag]).show(10, truncate=False)

        # Видалити рядки, де хоча б одна з вибраних колонок є викидом
        before = df.count()
        df = remove_outliers(df, cols=cols_to_flag, z_thresh=3.0, how='any')
        after = df.count()
        print(f"\nРядків до очистки: {before}, після видалення викидів: {after}")

    get_numeric_stats(df)

    vendor_lookup = spark.createDataFrame([
        Row(vendor_id=1, vendor_name="Creative Mobile Technologies"),
        Row(vendor_id=2, vendor_name="VeriFone Inc")
    ])
    payment_lookup = spark.createDataFrame([
        Row(payment_type=1, payment_name="Credit card"),
        Row(payment_type=2, payment_name="Cash"),
    ])

    filter_questions(df)
    groupby_questions(df)
    join_questions_safe(df, vendor_lookup, payment_lookup)
    join_and_window_questions_safe(df, vendor_lookup, payment_lookup)

    spark.stop()