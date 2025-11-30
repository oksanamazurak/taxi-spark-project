from pyspark.sql import SparkSession
from src.data_extraction import load_taxi_data
from src.transform import *
from src.questions import *
from pyspark.sql import Row
import sys
from src.linregression import train_and_evaluate_linear_regression
from src.logregression import train_and_evaluate_logistic_regression
from src.GBTRegressor import train_and_evaluate_gbt_regression 
from src.GBTClassifier import train_and_evaluate_gbt_classification
from src.RFRegressor import train_and_evaluate_rf_regression
from src.RFClassifier import train_and_evaluate_rf_classification
from src.evaluation import save_classification_comparison, save_regression_comparison

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("NYC Taxi Data Extraction") \
        .master("spark://spark-master:7077") \
        .config("spark.executorEnv.PYSPARK_PYTHON", "/usr/bin/python3") \
        .config("spark.pyspark.python", "/usr/bin/python3") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")
    spark.sparkContext.setLogLevel("WARN")

    file_path = "/app/data/yellow_tripdata_2015-01.csv"

    df = load_taxi_data(spark, file_path)

    print(f"Кількість рядків у DataFrame: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")

    # # Загальна інформація
    # get_dataset_info(df)
    #
    # # Аналіз пропущених значень
    # check_missing_values(df)

    # Видаляємо пропуски у колонці 'improvement_surcharge'
    df = remove_missing_in_column(df, 'improvement_surcharge')

    # Очищення комбіноване
    df = clean_taxi_data(df)

    # # Далі статистика, викиди і решта аналізу
    # get_numeric_stats(df)

    # # Статистика для числових стовпців
    # numeric_cols = list_numeric_columns(df)
    # print("\nЧислові колонки:", numeric_cols)
    #
    # stats = compute_column_stats(df, numeric_cols)
    # print("\n--- Mean / Stddev / Count non-null для числових колонок ---")
    # for col_name, (mean_v, std_v, cnt) in stats.items():
    #     print(f"{col_name}: mean={mean_v}, stddev={std_v}, non_null_count={cnt}")
    #
    # # Звіт по викидам (z-score)
    # out_summary = show_outliers_summary(df, cols=list_numeric_columns(df), z_thresh=3.0)
    # print("\n--- Підсумок по викидам (z-score > 3.0) ---")
    # out_summary.show(truncate=False)

    # --- Прапорці викидів для основних колонок ---
    cols_to_flag = [c for c in ['trip_distance', 'fare_amount'] if c in df.columns]
    if cols_to_flag:
        # df = add_outlier_flags(df, cols=cols_to_flag, z_thresh=3.0)
        # print("\nПоказати кілька рядків з прапорцями викидів:")
        # df.select(cols_to_flag + [f"{c}_is_outlier" for c in cols_to_flag]).show(10, truncate=False)

        # Видалити рядки, де хоча б одна з вибраних колонок є викидом
        before = df.count()
        df = remove_outliers(df, cols=cols_to_flag, z_thresh=3.0, how='any')
        after = df.count()
        print(f"\nРядків до очистки: {before}, після видалення викидів: {after}")

    # --- Зменшення вибірки для тренування логістичної регресії ---
    df_sample = df.sample(withReplacement=False, fraction=0.2, seed=42)
    print(f"\nВикористовується {df_sample.count()} рядків для класифікації (20% від усіх)")

    # Регресія (результати в терміналі, без збереження)
    reg_results = train_and_evaluate_linear_regression(
        df,
        feature_cols=['passenger_count', 'trip_distance'],
        categorical_cols=['vendor_id', 'payment_type'],
        label_col='total_amount',
        save=False,
        use_tvs=False
    )
    reg_results.show(truncate=False)

    gbt_results = train_and_evaluate_gbt_regression(
        df_sample,
        feature_cols=['passenger_count', 'trip_distance'],
        categorical_cols=['vendor_id', 'payment_type'],
        label_col='total_amount',
        save=False,
        use_tvs=False
    )
    gbt_results.show(truncate=False)

    rf_reg_results = train_and_evaluate_rf_regression(
        df_sample,
        feature_cols=['passenger_count', 'trip_distance'],
        categorical_cols=['vendor_id', 'payment_type'],
        label_col='total_amount',
        save=False,
        use_tvs=False
    )
    rf_reg_results.show(truncate=False)

    # Класифікація
    print("\nМультикласова Класифікація: Категорія Поїздки")
    # Класи: 0 (Short), 1 (Medium), 2 (Long)

    clf_results = train_and_evaluate_logistic_regression(
        df_sample,
        feature_cols=[
            'passenger_count',
            'total_amount',  # Ціна сильно корелює з відстанню
            'tolls_amount',  # Платні дороги часто означають довгу поїздку
            'tip_amount'
        ],
        categorical_cols=[
            'vendor_id',
            'payment_type'
        ],
        label_col='trip_category',  # Назва нової колонки з класами
        save=False
    )
    clf_results.show(truncate=False)

    print("\n---> GBT Classifier:")
    clf_gbt_results = train_and_evaluate_gbt_classification(
        df_sample,
        feature_cols=['passenger_count', 'total_amount', 'tolls_amount', 'tip_amount'],
        categorical_cols=['vendor_id', 'payment_type'],
        label_col='trip_category', # Використає вже існуючу колонку
        save=False,
    )
    clf_gbt_results.show(truncate=False)

    print("\n---> Random Forest Classifier:")
    clf_rf_results = train_and_evaluate_rf_classification(
        df_sample,
        feature_cols=['passenger_count', 'total_amount', 'tolls_amount', 'tip_amount'],
        categorical_cols=['vendor_id', 'payment_type'],
        label_col='trip_category',
        save=False
    )
    clf_rf_results.show(truncate=False)

    # --- Збереження метрик та побудова графіків ---
    # Регресія: збираємо доступні результати (перевіряємо, чи виконувались блоки)
    reg_results_list = []
    if 'reg_results' in locals():
        reg_results_list.append(reg_results)
    if 'gbt_results' in locals():
        reg_results_list.append(gbt_results)
    if 'rf_reg_results' in locals():
        reg_results_list.append(rf_reg_results)
    if reg_results_list:
        save_regression_comparison(reg_results_list, output_dir="/app/output")

    # Класифікація: збираємо доступні результати
    cls_results_list = []
    if 'clf_results' in locals():
        cls_results_list.append(clf_results)
    if 'clf_gbt_results' in locals():
        cls_results_list.append(clf_gbt_results)
    if 'clf_rf_results' in locals():
        cls_results_list.append(clf_rf_results)
    if cls_results_list:
        save_classification_comparison(cls_results_list, output_dir="/app/output")

    # get_numeric_stats(df)
    #
    # vendor_lookup = spark.createDataFrame([
    #     Row(vendor_id=1, vendor_name="Creative Mobile Technologies"),
    #     Row(vendor_id=2, vendor_name="VeriFone Inc")
    # ])
    # payment_lookup = spark.createDataFrame([
    #     Row(payment_type=1, payment_name="Credit card"),
    #     Row(payment_type=2, payment_name="Cash"),
    # ])
    #
    # filter_questions(df)
    # groupby_questions(df)
    # join_questions_safe(df, vendor_lookup, payment_lookup)
    # join_and_window_questions_safe(df, vendor_lookup, payment_lookup)

    spark.stop()
