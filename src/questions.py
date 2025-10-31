from pyspark.sql.functions import col, dayofweek, avg, to_date, row_number, rank
from pyspark.sql.window import Window
from pyspark.sql.functions import broadcast

def filter_questions(df):
    print("\n--- Filter запит 1: поїздки >10 миль і чайові > 5")
    df.filter((col("trip_distance") > 10) & (col("tip_amount") > 5)) \
      .select("tpep_pickup_datetime", "vendor_id", "trip_distance", "tip_amount", "total_amount", "passenger_count") \
      .limit(10).show(truncate=False)

    print("\n--- Filter запит 2: вихідні та >2 пасажири")
    df.filter(dayofweek(col("tpep_pickup_datetime")).isin([1, 7]) &
              (col("passenger_count") > 2)) \
      .select("tpep_pickup_datetime", "vendor_id", "passenger_count", "trip_distance", "total_amount") \
      .limit(10).show(truncate=False)

    print("\n--- Filter запит 3: total_amount > 30 і passenger_count > 3")
    df.filter((col("total_amount") > 30) & (col("passenger_count") > 3)) \
      .select("tpep_pickup_datetime", "vendor_id", "passenger_count", "trip_distance",
              "fare_amount", "tip_amount", "total_amount") \
      .limit(10).show(truncate=False)


def groupby_questions(df):
    print("\n--- GroupBy запит 1: середня total і tip по vendor_id")
    df.groupBy("vendor_id") \
      .agg(avg("total_amount").alias("avg_total"), avg("tip_amount").alias("avg_tip")) \
      .orderBy("vendor_id") \
      .show(truncate=False)

    print("\n--- GroupBy запит 2: середня total по кількості пасажирів")
    df.groupBy("passenger_count") \
      .agg(avg("total_amount").alias("avg_total")) \
      .orderBy("passenger_count") \
      .show(truncate=False)


def join_questions_safe(df, vendor_lookup, payment_lookup):
    # --- Join Question 1: Найдовша поїздка по vendor (використовує маленький lookup) ---
    from pyspark.sql.functions import col
    from pyspark.sql.window import Window
    from pyspark.sql.functions import rank, broadcast

    window_vendor = Window.partitionBy("vendor_id").orderBy(col("trip_distance").desc())
    df_top_vendor = df.withColumn("rank_trip", rank().over(window_vendor)) \
                      .filter(col("rank_trip") == 1)

    df_top_vendor_named = df_top_vendor.join(broadcast(vendor_lookup), on="vendor_id", how="left")

    print("\n--- Join запит 1: Найдовша поїздка по vendor ---")
    df_top_vendor_named.select("vendor_id", "vendor_name", "tpep_pickup_datetime",
                               "trip_distance", "fare_amount", "tip_amount", "total_amount") \
                       .show(truncate=False)

    # --- Join Question 2: Найбільша total_amount по payment_type (з lookup) ---
    window_payment = Window.partitionBy("payment_type").orderBy(col("total_amount").desc())
    df_top_payment = df.withColumn("rank_total", rank().over(window_payment)) \
                       .filter(col("rank_total") == 1)

    df_top_payment_named = df_top_payment.join(broadcast(payment_lookup), on="payment_type", how="left")

    print("\n--- Join запит 2: Найбільша total_amount по payment_type ---")
    df_top_payment_named.select("payment_type", "payment_name", "tpep_pickup_datetime",
                                "vendor_id", "fare_amount", "tip_amount", "total_amount") \
                        .show(truncate=False)


def join_and_window_questions_safe(df, vendor_lookup, payment_lookup):
    # --- Window Question 1: Топ-3 найдовші поїздки по кожному vendor ---
    window_vendor = Window.partitionBy("vendor_id").orderBy(col("trip_distance").desc())
    df_top3_vendor = df.withColumn("rank_trip", rank().over(window_vendor)) \
                        .filter(col("rank_trip") <= 3)

    df_top3_vendor_named = df_top3_vendor.join(broadcast(vendor_lookup), on="vendor_id", how="left")

    print("\n--- Window запит 1: Топ-3 найдовші поїздки по кожному vendor ---")
    df_top3_vendor_named.select("vendor_id", "vendor_name", "tpep_pickup_datetime",
                                "trip_distance", "fare_amount", "tip_amount", "total_amount", "rank_trip") \
                        .orderBy("vendor_id", "rank_trip") \
                        .limit(20).show(truncate=False)

    # --- Window Question 2: Топ-5 по total_amount на кожен день ---
    df_day = df.withColumn("day", to_date(col("tpep_pickup_datetime")))
    window_day = Window.partitionBy("day").orderBy(col("total_amount").desc())
    df_top5_day = df_day.withColumn("rank_total", row_number().over(window_day)) \
                        .filter(col("rank_total") <= 5)

    print("\n--- Window запит 2: Топ-5 по total_amount на день ---")
    df_top5_day.select("day", "tpep_pickup_datetime", "vendor_id", "trip_distance",
                       "fare_amount", "tip_amount", "total_amount", "rank_total") \
                .orderBy("day", "rank_total") \
                .limit(30).show(truncate=False)
