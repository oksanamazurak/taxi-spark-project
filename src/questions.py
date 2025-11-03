from pyspark.sql.functions import col, dayofweek, avg, to_date, row_number, rank, hour, minute, unix_timestamp, sum as spark_sum, count
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

    print("\n--- Filter запит 4: поїздки з total_amount між 20 і 50 та без чайових (tip_amount = 0) ---")
    df.filter((col("total_amount").between(20, 50)) & (col("tip_amount") == 0)) \
        .select("tpep_pickup_datetime", "vendor_id", "trip_distance", "tip_amount", "total_amount", "payment_type") \
        .limit(10).show(truncate=False)
    
    print("\n--- Filter запит 5: поїздки між 8:00-10:00 ранку, тривалість < 15 хвилин ---")
    df_with_duration = df.withColumn(
        "duration_minutes",
        (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
    )
    df_with_duration.filter(
        (hour(col("tpep_pickup_datetime")).between(8, 9)) &
        (col("duration_minutes") < 15) &
        (col("duration_minutes") > 0)
    ).select(
        "tpep_pickup_datetime", "tpep_dropoff_datetime", "duration_minutes",
        "vendor_id", "trip_distance", "total_amount"
    ).limit(10).show(truncate=False)

    print("\n--- Filter запит 6: відстань < 2 миль, але оплата > $20 ---")
    df.filter((col("trip_distance") < 2) & (col("total_amount") > 20)) \
        .select("tpep_pickup_datetime", "vendor_id", "trip_distance", "total_amount",
                "fare_amount", "extra", "tip_amount") \
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

    print("\n--- GroupBy запит 3: середня total_amount по payment_type (>5 миль) ---")
    df.filter(col("trip_distance") > 5) \
        .groupBy("payment_type") \
        .agg(avg("total_amount").alias("avg_total")) \
        .orderBy("payment_type") \
        .show(truncate=False)
    
    print("\n--- GroupBy запит 4: кількість поїздок і загальна сума по дню тижня ---")
    from pyspark.sql.functions import count
    df.groupBy(dayofweek(col("tpep_pickup_datetime")).alias("day_of_week")) \
        .agg(
            count("*").alias("trip_count"),
            spark_sum("total_amount").alias("total_revenue")
        ) \
        .orderBy("day_of_week") \
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

    print("\n--- Join запит 3: поїздки у вихідні з назвами vendor ---")
    df_weekend = df.filter(dayofweek(col("tpep_pickup_datetime")).isin([1, 7]))
    df_weekend_named = df_weekend.join(broadcast(vendor_lookup), on="vendor_id", how="left")
    df_weekend_named.select("vendor_name", "tpep_pickup_datetime", "trip_distance", "total_amount") \
        .limit(10).show(truncate=False)

    print("\n--- Join запит 4: середня сума чайових по payment_name ---")
    df_joined = df.join(broadcast(payment_lookup), on="payment_type", how="left")
    df_joined.groupBy("payment_name") \
        .agg(avg("tip_amount").alias("avg_tip")) \
        .orderBy("avg_tip", ascending=False) \
        .show(truncate=False)
    
    print("\n--- Join запит 5: середня оплата по районах (rate_code як proxy для зон) ---")
    zone_lookup = df.sparkSession.createDataFrame([
        (1, "Standard Rate Zone"),
        (2, "JFK Airport"),
        (3, "Newark Airport"),
        (4, "Nassau/Westchester"),
        (5, "Negotiated Fare Zone"),
        (6, "Group Ride Zone")
    ], ["rate_code", "zone_name"])
    
    df_with_zones = df.join(broadcast(zone_lookup), on="rate_code", how="left")
    df_with_zones.groupBy("zone_name") \
        .agg(avg("total_amount").alias("avg_payment")) \
        .orderBy("avg_payment", ascending=False) \
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

    print("\n--- Window запит 3: топ-3 найдовші поїздки (>20 total) ---")
    window_vendor = Window.partitionBy("vendor_id").orderBy(col("trip_distance").desc())
    df.filter(col("total_amount") > 20) \
        .withColumn("rank_trip", row_number().over(window_vendor)) \
        .filter(col("rank_trip") <= 3) \
        .select("vendor_id", "tpep_pickup_datetime", "trip_distance", "total_amount", "rank_trip") \
        .orderBy("vendor_id", "rank_trip") \
        .show(truncate=False)

    print("\n--- Window запит 4: топ-3 поїздки по кожному payment_type ---")
    window_payment = Window.partitionBy("payment_type").orderBy(col("total_amount").desc())
    df.withColumn("rank_total", rank().over(window_payment)) \
        .filter(col("rank_total") <= 3) \
        .select("payment_type", "tpep_pickup_datetime", "trip_distance", "total_amount", "rank_total") \
        .orderBy("payment_type", "rank_total") \
        .show(truncate=False)
    
    print("\n--- Window запит 5: кумулятивна сума заробітку по vendor_id ---")
    window_cumulative = Window.partitionBy("vendor_id").orderBy("tpep_pickup_datetime")
    df.withColumn("cumulative_earnings", spark_sum("total_amount").over(window_cumulative)) \
        .select("vendor_id", "tpep_pickup_datetime", "total_amount", "cumulative_earnings") \
        .orderBy("vendor_id", "tpep_pickup_datetime") \
        .limit(20).show(truncate=False)

    print("\n--- Window запит 6: ковзне середнє відстані (5 останніх поїздок) по vendor_id ---")
    window_moving_avg = Window.partitionBy("vendor_id") \
        .orderBy("tpep_pickup_datetime") \
        .rowsBetween(-4, 0)
    
    df.withColumn("moving_avg_distance", avg("trip_distance").over(window_moving_avg)) \
        .select("vendor_id", "tpep_pickup_datetime", "trip_distance", "moving_avg_distance") \
        .orderBy("vendor_id", "tpep_pickup_datetime") \
        .limit(20).show(truncate=False)
