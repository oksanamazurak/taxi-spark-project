from typing import List, Optional, Dict, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, mean, stddev, min, max, count


def get_dataset_info(df: DataFrame):
    """Виводить інформацію про DataFrame (рядки, колонки, схема)."""
    print("\n=== Інформація про набір даних ===")
    print(f"Кількість рядків: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")
    print("Назви колонок:", df.columns)
    print("\nСхема DataFrame:")
    df.printSchema()


def check_missing_values(df: DataFrame) -> DataFrame:
    """Перевіряє пропущені значення у всіх колонках і повертає DataFrame з підрахунками.

    Повертає Spark DataFrame з однією рядком та колонками для кожної початкової колонки,
    де значення — кількість "порожніх" (NULL або порожній рядок або строка 'null').
    """
    print("\n=== Перевірка пропусків і порожніх значень ===")

    expressions = [
        F.count(
            F.when(
                (col(c).isNull()) |
                ((F.trim(F.lower(col(c))) == "") & (F.lower(F.trim(col(c))).isNotNull())) |
                (F.lower(F.trim(col(c))) == "null"),
                c
            )
        ).alias(c)
        for c in df.columns
    ]

    null_counts = df.select(expressions)
    null_counts.show(truncate=False)

    total_missing = sum(null_counts.collect()[0].asDict().values())
    total_rows = df.count()
    print(f"Загальна кількість пропусків: {total_missing}")
    print(f"Відсоток пропусків від усіх значень: {round(total_missing / (total_rows * len(df.columns)) * 100, 4)}%")

    print("\nКолонки з пропусками:")
    for c, val in null_counts.collect()[0].asDict().items():
        if val > 0:
            print(f" - {c}: {val} ({round(val / total_rows * 100, 4)}%)")

    return null_counts


def remove_missing_in_column(df: DataFrame, column_name: str) -> DataFrame:
    """Видаляє рядки з пропусками у вказаній колонці (NULL або порожній рядок або 'null')."""
    if column_name not in df.columns:
        print(f"Колонка '{column_name}' не існує.")
        return df

    before = df.count()
    cleaned = df.filter(~(
        (col(column_name).isNull()) |
        (F.trim(F.lower(col(column_name))) == "") |
        (F.lower(F.trim(col(column_name))) == "null")
    ))
    after = cleaned.count()
    print(f"\n=== Видалення пропусків у колонці '{column_name}' ===")
    print(f"Видалено {before - after} рядків. Залишилось: {after}")
    return cleaned


# ---------------------- Робота з числовими колонками ----------------------

def list_numeric_columns(df: DataFrame) -> List[str]:
    """Повертає список колонок числових типів у DataFrame."""
    numeric_types = ("IntegerType", "LongType", "DoubleType", "FloatType", "DecimalType", "ShortType")
    return [f.name for f in df.schema.fields if any(t in str(f.dataType) for t in numeric_types)]


def compute_column_stats(df: DataFrame, cols: Optional[List[str]] = None) -> Dict[str, Tuple[float, float, int]]:
    """Обчислює mean, stddev і кількість ненульових(не-NULL) значень для кожної колонки.

    Повертає словник: {col: (mean, stddev, count_nonnull)}
    """
    if cols is None:
        cols = list_numeric_columns(df)

    stats: Dict[str, Tuple[float, float, int]] = {}
    for c in cols:
        agg = df.select(
            F.mean(col(c)).alias("mean"),
            F.stddev(col(c)).alias("stddev"),
            F.count(col(c)).alias("count_nonnull")
        ).collect()[0]

        stats[c] = (agg["mean"], agg["stddev"], agg["count_nonnull"])
    return stats


def show_outliers_summary(df: DataFrame, cols: Optional[List[str]] = None, z_thresh: float = 3.0) -> DataFrame:
    """Повертає Spark DataFrame з підсумком по викидах для кожної числової колонки.

    Колонки результату: column, mean, stddev, non_null_count, outlier_count, outlier_pct
    """
    spark = df.sql_ctx.sparkSession
    if cols is None:
        cols = list_numeric_columns(df)

    rows = []
    total_rows = df.count()
    stats = compute_column_stats(df, cols)

    for c in cols:
        mean_v, std_v, non_null = stats[c]
        if std_v is None or std_v == 0 or mean_v is None:
            outlier_count = 0
            outlier_pct = 0.0
        else:
            # обчислюємо фільтр викидів і підраховуємо
            expr = F.abs((col(c).cast("double") - F.lit(mean_v)) / F.lit(std_v)) > F.lit(z_thresh)
            outlier_count = df.filter(expr).count()
            outlier_pct = round(outlier_count / total_rows * 100, 4) if total_rows > 0 else 0.0

        rows.append((c, mean_v, std_v, int(non_null), int(outlier_count), float(outlier_pct)))

    schema = ["column", "mean", "stddev", "non_null_count", "outlier_count", "outlier_pct"]
    return spark.createDataFrame(rows, schema=schema)


def add_outlier_flags(df: DataFrame, cols: Optional[List[str]] = None, z_thresh: float = 3.0, suffix: str = "_is_outlier") -> DataFrame:
    """Додає для кожної числової колонки булеву колонку з прапорцем викиду.

    Наприклад, для колонки 'fare_amount' буде додано 'fare_amount_is_outlier'.
    Прапорець True означає, що abs(z-score) > z_thresh.
    Якщо std == 0 або std == None або mean == None — прапорець ставиться в False.
    """
    if cols is None:
        cols = list_numeric_columns(df)

    stats = compute_column_stats(df, cols)
    out = df
    for c in cols:
        mean_v, std_v, _ = stats[c]
        flag_col = f"{c}{suffix}"
        if std_v is None or std_v == 0 or mean_v is None:
            out = out.withColumn(flag_col, F.lit(False))
        else:
            z_expr = F.abs((col(c).cast("double") - F.lit(mean_v)) / F.lit(std_v)) > F.lit(z_thresh)
            out = out.withColumn(flag_col, F.when(z_expr, F.lit(True)).otherwise(F.lit(False)))
    return out


def remove_outliers(df: DataFrame, cols: Optional[List[str]] = None, z_thresh: float = 3.0, how: str = 'any') -> DataFrame:
    """Видаляє рядки з викидами по заданому набору колонок.

    Параметри:
      cols: список колонок. Якщо None — всі числові.
      z_thresh: поріг z-score.
      how: 'any' (за бажанням видалити рядок, якщо хоча б одна колонка — викид) або 'all' (видалити,
           якщо ВСІ колонки — викиди). 'any' за замовчуванням.

    Повертає новий DataFrame без рядків-викидів.
    """
    if cols is None:
        cols = list_numeric_columns(df)

    # Для продуктивності — краще додати тимчасові прапорці та відфільтрувати по ним
    flagged = add_outlier_flags(df, cols=cols, z_thresh=z_thresh, suffix='_tmp_out')

    flag_cols = [f"{c}_tmp_out" for c in cols]
    if how == 'any':
        # зберігаємо ті рядки, де жоден з прапорців не True
        predicate = None
        for fc in flag_cols:
            cond = (~col(fc)).cast("boolean")
            predicate = cond if predicate is None else (predicate & cond)
        cleaned = flagged.filter(predicate)
    elif how == 'all':
        # зберігаємо ті рядки, де хоча б один прапорець False
        predicate = None
        for fc in flag_cols:
            cond = (~col(fc)).cast("boolean")
            predicate = cond if predicate is None else (predicate | cond)
        cleaned = flagged.filter(predicate)
    else:
        raise ValueError("Аргумент how повинен бути 'any' або 'all'.")

    # Видаляємо тимчасові прапорці
    cleaned = cleaned.drop(*flag_cols)
    return cleaned

def clean_taxi_data(df: DataFrame) -> DataFrame:
    """
    Очищує NYC Taxi дані за географічними межами, натуральними межами
    і за допомогою динамічних меж (перцентилів) для фінансових значень.
    """
    print("\n=== Починаємо комплексну очистку NYC Taxi ===")
    before = df.count()

    # 1. Географічна фільтрація
    df = df.filter(
        (col("pickup_latitude").between(40.4774, 40.9176)) &
        (col("pickup_longitude").between(-74.2591, -73.7004)) &
        (col("dropoff_latitude").between(40.4774, 40.9176)) &
        (col("dropoff_longitude").between(-74.2591, -73.7004))
    )

    # 2. Мінімальні базові фільтри для числових полів
    df = df.filter(
        (col("fare_amount") >= 0) &
        (col("total_amount") >= 0) &
        (col("trip_distance") >= 0)
    )

    # 3. Динамічне обмеження для поля tip_amount на основі перцентиля
    # Наприклад, відсічемо дуже великі значення, які є аномаліями
    percentile = df.select(F.expr("percentile_approx(tip_amount, 0.99)")).collect()[0][0]
    # Використовуємо 99-й перцентиль як верхню межу
    max_tip = percentile if percentile is not None else 100.0
    print(f"Встановлюємо межу tip_amount = {max_tip:.2f} як 99-й перцентиль")

    df = df.filter(col("tip_amount") <= max_tip)

    after = df.count()
    print(f"Було рядків: {before}, після очищення: {after}, видалено {before - after}")

    return df

def get_numeric_stats(df: DataFrame):
    """
    Отримати статистику для числових колонок
    """
    print("\nСтатистика числових стовпців")

    # Визначаємо числові колонки
    numeric_cols = [
        f.name for f in df.schema.fields
        if "IntegerType" in str(f.dataType)
        or "DoubleType" in str(f.dataType)
        or "FloatType" in str(f.dataType)
        or "LongType" in str(f.dataType)
    ]

    if not numeric_cols:
        print("У наборі даних немає числових стовпців.")
        return

    for col_name in numeric_cols:
        print(f"\n--- {col_name} ---")
        stats = df.select(
            count(col_name).alias("count"),
            mean(col_name).alias("mean"),
            stddev(col_name).alias("stddev"),
            min(col_name).alias("min"),
            max(col_name).alias("max")
        ).collect()[0]

        print(f"Кількість значень: {stats['count']}")
        print(f"Середнє: {round(stats['mean'], 3) if stats['mean'] is not None else 'N/A'}")
        print(f"Стандартне відхилення: {round(stats['stddev'], 3) if stats['stddev'] is not None else 'N/A'}")
        print(f"Мінімум: {stats['min']}")
        print(f"Максимум: {stats['max']}")

