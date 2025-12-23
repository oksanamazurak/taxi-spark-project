from typing import List, Optional, Tuple, Dict
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col, when
from pyspark.storagelevel import StorageLevel
import os
from pyspark.sql.functions import rand


def prepare_features_classification(df: DataFrame,
                                    feature_cols: Optional[List[str]] = None,
                                    categorical_cols: Optional[List[str]] = None,
                                    label_col: str = "label") -> Tuple[Pipeline, DataFrame]:

    if feature_cols is None:
        suggested = ["passenger_count", "total_amount"]
        feature_cols = [c for c in suggested if c in df.columns]

    if categorical_cols is None:
        categorical_cols = [c for c in ["vendor_id", "payment_type"] if c in df.columns]

    stages = []
    ohe_cols = []

    # 1. Обробка категорій
    for c in categorical_cols:
        idx = f"{c}_idx"
        ohe = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe], handleInvalid="keep"))
        ohe_cols.append(ohe)

    # 2. Числові
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # 3. Збірка вектора
    assembler_inputs = numeric_cols + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")
    stages.append(assembler)

    # 4. Скалювання
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)
    fitted_pipeline = pipeline.fit(df)

    cols_to_select = list(set([label_col, "features"] + feature_cols))
    prepared = fitted_pipeline.transform(df).select(*cols_to_select)

    return pipeline, prepared


def _compute_prf(predictions_df: DataFrame, label_col: str) -> Dict[str, float]:

    # Конвертуємо в RDD (prediction, label) для mllib metrics
    rdd = predictions_df.select("prediction", label_col).rdd.map(lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(rdd)

    accuracy = metrics.accuracy
    f1 = metrics.weightedFMeasure()
    precision = metrics.weightedPrecision
    recall = metrics.weightedRecall

    try:
        conf_mat = metrics.confusionMatrix().toArray().tolist()
    except Exception:
        conf_mat = None

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_mat
    }


def train_and_evaluate_logistic_regression(df: DataFrame,
                                           feature_cols: Optional[List[str]] = None,
                                           categorical_cols: Optional[List[str]] = None,
                                           label_col: str = "trip_category",
                                           save: bool = False,
                                           output_dir: str = "/app/results") -> DataFrame:
    spark = df.sql_ctx.sparkSession
    if save:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    # 1. Створення Label
    if label_col not in df.columns:
        if "trip_distance" in df.columns:
            print(f"Створюємо колонку '{label_col}' на основі дистанції...")
            # 0.0 = Short, 1.0 = Medium, 2.0 = Long
            df = df.withColumn(label_col,
                               when(col("trip_distance") < 2, 0.0)
                               .when((col("trip_distance") >= 2) & (col("trip_distance") < 10), 1.0)
                               .otherwise(2.0)
                               )
        else:
            raise ValueError(f"Немає колонки {label_col} і trip_distance.")

    df = df.withColumn(label_col, col(label_col).cast("double"))
    df = df.filter(col(label_col).isNotNull())

    # 2. Підготовка
    pipeline, prepared = prepare_features_classification(df, feature_cols, categorical_cols, label_col)

    # Split
    train, test = prepared.randomSplit([0.8, 0.2], seed=42)
    train = train.repartition(20).persist(StorageLevel.MEMORY_AND_DISK)

    # 3. Модель
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=20,
        regParam=0.01,
        family="auto"
    )

    print(f"\n--- Тренування LogisticRegression (Label: {label_col}) ---")
    fitted = lr.fit(train)

    preds = fitted.transform(test)
    metrics = _compute_prf(preds, label_col)

    print("\n[TEST EVALUATION - Multiclass]")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1']:.4f}")
    if metrics.get("confusion_matrix") is not None:
        conf = metrics["confusion_matrix"]
        classes = ["Short", "Medium", "Long"]

        print("\n[CONFUSION MATRIX - Predicted \\ Actual]")
        # Друкуємо шапку
        print(f"{'':>10}", end="")
        for c in classes:
            print(f"{c:>10}", end="")
        print()

        # Друкуємо рядки
        for i, row in enumerate(conf):
            print(f"{classes[i]:>10}", end="")
            for val in row:
                print(f"{int(val):>10}", end="")
            print()

    # if metrics.get("confusion_matrix"):
    #     print("\n[CONFUSION MATRIX]")
    #     print("Cols = Actual, Rows = Predicted (0=Short, 1=Medium, 2=Long)")
    #     for row in metrics["confusion_matrix"]:
    #         print(row)


    print("\n[Sample predictions]")

    print("\nRows with trip_category = Long (2.0):")
    preds.filter(preds[label_col] == 2.0) \
        .select(label_col, "prediction", "probability") \
        .show(10, truncate=False)

    print("\nRows with trip_category = Medium (1.0):")
    preds.filter(preds[label_col] == 1.0) \
        .select(label_col, "prediction", "probability") \
        .show(10, truncate=False)

    print("\nRows with trip_category = Short (0.0):")
    preds.filter(preds[label_col] == 0.0) \
        .select(label_col, "prediction", "probability") \
        .show(10, truncate=False)

    print("\nRandom sample of predictions:")
    preds.orderBy(rand()) \
        .select(label_col, "prediction", "probability") \
        .show(10, truncate=False)

    preds.select(label_col, "prediction", "probability").show(10, truncate=False)

    results = [("LogReg_Multiclass", metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])]
    results_df = spark.createDataFrame(results, schema=["model_name", "accuracy", "w_precision", "w_recall", "w_f1"])

    return results_df