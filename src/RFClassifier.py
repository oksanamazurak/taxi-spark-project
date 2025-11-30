from typing import List, Optional, Tuple, Dict
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.storagelevel import StorageLevel
import os


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

    for c in categorical_cols:
        idx = f"{c}_idx"
        ohe = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe], handleInvalid="keep"))
        ohe_cols.append(ohe)

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    assembler_inputs = numeric_cols + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")
    stages.append(assembler)

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)
    fitted_pipeline = pipeline.fit(df)

    cols_to_select = list(set([label_col, "features"] + feature_cols))
    prepared = fitted_pipeline.transform(df).select(*cols_to_select)

    return pipeline, prepared


def _compute_metrics_safe(predictions_df: DataFrame, label_col: str) -> Dict[str, float]:
    evaluator_acc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedFMeasure")
    evaluator_prec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="weightedRecall")

    return {
        "accuracy": evaluator_acc.evaluate(predictions_df),
        "f1": evaluator_f1.evaluate(predictions_df),
        "precision": evaluator_prec.evaluate(predictions_df),
        "recall": evaluator_rec.evaluate(predictions_df)
    }


def train_and_evaluate_rf_classification(df: DataFrame,
                                         feature_cols: Optional[List[str]] = None,
                                         categorical_cols: Optional[List[str]] = None,
                                         label_col: str = "trip_category",
                                         save: bool = False,
                                         output_dir: str = "/app/results") -> DataFrame:
    spark = df.sparkSession

    if label_col not in df.columns:
        if "trip_distance" in df.columns:
            print(f"Створюємо колонку '{label_col}' на основі дистанції...")
            df = df.withColumn(label_col,
                               when(col("trip_distance") < 2, 0.0)
                               .when((col("trip_distance") >= 2) & (col("trip_distance") < 10), 1.0)
                               .otherwise(2.0)
                               )
        else:
            raise ValueError(f"Немає колонки {label_col} і trip_distance.")

    df = df.withColumn(label_col, col(label_col).cast("double"))
    df = df.filter(col(label_col).isNotNull())

    pipeline, prepared = prepare_features_classification(df, feature_cols, categorical_cols, label_col)

    train, test = prepared.randomSplit([0.8, 0.2], seed=42)
    train = train.repartition(20).persist(StorageLevel.MEMORY_AND_DISK)

    rf = RandomForestClassifier(featuresCol="features", labelCol=label_col, seed=42, numTrees=100, maxDepth=10)

    print(f"\n--- Тренування RandomForestClassifier ---")
    fitted = rf.fit(train)
    preds = fitted.transform(test)

    metrics = _compute_metrics_safe(preds, label_col)

    print("\n[TEST EVALUATION - RF Classifier]")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall:    {metrics['recall']:.4f}")

    if save:
        try:
            os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
            fitted.write().overwrite().save(os.path.join(output_dir, "models", "RFClassifier"))
            print(f"Модель збережена в: {os.path.join(output_dir, 'models', 'RFClassifier')}")
        except Exception as e:
            print(f"Не вдалося зберегти модель: {e}")

    train.unpersist()

    results = [("RF_Multiclass", metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"])]
    results_df = spark.createDataFrame(results, schema=["model_name", "accuracy", "w_precision", "w_recall", "w_f1"])

    return results_df
