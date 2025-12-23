#регресія GBT
from typing import List, Optional, Tuple
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import GBTRegressor  
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import rand
import os

def build_feature_pipeline(feature_cols: Optional[List[str]] = None,
                           categorical_cols: Optional[List[str]] = None) -> Tuple[Pipeline, List[str]]:
    if feature_cols is None:
        feature_cols = ["passenger_count", "trip_distance"]

    if categorical_cols is None:
        categorical_cols = ["vendor_id", "payment_type"]

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
    return pipeline, assembler_inputs


def train_and_evaluate_gbt_regression(df: DataFrame,
                                      feature_cols: Optional[List[str]] = None,
                                      categorical_cols: Optional[List[str]] = None,
                                      label_col: str = "total_amount",
                                      save: bool = False,
                                      output_dir: str = "/app/results",
                                      use_tvs: bool = True) -> DataFrame:

    spark = df.sparkSession

    if save:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    if feature_cols is None:
        suggested = ["passenger_count", "trip_distance"]
        feature_cols = [c for c in suggested if c in df.columns]

    if categorical_cols is None:
        categorical_cols = [c for c in ["vendor_id", "payment_type"] if c in df.columns]

    raw_train, raw_test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Split sizes -> train: {raw_train.count()}, test: {raw_test.count()}")

    feature_pipeline, assembler_inputs = build_feature_pipeline(feature_cols, categorical_cols)

    fitted_feature_pipeline = feature_pipeline.fit(raw_train)

    train = fitted_feature_pipeline.transform(raw_train).select(*([label_col, "features"] + [c for c in feature_cols if c in df.columns]))
    test = fitted_feature_pipeline.transform(raw_test).select(*([label_col, "features"] + [c for c in feature_cols if c in df.columns]))

    gbt = GBTRegressor(featuresCol="features", labelCol=label_col, seed=42)

    if use_tvs:
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxDepth, [2, 5]) \
            .addGrid(gbt.maxIter, [10, 20]) \
            .build()

        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")

        print("\n--- TrainValidationSplit (hyperparam tuning for GBT) ---")
        tvs = TrainValidationSplit(estimator=gbt,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=evaluator,
                                   trainRatio=0.8,
                                   parallelism=2)
        tvs_model = tvs.fit(train)
        best_model = tvs_model.bestModel
        print("TrainValidationSplit done.")
    else:
        print("\n--- Fitting GBTRegressor without TVS ---")
        best_model = gbt.fit(train)

    print("\n[TRAINING INFO]")
    print(f"Model used: {best_model}")
    print(f"Number of trees: {best_model.getNumTrees}")

    preds = best_model.transform(test).na.fill({"prediction": 0.0})

    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    print("\n[TEST EVALUATION]")
    print(f"GBTRegressor -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    try:
        importances = best_model.featureImportances
        print("\n[FEATURE IMPORTANCES]")
        print(f"Feature Importances vector: {importances}")
        print("(Note: Indices correspond to the 'features' vector from VectorAssembler)")
    except Exception:
        pass

    print("\n[Sample predictions (all types)]")

    print("\nRows with total_amount > 10:")
    preds.filter(preds[label_col] > 10).select(label_col, "prediction").show(10, truncate=False)

    print("\nRows with total_amount between 5 and 20:")
    preds.filter((preds[label_col] >= 5) & (preds[label_col] <= 20)) \
        .select(label_col, "prediction").show(10, truncate=False)

    print("\nRandom sample:")
    preds.orderBy(rand()).select(label_col, "prediction").show(10, truncate=False)

    if save:
        model_path = os.path.join(output_dir, "models", "GBTRegressor")
        try:
            best_model.write().overwrite().save(model_path)
            print(f"Модель збережена в: {model_path}")
        except Exception as e:
            print(f"Не вдалося зберегти модель: {e}")

        # Збереження метрик
        # Можна розкоментувати, якщо потрібно писати в CSV
        # results_df_save = spark.createDataFrame([("GBTRegressor", float(rmse), float(r2))], schema=["model_name", "rmse", "r2"])
        # results_df_save.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_dir, "regression_metrics_gbt"))

    results = [("GBTRegressor", float(rmse), float(r2))]
    results_df = spark.createDataFrame(results, schema=["model_name", "rmse", "r2"])
    return results_df