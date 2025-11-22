#регресія
from typing import List, Optional, Tuple
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import LinearRegression
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

    # Створюємо індексатори та OHE для категорій (стадії)
    for c in categorical_cols:
        idx = f"{c}_idx"
        ohe = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=[idx], outputCols=[ohe], handleInvalid="keep"))
        ohe_cols.append(ohe)

    # Numeric cols (ті, що не в categorical_cols)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    assembler_inputs = numeric_cols + ohe_cols

    # VectorAssembler + StandardScaler
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="raw_features", handleInvalid="keep")
    stages.append(assembler)

    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=False)
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)
    return pipeline, assembler_inputs


def train_and_evaluate_linear_regression(df: DataFrame,
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

    # Перевірка колонок (пропозиція фіч, якщо не задані)
    if feature_cols is None:
        suggested = ["passenger_count", "trip_distance"]
        feature_cols = [c for c in suggested if c in df.columns]

    if categorical_cols is None:
        categorical_cols = [c for c in ["vendor_id", "payment_type"] if c in df.columns]

    # 1) Split raw df BEFORE fitting pipeline (важливо)
    raw_train, raw_test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Split sizes -> train: {raw_train.count()}, test: {raw_test.count()}")

    # 2) Build (but NOT fit) feature pipeline
    feature_pipeline, assembler_inputs = build_feature_pipeline(feature_cols, categorical_cols)

    # 3) Fit pipeline only on train
    fitted_feature_pipeline = feature_pipeline.fit(raw_train)

    # 4) Transform train & test to get 'features' column
    # include label_col and optionally original feature_cols for debugging/inspection
    train = fitted_feature_pipeline.transform(raw_train).select(*([label_col, "features"] + [c for c in feature_cols if c in df.columns]))
    test = fitted_feature_pipeline.transform(raw_test).select(*([label_col, "features"] + [c for c in feature_cols if c in df.columns]))

    # 5) Prepare estimator
    lr = LinearRegression(featuresCol="features", labelCol=label_col, maxIter=100)

    # 6) Hyperparam tuning (TrainValidationSplit) - optional but recommended
    if use_tvs:
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.0, 0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
            .build()

        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")

        print("\n--- TrainValidationSplit (hyperparam tuning) ---")
        tvs = TrainValidationSplit(estimator=lr,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=evaluator,
                                   trainRatio=0.8,
                                   parallelism=2)
        tvs_model = tvs.fit(train)
        best_model = tvs_model.bestModel
        print("TrainValidationSplit done.")
    else:
        print("\n--- Fitting LinearRegression without TVS ---")
        best_model = lr.fit(train)

    # 7) Optionally show training summary (if present)
    try:
        training_summary = best_model.summary
        print("\n[TRAINING SUMMARY]")
        print(f"numIterations: {getattr(training_summary, 'totalIterations', 'n/a')}")
        print(f"training RMSE: {getattr(training_summary, 'rootMeanSquaredError', 'n/a'):.4f}")
        print(f"training r2: {getattr(training_summary, 'r2', 'n/a'):.4f}")
    except Exception:
        # Some models may not expose summary in this context
        pass

    # 8) Predict on test
    preds = best_model.transform(test).na.fill({"prediction": 0.0})

    # 9) Evaluate on test
    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    # 10) Print results
    print("\n[TEST EVALUATION]")
    print(f"LinearRegression -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Model coefficients and intercept (if available)
    try:
        coeffs = best_model.coefficients
        intercept = best_model.intercept
        print("\n[MODEL COEFFICIENTS]")
        print(f"Intercept: {intercept}")
        print(f"Coefficients (len={len(coeffs)}): {coeffs}")
    except Exception:
        pass

    # 11) Sample predictions (same style as у тебе)
    print("\n[Sample predictions (all types)]")

    print("\nRows with total_amount > 10:")
    preds.filter(preds[label_col] > 10).select(label_col, "prediction").show(10, truncate=False)

    print("\nRows with total_amount between 5 and 20:")
    preds.filter((preds[label_col] >= 5) & (preds[label_col] <= 20)) \
        .select(label_col, "prediction").show(10, truncate=False)

    print("\nRandom sample:")
    preds.orderBy(rand()).select(label_col, "prediction").show(10, truncate=False)

    # 12) Optional saving (закоментовано за замовчуванням)
    # if save:
    #     model_path = os.path.join(output_dir, "models", "LinearRegression")
    #     try:
    #         best_model.write().overwrite().save(model_path)
    #         print(f"Модель збережена в: {model_path}")
    #     except Exception as e:
    #         print(f"Не вдалося зберегти модель: {e}")
    #
    #     results = [("LinearRegression", float(rmse), float(r2))]
    #     results_df = spark.createDataFrame(results, schema=["model_name", "rmse", "r2"])
    #     results_df.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_dir, "regression_metrics"))
    #     print(f"Метрики збережені в: {os.path.join(output_dir, 'regression_metrics')}")

    # 13) Return summary DataFrame (for compatibility)
    results = [("LinearRegression", float(rmse), float(r2))]
    results_df = spark.createDataFrame(results, schema=["model_name", "rmse", "r2"])
    return results_df
