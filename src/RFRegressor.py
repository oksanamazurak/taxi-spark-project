from typing import List, Optional, Tuple
from pyspark.sql import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
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


def train_and_evaluate_rf_regression(df: DataFrame,
                                     feature_cols: Optional[List[str]] = None,
                                     categorical_cols: Optional[List[str]] = None,
                                     label_col: str = "total_amount",
                                     save: bool = False,
                                     output_dir: str = "/app/results",
                                     use_tvs: bool = False) -> DataFrame:
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

    rf = RandomForestRegressor(featuresCol="features", labelCol=label_col, seed=42)

    if use_tvs:
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.maxDepth, [5, 10]) \
            .addGrid(rf.numTrees, [50, 100]) \
            .build()

        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")

        print("\n--- TrainValidationSplit (hyperparam tuning for RF) ---")
        tvs = TrainValidationSplit(estimator=rf,
                                   estimatorParamMaps=paramGrid,
                                   evaluator=evaluator,
                                   trainRatio=0.8,
                                   parallelism=2)
        tvs_model = tvs.fit(train)
        best_model = tvs_model.bestModel
        print("TrainValidationSplit done.")
    else:
        print("\n--- Fitting RandomForestRegressor without TVS ---")
        best_model = rf.fit(train)

    print("\n[TRAINING INFO]")
    print(f"Model used: {best_model}")
    try:
        print(f"Number of trees: {best_model.getNumTrees}")
    except Exception:
        pass

    preds = best_model.transform(test).na.fill({"prediction": 0.0})

    evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(preds)
    r2 = evaluator_r2.evaluate(preds)

    print("\n[TEST EVALUATION]")
    print(f"RandomForestRegressor -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    print("\n[Sample predictions (all types)]")
    print("\nRows with total_amount > 10:")
    preds.filter(preds[label_col] > 10).select(label_col, "prediction").show(10, truncate=False)

    print("\nRows with total_amount between 5 and 20:")
    preds.filter((preds[label_col] >= 5) & (preds[label_col] <= 20)) \
        .select(label_col, "prediction").show(10, truncate=False)

    print("\nRandom sample:")
    preds.orderBy(rand()).select(label_col, "prediction").show(10, truncate=False)

    if save:
        model_path = os.path.join(output_dir, "models", "RFRegressor")
        try:
            best_model.write().overwrite().save(model_path)
            print(f"Модель збережена в: {model_path}")
        except Exception as e:
            print(f"Не вдалося зберегти модель: {e}")

    results = [("RFRegressor", float(rmse), float(r2))]
    results_df = spark.createDataFrame(results, schema=["model_name", "rmse", "r2"])
    return results_df
