import csv
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def log_predictions(predictions, logs_path="data/prediction_logs.csv"):
    """
    Append prediction outputs to a CSV file. Writes headers only if the file is newly created.
    """
    os.makedirs(os.path.dirname(logs_path), exist_ok=True)
    file_exists = os.path.exists(logs_path)

    with open(logs_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "customer_id", "probability", "prediction"])
        for pred in predictions:
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    pred["customer_id"],
                    pred["churn_probability"],
                    pred["prediction"],
                ]
            )


def calculate_drift(
    baseline_path="data/baseline_probabilities.npy",
    logs_path="data/prediction_logs.csv",
    min_samples=20,
    p_threshold=0.05,
):
    """
    Calculates prediction drift using Kolmogorov-Smirnov test on numerical prediction probabilities.
    Requires at least min_samples of logged predictions.
    """
    if not os.path.exists(baseline_path):
        return {
            "status": "error",
            "message": f"Baseline probabilities file '{baseline_path}' not found.",
        }

    if not os.path.exists(logs_path):
        return {
            "status": "insufficient_data",
            "count": 0,
            "message": "Logged predictions file does not exist yet.",
        }

    # Read prediction probabilities from log file
    try:
        df = pd.read_csv(logs_path)
        if "probability" not in df.columns or len(df) == 0:
            return {
                "status": "insufficient_data",
                "count": 0,
                "message": "Logged predictions file is empty or missing probability column.",
            }
        logged_probs = df["probability"].astype(float).values
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to read logged predictions: {e!s}",
        }

    logged_count = len(logged_probs)
    if logged_count < min_samples:
        return {
            "status": "insufficient_data",
            "count": logged_count,
            "message": f"Insufficient data: logged predictions count ({logged_count}) is less than the required minimum ({min_samples}) to perform statistical KS-test.",
        }

    # Load baseline probabilities
    try:
        baseline_probs = np.load(baseline_path)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load baseline probabilities: {e!s}",
        }

    # Run two-sample Kolmogorov-Smirnov test
    stat, p_value = ks_2samp(baseline_probs, logged_probs)

    return {
        "status": "success",
        "p_value": float(p_value),
        "statistic": float(stat),
        "drift_detected": bool(p_value < p_threshold),
        "logged_count": int(logged_count),
        "baseline_count": len(baseline_probs),
    }


def calculate_evidently_drift(
    baseline_path="data/baseline_probabilities.npy",
    logs_path="data/prediction_logs.csv",
    min_samples=20,
):
    """
    Calculates prediction drift using Evidently AI ColumnDriftMetric on probabilities.
    """
    if not os.path.exists(baseline_path) or not os.path.exists(logs_path):
        return None

    try:
        df = pd.read_csv(logs_path)
        if "probability" not in df.columns or len(df) < min_samples:
            return None
        logged_probs = df["probability"].astype(float).values
        baseline_probs = np.load(baseline_path)
    except Exception:
        return None

    try:
        from evidently.legacy.metrics import ColumnDriftMetric
        from evidently.legacy.report import Report

        ref_df = pd.DataFrame({"probability": baseline_probs})
        cur_df = pd.DataFrame({"probability": logged_probs})

        report = Report(metrics=[ColumnDriftMetric(column_name="probability")])
        report.run(reference_data=ref_df, current_data=cur_df)
        report_dict = report.as_dict()

        drift_result = report_dict["metrics"][0]["result"]
        drift_detected = drift_result["drift_detected"]
        drift_score = drift_result["drift_score"]

        return {
            "p_value": float(drift_score),
            "drift_detected": bool(drift_detected),
            "drift_score": float(drift_score),
        }
    except Exception as e:
        print(f"Evidently calculation warning: {e!s}")
        return None
