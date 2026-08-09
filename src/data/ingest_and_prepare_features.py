import os
from datetime import UTC, datetime

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Clean TotalCharges column (replace empty spaces with '0' and convert to float)
    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace("", "0"), errors="coerce"
    ).fillna(0.0)

    # 2. Ensure correct data types
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int64")
    df["tenure"] = df["tenure"].astype("int64")
    df["MonthlyCharges"] = df["MonthlyCharges"].astype("float32")
    df["TotalCharges"] = df["TotalCharges"].astype("float32")

    # 3. Add Feast required timestamp columns (UTC timezone-naive to avoid Feast issues)
    current_time = datetime.now(UTC).replace(tzinfo=None)
    df["event_timestamp"] = current_time
    df["created_timestamp"] = current_time
    return df


def main():
    raw_data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parquet_out_path = os.path.join("data", "churn_features.parquet")

    print(f"Reading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    df = preprocess_data(df)

    # 4. Save to Parquet format (offline store for Feast)
    print(f"Saving preprocessed features to {parquet_out_path}...")
    df.to_parquet(parquet_out_path, index=False)
    print("Ingestion and feature preparation complete!")


if __name__ == "__main__":
    main()
