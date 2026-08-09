import pandas as pd

from src.data.ingest_and_prepare_features import preprocess_data


def test_preprocess_data_clean_and_types():
    # Arrange: Mock a raw DataFrame with dirty TotalCharges and various formats
    raw_data = {
        "customerID": ["1", "2", "3"],
        "SeniorCitizen": [0, 1, 0],
        "tenure": [12, 0, 24],
        "MonthlyCharges": [29.85, 56.95, 108.15],
        "TotalCharges": [" 29.85 ", " ", "108.15"],  # Dirty spaces, empty string
    }
    df = pd.DataFrame(raw_data)

    # Act: Process raw data
    processed_df = preprocess_data(df)

    # Assert: TotalCharges cleaned
    assert processed_df.loc[0, "TotalCharges"] == 29.85
    assert processed_df.loc[1, "TotalCharges"] == 0.0  # Empty space resolved to 0.0
    assert processed_df.loc[2, "TotalCharges"] == 108.15

    # Assert: Types casted correctly
    assert processed_df["SeniorCitizen"].dtype == "int64"
    assert processed_df["tenure"].dtype == "int64"
    assert processed_df["MonthlyCharges"].dtype == "float32"
    assert processed_df["TotalCharges"].dtype == "float32"

    # Assert: Feast timestamp columns added
    assert "event_timestamp" in processed_df.columns
    assert "created_timestamp" in processed_df.columns
    assert pd.api.types.is_datetime64_any_dtype(processed_df["event_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(processed_df["created_timestamp"])
