from datetime import timedelta

from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# 1. Define the Entity (Primary Key identifier)
customer = Entity(
    name="customer_id",
    join_keys=["customerID"],
    value_type=ValueType.STRING,
    description="Customer ID entity for Telco Churn",
)

# 2. Define the Batch/Offline Data Source (Parquet file)
# The path is relative to the feature_store/ directory where we execute feast apply.
churn_source = FileSource(
    name="churn_source",
    path="../data/churn_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Define the Feature View
churn_feature_view = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=365),
    schema=[
        Field(name="gender", dtype=String),
        Field(name="SeniorCitizen", dtype=Int64),
        Field(name="Partner", dtype=String),
        Field(name="Dependents", dtype=String),
        Field(name="tenure", dtype=Int64),
        Field(name="PhoneService", dtype=String),
        Field(name="MultipleLines", dtype=String),
        Field(name="InternetService", dtype=String),
        Field(name="OnlineSecurity", dtype=String),
        Field(name="OnlineBackup", dtype=String),
        Field(name="DeviceProtection", dtype=String),
        Field(name="TechSupport", dtype=String),
        Field(name="StreamingTV", dtype=String),
        Field(name="StreamingMovies", dtype=String),
        Field(name="Contract", dtype=String),
        Field(name="PaperlessBilling", dtype=String),
        Field(name="PaymentMethod", dtype=String),
        Field(name="MonthlyCharges", dtype=Float32),
        Field(name="TotalCharges", dtype=Float32),
    ],
    online=True,
    source=churn_source,
    tags={"team": "churn_prediction_team"},
)
