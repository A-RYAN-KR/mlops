from pydantic import BaseModel


class ChurnRequest(BaseModel):
    customer_ids: list[str]


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: str


class ChurnResponse(BaseModel):
    predictions: list[ChurnPrediction]
