from pydantic import BaseModel
from datetime import datetime

class PredictionRequest(BaseModel):
    point_timestamp: str
    point_value: float

