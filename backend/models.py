from pydantic import BaseModel
from datetime import datetime

class PredictionRequest(BaseModel):
    point_timestamp: datetime
    point_value: float

