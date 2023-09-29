from fastapi import FastAPI, Query, Body
import uvicorn
from models import PredictionRequest
from typing import List

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/predict")
async def predict(
    date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: int = Query(..., description="Period"),
    request_data: List[PredictionRequest] = Body(
        ..., description="Data points for prediction"
    ),
):
    # Todo implement classify the model and use that mode to train , pred value to return
    prediction_info = {
        "date_from": date_from,
        "date_to": date_to,
        "period": period,
        "data": request_data,
    }

    return {"message": prediction_info}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)
