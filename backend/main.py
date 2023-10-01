from fastapi import FastAPI, Query, Body
import uvicorn
from models import PredictionRequest
from typing import List
from constants import method_to_predict
from utils import predict_values, check_all_models, process_data
import pandas as pd
import pickle
import warnings
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
warnings.filterwarnings("ignore")

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


# Predict API
@app.post("/predict")
async def predict(
    format: str = Query(..., description="Format"),
    date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: int = Query(..., description="Period"),
    request_data: List[PredictionRequest] = Body(
        ..., description="Data points for prediction"
    ),
):
    if period and period < 0:
        return {
            "message": "forecast number cannot be negative, It has to be 0 or greater"
        }

    if request_data:
        data = [
            {"point_timestamp": val.point_timestamp, "point_value": val.point_value}
            for val in request_data
        ]
        data = sorted(data, key=lambda x: x["point_timestamp"])
        sample_df = pd.DataFrame(data)

        sample_df["point_timestamp"] = pd.to_datetime(sample_df["point_timestamp"])
        sample_df.columns = ["date", "value"]

        sample_df.set_index(sample_df["date"], inplace=True)
        sample_df.sort_index(inplace=True)
        sample_df.rename_axis("index", inplace=True)

        model_input, error_response = process_data(
            sample_df, format, date_from, date_to
        )
        if error_response:
            return error_response

    forecast_dates = None
    model = None

    if format == "daily":
        forecast_dates = pd.date_range(
            start=pd.to_datetime(date_to) + pd.DateOffset(1),
            periods=period,
            freq="D",
        )
        with open("models/RandomForest0.6daily7.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    elif format == "monthly":
        forecast_dates = pd.date_range(
            start=pd.to_datetime(date_to),
            periods=period + 1,
            freq="MS",
        )[1:]
        with open("models/RandomForest0.35monthly7.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    elif format == "hourly":
        forecast_dates = pd.date_range(
            start=sample_df["date"][-1],
            periods=period + 1,
            closed="left",
            freq="H",
        )[1:]
        with open("models/RandomForest0.25hourly7.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    elif format == "weekly":
        forecast_dates = pd.date_range(
            start=pd.to_datetime(date_to) + pd.DateOffset(1),
            periods=period,
            freq="W",
        )
        with open("models/RandomForest0.75weekly7.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    model_to_predict = method_to_predict[model.predict([model_input])[0]]
    y_pred, mape = predict_values(model_to_predict, sample_df, period, date_to)
    dates, true_y = sample_df["date"].tolist(), sample_df["value"].tolist()
    dates.extend(forecast_dates)

    best_model = check_all_models(sample_df, period, date_to)
    logging.info("{} - {}".format(model_to_predict, best_model))

    res = []
    for i in range(len(y_pred)):
        if true_y:
            res.append(
                {
                    "point_timestamp": dates.pop(0),
                    "point_value": true_y.pop(0),
                    "yhat": round(y_pred.pop(0), 4),
                }
            )
        else:
            res.append(
                {
                    "point_timestamp": dates.pop(0),
                    "yhat": round(y_pred.pop(0), 4),
                }
            )

    prediction_info = {
        "model": model_to_predict,
        "mape": mape,
        "result": res,
    }

    return prediction_info


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8105)
