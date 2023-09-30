from fastapi import FastAPI, Query, Body
import uvicorn
from models import PredictionRequest
from typing import List
from constants import method_to_predict
from utils import predict_values
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import warnings
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True
warnings.filterwarnings("ignore")

app = FastAPI()


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}


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
    # Todo implement classify the model and use that mode to train , pred value to return
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

        mean = sample_df["value"].mean()
        variance = sample_df["value"].var()
        skewness = skew(sample_df["value"])
        kurt = kurtosis(sample_df["value"])

        result = seasonal_decompose(
            sample_df["value"], model="additive", extrapolate_trend="freq", period=1
        )
        trend_mean = result.trend.mean()
        seasonal_mean = result.seasonal.mean()
        residual_mean = result.resid.mean()
        model_input = [
            mean,
            variance,
            skewness,
            kurt,
            trend_mean,
            seasonal_mean,
            residual_mean,
        ]

    if format:
        if format == "daily":
            forecast_dates = pd.date_range(
                start=pd.to_datetime(date_to) + pd.DateOffset(1), periods=period, freq="D"
            )
            with open("models/daily-rf.pkl", "rb") as model_file:
                model = pickle.load(model_file)

        elif format == "monthly":
            pass
        elif format == "hourly":
            pass
        elif format == "weekly":
            pass

        model_to_predict = method_to_predict[model.predict([model_input])[0]]
        y_pred, mape = predict_values(model_to_predict, sample_df, period, date_to)
        dates, true_y = sample_df["date"].tolist(), sample_df["value"].tolist()
        dates.extend(forecast_dates)

        res = []
        for i in range(len(y_pred)):
            if true_y:
                res.append(
                    {
                        "point_timestamp": dates.pop(0),
                        "point_value": true_y.pop(0),
                        "yhat": round(y_pred.pop(0) ,4),
                    }
                )
            else:
                res.append(
                    {
                        "point_timestamp": dates.pop(0),
                        "yhat": round(y_pred.pop(0) ,4),
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
