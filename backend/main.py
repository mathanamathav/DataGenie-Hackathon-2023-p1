from fastapi import FastAPI, Query, Body
import uvicorn
from models import PredictionRequest
from typing import List
from constants import method_to_predict
from utils import predict_values, check_all_models, process_data
from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
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


@app.post("/predict")
async def predict(
    date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: int = Query(..., description="Period"),
    request_data: List[PredictionRequest] = Body(
        ..., description="Data points for prediction"
    ),
):
    """_summary_

    Args:
        date_from (str, optional): _description_. Defaults to Query(..., description="Start date (YYYY-MM-DD)").
        date_to (str, optional): _description_. Defaults to Query(..., description="End date (YYYY-MM-DD)").
        period (int, optional): _description_. Defaults to Query(..., description="Period").
        request_data (List[PredictionRequest], optional): _description_. Defaults to Body( ..., description="Data points for prediction" ).

    Returns:
        response : _description_ { "model" : "" , "mape" : 0 , "result" : [ date , value , predicted value ]}
    """
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

        if (
            pd.to_datetime(date_from).date() != sample_df["date"][0].date()
            or pd.to_datetime(date_to).date() != sample_df["date"][-1].date()
        ):
            return {
                "message": "Date mentioned in the parameters and payload is not matching!"
            }

        mean = sample_df["value"].mean()
        variance = sample_df["value"].var()
        kurt = kurtosis(sample_df["value"])

        try:
            result = seasonal_decompose(
                sample_df["value"], model="additive", extrapolate_trend="freq"
            )
        except:
            result = seasonal_decompose(
                sample_df["value"],
                model="additive",
                extrapolate_trend="freq",
                period=1,
            )

        trend_mean = result.trend.mean()
        seasonal_mean = result.seasonal.mean()
        residual_mean = result.resid.mean()
        model_input = [
            mean,
            variance,
            kurt,
            trend_mean,
            seasonal_mean,
            residual_mean,
        ]

    with open("models/combined0.4875.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    dates, true_y = sample_df["date"].tolist(), sample_df["value"].tolist()
    best_model = check_all_models(sample_df, period, date_to)

    model_to_predict = method_to_predict[model.predict([model_input])[0]]
    y_pred, mape = predict_values(model_to_predict, sample_df, period, date_to)

    sample_df["timedelta"] = sample_df["date"] - sample_df["date"].shift(1)
    mode_timedelta = sample_df["timedelta"].value_counts().idxmax()
    last_date = sample_df["date"].max()

    forecast_dates = pd.date_range(
        start=last_date, freq=mode_timedelta, periods=period + 1, inclusive="right"
    ).tolist()
    dates.extend(forecast_dates)

    logging.info(mean_absolute_percentage_error(y_pred[: len(true_y)], true_y))

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

    prediction_info = {"model": model_to_predict, "mape": mape, "result": res}

    return prediction_info


@app.post("/predict_with_format")
async def predict_with_format(
    format: str = Query(..., description="Format"),
    date_from: str = Query(..., description="Start date (YYYY-MM-DD)"),
    date_to: str = Query(..., description="End date (YYYY-MM-DD)"),
    period: int = Query(..., description="Period"),
    request_data: List[PredictionRequest] = Body(
        ..., description="Data points for prediction"
    ),
):
    """_summary_

    Args:
        format (str, optional): _description_. Defaults to Query(..., description="Format").
        date_from (str, optional): _description_. Defaults to Query(..., description="Start date (YYYY-MM-DD)").
        date_to (str, optional): _description_. Defaults to Query(..., description="End date (YYYY-MM-DD)").
        period (int, optional): _description_. Defaults to Query(..., description="Period").
        request_data (List[PredictionRequest], optional): _description_. Defaults to Body( ..., description="Data points for prediction" ).

    Returns:
        response : _description_ { "model" : "" , "mape" : 0 , "result" : [ date , value , predicted value ]}
    """
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
