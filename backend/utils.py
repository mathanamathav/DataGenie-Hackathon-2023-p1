import numpy as np
import pandas as pd
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pmdarima as pm

from prophet import Prophet

import xgboost as xgb

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf

from scipy.stats import skew, kurtosis
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import logging

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled = True
warnings.filterwarnings("ignore")


def process_data(sample_df, format, date_from, date_to):
    model_input = []

    mean = sample_df["value"].mean()
    model_input.append(mean)

    variance = sample_df["value"].var()
    model_input.append(variance)

    if (
        pd.to_datetime(date_from).date() != sample_df["date"][0].date()
        or pd.to_datetime(date_to).date() != sample_df["date"][-1].date()
    ):
        return (
            None,
            {
                "message": "Date mentioned in the parameters and payload is not matching!"
            },
        )

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

    if format == "hourly":
        seasonal_mean = result.seasonal.mean()
        model_input.append(seasonal_mean)

        residual_mean = result.resid.mean()
        model_input.append(residual_mean)

    if format == "daily" or format == "monthly" or format == "weekly":
        kurt = kurtosis(sample_df["value"])
        model_input.append(kurt)

        if format == "weekly":
            trend_mean = result.trend.mean()
            model_input.append(trend_mean)
        else:
            trend_mean = result.trend.mean()
            model_input.append(trend_mean)

            seasonal_mean = result.seasonal.mean()
            model_input.append(seasonal_mean)

            residual_mean = result.resid.mean()
            model_input.append(residual_mean)

    return model_input, None


def calculate_mape(true_values, predicted_values):
    epsilon = 1e-10
    abs_percentage_error = np.abs(
        (true_values - predicted_values) / (true_values + epsilon)
    )
    return round(np.mean(abs_percentage_error) * 100, 4)


def predict_values(model, df, forecast_period=0, pred_start=None):
    if model == "ARIMA":
        return forecast_autoarima(df, forecast_period)
    elif model == "XGBoost":
        return forecast_xgboost(df, forecast_period, pred_start)
    elif model == "ETS":
        return forecast_ets(df, forecast_period)
    elif model == "Prophet":
        return forecast_prophet(df, forecast_period)


def check_all_models(df, forecast_period=0, pred_start=None):
    y_pred_prophet, mape_prophet = forecast_prophet(df, forecast_period)
    y_pred_arima, mape_auto_arima = forecast_autoarima(df, forecast_period)
    y_pred_ets, mape_ets = forecast_ets(df, forecast_period)
    y_pred_xgboost, mape_xgboost = forecast_xgboost(df, forecast_period, pred_start)

    res = [
        ("Prophet", mape_prophet),
        ("ARIMA", mape_auto_arima),
        ("ETS", mape_ets),
        ("XGBoost", mape_xgboost),
    ]

    best_model = min(res, key=lambda x: x[1])[0]
    return best_model, res


def forecast_greykite(df, freq="D", forecast=False, forecast_period=100):
    metadata = MetadataParam(
        time_col="date",
        value_col="value",
        freq=freq,
        train_end_date=df["date"].iloc[-1],
    )
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=forecast_period,
            coverage=0.95,
            metadata_param=metadata,
        ),
    )

    if not forecast:
        return result.forecast.df["forecast"].tolist()[:-forecast_period]
    return result.forecast.df["forecast"].tolist()


def forecast_xgboost(df, forecast_period=0, date_to=None):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek

    X_train = df.drop(["value", "date"], axis=1)
    y_train = df["value"]

    model = xgb.XGBRegressor(max_depth=3, min_child_weight=1, reg_alpha=0.1)
    model.fit(X_train, y_train)

    past = model.predict(X_train).tolist()
    mape = calculate_mape(df["value"], past)

    if forecast_period and date_to:
        date_range = pd.date_range(
            start=pd.to_datetime(date_to) + pd.DateOffset(days=1),
            periods=forecast_period,
            freq="D",
        )
        future_data = pd.DataFrame(index=date_range)
        future_data["year"] = future_data.index.year
        future_data["month"] = future_data.index.month
        future_data["day"] = future_data.index.day
        future_data["day_of_week"] = future_data.index.dayofweek

        future_predictions = model.predict(future_data).tolist()
        past.extend(future_predictions)
        return past, mape

    return past, mape


def forecast_ets(df, forecast_period=0):
    try:
        model = ExponentialSmoothing(
            df["value"], trend="add", seasonal="add", seasonal_periods=12
        )
    except:
        model = ExponentialSmoothing(df["value"])

    result = model.fit()

    past = result.fittedvalues.tolist()
    mape = calculate_mape(df["value"], past)

    if forecast_period:
        future_data = result.forecast(steps=forecast_period)
        past.extend(future_data)
        return past, mape

    return past, mape


def forecast_prophet(df, forecast_period=0):
    df = df.rename(columns={"date": "ds", "value": "y"})

    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    mape = calculate_mape(df["y"], forecast["yhat"].tolist()[: len(df)])

    return forecast["yhat"].tolist(), mape


def forecast_autoarima(df, forecast_period=0):
    try:
        model = pm.auto_arima(df["value"].tolist(), seasonal=True, m=12)
    except:
        model = pm.auto_arima(df["value"].tolist(), seasonal=True)

    past = model.predict_in_sample().tolist()
    mape = calculate_mape(df["value"], past)
    if forecast_period:
        forecasts = model.predict(forecast_period).tolist()
        past.extend(forecasts)
        return past, mape
    return past, mape


def forecast_arima(df, forecast_period=0):
    try:
        result = adfuller(df["value"])
        p_value = result[1]

        if p_value < 0.05:
            d = 0
        else:
            d = 1

        pacf_values = pacf(df["value"], method="ols")
        acf_values = pacf(df["value"], method="ols")

        for p in range(1, len(pacf_values)):
            if abs(pacf_values[p]) < 1.96 / np.sqrt(len(df)):
                break

        for q in range(1, len(acf_values)):
            if abs(acf_values[q]) < 1.96 / np.sqrt(len(df)):
                break
    except:
        p, d, q = 1, 1, 1

    model = sm.tsa.arima.ARIMA(df["value"], order=(p, d, q))
    results = model.fit()
    past = results.fittedvalues.tolist()

    mape = calculate_mape(df["value"], past)

    if forecast_period:
        future_data = results.forecast(steps=forecast_period)
        past.extend(future_data)
        return past, mape

    return past, mape
