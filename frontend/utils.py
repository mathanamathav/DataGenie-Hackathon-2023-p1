import plotly.graph_objects as go
import pandas as pd


def plot_grapgh(data):
    result_data = data["result"]
    df = pd.DataFrame(result_data)

    df["point_timestamp"] = pd.to_datetime(df["point_timestamp"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["point_timestamp"],
            y=df["point_value"],
            mode="lines+markers",
            name="Point Value",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["point_timestamp"],
            y=df["yhat"],
            mode="lines+markers",
            name="yhat",
        )
    )

    fig.update_layout(
        title=f"Model: {data['model']} - MAPE: {data['mape']}",
        xaxis_title="Date",
        yaxis_title="Value",
    )

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    return fig
