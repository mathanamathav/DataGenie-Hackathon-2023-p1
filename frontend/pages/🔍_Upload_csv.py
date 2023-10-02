import streamlit as st
import json
import requests
import pandas as pd
from utils import plot_grapgh
import plotly.io as pio

pio.templates.default = "plotly"

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #008B8B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CSV UPLOAD")

Period = st.number_input("Enter Period", value=0)

csv_file = st.file_uploader("Choose a CSV file", type=["csv"])


def check_csv_format(dataframe, filename):
    if len(dataframe.columns) != 2:
        st.warning(
            "The CSV file should contain exactly two columns: 'date' and 'time'."
        )
        return False

    # if "date" not in dataframe.columns or "time" not in dataframe.columns:
    #     st.warning("The CSV file should contain columns named 'date' and 'time'.")
    #     return False

    return True


def make_api_call(date_from, date_to, period, payload):
    API_ENDPOINT = (
        "http://127.0.0.1:8105/predict?date_from={}&date_to={}&period={}".format(
            date_from, date_to, period
        )
    )

    try:
        response = requests.post(API_ENDPOINT, json=payload).json()
        if response:
            return response
        else:
            st.error(f"API call failed: {response}")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed with error: {str(e)}")
    return None


if st.button("Do the Magic"):
    if csv_file is not None:
        try:
            csv_data = pd.read_csv(csv_file)

            if check_csv_format(csv_data, csv_file.name):
                csv_data.columns = ["date", "value"]

                csv_data["date"] = pd.to_datetime(csv_data["date"])

                csv_data.set_index(csv_data["date"], inplace=True)
                csv_data.sort_index(inplace=True)
                csv_data.rename_axis("index", inplace=True)

                date_from = csv_data["date"][0].date()
                date_to = csv_data["date"][-1].date()

                csv_data["date"] = csv_data["date"].astype(str)
                csv_data.columns = ["point_timestamp", "point_value"]

                api_response = make_api_call(
                    date_from=date_from,
                    date_to=date_to,
                    period=Period,
                    payload=csv_data.to_dict(orient="records"),
                )

                if api_response is not None:
                    st.plotly_chart(plot_grapgh(api_response), use_container_width=True)
                    st.subheader("API Response:")
                    st.json(api_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a CSV file.")
