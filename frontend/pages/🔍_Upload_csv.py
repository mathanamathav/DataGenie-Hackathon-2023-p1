import streamlit as st
import json
import requests
import pandas as pd

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

    if not (
        "daily" not in filename
        or "weekly" not in filename
        or "monthly" not in filename
        or "hourly" not in filename
    ):
        st.warning(
            "The file name should adhere to the format daily, hourly, weekly, monthly"
        )
        return False
    return True


def make_api_call(format, date_from, date_to, period, payload):
    API_ENDPOINT = "http://127.0.0.1:8105/predict?format={}&date_from={}&date_to={}&period={}".format(
        format, date_from, date_to, period
    )

    try:
        response = requests.post(API_ENDPOINT, json=payload).json().get("message")
        if response:
            return response
        else:
            st.error(f"API call failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed with error: {str(e)}")
    return None


if st.button("Do the Magic"):
    if csv_file is not None:
        try:
            csv_data = pd.read_csv(csv_file , index_col = 0)

            if check_csv_format(csv_data, csv_file.name):
                api_response = make_api_call(
                    format="daily",
                    date_from="2023-09-29",
                    date_to="2023-09-29",
                    period=Period,
                    payload=csv_data.to_dict(orient="records"),
                )

                if api_response is not None:
                    st.subheader("API Response:")
                    st.json(api_response)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please upload a CSV file.")

