import streamlit as st
import json
import requests

st.set_page_config(page_title="DataGenie Hackathon 2023", page_icon="🌐", layout="wide")

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

st.title("API CALL📞")

format = st.text_input("Format")
date_from = st.date_input("Start Date")
date_to = st.date_input("End Date")
period = st.number_input("Period", min_value=0, step=1)
json_file = st.file_uploader("Upload JSON File", type=["json"])

if st.button("Make API Call"):
    if format and date_from and date_to and json_file:
        try:
            json_content_bytes = json_file.read()
            json_content_str = json_content_bytes.decode("utf-8")
            payload = json.loads(json_content_str)

            API_ENDPOINT = "http://127.0.0.1:8105/predict?format={}&date_from={}&date_to={}&period={}".format(
                format, date_from, date_to, period
            )

            response = requests.post(API_ENDPOINT, json=payload).json()
            st.subheader("API Response:")
            st.json(response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please Fill all the values")
