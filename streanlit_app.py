import streamlit as st
import requests
import pandas as pd
import os

API_URL = os.getenv("API_URL",   "http://localhost:8000")

st.set_page_config(page_title= "Shopease Sentiment Dashboard", layout="wide")

st.title("Shopease Sentiment Analysis Dashboard")
st.markdown("Analysis customer reviews")

st.header("Single Review Prediction")

#Add a UNIQUE key here
user_input = st.text_area("Enter customer review:", key="single_review_input")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": user_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Sentiment: {result['sentiment']}")
                else:
                    st.error("Error from API")
            except Exception as e:
                st.error(f"Connection error: {e}")
                
st.divider()

st.header("Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV with 'review' column", type=["csv"], key="batch_file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if st.button("Run Batch Prediction", key="batch_predict"):
        with st.spinner("Processing batch prediction"):
            try:
                response = requests.post(
                    f"{API_URL}/predict/batch",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                )

                if response.status_code == 200:
                    results = pd.DataFrame(response.json())

                    st.success("Batch Prediction Completed")
                    st.dataframe(results)

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        key="download_results"
                    )
                else:
                    st.error("Error from API")

            except Exception as e:
                st.error(f"Error> {e}")

st.divider()

st.header("Model Training")

st.warning("Note: this may take time.")

if st.button("Retrain Model", key="retrain"):
    with st.spinner("Training model..."):
        try:
            response = requests.get(f"{API_URL}/train")

            if response.status_code==200:
                st.success("Training triggered successfully")
            else:
                st.error("Training failed")
        
        except Exception as e:
            st.error(f"Error: {e}")