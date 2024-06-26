import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import requests
import os

# Function to download the model file from GitHub
def download_model_from_github(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(output_path, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return False

# Define the GitHub URL and local model path
github_url = "https://github.com/tar37583/Stock-Price-Web-App/blob/d5190121d337ff6008c186ce509b8dfcfb7f9879/Stock%20Predictions%20Model.keras"
local_model_path = "downloaded_model.keras"

# Download and load the model
if download_model_from_github(github_url, local_model_path):
    model = load_model(local_model_path)
else:
    st.error("Failed to download and load the model")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price vs MA50')
plt.legend(['MA50', 'Close'])
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price vs MA50 vs MA100')
plt.legend(['MA50', 'MA100', 'Close'])
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price vs MA100 vs MA200')
plt.legend(['MA100', 'MA200', 'Close'])
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original Price vs Predicted Price')
plt.legend()
st.pyplot(fig4)
