import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the Keras model
model = load_model("C:\StockPrice\StockModel.h5")

st.header('Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

try:
    data = yf.download(stock, start, end)
except Exception as e:
    st.error(f"Error fetching data for {stock}: {e}")
    st.stop()

if data.empty:
    st.warning(f"No data available for {stock} in the specified date range.")
    st.stop()

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Use the same scaler for both training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])
data_train_scaled = scaler.transform(data_train)
data_test_scaled = scaler.transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data.Close, 'g', label='Closing Price')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data.Close, 'g', label='Closing Price')
ax2.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data.Close, 'g', label='Closing Price')
ax3.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

# ... (rest of your code)

st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(predict, 'r', label='Predicted Price')
ax4.plot(y, 'g', label='Original Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)