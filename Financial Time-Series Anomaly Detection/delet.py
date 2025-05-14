
---

### 🔧 **Tools & Libraries Required**

```bash
pip install yfinance pandas numpy matplotlib scikit-learn seaborn ta keras tensorflow
```

---

## ✅ **Step-by-Step Breakdown**

---

### **1. Download and Preprocess Historical Stock Price Data**

Use `yfinance` to get historical stock data for selected companies like Apple (AAPL), Microsoft (MSFT), etc.

```python
import yfinance as yf
import pandas as pd

# Download historical stock data
companies = ['AAPL', 'MSFT', 'GOOG']
data = {ticker: yf.download(ticker, start='2018-01-01', end='2023-01-01') for ticker in companies}
```

Preprocess (keep only Close price, drop NA, normalize if needed):

```python
for ticker in data:
    data[ticker] = data[ticker][['Close']].dropna()
```

---

### **2. Calculate Financial Indicators**

Use `ta` (technical analysis) library to calculate SMA, EMA, RSI, Bollinger Bands.

```python
import ta

for ticker in data:
    df = data[ticker]
    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    data[ticker] = df.dropna()
```

---

### **3. Unsupervised Anomaly Detection (Isolation Forest / DBSCAN)**

#### Using Isolation Forest:

```python
from sklearn.ensemble import IsolationForest

ticker = 'AAPL'
df = data[ticker].copy()

features = ['Close', 'SMA', 'EMA', 'RSI', 'BB_high', 'BB_low']
model = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = model.fit_predict(df[features])
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})
```

---

### **4. LSTM Time-Series Forecasting Model**

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare data
df = data['AAPL'].copy()
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Close']])

# Create sequences
def create_sequences(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(df_scaled)

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)

# Forecast
predictions = model.predict(X)
```

---

### **5. Visualize Anomalies on Stock Price Trends**

```python
import matplotlib.pyplot as plt

df_plot = df[-len(predictions):].copy()
df_plot['Predicted'] = scaler.inverse_transform(predictions)

plt.figure(figsize=(14, 6))
plt.plot(df_plot['Close'], label='Actual Price')
plt.plot(df_plot['Predicted'], label='Predicted Price', alpha=0.7)
plt.scatter(df.index[df['anomaly'] == 1], df[df['anomaly'] == 1]['Close'], color='red', label='Anomaly', marker='X')
plt.title("Stock Price with Anomaly Detection")
plt.legend()
plt.show()
```

---

## 🧠 **Bonus Tips**

* Try other models like **Prophet**, **ARIMA**, or **AutoEncoder** for experimentation.
* Use **multivariate LSTM** if you want to include indicators in forecasting.

---

Would you like me to generate a full, ready-to-run `.ipynb` Jupyter notebook for this?
