stocks = [
    'CSCO', 'ROK', 'VIS', 'LUNR', 'NUKZ', 'AMZN', 'AAPL', 'ORCL',
    'JPM', 'BAC', 'KO', 'C', 'PEP', 'RKLB', 'MS', 'COF', 'BA', 'CAT', 'WFC',
    'COST', 'HON', 'XOM', 'APD', 'VZ', 'PLD', 'BYND', 'NVDA', 'AVGO',
    'META', 'BRK-B', 'TCEHY', 'LLY', 'V', 'MA', 'NFLX', 'PLTR', 'ABBV',
    'HD', '601288.SS', 'PG', 'SAP', 'GE', 'AZN', 'TMUS', 'RTX', '601988.SS',
    'AOS', 'ABT', 'AYI', 'ADBE', 'AFL', 'AMCR', 'T', 'AZO',
    'BIIB', 'BLK', 'CAG', 'CMI', 'DAL', 'D', 'EA', 'GM',
    'NLR', 'JNJ', 'MCD', 'NEE', 'MDB', 'BHP', 'RIO.L', '1COV.DE',
    'HSBA.L', 'DBK.DE', 'AA', 'AAME', 'AAOI', 'AAON', 'AAPG',
    'AAUC', 'AB', 'NEM', 'PPTA', 'HYMC', 'ABTC', 'ACB',
    'ACEL', 'ACHR', 'ACNT', 'AEP', 'AFRM', 'AGI', 'AI', 'ANVS', 'AOSL',
    'APEI', 'APPF', 'APWC', 'ARR', 'AS', 'ASTL', 'TAK',
    'RDY', 'HLN', 'USB', 'DB', 'MTB', 'MFG', 'RGTI', 'QBTS', 'NOK',
    'BBAI', 'IONQ', 'SOFI', 'SMCI', 'SOXS', 'TSLL', 'SQQQ',
    'MSTU', 'SPY', 'QQQ', 'DASH', 'CHWY', 'W', 'ETSY', 'ITGR', 'SMPL',
    'MOH', 'R', 'URI', 'KNX', 'OII', 'WH', 'INTC', 'KHC',
    'WBA', 'SAH', 'STM', 'UUUU', 'UNF',
    'EXPE', 'OWL', 'SRPT', 'BHVN', 'NEOG', 'PCVX', 'FLNC',
    'GLOB', 'VSCO', 'JANX', 'ADP', 'ALL', 'AMGN', 'ANET', 'ATO', 'AVB', 'BDX', 'BKR',
    'CNC', 'CLX', 'CMG', 'CSX', 'CTSH', 'CTAS', 'DD', 'DHI', 'DHR',
    'DLR', 'DTE', 'DXC', 'EBAY', 'ECL', 'EL', 'EMN', 'ETN', 'EXC',
    'F', 'FAST', 'FIS', 'FTNT', 'GPC', 'GS', 'HAL', 'HAS',
    'HIG', 'HOLX', 'HUM', 'IDXX', 'ITW', 'KMI', 'KEYS',
    'KLAC', 'LHX', 'LIN', 'LKQ', 'LRCX', 'MCK', 'MDT', 'MET', 'MGM'
]


dd = pd.DataFrame()
for stock in stocks:
  s = yf.download(stock, period='24mo', interval='1d')
  dd = pd.concat([dd, s['Close'].T], ignore_index=True)

#dd = pd.concat([data['Close'].T,data2['Close'].T, data3['Close'].T, data4['Close'].T, data5['Close'].T], ignore_index=True)
dd = dd.T
dd.columns = stocks
dd = dd.ffill()
dd = dd.bfill()
dd = dd.dropna(axis=1, how='all')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Fourier smoothing function
# -----------------------------
def fourier_smooth_series(x, keep_ratio=0.1):
    n = len(x)
    if n < 2 or keep_ratio >= 1.0:
        return x.copy()
    Xf = np.fft.rfft(x)
    n_freq = Xf.size
    keep = max(1, int(np.ceil(n_freq * keep_ratio)))
    Xf_filtered = np.zeros_like(Xf)
    Xf_filtered[:keep] = Xf[:keep]
    x_smooth = np.fft.irfft(Xf_filtered, n=n)
    return x_smooth

# -----------------------------
# Apply smoothing column by column
# -----------------------------
def fourier_smooth_dataframe(df, keep_ratio=0.1):
    df_smooth = df.copy()
    for c in df.columns:
        df_smooth[c] = fourier_smooth_series(df[c].values.astype(float), keep_ratio=keep_ratio)
    return df_smooth

# -----------------------------
# LSTM sequence creation (1D series)
# -----------------------------
def create_lstm_sequences_1d(series, time_steps=10, n_output_steps=1):
    X, y = [], []
    for i in range(len(series) - time_steps - n_output_steps + 1):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps:i+time_steps+n_output_steps])
    return np.array(X), np.array(y)

# -----------------------------
# Parameters
# -----------------------------
KEEP_RATIO = 1.0
TIME_STEPS = 10
N_OUTPUT_STEPS = 1
EPOCHS = 25
BATCH_SIZE = 32

# -----------------------------
# Prepare data
# -----------------------------
lstm = dd.copy()
final_df = dd.copy()  # Original unsmoothed data
lstm_smooth = fourier_smooth_dataframe(lstm, keep_ratio=KEEP_RATIO)

# -----------------------------
# Train/predict each column individually
# -----------------------------
results = {}

maes = []
mses = []
mae_p = []

# Build LSTM model
model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=13, kernel_size=3, activation='tanh', input_shape=(TIME_STEPS, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(32, activation='tanh', return_sequences=False),
        tf.keras.layers.Dense(N_OUTPUT_STEPS)
])

model.compile(optimizer='adam', loss='mse')

for col in lstm_smooth.columns:
    print(f"\n=== Processing column: {col} ===")

    series_smooth = lstm_smooth[col].values.reshape(-1, 1)
    X, y = create_lstm_sequences_1d(series_smooth, TIME_STEPS, N_OUTPUT_STEPS)

    # -----------------------------
    # Train-test split BEFORE scaling (prevents leakage)
    # -----------------------------
    n_seq = X.shape[0]
    train_size = int(n_seq * 0.75)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # -----------------------------
    # Fit scalers ONLY on training data
    # -----------------------------
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(y_train.shape[0], -1))

    # Apply same scaler to test data (no fitting)
    X_test_scaled = scaler_X.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(y_test.shape[0], -1))

    # -----------------------------
    # Train model
    # -----------------------------
    model.fit(X_train_scaled, y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_test_scaled, y_test_scaled))

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test_scaled)

    # -----------------------------
    # Align with original data
    # -----------------------------
    n_test = len(y_test)
    y_indices = np.arange(TIME_STEPS + train_size, TIME_STEPS + train_size + n_test)
    unsmoothed_aligned = final_df[[col]].tail(n_test).values.flatten()

    # -----------------------------
    # Store and evaluate
    # -----------------------------
    results[col] = (y_test_inv.flatten(), y_pred.flatten())

    mae = np.mean(np.abs(y_pred.flatten() - unsmoothed_aligned))
    mse = np.mean((y_pred.flatten() - unsmoothed_aligned)**2)
    maep = np.mean(np.abs((y_pred.flatten() - unsmoothed_aligned) / unsmoothed_aligned))

    maes.append(mae)
    mses.append(mse)
    mae_p.append(maep)

    print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, MAE%: {maep:.6f}")
    print(f"Running Avg MAE%: {np.mean(mae_p):.6f}")

# -----------------------------
# Final aggregate results
# -----------------------------
print(f"\nFinal Results:")
print(f"MAE: {np.mean(maes):.6f} | MSE: {np.mean(mses):.6f} | MAE%: {np.mean(mae_p):.6f}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

stocks2 = ['ADSK', 'AMAT', 'BKNG', 'BMY', 'CARR', 'CL', 'CMCSA', 'CVS',
 'DE', 'DIS', 'EMR', 'FDX', 'GD', 'GILD', 'GIS', 'HCA',
 'IBM', 'ISRG', 'KMB', 'LMT', 'LOW', 'MAR', 'MDLZ', 'MMM',
 'MRK', 'NKE', 'NOW', 'ORLY', 'PANW', 'PAYX', 'PFE', 'PYPL',
 'SBUX', 'SHEL', 'SLB', 'TGT', 'TXN', 'UNH', 'UPS', 'WMT',
           'HP', 'OI', 'SDRL', 'KLIC', 'PLUG',
           'OKLO', 'CCJ', 'GOOG', 'TSLA', 'NIO',
           'ARM', 'MSTR', 'APA', 'WYNN', 'CZR', 'FCX',
           'MRO', 'IPGP', 'UPST', 'ZM', 'ASAN', 'NET',
           'DDOG', 'SHOP', 'APP', 'DXCM', 'BBWI',
           'PALI', 'ARBK', 'WDC', 'CELH', 'DUOL',
           'SWAV', 'LITE', 'NUE', 'STLD',
           'CLF', 'GM', 'AVGO', 'BABA', 'TCEHY', 'NTES',
           'LI', 'XPENG', 'IQ', 'HTHT', 'HUYA', 'STNE', 'PETR4',
           'BBAS3',
           'TROW', 'AMP', 'BEN', 'AIG', 'PRU', 'CINF',
    'VRTX', 'INCY', 'ILMN', 'NVCR',
    'AER', 'ALK', 'UAL', 'DAL',
    'LEN', 'PHM', 'TOL',
    'EQIX', 'AMT', 'CCI', 'SBAC',
    'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLY', 'XLC', 'XBI', 'CRWD']


dd2 = pd.DataFrame()
for stock in stocks2:
  s = yf.download(stock, period='24mo', interval='1d')
  dd2 = pd.concat([dd2, s['Close'].T], ignore_index=True)

#dd = pd.concat([data['Close'].T,data2['Close'].T, data3['Close'].T, data4['Close'].T, data5['Close'].T], ignore_index=True)
dd2 = dd2.T
dd2.columns = stocks2
dd2 = dd2.ffill()
dd2 = dd2.bfill()
dd2 = dd2.dropna(axis=1, how='all')

# -----------------------------
# Fourier smoothing function
# -----------------------------
def fourier_smooth_series(x, keep_ratio=0.1):
    n = len(x)
    if n < 2 or keep_ratio >= 1.0:
        return x.copy()
    Xf = np.fft.rfft(x)
    n_freq = Xf.size
    keep = max(1, int(np.ceil(n_freq * keep_ratio)))
    Xf_filtered = np.zeros_like(Xf)
    Xf_filtered[:keep] = Xf[:keep]
    x_smooth = np.fft.irfft(Xf_filtered, n=n)
    return x_smooth

# -----------------------------
# Apply smoothing column by column
# -----------------------------
def fourier_smooth_dataframe(df, keep_ratio=0.1):
    df_smooth = df.copy()
    for c in df.columns:
        df_smooth[c] = fourier_smooth_series(df[c].values.astype(float), keep_ratio=keep_ratio)
    return df_smooth

# -----------------------------
# LSTM sequence creation (1D series)
# -----------------------------
def create_lstm_sequences_1d(series, time_steps=10, n_output_steps=1):
    X, y = [], []
    for i in range(len(series) - time_steps - n_output_steps + 1):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps:i+time_steps+n_output_steps])
    return np.array(X), np.array(y)

# -----------------------------
# Parameters
# -----------------------------
KEEP_RATIO = 1.0
TIME_STEPS = 10
N_OUTPUT_STEPS = 1
EPOCHS = 10
BATCH_SIZE = 10

# -----------------------------
# Prepare data
# -----------------------------
lstm = dd2.copy()
final_df = dd2.copy()  # Original unsmoothed data
lstm_smooth = fourier_smooth_dataframe(lstm, keep_ratio=KEEP_RATIO)

# -----------------------------
# Train/predict each column individually
# -----------------------------
results = {}

maes = []
mses = []
mae_p = []
for col in lstm_smooth.columns:
    print(f"\n=== Processing column: {col} ===")

    series_smooth = lstm_smooth[col].values.reshape(-1, 1)
    X, y = create_lstm_sequences_1d(series_smooth, TIME_STEPS, N_OUTPUT_STEPS)

    # -----------------------------
    # Train-test split BEFORE scaling (prevents leakage)
    # -----------------------------
    n_seq = X.shape[0]
    train_size = int(n_seq * 0.75)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # -----------------------------
    # Fit scalers ONLY on training data
    # -----------------------------
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(y_train.shape[0], -1))

    # Apply same scaler to test data (no fitting)
    X_test_scaled = scaler_X.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(y_test.shape[0], -1))

    # Predict
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test_scaled)

    # -----------------------------
    # Align original unsmoothed data for plotting
    # -----------------------------
    n_test = len(y_test)
    y_indices = np.arange(TIME_STEPS + train_size, TIME_STEPS + train_size + n_test)
    unsmoothed_aligned = final_df[[col]].tail(n_test).values

    unsmoothed_aligned = unsmoothed_aligned[:,0]

    # -----------------------------
    # Store results
    # -----------------------------
    results[col] = (y_test_inv.flatten(), y_pred.flatten())

    # -----------------------------
    # Plot
    # -----------------------------

    plt.figure(figsize=(10,4))
    plt.plot(y_indices, y_pred.flatten(), label='Predicted', color='red')
    plt.plot(y_indices, unsmoothed_aligned, label='Original Data', alpha=0.5, color='black')
    plt.title(f"{col}: Original vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # -----------------------------
    # Metrics
    # -----------------------------
    print(unsmoothed_aligned)
    print(y_pred.shape)
    mae = np.mean(np.abs(y_pred.flatten() - unsmoothed_aligned.flatten()))
    mse = np.mean((y_pred.flatten() - unsmoothed_aligned.flatten())**2)
    maep = np.mean(np.abs((y_pred.flatten() - unsmoothed_aligned.flatten())/unsmoothed_aligned.flatten()))

    maes.append(mae)
    mses.append(mse)
    mae_p.append(maep)
    print(f"MAE: {mae:.6f}, MSE: {mse:.6f}, MAE P: {maep:.6f}")

maes = np.array(maes)
maes = np.mean(maes)
mses = np.array(mses)
mses = np.mean(mses)
mae_p = np.array(mae_p)
mae_p = np.mean(mae_p)

print(f" MAE: {maes} | MSE: {mses} | MAE Percent: {mae_p}")
