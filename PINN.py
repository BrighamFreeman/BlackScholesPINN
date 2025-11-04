!pip install yfinance pandas matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, differential_evolution
import yfinance as yf
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

#np.random.seed(42)

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
    'KLAC', 'LHX', 'LIN', 'LKQ', 'LRCX', 'MCK', 'MDT', 'MET', 'MGM',
    'MMC', 'MOS', 'MSCI', 'MTD', 'NDAQ', 'NTRS', 'NVR',
    'NXPI', 'ODFL', 'OKE', 'OLN', 'OTIS', 'PAYC', 'PH',
    'PNR', 'PPG', 'PSX', 'PTC', 'PWR', 'RCL', 'REGN', 'ROL',
    'ROST', 'SRE', 'STZ', 'SWK', 'SYY', 'TDG', 'TER', 'TMO',
    'TRV', 'TSCO', 'TXT', 'VLO', 'VMC', 'VRSK', 'VTR', 'WEC', 'WELL',
    'WMB', 'WRB', 'XEL', 'XYL', 'ZBH', 'ZTS',
    'FICO', 'CRWD', 'SNOW', 'OKTA', 'TEAM', 'DOCU', 'ZS', 'PINS',
    'ABNB', 'RBLX', 'U', 'BILL', 'PATH', 'ROKU', 'DKNG', 'TTD',
    'ENPH', 'SEDG', 'RUN', 'FSLR', 'CSIQ',
    'DVN', 'EOG', 'OXY', 'COP',
    'CVNA', 'RIVN', 'LCID', 'BLNK', 'CHPT',
    'CRH', 'MLM', 'MTZ', 'JCI',
    'ZBRA', 'IRDM', 'TDY', 'AVT',
    'WING', 'TXRH', 'DRI', 'DPZ',
    'RSG', 'WM', 'GFL',
    'TROW', 'AMP', 'BEN', 'AIG', 'PRU', 'CINF',
    'VRTX', 'INCY', 'ILMN', 'NVCR',
    'AER', 'ALK', 'UAL', 'DAL',
    'LEN', 'PHM', 'TOL',
    'EQIX', 'AMT', 'CCI', 'SBAC',
    'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLY', 'XLC', 'XBI'
]

# Download stock data from yfinance
dd = pd.DataFrame()
for stock in stocks:
  s = yf.download(stock, period='6mo', interval='1d')
  dd = pd.concat([dd, s['Close'].T], ignore_index=True)

# Format DataFrame and remove NaN
dd = dd.T
dd.columns = stocks
dd = dd.ffill()
dd = dd.bfill()
dd = dd.dropna(axis=1, how='all')


df = dd

# ---------------------------
#     Start of PINN
# ---------------------------

import tensorflow as tf
import random
# -----------------------------
# Define trainable parameters
# -----------------------------

r = 0.01

# -------------------------------------
#
#   Define observation function
#
# -------------------------------------

def V_obs_function_tf(S, A, dt=1.0):
    """
    Geometric jump-diffusion observation function (Merton model)
    S: tf.Tensor, shape (N,1)
    A: list of params [mu, sigma, lam, jump_mean, jump_std]
    dt: time step size
    """
    mu, sigma, lam, jump_mean, jump_std = A
    N = tf.shape(S)[0]

    # Diffusion term (scaled by S)
    dW = tf.random.normal((N,1), dtype=tf.float32) * tf.sqrt(dt)
    diffusion = mu * S * dt + sigma * S * dW

    # Jump term: approximate Poisson jumps
    # Sample number of jumps using Bernoulli approximation
    uniform_noise = tf.random.uniform((N,1), 0, 1)
    jump_prob_soft = tf.sigmoid(2 * (lam*dt - uniform_noise))  # adjust dt
    jump_sizes = jump_mean + jump_std * tf.random.normal((N,1), dtype=tf.float32)
    jumps = S * jump_prob_soft * (tf.exp(jump_sizes) - 1)

    # Return next step
    return S + diffusion + jumps

# lambda = 0.02 also works well
class ParamNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mu = self.add_weight(name="mu", shape=(), initializer=tf.constant_initializer(0.002), trainable=True)
        self.sigma = self.add_weight(name="sigma", shape=(), initializer=tf.constant_initializer(0.2), trainable=True)
        self.lam = self.add_weight(name="lam", shape=(), initializer=tf.constant_initializer(0.01), trainable=True)
        self.jump_mean = self.add_weight(name="jump_mean", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.jump_std = self.add_weight(name="jump_std", shape=(), initializer=tf.constant_initializer(0.1), trainable=True)

    def call(self, x):
        return self.mu, self.sigma, self.lam, self.jump_mean, self.jump_std



# -----------------------------
# Define the neural network
# -----------------------------
class VHat(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(128, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(96, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(64, activation='tanh')
        self.hidden4 = tf.keras.layers.Dense(48, activation='tanh')
        self.hidden5 = tf.keras.layers.Dense(32, activation='tanh')
        self.hidden6 = tf.keras.layers.Dense(16, activation='tanh')
        self.out = tf.keras.layers.Dense(1)

    def call(self, S, t):
        x = tf.concat([S, t], axis=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        return self.out(x)


# -------------------------
# Define models
# -------------------------
v_hat_model = VHat()
param_net = ParamNet()

param_net(tf.zeros((1,0)))

# -----------------------------
# Optimizer
# -----------------------------

epochs = 4000

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=epochs
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


for epoch in range(epochs):


    #random_idx = random.randint(0, len(df))
    random_idx = random.randint(0, len(df.T)-1)

    in_df = df.T.iloc[random_idx].to_numpy()

    row_scaler = MinMaxScaler(feature_range=(0, 1))
    in_df = row_scaler.fit_transform(in_df.reshape(-1,1)).flatten()

    # Add Gaussian noise
    noise = np.random.normal(0, 0.05, size=in_df.shape)
    in_df = np.clip(in_df + noise, 0, 1)  # Keep within (0,1)

    #print(in_df)
    N = len(in_df)

    S = tf.convert_to_tensor(in_df.reshape(-1,1), dtype=tf.float32)
    t_values = (df.index - df.index.min()).days
    t_values = t_values / t_values.max()  # normalize between 0 and 1
    t_values = t_values.values.reshape(-1, 1)
    t = tf.convert_to_tensor(t_values.reshape(-1, 1), dtype=tf.float32)  # <-- define t


    #print(f"V_obs: {V_obs}")

    with tf.GradientTape(persistent=True) as tape1:
        # Set the model to watch parameters S and t
        mu, sigma, lam, jump_mean, jump_std = param_net(tf.zeros((1,0)))
        # average across multiple possibilities
        V_obs = tf.reduce_mean(
        [V_obs_function_tf(S, [mu, sigma, lam, jump_mean, jump_std]) for _ in range(25)],
        axis=0
        )

        tape1.watch([S, t])
        # Forward pass through the *instance* of the model
        V = v_hat_model(S, t)

        # First derivatives
        V_S = tape1.gradient(V, S)
        V_t = tape1.gradient(V, t)

        # Second derivative (∂²V/∂S²)
        V_SS = tape1.gradient(V_S, S)

        expected_jump = lam * (jump_mean + 0.5 * jump_std**2)

        residual = V_t + mu*S*V_S + 0.5*sigma**2*S**2*V_SS - r*V + expected_jump

        # Loss = MSE of residuals
        loss_data = tf.reduce_mean(tf.square(V - V_obs))  # match network output to observed value
        loss_pde = tf.reduce_mean(tf.square(residual))
        loss = 0.5 * loss_data + loss_pde
        # Anything in the range 0.4 - 0.6 works here

    # Compute gradients for model and PDE parameters
    grads = tape1.gradient(loss, v_hat_model.trainable_variables + param_net.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, v_hat_model.trainable_variables + param_net.trainable_variables))

    # Track moving average of loss
    if epoch == 0:
        best_loss = float('inf')
        stagnant_epochs = 0
        best_weights_vhat = None
        best_weights_param = None

    if loss.numpy() < best_loss - 1e-5:
        best_loss = loss.numpy()
        stagnant_epochs = 0

        best_weights_vhat = v_hat_model.get_weights()
        best_weights_param = param_net.get_weights()

    else:
        stagnant_epochs += 1

    if stagnant_epochs > 300:  # no improvement for 300 epochs
        print("Restoring best weights...")
        v_hat_model.set_weights(best_weights_vhat)
        param_net.set_weights(best_weights_param)
        print("Applying random mutation to escape stagnation...")
        for var in param_net.trainable_variables:
            noise = tf.random.normal(var.shape, stddev=0.1 * tf.abs(var) + 1e-3)
            var.assign_add(noise)
        stagnant_epochs = 0

    if epoch == 200 or epoch == 400 or epoch == 800 or epoch == 1600 or epoch == 3200:
      for var in param_net.trainable_variables:
            noise = tf.random.normal(var.shape, stddev=0.1 * tf.abs(var) + 1e-3)
            var.assign_add(noise)
      print("Mutated variables")


    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}, mu: {mu.numpy():.3f}, sigma: {sigma.numpy():.3f}, lam: {lam.numpy():.3f}")
        print(f"Best weights: {best_weights_param}")
        print(f"Best loss: {best_loss}")


# -----------------------------
# Restore best weights
#------------------------------

v_hat_model.set_weights(best_weights_vhat)
param_net.set_weights(best_weights_param)
print(f"Restored best weights from loss={best_loss:.6f}")

# Output final parameters
print(f"mu: {mu} | lam: {lam} | sigma: {sigma}")

mu_ = mu.numpy().item()
lam_ = lam.numpy().item()
sigma_ = sigma.numpy().item()
jump_mean_ = jump_mean.numpy().item()
jump_std_ = jump_std.numpy().item()

print(f"mu: {mu_} | lam: {lam_} | sigma: {sigma_} | jump_mean: {jump_mean_} | jump_std: {jump_std_}")



def simulate_jump_diffusion(params, row_scaled, dt=1):
    m, s, l, jm, js = params
    T = len(row_scaled)
    X = np.zeros(T)
    X[0] = row_scaled[0]

    for t in range(1, T):
        try:
          Nj = np.random.poisson(l * dt)
        except:
          Nj = np.random.poisson(0.001 * dt)
        jump = np.sum(np.random.normal(jm, js, Nj))
        diffusion = m * dt + s * np.sqrt(dt) * np.random.randn()
        X[t] = X[t-1] + diffusion + jump

    # Optional: clip within scaled range to prevent blow-ups
    #X_sim = np.clip(X, 0, np.max(row_scaled))
    return X

def loss(params, row_scaled, sims=150):
    mses = []
    for _ in range(sims):
        X_sim = simulate_jump_diffusion(params, row_scaled)
        mses.append(np.mean((X_sim - row_scaled)**2))
    return np.mean(mses)


mse = []
mae = []
maep = []
for i in range(len(df.T)):
  row = df.T.iloc[i].to_numpy(dtype=float)

  scaler = MinMaxScaler(feature_range=(0, 1))
  row_scaled = scaler.fit_transform(row.reshape(-1,1)).flatten()

  # Simulate using optimized parameters
  X_sim_scaled = simulate_jump_diffusion([mu_, sigma_, lam_, jump_mean_, jump_std_], row_scaled)

  # Inverse transform for plotting
  X_sim = scaler.inverse_transform(X_sim_scaled.reshape(-1,1)).flatten()

  # Plot

  plt.plot(row, label="Actual")
  plt.plot(X_sim, label=f"Jump Diffusion Simulation {df.columns[i]}")
  plt.legend()
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.title("Jump Diffusion Simulation")
  plt.show()

  print(f"Error: {np.mean((X_sim - row)**2)}")
  mse.append(np.mean((X_sim - row)**2))
  mae.append(np.mean((X_sim - row)))
  maep.append(np.mean(np.abs((X_sim - row)/row)))


mse = np.array(mse)
mae = np.array(mae)
maep = np.array(maep)
print(f"========= Final Error Seen: {mse.mean()} ==============\n")
print(f"========= Final MAE Seen: {mae.mean()} ==============\n")
print(f"========= Final MAE Percent Seen: {maep.mean()} ==============\n")


# ========================================
# ========= TEST ON UNSEEN =============
# ========================================
stocks2 = ['ADSK', 'AMAT', 'BKNG', 'BMY', 'CARR', 'CL', 'CMCSA', 'CVS',
 'DE', 'DIS', 'EMR', 'FDX', 'GD', 'GILD', 'GIS', 'HCA',
 'IBM', 'ISRG', 'KMB', 'LMT', 'LOW', 'MAR', 'MDLZ', 'MMM',
 'MRK', 'NKE', 'NOW', 'ORLY', 'PANW', 'PAYX', 'PFE', 'PYPL',
 'SBUX', 'SHEL', 'SLB', 'TGT', 'TXN', 'UNH', 'UPS', 'WMT',
           'HP', 'OI', 'SDRL', 'KLIC', 'PLUG',
           'OKLO', 'CCJ']


# ===================================
# Download new stocks
# ===================================
dd2 = pd.DataFrame()
for stock in stocks2:
  s = yf.download(stock, period='6mo', interval='1d')
  dd2 = pd.concat([dd2, s['Close'].T], ignore_index=True)

dd2 = dd2.T
dd2.columns = stocks2
dd2 = dd2.ffill()
dd2 = dd2.bfill()


def simulate_jump_diffusion(params, row_scaled, dt=1):
    m, s, l, jm, js = params
    T = len(row_scaled)
    X = np.zeros(T)
    X[0] = row_scaled[0]  # already scaled, strictly positive

    for t in range(1, T):
        try:
          Nj = np.random.poisson(l * dt)
        except:
          Nj = np.random.poisson(0.001 * dt)
        jump = np.sum(np.random.normal(jm, js, Nj))
        diffusion = m * dt + s * np.sqrt(dt) * np.random.randn()
        X[t] = X[t-1] + diffusion + jump

    # Optional: clip within scaled range to prevent blow-ups
    X_sim = np.clip(X, 0, np.max(row_scaled))
    return X

def loss(params, row_scaled, sims=150):
    mses = []
    for _ in range(sims):
        X_sim = simulate_jump_diffusion(params, row_scaled)
        mses.append(np.mean((X_sim - row_scaled)**2))
    return np.mean(mses)

df2 = dd2
mse = []
mae = []
maep = []

for i in range(len(df2.T)):
  row = df2.T.iloc[i].to_numpy(dtype=float)

  scaler = MinMaxScaler(feature_range=(0, 1))
  row_scaled = scaler.fit_transform(row.reshape(-1,1)).flatten()

  X_sim_scaled = simulate_jump_diffusion([mu_, sigma_, lam_, jump_mean_, jump_std_], row_scaled)

  # Inverse transform for plotting
  X_sim = scaler.inverse_transform(X_sim_scaled.reshape(-1,1)).flatten()
  #X_sim = X_sim_scaled
  # Plot
  plt.plot(row, label="Actual")
  plt.plot(X_sim, label=f"Jump Diffusion Simulation {df2.columns[i]}")
  plt.legend()
  plt.xlabel("Time")
  plt.ylabel("Price")
  plt.title("Jump Diffusion Simulation")
  plt.show()

  print(f"Error: {np.mean((X_sim - row)**2)}")
  mse.append(np.mean((X_sim - row)**2))
  mae.append(np.mean((X_sim - row)))
  maep.append(np.mean(np.abs((X_sim - row)/row)))


mse = np.array(mse)
mae = np.array(mae)
maep = np.array(maep)
print(f"========= Final Error: {mse.mean()} ==============")
print(f"========= Final MAE: {mae.mean()} ==============\n")
print(f"========= Final MAE: {maep.mean()} ==============\n")


print(f"========== Average: =============")


row_sum = df2.T.to_numpy(dtype=float)

row_sum = row_sum.mean(axis=0)

fin_scaler = MinMaxScaler(feature_range=(0, 1))
row_scaled = fin_scaler.fit_transform(row_sum.reshape(-1,1)).flatten()

# Simulate using optimized parameters
X_sim_scaled = simulate_jump_diffusion([mu_, sigma_, lam_, jump_mean_, jump_std_], row_scaled)

# Inverse transform for plotting
X_sim = fin_scaler.inverse_transform(X_sim_scaled.reshape(-1,1)).flatten()

real = fin_scaler.inverse_transform(row_scaled.reshape(-1,1)).flatten()
# Plot
plt.plot(real, label="Simulation vs Averaged Market Data")
plt.plot(X_sim, label=f"Jump Diffusion Simulation")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Jump Diffusion Simulation")
plt.show()

final_error = (np.mean((X_sim - real)))

print(f"============ MAE From Average: {final_error} =================")

