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

def V_obs_function_tf(S, A):
    mu, sigma, lam, jump_mean, jump_std = A
    N = tf.shape(S)[0]

    # Diffusion term
    diffusion = mu + sigma * tf.random.normal((N,1), dtype=tf.float32)

    uniform_noise = tf.random.uniform((N,1), 0, 1)
    jump_prob_soft = tf.sigmoid(2 * (lam - uniform_noise))

    # Jump term: approximate Poisson jump by Bernoulli trial
    jumps = jump_prob_soft * (jump_mean + jump_std * tf.random.normal((N,1), dtype=tf.float32))
    expected_jump = lam * (jump_mean + 0.5 * jump_std**2)

    # TODO: update to use optimizer to find global solution?

    # Return next step
    return S + diffusion + expected_jump


def initialize_params():
  mu.assign(0.1)
  sigma.assign(0.2)
  lam.assign(0.05)
  jump_mean.assign(0.0)
  jump_std.assign(0.1)
  return mu, sigma, lam, jump_mean, jump_std


class ParamNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mu = self.add_weight(name="mu", shape=(), initializer=tf.constant_initializer(0.002), trainable=True)
        self.sigma = self.add_weight(name="sigma", shape=(), initializer=tf.constant_initializer(0.2), trainable=True)
        # sigma was good at 0.15
        self.lam = self.add_weight(name="lam", shape=(), initializer=tf.constant_initializer(0.05), trainable=True)
        self.jump_mean = self.add_weight(name="jump_mean", shape=(), initializer=tf.constant_initializer(0.0), trainable=True)
        self.jump_std = self.add_weight(name="jump_std", shape=(), initializer=tf.constant_initializer(0.1), trainable=True)

    def call(self, x):
        # Ensure positive where needed
        sigma = tf.nn.softplus(self.sigma)
        lam = tf.nn.softplus(self.lam)
        jump_std = tf.nn.softplus(self.jump_std)
        return self.mu, self.sigma, self.lam, self.jump_mean, self.jump_std



# -----------------------------
# Define the neural network
# -----------------------------
class VHat(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden1 = tf.keras.layers.Dense(96, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(64, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(48, activation='tanh')
        self.hidden4 = tf.keras.layers.Dense(32, activation='tanh')
        self.hidden5 = tf.keras.layers.Dense(16, activation='tanh')
        # increased from 64, 48, 32, 24
        self.out = tf.keras.layers.Dense(1)

    def call(self, S, t):
        x = tf.concat([S, t], axis=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        return self.out(x)

v_hat_model = VHat()
param_net = ParamNet()

param_net(tf.zeros((1,0)))

# -----------------------------
# Collocation points (S, t)
# -----------------------------

# -----------------------------
# Optimizer
# -----------------------------

epochs = 4000

tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=200,
    t_mul=2,
    m_mul=1.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# -----------------------------
# Training loop
# -----------------------------
# 1500 is good
for epoch in range(epochs):

    '''
      Here is where we would expose the model to different stock data

    '''
    #random_idx = random.randint(0, len(df))
    random_idx = random.randint(0, len(df.T)-1)

    in_df = df.T.iloc[random_idx].to_numpy()

    row_scaler = MinMaxScaler(feature_range=(0, 1))
    in_df = scaler.fit_transform(in_df.reshape(-1,1)).flatten()

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
        [V_obs_function_tf(S, [mu, sigma, lam, jump_mean, jump_std]) for _ in range(20)],
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
        # 0.6 is good

    # Compute gradients for model and PDE parameters
    # tf.gradient() is the backpropogation algorithm
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

# -----------------------------
# Test prediction
# -----------------------------

# -----------------------------
# Restore best weights
#------------------------------

v_hat_model.set_weights(best_weights_vhat)
param_net.set_weights(best_weights_param)
print(f"Restored best weights from loss={best_loss:.6f}")

print(f"mu: {mu} | lam: {lam} | sigma: {sigma}")
# ========= Final Error: 3177.8146323154133 ==============
# 3000 epoch: # ========= Final Error: 2652.520220184404 ==============
# Unseen loss 1500 epoch: ========= Final Error: 12181.404879878626 ==============
# Unseen loss 3000 epoch: ========= Final Error: 2233.3614817647554 ==============

mu_ = mu.numpy().item()
lam_ = lam.numpy().item()
sigma_ = sigma.numpy().item()
jump_mean_ = jump_mean.numpy().item()
jump_std_ = jump_std.numpy().item()

print(f"mu: {mu_} | lam: {lam_} | sigma: {sigma_} | jump_mean: {jump_mean_} | jump_std: {jump_std_}")

#data = yf.download("CSCO", period="6mo", interval="1d")
#data2 = yf.download("ROK", period='6mo', interval='1d')

#data = pd.concat([data, data2])

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


mse = []
mae = []
for i in range(len(df.T)):
  row = df.T.iloc[i].to_numpy(dtype=float)

  scaler = MinMaxScaler(feature_range=(0, 1))
  row_scaled = scaler.fit_transform(row.reshape(-1,1)).flatten()

  # Simulate using optimized parameters
  X_sim_scaled = simulate_jump_diffusion([mu_, sigma_, lam_, jump_mean_, jump_std_], row_scaled)

  # Inverse transform for plotting
  X_sim = scaler.inverse_transform(X_sim_scaled.reshape(-1,1)).flatten()

  # Plot
  '''
  plt.plot(row, label="Actual")
  plt.plot(X_sim, label=f"Jump Diffusion Simulation {df.columns[i]}")
  plt.legend()
  plt.xlabel("Time")
  plt.ylabel("Metric")
  plt.title("Jump Diffusion Simulation")
  plt.show()
  '''
  print(f"Error: {np.mean((X_sim - row)**2)}")
  mse.append(np.mean((X_sim - row)**2))
  mae.append(np.mean((X_sim - row)))

mse = np.array(mse)
mae = np.array(mae)
print(f"========= Final Error Seen: {mse.mean()} ==============\n")
print(f"========= Final MAE Seen: {mae.mean()} ==============\n")

# ========= Final Error: 2961.9540288669423 ==============


# ========= TEST ON UNSEEN =============
stocks2 = ['ADSK', 'AMAT', 'BKNG', 'BMY', 'CARR', 'CL', 'CMCSA', 'CVS',
 'DE', 'DIS', 'EMR', 'FDX', 'GD', 'GILD', 'GIS', 'HCA',
 'IBM', 'ISRG', 'KMB', 'LMT', 'LOW', 'MAR', 'MDLZ', 'MMM',
 'MRK', 'NKE', 'NOW', 'ORLY', 'PANW', 'PAYX', 'PFE', 'PYPL',
 'SBUX', 'SHEL', 'SLB', 'TGT', 'TXN', 'UNH', 'UPS', 'WMT']


dd2 = pd.DataFrame()
for stock in stocks2:
  s = yf.download(stock, period='6mo', interval='1d')
  dd2 = pd.concat([dd2, s['Close'].T], ignore_index=True)

#dd = pd.concat([data['Close'].T,data2['Close'].T, data3['Close'].T, data4['Close'].T, data5['Close'].T], ignore_index=True)
dd2 = dd2.T
dd2.columns = stocks2
dd2 = dd2.ffill()
dd2 = dd2.bfill()

#data = yf.download("CSCO", period="6mo", interval="1d")
#data2 = yf.download("ROK", period='6mo', interval='1d')

#data = pd.concat([data, data2])

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
for i in range(len(df2.T)):
  row = df2.T.iloc[i].to_numpy(dtype=float)

  scaler = MinMaxScaler(feature_range=(0, 1))
  row_scaled = scaler.fit_transform(row.reshape(-1,1)).flatten()

  # Simulate using optimized parameters
  X_sim_scaled = simulate_jump_diffusion([mu_, sigma_, lam_, jump_mean_, jump_std_], row_scaled)

  # Inverse transform for plotting
  X_sim = scaler.inverse_transform(X_sim_scaled.reshape(-1,1)).flatten()

  # Plot
  plt.plot(row, label="Actual")
  plt.plot(X_sim, label=f"Jump Diffusion Simulation {df2.columns[i]}")
  plt.legend()
  plt.xlabel("Time")
  plt.ylabel("Metric")
  plt.title("Jump Diffusion Simulation")
  plt.show()

  print(f"Error: {np.mean((X_sim - row)**2)}")
  mse.append(np.mean((X_sim - row)**2))
  mae.append(np.mean((X_sim - row)))

mse = np.array(mse)
mae = np.array(mae)
print(f"========= Final Error: {mse.mean()} ==============")
print(f"========= Final MAE: {mae.mean()} ==============\n")


print(f"========== Average: =============")
# 1000 size: ========= Final Error Seen: 16623.056459472646 ==============
# ========= Final Error: 10624.528782165675 ==============

# 500 size: ========= Final Error: 5680.735928301335 ==============
# ========= Final MAE: 4.520269924836572 ==============


#data = yf.download("CSCO", period="6mo", interval="1d")
#data2 = yf.download("ROK", period='6mo', interval='1d')

#data = pd.concat([data, data2])


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
plt.ylabel("Metric")
plt.title("Jump Diffusion Simulation")
plt.show()

final_error = (np.mean((X_sim - real)))

print(f"============ MAE From Average: {final_error} =================")
