import tensorflow as tf
import random
# -----------------------------
# Define trainable parameters
# -----------------------------

r = 0.03

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
    # Return next step
    return S + diffusion + jumps


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
        self.mu = self.add_weight(name="mu", shape=(), initializer=tf.constant_initializer(0.01), trainable=True)
        self.sigma = self.add_weight(name="sigma", shape=(), initializer=tf.constant_initializer(0.2), trainable=True)
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
        self.hidden1 = tf.keras.layers.Dense(48, activation='tanh')
        self.hidden2 = tf.keras.layers.Dense(32, activation='tanh')
        self.hidden3 = tf.keras.layers.Dense(16, activation='tanh')
        self.out = tf.keras.layers.Dense(1)

    def call(self, S, t):
        x = tf.concat([S, t], axis=1)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
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
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# -----------------------------
# Training loop
# -----------------------------
epochs = 100
for epoch in range(epochs):

    '''
      Here is where we would expose the model to different stock data

    '''
    #random_idx = random.randint(0, len(df))
    random_idx = random.randint(0, len(df.T)-1)
    
    in_df = df.T.iloc[random_idx].to_numpy()
    
    row_scaler = MinMaxScaler(feature_range=(0, 1))
    in_df = scaler.fit_transform(in_df.reshape(-1,1)).flatten()
    
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
        [V_obs_function_tf(S, [mu, sigma, lam, jump_mean, jump_std]) for _ in range(5)],
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
        loss = 0.75 * loss_data + loss_pde

    # Compute gradients for model and PDE parameters
    # tf.gradient() is the backpropogation algorithm
    grads = tape1.gradient(loss, v_hat_model.trainable_variables + param_net.trainable_variables)

    # Apply gradients
    optimizer.apply_gradients(zip(grads, v_hat_model.trainable_variables + param_net.trainable_variables))

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy():.6f}, mu: {mu.numpy():.3f}, sigma: {sigma.numpy():.3f}, lam: {lam.numpy():.3f}")

# -----------------------------
# Test prediction
# -----------------------------
S_test = tf.constant([[150.0]], dtype=tf.float32)
t_test = tf.constant([[0.5]], dtype=tf.float32)
V_pred = v_hat_model(S_test, t_test)
print("Predicted V_hat(S=150, t=0.5):", V_pred.numpy())

print(f"mu: {mu} | lam: {lam} | sigma: {sigma}")
