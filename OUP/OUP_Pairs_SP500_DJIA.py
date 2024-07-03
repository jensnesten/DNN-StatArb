import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

sp500 = yf.download('^GSPC', start='2000-01-01', end='2024-05-01')['Adj Close']
djia = yf.download('^DJI', start='2000-01-01', end='2024-05-01')['Adj Close']
log_returns = (np.log(sp500 / sp500.shift(1)).dropna()) - (np.log(djia / djia.shift(1)).dropna())

# Define the negative log-likelihood function
def neg_log_likelihood(params):
    theta, mu, sigma = params
    X = log_returns.values
    dt = 1
    n = len(X)
    X_diff = np.diff(X)
    X_mean = X[:-1]
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2 * dt) - 1/(2 * sigma**2 * dt) * np.sum((X_diff - theta * (mu - X_mean) * dt)**2)
    return -log_likelihood

# Initial parameter guesses
initial_params = [0.1, log_returns.mean(), log_returns.std()]

# Minimize the negative log-likelihood
result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(0, None), (None, None), (0, None)])
theta, mu, sigma = result.x

print(f"Estimated parameters: theta={theta}, mu={mu}, sigma={sigma}")


def simulate_ou_process(theta, mu, sigma, X0, T, dt):
    n = int(T / dt)
    t = np.linspace(0, T, n)
    X = np.zeros(n)
    X[0] = X0
    for i in range(1, n):
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return t, X

# Parameters
T = len(log_returns)  # Total time
dt = 1  # Time step
X0 = log_returns.iloc[0]  # Initial value

# Simulate the process
t, X_sim = simulate_ou_process(theta, mu, sigma, X0, T, dt)

X_simulated_aligned = X_sim[:len(log_returns)]

deviation = log_returns.values - X_simulated_aligned

max_deviation = np.max(np.abs(deviation))
std_deviation = np.std(deviation)

print(f"Max deviation: {max_deviation}")
print(f"Standard deviation of deviation: {std_deviation}")





log_returns_cumsum = log_returns.cumsum()
X_sim_cumsum = np.cumsum(X_sim)

# Create a DataFrame with the data
df = pd.DataFrame({
    'Date': log_returns.index,
    'Actual': log_returns_cumsum,
    'Simulated': X_sim_cumsum
})

df['Date'] = df['Date'].dt.strftime('%Y%m%d')

# Save to CSV
df.to_csv('../Data/sp500_spreadlog_returns.csv', index=False)



# Plot the results
plt.figure(figsize=(12, 6))

plt.plot(log_returns.index, log_returns, label='Actual Spread Log Returns')
plt.plot(log_returns.index, X_sim, label='Simulated OU Process')
plt.text(0.07, 0.05, f'Max_dev: {max_deviation:.4f}', transform=plt.gca().transAxes)
plt.text(0.07, 0.10, f'Std_dev: {std_deviation:.4f}', transform=plt.gca().transAxes)
plt.xticks(rotation=45)
plt.legend()
plt.title('Spread Log Returns vs. Simulated OU Process')
plt.xlabel('Date')
plt.tight_layout()
plt.ylabel('Log Returns')

plt.show()
