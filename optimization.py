# Install required packages
!pip install yfinance
!pip install PyPortfolioOpt

import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Define stock tickers across various sectors
tickers = ['PFE', 'LLY', 'JNJ', 'CVS', 'UNH', 'VRNA', 'AMGN', 'SYK',
          'EOG', 'MPC', 'CVX', 'OXY', 'KMI', 'HES', 'OKE', 'WMB',
          'JPM', 'V', 'C', 'GS', 'BLK', 'AXP', 'BAC', 'FNF',
          'TMUS', 'VZ', 'T', 'NXST', 'NFLX', 'DIS', 'FOXA', 'NWSA',
          'SCCO', 'SHW', 'PPG', 'VMC', 'NEM', 'NUE', 'FCX', 'CTVA',
          'NVDA', 'MSFT', 'AVGO', 'ORCL', 'AAPL', 'GOOGL', 'AMZN', 'META',
          'BA', 'GE', 'DE', 'LMT', 'RTX', 'CAT', 'INTC', 'APH',
          'TSLA', 'NKE', 'SBUX', 'MCD',
          'WMT', 'PM', 'PG', 'KO', 'NSRGY', 'PEP', 'CL', 'GIS', 'KMB', 'CHD']

# Download 'Close' prices (auto-adjusted)
data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Close']

# Drop any columns (stocks) that are all NaNs
data = data.dropna(axis=1, how='all')

# Calculate daily returns
returns = data.pct_change().dropna()

# Display first few rows of returns
print("First 5 rows of returns data:")
print(returns.head())
print("\n" + "="*50 + "\n")

# Portfolio Optimization using PyPortfolioOpt
print("Starting Portfolio Optimization...")

# 1. Calculate expected returns
mu = expected_returns.mean_historical_return(data)

# 2. Calculate the sample covariance matrix of returns
S = risk_models.sample_cov(data)

# 3. Set up the optimizer
ef = EfficientFrontier(mu, S)

# 4. Add constraint: max 3% per asset (note: original comment said 10% but constraint was 0.03)
ef.add_constraint(lambda w: w <= 0.03)

# 5. Maximize Sharpe Ratio
weights = ef.max_sharpe()

# 6. Clean the weights (remove tiny allocations)
cleaned_weights = ef.clean_weights()
print("Optimized Portfolio Weights:")
print(cleaned_weights)
print("\n" + "="*50 + "\n")

# 7. Get performance metrics
print("Portfolio Performance Metrics:")
performance = ef.portfolio_performance(verbose=True)

print("\n" + "="*50 + "\n")

# Plotting the Efficient Frontier
from pypfopt import plotting
import matplotlib.pyplot as plt

print("Generating Efficient Frontier Plot...")

# Re-initialize with a less verbose solver
ef = EfficientFrontier(mu, S, solver="SCS")
ef.add_constraint(lambda w: w <= 0.03)

# Plot efficient frontier
fig, ax = plt.subplots(figsize=(10, 6))
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Add max Sharpe point
ret_tangent, std_tangent, _ = ef.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", color="r", s=100, label="Max Sharpe")

ax.set_title("Efficient Frontier with Max Sharpe Portfolio")
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()