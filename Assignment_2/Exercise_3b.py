import yfinance as yf
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# Define the ticker symbol
ticker = "AAPL"
today = dt.date(2024, 3, 25)
apple_dataframe = pd.DataFrame()
tk = yf.Ticker(ticker)

expirations = tk.options
for e in expirations:
    current_dataframe = tk.option_chain(e).calls
    current_dataframe['timeToMaturity'] = (dt.date.fromisoformat(e) - today) / dt.timedelta(days=1)
    q = current_dataframe["impliedVolatility"].quantile(0.95)
    current_dataframe = current_dataframe[current_dataframe["impliedVolatility"] < q]
    apple_dataframe = pd.concat([apple_dataframe, current_dataframe[['timeToMaturity', 'strike', 'impliedVolatility']]], ignore_index=True)

# Remove outliers
apple_dataframe = apple_dataframe.drop(apple_dataframe[apple_dataframe['impliedVolatility'] <= 0.15].index)
apple_dataframe = apple_dataframe.drop(apple_dataframe[apple_dataframe['strike'] >= 250].index)
apple_dataframe = apple_dataframe.drop(apple_dataframe[apple_dataframe['strike'] <= 60].index)
apple_dataframe = apple_dataframe.drop(apple_dataframe[apple_dataframe['impliedVolatility'] >= 2].index)
# apple_dataframe = apple_dataframe.drop(apple_dataframe[apple_dataframe['timeToMaturity'] >= 100].index)

# Plot 3D surface
x = apple_dataframe['timeToMaturity']
y = apple_dataframe['strike']
z = apple_dataframe['impliedVolatility']

ax = plt.axes(projection='3d')
surf = ax.plot_trisurf(x, y, z, alpha=0.95, cmap='viridis')
ax.set_xlabel("Time To Maturity in days")
ax.set_ylabel("Strike")
ax.set_zlabel("Implied volatility")
ax.set_title("Apple Inc. implied volatility surface (call-options)")

# Plot implied volatility line for every time to maturity
unique_times = apple_dataframe['timeToMaturity'].unique()
for time in unique_times:
    df_subset = apple_dataframe[apple_dataframe['timeToMaturity'] == time]
    ax.plot(df_subset['timeToMaturity'], df_subset['strike'], df_subset['impliedVolatility'], color='red', linewidth=2)  # Adjust linewidth here


plt.show()
