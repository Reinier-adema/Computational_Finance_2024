import yfinance as yf

# Define the ticker symbol
ticker = "AAPL"

# Get the option chain
option_chain = yf.Ticker(ticker).option_chain('2024-04-12')

calls = option_chain.calls

# Display today's call option prices for all strike prices and maturities
print(calls)
