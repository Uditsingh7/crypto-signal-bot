# main.py
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_CONFIG

# Step 1: Database connection function using SQLAlchemy
def connect_to_db():
    try:
        # Create connection URL for SQLAlchemy
        connection_url = f"postgresql+psycopg2://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
        engine = create_engine(connection_url)
        return engine
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

# Step 2: Fetch latest 200 days of data for a specific coin by symbol
def fetch_coin_data(coin_symbol):
    query = """
    SELECT cp.recorded_at, cp.value_usd
    FROM coin_prices_daily cp
    JOIN coins c ON c.id = cp.coin_id
    WHERE c.symbol = %s
    ORDER BY cp.recorded_at DESC
    LIMIT 200;
    """
    engine = connect_to_db()
    if engine:
        try:
            # Use SQLAlchemy engine for the query
            df = pd.read_sql_query(query, engine, params=(coin_symbol,))
            df = df.sort_values(by='recorded_at')  # Sort by date ascending
            return df
        except Exception as e:
            print("Error fetching data:", e)
    return None

# Step 3: Calculate Moving Averages and Generate Signals
def calculate_moving_averages(df):
    # Calculate 100-day and 200-day SMA
    df['SMA_100'] = df['value_usd'].rolling(window=100).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=200).mean()
    
    # Signal generation based on SMA crossovers
    df['signal'] = 'Hold'  # Default signal
    df.loc[(df['SMA_100'] > df['SMA_200']) & (df['SMA_100'].shift(1) <= df['SMA_200'].shift(1)), 'signal'] = 'Bullish'
    df.loc[(df['SMA_100'] < df['SMA_200']) & (df['SMA_100'].shift(1) >= df['SMA_200'].shift(1)), 'signal'] = 'Bearish'
    
    # Return the most recent signal
    latest_signal = df.iloc[-1][['recorded_at', 'value_usd', 'SMA_100', 'SMA_200', 'signal']]
    return latest_signal

# Bullish Crossover: (df['SMA_100'] > df['SMA_200']) checks if the 100-day SMA is now above the 200-day SMA, 
# and df['SMA_100'].shift(1) <= df['SMA_200'].shift(1) checks if it was below or equal to the 200-day SMA on the previous day. 
# This combination detects when the 100-day SMA crosses above the 200-day SMA, indicating a bullish trend.

# Bearish Crossover: (df['SMA_100'] < df['SMA_200']) checks if the 100-day SMA is now below the 200-day SMA, 
# and df['SMA_100'].shift(1) >= df['SMA_200'].shift(1) checks if it was above or equal to the 200-day SMA on the previous day. 
# This combination detects when the 100-day SMA crosses below the 200-day SMA, indicating a bearish trend.

################################################################################################################################
# RSI: Helps identify overbought or oversold conditions, which can prevent false signals in strong trends or during reversals.
# MACD: Adds momentum-based insights by comparing short-term and long-term EMAs, helping confirm the trend's strength or potential changes.


# Calculate RSI: Use a 14-day RSI (a standard setting) to measure the strength and speed of price changes. 
# RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.

# Calculate MACD: The MACD is calculated by subtracting the 26-day EMA from the 12-day EMA. 
# The MACD Signal Line (usually a 9-day EMA of the MACD) helps identify crossovers to signal trend changes.


# SMA provides the base signal (Bullish in this case).
# RSI indicates the asset is overbought, which suggests caution and the possibility of a reversal.
# MACD confirms bullish momentum, but since RSI is overbought, it adjusts the final signal to Overbought - Potential Reversal.


# SMAs and MACD both track trend and momentum, making them trend-following indicators. 
# RSI, on the other hand, provides a contrarian signal by highlighting extreme market conditions where the price may reverse.

#### EMA Crossovers:
# Bullish Crossover: (df['EMA_50'] > df['EMA_100']) checks if the 50-day EMA is now above the 100-day EMA,
# and df['EMA_50'].shift(1) <= df['EMA_100'].shift(1) checks if it was below or equal to the 100-day EMA on the previous day.
# This combination detects when the 50-day EMA crosses above the 100-day EMA, indicating a bullish trend.

# Bearish Crossover: (df['EMA_50'] < df['EMA_100']) checks if the 50-day EMA is now below the 100-day EMA,
# and df['EMA_50'].shift(1) >= df['EMA_100'].shift(1) checks if it was above or equal to the 100-day EMA on the previous day.
# This combination detects when the 50-day EMA crosses below the 100-day EMA, indicating a bearish trend.


import pandas as pd

def calculate_indicators(df):
    # Ensure there is enough data for calculations
    if len(df) < 200:
        print("Insufficient data for reliable indicator calculation.")
        return None

    # Calculate 100-day and 200-day SMA (for long-term trends)
    df['SMA_100'] = df['value_usd'].rolling(window=100).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=200).mean()

    # Calculate 50-day and 100-day EMA (for short-term trends)
    df['EMA_50'] = df['value_usd'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['value_usd'].ewm(span=100, adjust=False).mean()

    # Corrected RSI Calculation (14-day)
    delta = df['value_usd'].diff(1)
    gain = delta.where(delta > 0, 0)  # Keep only positive gains
    loss = -delta.where(delta < 0, 0)  # Invert negative losses to positive

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Set initial RSI to neutral if NaN

    # Calculate MACD and Signal Line
    df['EMA_12'] = df['value_usd'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value_usd'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate Bollinger Bands for volatility assessment
    df['Bollinger_Mid'] = df['value_usd'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['value_usd'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (2 * df['Bollinger_Std'])
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (2 * df['Bollinger_Std'])

    # Generate signals based on SMA and EMA crossovers
    df['signal'] = 'Hold'  # Default signal
    df.loc[(df['SMA_100'] > df['SMA_200']) & (df['SMA_100'].shift(1) <= df['SMA_200'].shift(1)), 'signal'] = 'Bullish'
    df.loc[(df['SMA_100'] < df['SMA_200']) & (df['SMA_100'].shift(1) >= df['SMA_200'].shift(1)), 'signal'] = 'Bearish'
    df.loc[(df['EMA_50'] > df['EMA_100']) & (df['EMA_50'].shift(1) <= df['EMA_100'].shift(1)), 'signal'] = 'Bullish'
    df.loc[(df['EMA_50'] < df['EMA_100']) & (df['EMA_50'].shift(1) >= df['EMA_100'].shift(1)), 'signal'] = 'Bearish'

    # Enhanced signal generation considering SMA, EMA, RSI, MACD, and Bollinger Bands
    df['signal'] = df.apply(lambda row: generate_signal(row), axis=1)

    # Return the most recent signal with indicators
    latest_signal = df.iloc[-1][[
        'recorded_at', 'value_usd', 'SMA_100', 'SMA_200', 'EMA_50', 'EMA_100', 
        'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'signal'
    ]]
    return latest_signal

# Helper function for generating a combined signal
def generate_signal(row):
    # Define thresholds for RSI, with flexibility for strong trends
    rsi_signal = 'Neutral'
    if row['RSI'] > 80:
        rsi_signal = 'Overbought'
    elif row['RSI'] < 20:
        rsi_signal = 'Oversold'
    elif row['RSI'] > 70:
        rsi_signal = 'Bullish Bias'
    elif row['RSI'] < 30:
        rsi_signal = 'Bearish Bias'

    # MACD signal based on MACD and Signal Line
    macd_signal = 'Neutral'
    if row['MACD'] > row['MACD_Signal']:
        macd_signal = 'Bullish'
    elif row['MACD'] < row['MACD_Signal']:
        macd_signal = 'Bearish'

    # Combine signals for a more reliable indicator
    if row['signal'] == 'Bullish' and macd_signal == 'Bullish' and rsi_signal not in ['Overbought', 'Bearish Bias']:
        return 'Strong Bullish'
    elif row['signal'] == 'Bearish' and macd_signal == 'Bearish' and rsi_signal not in ['Oversold', 'Bullish Bias']:
        return 'Strong Bearish'
    elif rsi_signal == 'Overbought' or (row['value_usd'] > row['Bollinger_Upper']):
        return 'Overbought - Potential Reversal'
    elif rsi_signal == 'Oversold' or (row['value_usd'] < row['Bollinger_Lower']):
        return 'Oversold - Potential Reversal'
    elif row['signal'] == 'Bullish' and rsi_signal not in ['Overbought', 'Bearish Bias']:
        return 'Bullish'
    elif row['signal'] == 'Bearish' and rsi_signal not in ['Oversold', 'Bullish Bias']:
        return 'Bearish'
    else:
        return 'Hold'


# Main function to get signals for each coin
def main():
    coin_symbols = ['btc', 'eth', 'sol']  # Replace with the coin symbols you use in your database
    for coin_symbol in coin_symbols:
        print(f"\nFetching data for {coin_symbol.upper()}")
        df = fetch_coin_data(coin_symbol)  # Replace this with your actual data-fetching function
        if df is not None and len(df) >= 200:
            result = calculate_indicators(df)
            if result is not None:
                print(f"Latest Signal for {coin_symbol.upper()}:")
                print(result)
            else:
                print(f"Calculation error for {coin_symbol.upper()}")
        else:
            print(f"Not enough data to calculate indicators for {coin_symbol.upper()}")

if __name__ == "__main__":
    main()



    

# Interpretation: Although both the SMA and MACD are bullish (indicating upward momentum), the high RSI signals caution. 
# The asset might be overbought and could be due for a price correction or reversal.
# Advisory Use: An investment advisory platform could use this signal to advise clients to be cautious about new purchases or 
# even consider taking profits, given the potential for a pullback.



# RSI above 80 (for BTC and ETH, close for SOL) suggests a strong overbought condition.
# Bollinger Bands: Prices exceeding the upper Bollinger Band reinforce the overbought signal, as they are outside normal volatility ranges.
# MACD shows bullish momentum, but this doesn’t override the overbought condition; 
# rather, it suggests that the price has been trending up but may be due for a correction.
