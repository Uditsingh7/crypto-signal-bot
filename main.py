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

# Step 4: Main function to get signals for each coin
def main():
    coin_symbols = ['btc', 'eth', 'sol']  # Replace with the coin symbols you use in your database
    for coin_symbol in coin_symbols:
        print(f"\nFetching data for {coin_symbol.upper()}")
        df = fetch_coin_data(coin_symbol)
        if df is not None and len(df) >= 200:
            result = calculate_moving_averages(df)
            print(f"Latest Signal for {coin_symbol.upper()}:")
            print(result)
        else:
            print(f"Not enough data to calculate moving averages for {coin_symbol.upper()}")

if __name__ == "__main__":
    main()
