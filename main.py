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
# def fetch_coin_data(coin_symbol):
#     query = """
#     SELECT cp.recorded_at, cp.value_usd
#     FROM coin_prices_daily cp
#     JOIN coins c ON c.id = cp.coin_id
#     WHERE c.symbol = %s
#     ORDER BY cp.recorded_at DESC
#     LIMIT 200;
#     """
#     engine = connect_to_db()
#     if engine:
#         try:
#             # Use SQLAlchemy engine for the query
#             df = pd.read_sql_query(query, engine, params=(coin_symbol,))
#             df = df.sort_values(by='recorded_at')  # Sort by date ascending
#             return df
#         except Exception as e:
#             print("Error fetching data:", e)
#     return None

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



# Step 2: Fetch data for a specific coin
def fetch_coin_data(coin_symbol):
    query = """
    SELECT cp.recorded_at, cp.value_usd, cp.twenty_four_hour_trading_volume_usd
    FROM coin_prices_daily cp
    JOIN coins c ON c.id = cp.coin_id
    WHERE c.symbol = %s
    ORDER BY cp.recorded_at DESC
    LIMIT 364;
    """
    engine = connect_to_db()
    if engine:
        try:
            df = pd.read_sql_query(query, engine, params=(coin_symbol,))
            df = df.sort_values(by='recorded_at')  # Sort by date ascending
            return df
        except Exception as e:
            print("Error fetching data:", e)
    return None


def calculate_atr(df, period=14):
    # If only closing price (value_usd) is available, use its daily change as a proxy for volatility
    df['TR'] = df['value_usd'].diff().abs()  # Use absolute daily price change
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df



# Step 3: Calculate Indicators Efficiently
def calculate_indicators(df):
    # Handle missing values by forward filling for value_usd and interpolating volume
    df['value_usd'] = df['value_usd'].ffill()  # Forward fill for value_usd
    df['twenty_four_hour_trading_volume_usd'] = df['twenty_four_hour_trading_volume_usd'].interpolate(method='linear')


    if len(df) < 200:
        print("Insufficient data for reliable indicator calculation.")
        return None

    # Simple Moving Averages (100-day and 200-day SMA)
    df['SMA_100'] = df['value_usd'].rolling(window=100).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=200).mean()

    # Exponential Moving Averages (50-day and 100-day EMA)
    df['EMA_50'] = df['value_usd'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['value_usd'].ewm(span=100, adjust=False).mean()

    # MACD and Signal Line
    df['EMA_12'] = df['value_usd'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value_usd'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI Calculation
    delta = df['value_usd'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs)).fillna(50)

    # Bollinger Bands
    rolling_mean = df['value_usd'].rolling(window=20).mean()
    rolling_std = df['value_usd'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + (2 * rolling_std)
    df['Bollinger_Lower'] = rolling_mean - (2 * rolling_std)

    # On-Balance Volume (OBV) and its 14-day Moving Average
    df['OBV'] = (df['twenty_four_hour_trading_volume_usd'] *
                 (df['value_usd'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=14).mean()

    # Calculate ATR for volatility analysis
    df = calculate_atr(df)  # Adds ATR to the DataFrame

    # Generate signal based on indicators (with ATR consideration)
    df['signal'] = df.apply(generate_signal, axis=1)
    
    # Return the latest row with calculated indicators and signal
    latest_signal = df.iloc[-1][[
        'recorded_at', 'value_usd', 'SMA_100', 'SMA_200', 'EMA_50', 'EMA_100',
        'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'OBV', 'OBV_MA', 'ATR', 'signal'
    ]]
    return latest_signal

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
    macd_signal = 'Bullish' if row['MACD'] > row['MACD_Signal'] else 'Bearish'

    # OBV trend check: comparing OBV to its 14-day moving average
    obv_signal = 'Bullish' if row['OBV'] > row['OBV_MA'] else 'Bearish'

    # Volatility check using ATR: Define high volatility as ATR > 2% of current price
    high_volatility = row['ATR'] > row['value_usd'] * 0.02

    # Combine signals for a more reliable indicator with ATR consideration
    if macd_signal == 'Bullish' and rsi_signal not in ['Overbought', 'Bearish Bias'] and obv_signal == 'Bullish':
        return 'Strong Bullish - High Volatility' if high_volatility else 'Strong Bullish'
    elif macd_signal == 'Bearish' and rsi_signal not in ['Oversold', 'Bullish Bias'] and obv_signal == 'Bearish':
        return 'Strong Bearish - High Volatility' if high_volatility else 'Strong Bearish'
    elif rsi_signal == 'Overbought' or (row['value_usd'] > row['Bollinger_Upper']):
        return 'Overbought - Potential Reversal'
    elif rsi_signal == 'Oversold' or (row['value_usd'] < row['Bollinger_Lower']):
        return 'Oversold - Potential Reversal'
    elif macd_signal == 'Bullish' and obv_signal == 'Bullish':
        return 'Bullish - High Volatility' if high_volatility else 'Bullish'
    elif macd_signal == 'Bearish' and obv_signal == 'Bearish':
        return 'Bearish - High Volatility' if high_volatility else 'Bearish'
    else:
        return 'Hold'





def predict_price(df, days=7):
    latest = df.iloc[-1]
    prediction = latest['value_usd']
    
    # Define a base multiplier for the number of days (e.g., 7 days has a slight multiplier increase)
    days_multiplier = 1 + (days / 100)  # Small increase per day to account for a longer timeframe

    # Apply adjustments using EMA, SMA, and Bollinger Bands for price prediction
    if latest['EMA_50'] > latest['EMA_100'] and latest['value_usd'] > latest['SMA_100']:
        prediction = min(latest['Bollinger_Upper'], latest['value_usd'] * 1.05 * days_multiplier)
    elif latest['EMA_50'] < latest['EMA_100'] and latest['value_usd'] < latest['SMA_100']:
        prediction = max(latest['Bollinger_Lower'], latest['value_usd'] * 0.95 * days_multiplier)

    # MACD and RSI based adjustment
    if latest['MACD'] > 0 and latest['RSI'] < 70:
        prediction *= 1.02
    elif latest['MACD'] < 0 and latest['RSI'] > 30:
        prediction *= 0.98

    # OBV trend adjustment
    if pd.notna(latest['OBV']) and pd.notna(latest['OBV_MA']):
        if latest['OBV'] > latest['OBV_MA']:
            prediction *= 1.01  # Slightly increase prediction if OBV is above its moving average
        else:
            prediction *= 0.99  # Slightly decrease if OBV is below its moving average

    # Overbought/Oversold check with Bollinger Bands
    if latest['value_usd'] > latest['Bollinger_Upper']:
        prediction = max(latest['Bollinger_Lower'], latest['value_usd'] * 0.97)
    elif latest['value_usd'] < latest['Bollinger_Lower']:
        prediction = min(latest['Bollinger_Upper'], latest['value_usd'] * 1.03)

    # ATR-based adjustment for volatility sensitivity
    atr_multiplier = 1 + (latest['ATR'] / latest['value_usd'])

    # Apply ATR multiplier to increase or decrease prediction range based on volatility
    if latest['ATR'] > latest['value_usd'] * 0.02:  # High volatility
        prediction *= atr_multiplier * days_multiplier
    else:  # Low volatility, apply minimal ATR adjustment
        prediction *= 0.995 * days_multiplier if latest['value_usd'] > latest['Bollinger_Lower'] else 1 / atr_multiplier

    return round(prediction, 2)




# Step 5: Main function to get signals for each coin
def main():
    coin_symbols = ['btc']  # Add your desired coin symbols here
    for coin_symbol in coin_symbols:
        print(f"\nFetching data for {coin_symbol.upper()}")
        df = fetch_coin_data(coin_symbol)
        if df is not None and len(df) >= 200:
            result = calculate_indicators(df)
            if result is not None:
                print(f"Latest Signal for {coin_symbol.upper()}:")
                print(result)
                
                # Generate and display the 7-day price prediction
                predicted_7d_price = predict_price(df, days=7)
                print(f"Predicted 7-day price for {coin_symbol.upper()}: {predicted_7d_price}")
                
                # Generate and display the 14-day price prediction
                predicted_14d_price = predict_price(df, days=14)
                print(f"Predicted 14-day price for {coin_symbol.upper()}: {predicted_14d_price}")
            else:
                print(f"Calculation error for {coin_symbol.upper()}")
        else:
            print(f"Not enough data to calculate indicators for {coin_symbol.upper()}")


if __name__ == "__main__":
    main()






    
