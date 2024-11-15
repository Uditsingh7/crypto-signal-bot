# main.py
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_CONFIG
import requests
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

def calculate_adx(df, period=14):
    # Calculate Directional Movements
    df['UpMove'] = df['value_usd'].diff().apply(lambda x: x if x > 0 else 0)
    df['DownMove'] = -df['value_usd'].diff().apply(lambda x: x if x < 0 else 0)
    
    # Smooth the directional movements
    df['+DM'] = df['UpMove'].rolling(window=period).sum()
    df['-DM'] = df['DownMove'].rolling(window=period).sum()

    # Calculate True Range (TR)
    df['TR'] = df['value_usd'].diff().abs()  # Using absolute price change as a proxy for TR

    # Smooth True Range and Directional Indicators
    df['TR'] = df['TR'].rolling(window=period).sum()
    df['+DI'] = 100 * (df['+DM'] / df['TR'])
    df['-DI'] = 100 * (df['-DM'] / df['TR'])

    # Calculate Directional Index and ADX
    df['DX'] = (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()  # ADX as the smoothed DX

    return df

def calculate_mfi(df, period=14):
    # Calculate the Price Change
    df['Price_Change'] = df['value_usd'].diff()

    # Calculate Raw Money Flow (Price Change x Volume)
    df['Raw_Money_Flow'] = df['Price_Change'] * df['twenty_four_hour_trading_volume_usd']

    # Separate positive and negative money flows
    positive_flow = df['Raw_Money_Flow'].apply(lambda x: x if x > 0 else 0)
    negative_flow = df['Raw_Money_Flow'].apply(lambda x: -x if x < 0 else 0)

    # Calculate Money Flow Ratio
    money_flow_ratio = positive_flow.rolling(window=period).sum() / negative_flow.rolling(window=period).sum()

    # Calculate MFI
    df['MFI'] = 100 - (100 / (1 + money_flow_ratio))
    return df



def get_fear_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return int(data['data'][0]['value'])  # Returns current index (0-100)
    else:
        print("Failed to fetch Fear & Greed Index")
        return None
    
def categorize_by_volume(volume):
    """
    Categorizes a coin based on its 24-hour trading volume.
    - High Volume: > $1 billion
    - Medium Volume: $100 million to $1 billion
    - Low Volume: < $100 million
    """
    if volume > 1_000_000_000:
        return 'High Liquidity'
    elif volume > 100_000_000:
        return 'Medium Liquidity'
    else:
        return 'Low Liquidity'

    
# Step 3: Calculate Indicators Efficiently
def calculate_indicators(df):
    # Handle missing values
    df['value_usd'] = df['value_usd'].ffill()
    df['twenty_four_hour_trading_volume_usd'] = df['twenty_four_hour_trading_volume_usd'].interpolate(method='linear')

    if len(df) < 200:
        print("Insufficient data for reliable indicator calculation.")
        return None

    # Calculate liquidity category
    latest_volume = df['twenty_four_hour_trading_volume_usd'].iloc[-1]
    liquidity_category = categorize_by_volume(latest_volume)

    # Simple Moving Averages (SMA)
    df['SMA_50'] = df['value_usd'].rolling(window=50).mean()
    df['SMA_100'] = df['value_usd'].rolling(window=100).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=200).mean()

    # Exponential Moving Averages (EMA)
    df['EMA_50'] = df['value_usd'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['value_usd'].ewm(span=100, adjust=False).mean()
    df['EMA_20'] = df['value_usd'].ewm(span=20, adjust=False).mean()

    # VWAP
    df['VWAP'] = (df['value_usd'] * df['twenty_four_hour_trading_volume_usd']).cumsum() / df['twenty_four_hour_trading_volume_usd'].cumsum()

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

    # OBV and its moving average
    df['OBV'] = (df['twenty_four_hour_trading_volume_usd'] *
                 (df['value_usd'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=14).mean()

    # ATR and ADX
    df = calculate_atr(df)
    df = calculate_adx(df)

    # MFI
    df = calculate_mfi(df)

    # Generate signal
    df['signal'] = df.apply(generate_signal, axis=1)

    # Return the latest row with indicators and signal
    latest_signal = df.iloc[-1][[
        'recorded_at', 'value_usd', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_50', 'EMA_100', 'EMA_20',
        'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower',
        'OBV', 'OBV_MA', 'ATR', 'ADX', 'MFI', 'VWAP', 'signal'
    ]].to_dict()
    
    latest_signal['Liquidity'] = liquidity_category  # Add liquidity category to the result
    
    return latest_signal

def generate_signal(row):
    # Define thresholds for RSI and MFI
    rsi_signal = 'Neutral'
    if row['RSI'] > 80:
        rsi_signal = 'Overbought'
    elif row['RSI'] < 20:
        rsi_signal = 'Oversold'

    mfi_signal = 'Neutral'
    if row['MFI'] > 80:
        mfi_signal = 'Overbought'
    elif row['MFI'] < 20:
        mfi_signal = 'Oversold'

    # MACD, OBV, and ADX-based signals
    macd_signal = 'Bullish' if row['MACD'] > row['MACD_Signal'] else 'Bearish'
    obv_signal = 'Bullish' if row['OBV'] > row['OBV_MA'] else 'Bearish'
    trend_strength = "Strong" if row['ADX'] >= 25 else "Weak"

    # VWAP check for momentum confirmation
    vwap_signal = 'Above' if row['value_usd'] > row['VWAP'] else 'Below'

    # EMA crossover signal (20-day and 50-day)
    ema_crossover_signal = 'Bullish' if row['EMA_20'] > row['EMA_50'] else 'Bearish'

    # Combined Signal
    if macd_signal == 'Bullish' and rsi_signal not in ['Overbought'] and obv_signal == 'Bullish' and trend_strength == "Strong" and mfi_signal != 'Overbought' and vwap_signal == 'Above' and ema_crossover_signal == 'Bullish':
        return 'Strong Bullish'
    elif macd_signal == 'Bearish' and rsi_signal not in ['Oversold'] and obv_signal == 'Bearish' and trend_strength == "Strong" and mfi_signal != 'Oversold' and vwap_signal == 'Below' and ema_crossover_signal == 'Bearish':
        return 'Strong Bearish'
    elif rsi_signal == 'Overbought' or mfi_signal == 'Overbought' or (row['value_usd'] > row['Bollinger_Upper']):
        return 'Overbought - Potential Reversal'
    elif rsi_signal == 'Oversold' or mfi_signal == 'Oversold' or (row['value_usd'] < row['Bollinger_Lower']):
        return 'Oversold - Potential Reversal'
    elif trend_strength == "Weak":
        return 'Hold - Weak Trend'
    elif macd_signal == 'Bullish' and obv_signal == 'Bullish':
        return 'Bullish'
    elif macd_signal == 'Bearish' and obv_signal == 'Bearish':
        return 'Bearish'
    else:
        return 'Hold'

def predict_price(df, days=7):
    latest = df.iloc[-1]
    current_price = latest['value_usd']
    prediction = current_price
    days_multiplier = 1 + (days / 100)

    # Trend and crossover analysis
    if latest['EMA_50'] > latest['EMA_100'] and current_price > latest['SMA_100']:
        prediction = min(latest['Bollinger_Upper'], current_price * 1.05 * days_multiplier)
    elif latest['EMA_50'] < latest['EMA_100'] and current_price < latest['SMA_100']:
        prediction = max(latest['Bollinger_Lower'], current_price * 0.95 * days_multiplier)

    # VWAP adjustment
    if current_price > latest['VWAP']:
        prediction *= 1.01
    else:
        prediction *= 0.99

    # Mean reversion adjustment
    if latest['RSI'] > 80 or latest['MFI'] > 80 or current_price > latest['Bollinger_Upper']:
        prediction = (prediction + latest['SMA_50']) / 2
    elif latest['RSI'] < 20 or latest['MFI'] < 20 or current_price < latest['Bollinger_Lower']:
        prediction = (prediction + latest['SMA_50']) / 2

    # ATR and ADX adjustment
    atr_multiplier = 1 + (latest['ATR'] / current_price)
    if latest['ADX'] >= 25:
        prediction *= atr_multiplier * days_multiplier
    else:
        prediction *= 0.98 * days_multiplier

    # Liquidity adjustment
    if categorize_by_volume(latest['twenty_four_hour_trading_volume_usd']) == 'Low Liquidity':
        prediction *= 0.95  # More conservative for low-liquidity coins

    # Calculate percentage change
    percentage_change = ((prediction - current_price) / current_price) * 100

    return round(percentage_change, 2)


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
                
                # Generate and display the 1-day price prediction
                predicted_24h_price = predict_price(df, days=1)
                print(f"Predicted 24-hour price: {coin_symbol.upper()}: {predicted_24h_price}")            
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






    
