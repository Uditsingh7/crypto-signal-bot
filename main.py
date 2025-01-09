# main.py
import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_CONFIG
import requests
# import talib
import numpy as np
import json
import math
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
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
    SELECT cp.recorded_at, cp.value_usd, cp.twenty_four_hour_trading_volume_usd, c.image_url, c.name
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
# New helper function for dynamic period calculation
def get_dynamic_period(df, base_period=14):
    """
    Helper function to calculate dynamic periods based on market volatility
    Returns adjusted period length for technical indicators
    """
    volatility = df['value_usd'].pct_change().std()
    if volatility > 0.05:  # High volatility
        return max(int(base_period * 0.7), 5)
    elif volatility < 0.02:  # Low volatility
        return min(int(base_period * 1.3), 30)
    return base_period

def calculate_atr(df, period=14):
    """
    Enhanced ATR with dynamic period adjustment
    """
    period = get_dynamic_period(df, period)
    df['TR'] = df['value_usd'].diff().abs()
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Add volatility normalized ATR
    df['ATR_normalized'] = df['ATR'] / df['value_usd'] * 100
    return df

def calculate_adx(df, period=14):
    period = get_dynamic_period(df, period)
    df['UpMove'] = df['value_usd'].diff().apply(lambda x: x if x > 0 else 0)
    df['DownMove'] = -df['value_usd'].diff().apply(lambda x: x if x < 0 else 0)
    
    df['+DM'] = df['UpMove'].rolling(window=period).sum()
    df['-DM'] = df['DownMove'].rolling(window=period).sum()
    
    df['TR'] = df['value_usd'].diff().abs()
    df['TR'] = df['TR'].rolling(window=period).sum()
    
    df['+DI'] = 100 * (df['+DM'] / df['TR'])
    df['-DI'] = 100 * (df['-DM'] / df['TR'])
    
    df['DX'] = abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    return df

# New function to calculate confidence score
def calculate_signal_confidence(row):
    """
    Calculate confidence score (0-100) for trading signals
    Higher score indicates stronger signal reliability
    """
    confidence = 0
    
    # Trend agreement checks (40 points max)
    if row['EMA_20'] > row['SMA_50'] and row['MACD'] > row['MACD_Signal']:
        confidence += 40
    elif row['EMA_20'] < row['SMA_50'] and row['MACD'] < row['MACD_Signal']:
        confidence += 40
    
    # RSI confirmation (20 points max)
    if (row['RSI'] > 70) or (row['RSI'] < 30):
        confidence += 20
    
    # Volume confirmation (20 points max)
    if isinstance(row['OBV'], (int, float)) and isinstance(row['OBV_MA'], (int, float)):
        if row['OBV'] > row['OBV_MA']:
            confidence += 20
    
    # Price vs VWAP (20 points max)
    if isinstance(row['VWAP'], (int, float)):
        current_price = row['value_usd']
        if (current_price > row['VWAP'] and confidence >= 40) or \
           (current_price < row['VWAP'] and confidence >= 40):
            confidence += 20
    
    return confidence


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

def calculate_rsi(df):
    """
    Enhanced RSI calculation with dynamic periods and volume weighting
    """
    period = get_dynamic_period(df, 14)
    delta = df['value_usd'].diff()
    
    # Add volume weighting
    volume_weight = df['twenty_four_hour_trading_volume_usd'] / df['twenty_four_hour_trading_volume_usd'].mean()
    gain = (delta.where(delta > 0, 0) * volume_weight).fillna(0)
    loss = (-delta.where(delta < 0, 0) * volume_weight).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def adjust_for_small_dataset(df):
    """
    Adjust window lengths dynamically for small datasets.
    """
    max_len = len(df)
    df['SMA_50'] = df['value_usd'].rolling(window=min(max_len, 50)).mean()
    df['SMA_100'] = df['value_usd'].rolling(window=min(max_len, 100)).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=min(max_len, 200)).mean()
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

# Step 3: Calculate Indicators Efficiently
def calculate_indicators(df):
    """
    Enhanced indicator calculation while maintaining original return structure
    Adds volatility-based adjustments and improved calculations
    """
    # Calculate base volatility for dynamic adjustments
    df['volatility'] = df['value_usd'].pct_change().rolling(window=20).std()
    
    # Handle missing values
    df['value_usd'] = df['value_usd'].ffill().bfill()
    df['twenty_four_hour_trading_volume_usd'] = df['twenty_four_hour_trading_volume_usd'].interpolate(method='linear')

    if len(df) < 200:
        print("Insufficient data for reliable indicator calculation. Using reduced window lengths.")
        df = adjust_for_small_dataset(df)

    # Cache reusable calculations
    rolling_50 = df['value_usd'].rolling(window=50)
    rolling_100 = df['value_usd'].rolling(window=100)
    rolling_200 = df['value_usd'].rolling(window=200)
    rolling_20 = df['value_usd'].rolling(window=20)

    # Liquidity category
    latest_volume = df['twenty_four_hour_trading_volume_usd'].iloc[-1]
    liquidity_category = categorize_by_volume(latest_volume)

    # Enhanced Moving Averages with volatility adjustment
    volatility_factor = 1 + df['volatility']
    df['SMA_50'] = rolling_50.mean() * volatility_factor
    df['SMA_100'] = rolling_100.mean() * volatility_factor
    df['SMA_200'] = rolling_200.mean() * volatility_factor

    df['EMA_50'] = df['value_usd'].ewm(span=50, adjust=False).mean() * volatility_factor
    df['EMA_100'] = df['value_usd'].ewm(span=100, adjust=False).mean() * volatility_factor
    df['EMA_20'] = df['value_usd'].ewm(span=20, adjust=False).mean() * volatility_factor

    # Enhanced VWAP with volume normalization
    cumulative_volume = df['twenty_four_hour_trading_volume_usd'].cumsum()
    cumulative_volume = cumulative_volume.replace(0, 1e-12)
    df['VWAP'] = (df['value_usd'] * df['twenty_four_hour_trading_volume_usd']).cumsum() / cumulative_volume

    # Enhanced MACD with volatility adaptation
    volatility_mean = df['volatility'].mean()
    ema_12 = df['value_usd'].ewm(span=12, adjust=False).mean()
    ema_26 = df['value_usd'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema_12 - ema_26) * (1 + volatility_mean)
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI with volume weighting
    delta = df['value_usd'].diff()
    volume_weight = df['twenty_four_hour_trading_volume_usd'] / df['twenty_four_hour_trading_volume_usd'].mean()
    gain = (delta.where(delta > 0, 0) * volume_weight).fillna(0)
    loss = (-delta.where(delta < 0, 0) * volume_weight).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Enhanced Bollinger Bands with volatility adjustment
    rolling_mean_20 = rolling_20.mean()
    rolling_std_20 = rolling_20.std()
    volatility_multiplier = 2 * (1 + df['volatility'])
    df['Bollinger_Upper'] = rolling_mean_20 + (volatility_multiplier * rolling_std_20)
    df['Bollinger_Lower'] = rolling_mean_20 - (volatility_multiplier * rolling_std_20)

    # ATR with volatility adjustment
    df = calculate_atr(df)
    df['ATR'] = df['ATR'] * (1 + df['volatility'])

    # ADX with improved trend strength calculation
    df = calculate_adx(df)

    # Enhanced MFI
    df = calculate_mfi(df)

    # Enhanced OBV with volume trend
    try:
        price_direction = df['value_usd'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df['OBV'] = (df['twenty_four_hour_trading_volume_usd'] * price_direction).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=14).mean()
    except Exception as e:
        print(f"Error calculating OBV: {e}")
        df['OBV'] = None
        df['OBV_MA'] = None

    # Generate signal with the latest row
    try:
        latest_signal = df.iloc[-1][[
            'recorded_at', 'value_usd', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_50', 'EMA_100', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower',
            'OBV', 'OBV_MA', 'ATR', 'ADX', 'MFI', 'VWAP'
        ]].to_dict()
    except KeyError as e:
        print(f"Error accessing required columns: {e}")
        latest_signal = {"error": f"Missing required data: {e}"}

    # Add sentiment and market condition
    result = generate_signal(df.iloc[-1])
    latest_signal.update(result)
    latest_signal['Liquidity'] = liquidity_category

    return latest_signal

def generate_signal(row):
    """
    Enhanced signal generation with comprehensive technical analysis
    Returns detailed market sentiment and conditions
    """
    # Initialize scoring components
    trend_score = 0
    momentum_score = 0
    volume_score = 0
    strength_score = 0

    # Trend Analysis (40 points)
    # EMA Analysis
    if row['EMA_20'] > row['EMA_50'] and row['EMA_50'] > row['EMA_100']:
        trend_score += 20
        if row['value_usd'] > row['EMA_20']:  # Strong uptrend
            trend_score += 10
    elif row['EMA_20'] < row['EMA_50'] and row['EMA_50'] < row['EMA_100']:
        trend_score -= 20
        if row['value_usd'] < row['EMA_20']:  # Strong downtrend
            trend_score -= 10
    
    # MACD Analysis
    if row['MACD'] > row['MACD_Signal']:
        trend_score += 10
    else:
        trend_score -= 10

    # Momentum Analysis (30 points)
    # RSI Analysis
    if row['RSI'] > 70:
        momentum_score -= 15
    elif row['RSI'] < 30:
        momentum_score += 15
    else:
        momentum_score += ((row['RSI'] - 30) / 40) * 15

    # MFI Analysis
    if row['MFI'] > 80:
        momentum_score -= 15
    elif row['MFI'] < 20:
        momentum_score += 15
    else:
        momentum_score += ((row['MFI'] - 20) / 60) * 15

    # Volume Analysis (30 points)
    if row['OBV'] > row['OBV_MA']:
        volume_score += 15
        if row['value_usd'] > row['VWAP']:  # Price confirms volume
            volume_score += 5
    else:
        volume_score -= 15
        if row['value_usd'] < row['VWAP']:  # Price confirms volume
            volume_score -= 5

    # ATR and ADX for Trend Strength
    if row['ADX'] > 25:  # Strong trend
        strength_score = 20 * (row['ADX'] / 100)
        if trend_score > 0:  # Amplify existing trend
            trend_score *= 1.2
        else:
            trend_score *= 1.2

    # Calculate total score
    total_score = trend_score + momentum_score + volume_score

    # Determine detailed sentiment
    if total_score >= 80 and row['ADX'] > 25:
        sentiment = 'Strong Bullish with Trend Confirmation'
    elif total_score >= 80:
        sentiment = 'Strong Bullish with Caution'
    elif total_score >= 40 and volume_score > 0:
        sentiment = 'Bullish with Volume Support'
    elif total_score >= 40:
        sentiment = 'Moderately Bullish'
    elif total_score <= -80 and row['ADX'] > 25:
        sentiment = 'Strong Bearish with Trend Confirmation'
    elif total_score <= -80:
        sentiment = 'Strong Bearish with Caution'
    elif total_score <= -40 and volume_score < 0:
        sentiment = 'Bearish with Volume Confirmation'
    elif total_score <= -40:
        sentiment = 'Moderately Bearish'
    else:
        if row['ADX'] < 20:
            sentiment = 'Neutral - Ranging Market'
        else:
            sentiment = 'Neutral with Developing Trend'

    # Enhanced Market Condition Analysis
    bb_position = (row['value_usd'] - row['Bollinger_Lower']) / (row['Bollinger_Upper'] - row['Bollinger_Lower'])
    
    if row['RSI'] > 75 and row['MFI'] > 80:
        if row['ADX'] > 25 and volume_score > 0:
            market_condition = 'Overbought - Trend Continuation Likely'
        else:
            market_condition = 'Strongly Overbought - Reversal Risk'
    elif row['RSI'] > 70 or row['value_usd'] > row['Bollinger_Upper']:
        if volume_score > 0 and row['ADX'] > 25:
            market_condition = 'Overbought with Strong Momentum'
        else:
            market_condition = 'Overbought - Watch for Reversal'
    elif row['RSI'] < 25 and row['MFI'] < 20:
        if row['ADX'] > 25 and volume_score < 0:
            market_condition = 'Oversold - Trend Continuation Likely'
        else:
            market_condition = 'Strongly Oversold - Reversal Potential'
    elif row['RSI'] < 30 or row['value_usd'] < row['Bollinger_Lower']:
        if volume_score < 0 and row['ADX'] > 25:
            market_condition = 'Oversold with Strong Momentum'
        else:
            market_condition = 'Oversold - Watch for Reversal'
    elif row['ADX'] > 25:
        if bb_position > 0.8:
            market_condition = 'Strong Trend - Upper Range'
        elif bb_position < 0.2:
            market_condition = 'Strong Trend - Lower Range'
        else:
            market_condition = 'Strong Trend - Mid Range'
    else:
        if 0.4 <= bb_position <= 0.6:
            market_condition = 'Neutral - Consolidating'
        else:
            market_condition = 'Neutral - Range Bound'

    # Calculate confidence based on indicator agreement
    confidence_factors = [
        abs(trend_score) / 40,  # Trend agreement
        abs(momentum_score) / 30,  # Momentum strength
        abs(volume_score) / 30,  # Volume confirmation
        strength_score / 20  # Trend strength
    ]
    confidence_score = (sum(confidence_factors) / len(confidence_factors)) * 100

    return {
        'sentiment': sentiment,
        'marketCondition': market_condition,
        'confidence_score': confidence_score,
        'trend_strength': strength_score,
        'volume_quality': volume_score
    }
    
def predict_price(df, days=7):
    """
    Enhanced price prediction with more realistic bounds and market conditions
    """
    latest = df.iloc[-1]
    current_price = latest['value_usd']
    
    # Calculate historical volatility
    historical_volatility = df['value_usd'].pct_change().std() * math.sqrt(252)  # Annualized
    
    # Calculate max realistic daily move based on historical data
    max_daily_move = df['value_usd'].pct_change().abs().quantile(0.95)  # 95th percentile of daily moves
    
    # Base prediction setup
    days_multiplier = 1 + (days / 365)  # More conservative time scaling
    max_move = max_daily_move * math.sqrt(days)  # Scale with square root of time
    
    # Initialize prediction components
    trend_component = 0
    momentum_component = 0
    volatility_component = 0
    
    # Trend Analysis (30% weight)
    if latest['EMA_20'] > latest['EMA_50'] and latest['EMA_50'] > latest['EMA_100']:
        trend_component = 1
    elif latest['EMA_20'] < latest['EMA_50'] and latest['EMA_50'] < latest['EMA_100']:
        trend_component = -1
        
    # Momentum Analysis (30% weight)
    rsi_factor = (latest['RSI'] - 50) / 50  # Normalize RSI to -1 to 1
    macd_factor = 1 if latest['MACD'] > latest['MACD_Signal'] else -1
    momentum_component = (rsi_factor + macd_factor) / 2
    
    # Volatility Component (40% weight)
    bb_position = (current_price - latest['Bollinger_Lower']) / (latest['Bollinger_Upper'] - latest['Bollinger_Lower'])
    volatility_component = 0.5 - bb_position  # Positive when price is below middle band
    
    # Combine components
    total_score = (
        trend_component * 0.3 +
        momentum_component * 0.3 +
        volatility_component * 0.4
    )
    
    # Calculate base prediction
    base_change = total_score * max_move * 100  # Convert to percentage
    
    # Apply constraints
    max_allowed_change = max_move * 100 * math.sqrt(days)
    base_change = max(min(base_change, max_allowed_change), -max_allowed_change)
    
    # Adjust for liquidity
    if categorize_by_volume(latest['twenty_four_hour_trading_volume_usd']) == 'Low Liquidity':
        base_change *= 0.7  # More conservative for low liquidity
    
    # Calculate prediction ranges
    prediction = current_price * (1 + base_change/100)
    confidence_interval = historical_volatility * math.sqrt(days/252) * current_price
    
    return {
        'predicted_change': round(base_change, 2),
        'lower_bound': round(prediction - confidence_interval, 2),
        'upper_bound': round(prediction + confidence_interval, 2),
        'confidence_score': round(abs(total_score * 100), 2),
        'max_daily_move': round(max_daily_move * 100, 2)
    }

# Step 5: Main function to get signals for each coin
def main():
    # coin_symbols = ['btc', 'eth', 'xrp', 'sol', 'bnb', 'ton', 'apt', 'arb', 'tao', 'om', 'render', 'super', 'ondo', 'sui', 
    #                 'ar', 'op', 'tia', 'mkr', 'doge', 'ada', 'shib', 'trx', 'avax', 'link', 'pepe', 'dot', 'spec', 'near', 
    #                 'aero', 'uni', 'neural', 'chex', 'cpool', 'fantom', 'beam', 'grass', 'ray', 'jup']

    coin_symbols = ['btc', 'eth', 'sol', 'xrp']
    crypto_predictions = []

    for coin_symbol in coin_symbols: 
        print(f"\nFetching data for {coin_symbol.upper()}")
        df = fetch_coin_data(coin_symbol)
        if df is not None and len(df) >= 200:
            result = calculate_indicators(df)
            if result is not None:
                print(f"Latest Signal for {coin_symbol.upper()}:")
                print(result)
                
                # Generate price predictions
                predicted_24h_price = predict_price(df, days=1)
                # Generate price predictions with new dictionary returns
                predicted_7d = predict_price(df, days=7)
                predicted_14d = predict_price(df, days=14)

                # Map data to frontend format
                prediction_data = {
                    "coin": df["name"].iloc[0],
                    "symbol": coin_symbol.upper(),
                    "imageUrl": df["image_url"].iloc[0],
                    "currentPrice": result["value_usd"],
                    "sentiment": result["sentiment"],  # Use signal as sentiment
                    "marketCondition": result["marketCondition"],  # Use signal as market condition
                    "confidenceScore":result["confidence_score"],
                    "trend_strength": result["trend_strength"],
                    "volume_quality": result["volume_quality"],
                    "sevenDayPrediction": round(result["value_usd"] * (1 + predicted_7d['predicted_change'] / 100), 2),
                    "fourteenDayPrediction": round(result["value_usd"] * (1 + predicted_14d['predicted_change'] / 100), 2),
                    "sevenDayRange": {  # New field with prediction ranges
                        "lower": predicted_7d['lower_bound'],
                        "upper": predicted_7d['upper_bound']
                    },
                    "fourteenDayRange": {  # New field with prediction ranges
                        "lower": predicted_14d['lower_bound'],
                        "upper": predicted_14d['upper_bound']
                    },
                    "tradingVolume": result.get("Liquidity", "Unknown"),  # Placeholder if Liquidity is not calculated
                    "keyEvents": "Harvest 2.0",  # Placeholder for actual events
                    "technicalIndicators": {
                        "SMA_50": round(result["SMA_50"], 2),
                        "SMA_100": round(result["SMA_100"], 2),
                        "SMA_200": round(result["SMA_200"], 2),
                        "EMA_50": round(result["EMA_50"], 2),
                        "EMA_100": round(result["EMA_100"], 2),
                        "EMA_20": round(result["EMA_20"], 2),
                        "RSI": round(result["RSI"], 2),
                        "MACD": round(result["MACD"], 2),
                        "MACD_Signal": round(result["MACD_Signal"], 2),
                        "Bollinger_Upper": round(result["Bollinger_Upper"], 2),
                        "Bollinger_Lower": round(result["Bollinger_Lower"], 2),
                        "OBV": round(result["OBV"], 2),
                        "OBV_MA": round(result["OBV_MA"], 2),
                        "ATR": round(result["ATR"], 2),
                        "ADX": round(result["ADX"], 2),
                        "MFI": round(result["MFI"], 2),
                        "VWAP": round(result["VWAP"], 2),
                        "Predicted_24h_Price_Change_Percentage": predicted_24h_price['predicted_change']
                    }
                }
                crypto_predictions.append(prediction_data)
            else:
                print(f"Calculation error for {coin_symbol.upper()}")
        else:
            print(f"Not enough data to calculate indicators for {coin_symbol.upper()}")

    # Convert the predictions to JSON format and save to a file
    output_json = json.dumps(crypto_predictions, indent=2)
    
    # Save the output to a fileg
    with open("crypto_predictions.json", "w") as json_file:
        json_file.write(output_json)

    print("\nThe predictions have been saved to 'crypto_predictions.json'.")


if __name__ == "__main__":
    main()






    
