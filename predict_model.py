import pandas as pd
from sqlalchemy import create_engine
from config import DATABASE_CONFIG
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def connect_to_db():
    try:
        connection_url = f"postgresql+psycopg2://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
        engine = create_engine(connection_url)
        return engine
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

# Step 2: Fetch latest 200 days of data for a specific coin by symbol
def fetch_coin_data(coin_symbol):
    query = """
    SELECT cp.recorded_at, cp.value_usd, cp.twenty_four_hour_trading_volume_usd
    FROM coin_prices_daily cp
    JOIN coins c ON c.id = cp.coin_id
    WHERE c.symbol = %s
    ORDER BY cp.recorded_at DESC
    LIMIT 200;
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

# Step 3: Calculate Technical Indicators with Improved Logic
def calculate_indicators(df):
    if len(df) < 200:
        print("Insufficient data for reliable indicator calculation.")
        return None

    # Preprocess: Smooth the price data to remove noise using a rolling median
    df['value_usd'] = df['value_usd'].rolling(window=3, min_periods=1).mean()

    # Moving Averages (100-day and 200-day SMA for long-term trends)
    df['SMA_100'] = df['value_usd'].rolling(window=100).mean()
    df['SMA_200'] = df['value_usd'].rolling(window=200).mean()
    
    # Exponential Moving Averages (50-day and 100-day for short-term trends)
    df['EMA_50'] = df['value_usd'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['value_usd'].ewm(span=100, adjust=False).mean()

    # VWAP (Volume Weighted Average Price) for volume-based trend confirmation
    df['VWAP'] = (df['value_usd'] * df['twenty_four_hour_trading_volume_usd']).cumsum() / df['twenty_four_hour_trading_volume_usd'].cumsum()

    # On-Balance Volume (OBV) for buying/selling pressure
    df['OBV'] = (df['twenty_four_hour_trading_volume_usd'] * (df['value_usd'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))).cumsum()

    # Corrected RSI Calculation (14-day)
    delta = df['value_usd'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    # MACD and Signal Line
    df['EMA_12'] = df['value_usd'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['value_usd'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands for volatility
    df['Bollinger_Mid'] = df['value_usd'].rolling(window=20).mean()
    df['Bollinger_Std'] = df['value_usd'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + (2 * df['Bollinger_Std'])
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - (2 * df['Bollinger_Std'])

    # Advanced Indicators

    # ATR (Average True Range)
    df['High_Low'] = df['value_usd'].rolling(window=1).max() - df['value_usd'].rolling(window=1).min()
    df['High_Close'] = abs(df['value_usd'].rolling(window=1).max() - df['value_usd'].shift())
    df['Low_Close'] = abs(df['value_usd'].rolling(window=1).min() - df['value_usd'].shift())
    df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=14).mean()

    # Chaikin Money Flow (CMF)
    df['Money_Flow_Multiplier'] = ((df['value_usd'] - df['value_usd'].rolling(window=1).min()) - (df['value_usd'].rolling(window=1).max() - df['value_usd'])) / (df['value_usd'].rolling(window=1).max() - df['value_usd'].rolling(window=1).min())
    df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['twenty_four_hour_trading_volume_usd']
    df['CMF'] = df['Money_Flow_Volume'].rolling(window=20).sum() / df['twenty_four_hour_trading_volume_usd'].rolling(window=20).sum()

    # Stochastic Oscillator
    df['Lowest_Low'] = df['value_usd'].rolling(window=14).min()
    df['Highest_High'] = df['value_usd'].rolling(window=14).max()
    df['%K'] = (df['value_usd'] - df['Lowest_Low']) * 100 / (df['Highest_High'] - df['Lowest_Low'])
    df['%D'] = df['%K'].rolling(window=3).mean()

    return df  # Returns the full DataFrame with all indicators

# Define targets function with data integrity checks
def define_targets(df):
    df['target_7d'] = df['value_usd'].shift(-7)
    df['target_30d'] = df['value_usd'].shift(-30)
    df.dropna(subset=['target_7d', 'target_30d'], inplace=True)
    return df

# Prepare LSTM data function with scaling and sequence handling
def prepare_lstm_data(df, sequence_length=30, target_column='target_7d'):
    data = df.copy()
    data.dropna(subset=[target_column], inplace=True)
    print("Shape after dropping NaNs in target columns:", data.shape)

    # Scale features
    feature_columns = [col for col in data.columns if col not in ['recorded_at', 'target_7d', 'target_30d']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[feature_columns].values)
    
    # Scale target values
    target_scaler = MinMaxScaler()
    scaled_targets = target_scaler.fit_transform(data[target_column].values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(scaled_features[i:i + sequence_length])
        y.append(scaled_targets[i + sequence_length])
    
    X, y = np.array(X), np.array(y)
    print("Number of sequences (X):", len(X))
    print("Number of targets (y):", len(y))

    if len(X) == 0 or len(y) == 0:
        print("Warning: No sequences generated. Check data or sequence length.")

    # Check for NaN/Inf before splitting
    print("NaN in X:", np.isnan(X).any(), "NaN in y:", np.isnan(y).any())
    print("Inf in X:", np.isinf(X).any(), "Inf in y:", np.isinf(y).any())

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_val, y_train, y_val, target_scaler

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Output for regression

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Last time step
        return out

# Main Execution Flow
if __name__ == "__main__":
    # Assuming fetch_coin_data and calculate_indicators are defined elsewhere
    df = fetch_coin_data("btc")
    print("Initial DataFrame shape:", df.shape)

    if df is not None:
        df = calculate_indicators(df)
        print("Shape after calculating indicators:", df.shape)
        df = define_targets(df)
        print("Shape after defining targets:", df.shape)
        print("DataFrame with Targets Preview:\n", df[['value_usd', 'target_7d', 'target_30d']].head())

        # Prepare data for LSTM with scaling and data validation
        X_train, X_val, y_train, y_val, target_scaler = prepare_lstm_data(df, sequence_length=30, target_column='target_7d')
        
        print("LSTM Input Shape (X_train):", X_train.shape)
        print("LSTM Target Shape (y_train):", y_train.shape)
        print("Validation Input Shape (X_val):", X_val.shape)
        print("Validation Target Shape (y_val):", y_val.shape)

        # Convert to PyTorch tensors and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        # Build the LSTM model
        input_size = X_train.shape[2]
        model = LSTMModel(input_size=input_size).to(device)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

        # Training loop with loss tracking
        num_epochs = 50
        batch_size = 16
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(X_train_tensor.size(0))
            epoch_loss = 0.0
            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.view(-1), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / (X_train_tensor.size(0) / batch_size)
            train_losses.append(avg_epoch_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.view(-1), y_val_tensor)
                val_losses.append(val_loss.item())
            
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Training Loss: {avg_epoch_loss:.4f}, "
                  f"Validation Loss: {val_loss.item():.4f}")

        # Plot training and validation loss
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('PyTorch LSTM Model Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



