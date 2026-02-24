import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define constants based on the report's architecture
DATA_FILE = '../Climate_Energy_Consumption_Dataset_2020_2024.csv'
TIME_STEPS = 7 # 7-day sliding window for daily data

def create_sequences(data, time_steps):
    """Creates 3D tensors for LSTM input (Sliding Window Cross-Validation)."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :-1]) # All features
        y.append(data[i + time_steps, -1])     # Target variable (energy_consumption)
    return np.array(X), np.array(y)

print("1. Data Ingestion Layer...")
df = pd.read_csv(DATA_FILE)

print("2. Data Preprocessing & Feature Engineering Layer...")
# Convert date to datetime and extract temporal features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Cyclical encoding for time variables (Sine/Cosine transformations)
df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

# Encode categorical 'country' variable
label_encoder = LabelEncoder()
df['country_encoded'] = label_encoder.fit_transform(df['country'])

# Drop non-numeric/original categorical columns to prepare for scaling
df = df.drop(['date', 'country', 'month'], axis=1)

# Ensure 'energy_consumption' is the last column for easier sequence generation
cols = [c for c in df.columns if c != 'energy_consumption'] + ['energy_consumption']
df = df[cols]

print("3. Feature Selection Layer (Random Forest)...")
X_rf = df.drop('energy_consumption', axis=1)
y_rf = df['energy_consumption']

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_rf, y_rf)

# Display feature importances as dictated by the report
importances = pd.Series(rf_model.feature_importances_, index=X_rf.columns).sort_values(ascending=False)
print("\nFeature Importances Ranking:")
print(importances)
print("\n*Note: In a full pipeline, we would drop features with near-zero importance here.*")

print("\n4. Normalization & Scaling...")
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale features and target separately so we can inverse_transform the predictions later
X_scaled = scaler_X.fit_transform(df.drop('energy_consumption', axis=1))
y_scaled = scaler_y.fit_transform(df[['energy_consumption']])

# Recombine temporarily to create sequences
scaled_data = np.hstack((X_scaled, y_scaled))

print("5. Time-Series Split (Sliding Window)...")
X, y = create_sequences(scaled_data, TIME_STEPS)

# Chronological Split (80% Train, 20% Test) to prevent data leakage
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

print("6. AI Modelling Core (LSTM Network)...")
model = Sequential()
# Input layer matches the 3D tensor shape (Time_Steps, Features)
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Mitigates overfitting
model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1)) # Output layer for predicting a single energy consumption value

model.compile(optimizer='adam', loss='mean_squared_error')

print("Training model (this may take a moment)...")
# Epochs kept relatively low for rapid prototyping
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)

print("\n7. Saving Models and Encoders for Decision Support Layer...")
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/lstm_energy_model.h5')
joblib.dump(scaler_X, 'saved_models/scaler_X.pkl')
joblib.dump(scaler_y, 'saved_models/scaler_y.pkl')
joblib.dump(label_encoder, 'saved_models/label_encoder.pkl')

print("Pipeline Complete! The model and scalers are ready to be loaded by your Tkinter UI.")