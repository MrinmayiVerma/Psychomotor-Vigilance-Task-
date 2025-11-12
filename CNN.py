# Install packages (if not already installed)
!pip install kaggle tensorflow scikit-learn pandas matplotlib -q

# Upload Kaggle API key
from google.colab import files
files.upload()  # Upload kaggle.json

import os
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and unzip dataset
!kaggle datasets download -d sacramentotechnology/sleep-deprivation-and-cognitive-performance
!unzip -o sleep-deprivation-and-cognitive-performance.zip -d pvt_dataset

# Load CSV
import pandas as pd
csv_files = [f for f in os.listdir('pvt_dataset') if f.endswith('.csv')]
df = pd.read_csv(f'pvt_dataset/{csv_files[0]}')
print(df.head())

# Drop Participant_ID if exists
if 'Participant_ID' in df.columns:
    df.drop(columns=['Participant_ID'], inplace=True)

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Define features and target
target_col = 'PVT_Reaction_Time'
X = df.drop(columns=[target_col])
y = df[target_col]
X = X.select_dtypes(include=['number'])

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize target
y_min = y.min()
y_max = y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Reshape for 1D CNN (samples, timesteps, features)
import numpy as np
X_train_cnn = np.expand_dims(X_train, axis=2)
X_test_cnn = np.expand_dims(X_test, axis=2)

# Build 1D CNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

model_cnn = Sequential([
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model_cnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True)

# Train model
history = model_cnn.fit(X_train_cnn, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[early_stop], verbose=1)

# Evaluate model
loss, mae_scaled = model_cnn.evaluate(X_test_cnn, y_test)
mae = mae_scaled * (y_max - y_min)
print(f"Test Mean Absolute Error (ms): {mae:.2f}")

# Plot training history
import matplotlib.pyplot as plt
plt.plot(history.history['mae'], label='MAE (train)')
plt.plot(history.history['val_mae'], label='MAE (val)')
plt.title('1D CNN Training History')
plt.xlabel('Epochs')
plt.ylabel('MAE (scaled)')
plt.legend()
plt.show()

# Make predictions
y_pred_scaled = model_cnn.predict(X_test_cnn)
y_pred = y_pred_scaled * (y_max - y_min)
