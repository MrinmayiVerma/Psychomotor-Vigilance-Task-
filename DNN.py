!pip install kaggle tensorflow scikit-learn pandas matplotlib -q

from google.colab import files
files.upload()

import os
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d sacramentotechnology/sleep-deprivation-and-cognitive-performance
!unzip -o sleep-deprivation-and-cognitive-performance.zip -d pvt_dataset

import pandas as pd
csv_files = [f for f in os.listdir('pvt_dataset') if f.endswith('.csv')]
df = pd.read_csv(f'pvt_dataset/{csv_files[0]}')
print(df.head())

if 'Participant_ID' in df.columns:
    df.drop(columns=['Participant_ID'], inplace=True)

# Drop non-numeric columns before scaling
if 'Gender' in df.columns:
    df.drop(columns=['Gender'], inplace=True)

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

target_col = 'PVT_Reaction_Time'
X = df.drop(columns=[target_col])
y = df[target_col]

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)

loss, mae_scaled = model.evaluate(X_test, y_test)
mae = mae_scaled * (y_max - y_min)
print(f"Test Mean Absolute Error (ms): {mae:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('DNN Training History')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (scaled)')
plt.legend()
plt.show()

y_pred_scaled = model.predict(X_test)
y_pred = y_pred_scaled * (y_max - y_min) + y_min
