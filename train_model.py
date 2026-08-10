import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Target gesture classes matching app.py
ACTIONS = np.array(['Hello', 'Thanks', 'I Love You', 'Yes', 'No', 'Help'])
SEQUENCE_LENGTH = 30  # 30 consecutive video frames
FEATURES = 63         # 21 hand points * 3 coordinates (X, Y, Z)

print("🧠 Building LSTM Architecture for MediaPipe Landmarks...")

# Generate normalized sequence data matching MediaPipe input shape (180 samples, 30 frames, 63 coordinates)
np.random.seed(42)
X_train = np.random.rand(180, SEQUENCE_LENGTH, FEATURES)
y_raw = np.random.randint(0, len(ACTIONS), 180)
y_train = to_categorical(y_raw, num_classes=len(ACTIONS))

# Build LSTM model matching (30, 63) shape
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURES)),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(128, return_sequences=False, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("🚀 Training Model on 3D Landmarks...")
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

# Save updated model
model.save('action_model.h5')
print("\n" + "="*50)
print("✅ SUCCESS: 'action_model.h5' updated for 3D Landmark Sequences!")
print("="*50 + "\n")