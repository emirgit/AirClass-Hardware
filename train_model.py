#%% Import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.layers import Input

df = pd.read_csv("all_landmarks/all_landmarks_combined.csv")


X = df.drop("label", axis=1)
y = df["label"]
le = LabelEncoder()
y = le.fit_transform(y)

# Normalizasyon - MinMaxScaler kullanarak 0-1 aralığına getir.
# Daha hizli egitilmesi icin
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Veri Augmentasyonu için fonksiyon.
# Daha genellestirilmis datada calismasi icin
# Datalar cesitlendiriliyor
def augment_landmarks(landmarks, noise_level=0.01):
    """Landmark verilerine küçük rastgele gürültü ekler"""
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

# First split %70 train %30 validation and test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

# Second split %15 validation and %15 test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Veri augmantasyonu
X_train_augmented = augment_landmarks(X_train.values)
# Orijinal ve augmente edilmiş verileri birleştir
X_train_combined = np.vstack([X_train.values, X_train_augmented])
y_train_combined = np.hstack([y_train, y_train])

# Import Dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Build and compile model
num_classes = len(le.classes_)
model = Sequential([
    Input(shape=(len(X_train.columns),)),
    Dense(64, activation='relu'),
    Dropout(0.2),  # Aşırı öğrenmeyi önlemek için
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler ekleyerek eğitimi iyileştirme
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Model kaydedilecek dosya ismini ve yolu sor
default_model_name = "gesture_recognizer.keras"
default_model_dir = os.getcwd()
print(f"\nModel will be saved in: {default_model_dir}")
model_name = input(f"Model file name to save (default: {default_model_name}): ").strip()
if not model_name:
    model_name = default_model_name

model_path = os.path.join(default_model_dir, model_name)
print(f"\nModel will be saved as: {model_path}")

confirm = input("Proceed with this model file? (y/n): ").strip().lower()
if confirm != 'y':
    print("Operation cancelled by user.")
    exit()

# ModelCheckpoint ve model.save için model_path kullan
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max')

# Fit, Predict, and Evaluate
history = model.fit(X_train_combined, y_train_combined, 
                    epochs=100, 
                    batch_size=32, 
                    verbose=1,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr, checkpoint])

y_hat = model.predict(X_test)
y_hat = y_hat.argmax(axis=1)

print("Test accuracy: ", accuracy_score(y_test, y_hat))

model.save(model_path)
model.summary()

# Label encoder'in kaydedilmesi lazim
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
    
# Scaler'ı kaydet (tahmin için gerekecek)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Eğitim performansını görselleştirme
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
