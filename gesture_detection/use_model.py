import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import time

DEBUG = False
gesture_threshold = 400
GESTURE_COOLDOWN = 1
model = load_model('gesture_recognizer.keras')

# MediaPipe el tespit edici
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

# Etiket dönüştürücüyü yükle veya yeniden oluştur
try:
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        print(f"Taninabilen gesture'lar: {', '.join(le.classes_)}")
except FileNotFoundError:
    print("You should provide label_encoder")
    exit(-1)

# Görüntüyü el özelliklerine dönüştüren fonksiyon
def extract_hand_features(hand_landmarks):
    features = {}
    # Her landmark için x, y, z koordinatlarını al
    for i, landmark in enumerate(hand_landmarks.landmark):
        features[f'x{i}'] = landmark.x
        features[f'y{i}'] = landmark.y
        features[f'z{i}'] = landmark.z
    return pd.DataFrame([features])

# Added gesture handling function similar to the one in paste-1.txt
last_gesture_time = 0

def send_websocket(command):
    with open('websocket.txt', 'a') as f:
        f.write(command + '\n')

def handle_gesture(frame, gesture_text, landmarks):
    global last_gesture_time, gesture_threshold
    current_time = time.time()
    
    # Only process if we have hand landmarks
    if landmarks:
        # Check cooldown and process gesture
        if current_time - last_gesture_time >= GESTURE_COOLDOWN:
            frame_height = frame.shape[0]
            # Her gesture için ayrı işlem bloğu
            if gesture_text == "call":
                # call için özel işlem
                send_websocket("call")
            elif gesture_text == "dislike":
                # dislike için özel işlem
                send_websocket("dislike")
            elif gesture_text == "like":
                send_websocket("like")
            elif gesture_text == "mute":
                send_websocket("mute")
            elif gesture_text == "ok":
                gesture_threshold = int(landmarks[8].y * frame_height)
                send_websocket("ok")
            elif gesture_text == "one":
                send_websocket("one")
            elif gesture_text == "stop":
                send_websocket("stop")
            elif gesture_text == "two_up":
                send_websocket("two_up")
            elif gesture_text == "rock":
                send_websocket("rock")
            elif gesture_text == "three":
                send_websocket("three")
            elif gesture_text == "two_up_inverse":
                send_websocket("two_up_inverse")
            elif gesture_text == "stop_inverse":
                send_websocket("stop_inverse")
            elif gesture_text == "three2":
                send_websocket("three2")
            elif gesture_text == "timeout":
                send_websocket("timeout")
            
            # Update last gesture time
            last_gesture_time = current_time
    
    return frame

# Kamerayı aç
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera acilamadi! Kamera numarasi hatali")
    exit(1)
    
# Cikti zamanlamasi
last_output_time = time.time()
current_gesture = "IDLE"
gesture_start_time = None
stable_duration = 0.4
gesture_text = "IDLE"

# Scaler'ı yükle
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Scaler dosyası bulunamadı! Normalizasyon yapılamayacak.")
    # Basit bir scaler oluştur
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
# Clear websocket output file at startup
with open("websocket.txt", "w") as f:
    f.write("# Gesture Commands Log\n")
    
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamıyor.")
        break
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # El tespit et
    results = hands.process(rgb_frame)
    
    # Varsayılan durum "IDLE"
    gesture_text = "IDLE"
    current_hand_landmarks = None
    
    # Eğer el tespit edildiyse
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Eli görüntü üzerinde çiz
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)
            
            current_hand_landmarks = hand_landmarks.landmark
            
            # El özelliklerini çıkar
            features_df = extract_hand_features(hand_landmarks)
            
            expected_columns = [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
            for col in expected_columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0
                    
            # Sadece gereken sutunlar secilecek
            features_df = features_df[expected_columns]
            
            # Modelle tahmin
            features_array = scaler.transform(features_df)
            
            prediction = model.predict(features_array, verbose=0)
            
            gesture_index = np.argmax(prediction)
            
            gesture = le.classes_[gesture_index]
            
            # Tahmin güvenini al
            confidence = prediction[0][gesture_index]

            if confidence > 0.85:
                detected_gesture = gesture  # Algılanan hareket
                
                # Algılanan hareket değişirse, yeni zamanlayıcı başlat
                if detected_gesture != current_gesture:
                    current_gesture = detected_gesture
                    gesture_start_time = time.time()
                
                # Hareket sabit kalmışsa ve yeterince süre geçmişse göster
                if time.time() - gesture_start_time >= stable_duration:
                    gesture_text = current_gesture
                
                # Konsol çıktısı (1 saniye aralıklarla)
                if time.time() - last_output_time >= 1.0:
                    print(f"Gesture: {gesture_text}, (Güven: {confidence:.2f})")
                    last_output_time = time.time()
            else:
                # El tespit edildi ama yetersiz confidence degere sahip
                if (DEBUG):
                    gesture_text = "Uncertain"
                    print(f"Uncertain gesture confidence: {confidence:.2f})")
                current_gesture = "IDLE"
    else:
        # Hand not detected
        gesture_text = "IDLE"
        current_gesture = "IDLE"
    
    # Handle gesture processing
    frame = handle_gesture(frame, gesture_text, current_hand_landmarks)
                
    # Ekranın sol üstüne hareket adini yazar
    cv2.putText(
        frame, gesture_text, (30, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
    )
    
    cv2.line(frame, (0, gesture_threshold), (frame.shape[1], gesture_threshold), (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Q tuşu ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Temizle
cap.release()
cv2.destroyAllWindows()
