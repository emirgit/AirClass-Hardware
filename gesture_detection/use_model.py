import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import math
import json
import os
import stat
import signal
import sys

DEBUG = True
gesture_threshold = 400
GESTURE_COOLDOWN = 1
TWO_UP_DISTANCE_THRESHOLD = 0.05  # İki parmak arası mesafe eşiği 
DRAWING_MODE = False  # İzleme modu başlangıçta kapalı
last_tracked_gesture = None
model = load_model('gesture_recognizer.keras')

# Named pipe for interprocess communication
PIPE_PATH = "/tmp/gesture_pipe"

# Updated setup_pipe function for the Python script
def setup_pipe():
    global pipe_fd
    pipe_fd = None
    
    try:
        # Remove the pipe if it already exists
        if os.path.exists(PIPE_PATH):
            os.unlink(PIPE_PATH)
        
        # Create the pipe
        os.mkfifo(PIPE_PATH, mode=0o666)
        print(f"Created named pipe: {PIPE_PATH}")
        
        # Don't open the pipe for writing yet - wait for reader
        print("Named pipe created successfully. Waiting for C++ client to connect...")
        
        # Register cleanup handler
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        
    except Exception as e:
        print(f"Error setting up pipe: {e}")
        sys.exit(1)

# Updated cleanup function
def cleanup_handler(sig, frame):
    print("\nCleaning up and exiting...")
    try:
        if pipe_fd is not None and pipe_fd > 0:
            os.close(pipe_fd)
        if os.path.exists(PIPE_PATH):
            os.unlink(PIPE_PATH)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

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

# Function to calculate distance between two landmarks
def calculate_landmark_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    dx = landmark1.x - landmark2.x
    dy = landmark1.y - landmark2.y
    dz = landmark1.z - landmark2.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# Function to calculate angle between three landmarks
def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# Added gesture handling function similar to the one in paste-1.txt
last_gesture_time = 0

# Updated send_websocket function with DEBUG mode
def send_websocket(command, position_data=None):
    """
    Send command to the C++ program via the named pipe
    When DEBUG is True, writes to websocket.txt file instead
    """
    global pipe_fd
    
    # Create a JSON message that matches the expected format in the C++ client
    message = {
        "command": command,
    }
    
    # Add position data if provided (for tracking mode)
    if position_data:
        message["position"] = position_data
    
    # Convert to JSON string and add newline for message separation
    json_str = json.dumps(message) + "\n"
    
    # In DEBUG mode, write to file instead of pipe
    if DEBUG:
        try:
            with open('websocket.txt', 'a') as f:
                f.write(json_str)
            print(f"DEBUG: Wrote to websocket.txt: {command}")
            return
        except Exception as e:
            print(f"DEBUG: Error writing to file: {e}")
            return
    
    # Non-DEBUG mode: Use named pipe as before
    try:
        # Open pipe for writing if not already open
        if pipe_fd is None:
            try:
                pipe_fd = os.open(PIPE_PATH, os.O_WRONLY | os.O_NONBLOCK)
                print("Pipe opened for writing")
            except OSError as e:
                if e.errno == 6:  # No such device or address
                    # No reader connected yet, skip this message
                    if DEBUG:
                        print(f"No reader connected, skipping: {command}")
                    return
                else:
                    raise
        
        # Write to the pipe
        try:
            os.write(pipe_fd, json_str.encode('utf-8'))
            if DEBUG:
                print(f"Sent: {command}")
        except BlockingIOError:
            # If pipe is full, this is non-critical - we can drop the message
            if DEBUG:
                print("Pipe is full, skipping message")
        except BrokenPipeError:
            # Reader disconnected, close and reset pipe
            print("Reader disconnected, closing pipe")
            if pipe_fd is not None:
                os.close(pipe_fd)
                pipe_fd = None
            
    except Exception as e:
        print(f"Error sending command: {e}")
        if pipe_fd is not None:
            try:
                os.close(pipe_fd)
            except:
                pass
            pipe_fd = None

def handle_gesture(frame, gesture_text, landmarks, is_right_hand=True):
    global last_gesture_time, gesture_threshold, DRAWING_MODE, last_tracked_gesture
    current_time = time.time()
    
    # Only process if we have hand landmarks
    if landmarks:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        if gesture_text == "two_up":
            index_tip = landmarks[8]  # Index fingertip
            # Convert normalized coordinates to pixel values
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            
            # Send position data in normalized format (0-1 range)
            position_data = {"x": index_tip.x, "y": index_tip.y}
            send_websocket("two_up", position_data)
            
            # Draw tracking indicator on frame - use a different color from tracking mode
            cv2.circle(frame, (x, y), 15, (255, 165, 0), -1)  # Orange circle
            cv2.putText(frame, "TWO UP POINTING", (10, frame_height - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
            
            # We still need to disable regular tracking mode when two_up is detected
            DRAWING_MODE = False

        elif gesture_text == "one" and last_tracked_gesture == "two_up":
            DRAWING_MODE = True  # Enable tracking mode
        
        if DRAWING_MODE and landmarks:
            index_tip = landmarks[8]  # Index fingertip
            # Convert normalized coordinates to pixel values
            x = int(index_tip.x * frame_width)
            y = int(index_tip.y * frame_height)
            
            # Send position data in normalized format (0-1 range)
            position_data = {"x": x, "y": y}
            send_websocket("one", position_data)
            
            # Draw tracking indicator on frame
            cv2.circle(frame, (x, y), 15, (0, 255, 255), -1)  # Yellow circle
            cv2.putText(frame, "DRAWING MODE", (10, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"X: {index_tip.x:.3f}, Y: {index_tip.y:.3f}", (10, frame_height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
        # Check cooldown and process gesture
        if current_time - last_gesture_time >= GESTURE_COOLDOWN:
            frame_height = frame.shape[0]
            # Her gesture için ayrı işlem bloğu
            if gesture_text == "call":
                # call için özel işlem
                send_websocket("call")
            elif gesture_text == "dislike":
                # dislike için özel işlem
                angle = calculate_angle(landmarks[4], landmarks[2], landmarks[5])
                
                # Adjust finger closed logic based on hand orientation
                if is_right_hand:
                    # For right hand: finger tips should be to the left of MCP joints when making thumbs down
                    index_closed = landmarks[8].x > landmarks[5].x
                    middle_closed = landmarks[12].x > landmarks[9].x
                    ring_closed = landmarks[16].x > landmarks[13].x
                    pinky_closed = landmarks[20].x > landmarks[17].x
                else:
                    # For left hand: finger tips should be to the right of MCP joints when making thumbs down
                    index_closed = landmarks[8].x < landmarks[5].x
                    middle_closed = landmarks[12].x < landmarks[9].x
                    ring_closed = landmarks[16].x < landmarks[13].x
                    pinky_closed = landmarks[20].x < landmarks[17].x
                
                print(f"Hand: {'Right' if is_right_hand else 'Left'}")
                print(f"Landmark 20 x: {landmarks[20].x}, Landmark 17 x: {landmarks[17].x}")
                print(f"Thumb to index angle: {angle:.2f} degrees")
                
                if 75 < angle < 120 and index_closed and middle_closed and ring_closed and pinky_closed:
                    send_websocket("dislike")
            # In the like gesture section:
            elif gesture_text == "like":
                # Angle between thumb tip, wrist, and index tip
                angle = calculate_angle(landmarks[4], landmarks[2], landmarks[5])
                print(f"Thumb to index angle: {angle:.2f} degrees")
                if 75 < angle < 120:  # Thumb is roughly perpendicular to index
                    send_websocket("like")
            elif gesture_text == "ok":
                gesture_threshold = int(landmarks[8].y * frame_height)
                send_websocket("ok")
            elif gesture_text == "rock":
                send_websocket("rock")
            elif gesture_text == "three":
                send_websocket("three")
            elif gesture_text == "three2":
                send_websocket("three2")
            elif gesture_text == "timeout":
                send_websocket("timeout")
            elif gesture_text == "three_gun":
                thumb_tip = landmarks[4]  # Thumb tip
                index_tip = landmarks[8]  # Index finger tip
                
                # Calculate direction based on thumb and index finger position
                if thumb_tip.x < index_tip.x:  # Thumb is to the right of index finger
                    send_websocket("previous_slide")  # Map to LEFT command for WebSocket
                else:  # Thumb is to the left of index finger
                    send_websocket("next_slide")  # Map to RIGHT command for WebSocket
            elif gesture_text == "palm" and int(landmarks[1].y * frame_height) < gesture_threshold:
                send_websocket("Attendance")
            elif gesture_text == "take_picture":
                send_websocket("take_picture")
            elif gesture_text == "heart":
                send_websocket("heart")
            elif gesture_text == "heart2":
                send_websocket("heart2")
            elif gesture_text == "mid_finger":
                send_websocket("mid_finger")
            elif gesture_text == "four":
                send_websocket("four")
            elif gesture_text == "thumb_index":
                send_websocket("thumb_index")
            elif gesture_text == "holy":
                send_websocket("holy")
                
            last_tracked_gesture = gesture_text

            # Update last gesture time
            last_gesture_time = current_time
    
    return frame

# Setup the named pipe before starting camera
setup_pipe()

# Send an initial message to indicate Python process is ready
send_websocket("init", {"status": "ready"})

# Kamerayı aç
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Kamera acilamadi! Kamera numarasi hatali")
    cleanup_handler(None, None)
    
# Cikti zamanlamasi
last_output_time = time.time()
current_gesture = "IDLE"
gesture_start_time = None
stable_duration = 0.1
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
    
print(f"Gesture recognition system started. Sending commands through pipe: {PIPE_PATH}")
print("Press 'q' to quit")

try:
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
        is_right_hand = True  # Default assumption
        
        # Eğer el tespit edildiyse
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand classification
                if results.multi_handedness:
                    hand_label = results.multi_handedness[hand_idx].classification[0].label
                    is_right_hand = hand_label == "Right"

                # ⬇️ Bu çizim işlemi sadece DEBUG modda yapılacak ⬇️
                if DEBUG:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS)

                current_hand_landmarks = hand_landmarks.landmark

                
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
                        # print(f"Gesture: {gesture_text}, (Güven: {confidence:.2f})")
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
        frame = handle_gesture(frame, gesture_text, current_hand_landmarks, is_right_hand)
                    
    
        # Ekranın sol üstüne hareket adini yazar
        if DEBUG:
            cv2.putText(
                frame, gesture_text, (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
            )

            if current_hand_landmarks:
                hand_text = "Right Hand" if is_right_hand else "Left Hand"
                cv2.putText(
                    frame, hand_text, (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA
                )

            cv2.line(frame, (0, gesture_threshold), (frame.shape[1], gesture_threshold), (0, 0, 255), 2)

            # Görüntüyü göster
            cv2.imshow('Hand Gesture Recognition', frame)

        
        # Q tuşu ile çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nKeyboard interrupt received. Exiting...")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Temizle
    cap.release()
    cv2.destroyAllWindows()
    cleanup_handler(None, None)