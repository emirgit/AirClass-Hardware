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
from picamera2 import Picamera2
from libcamera import controls

DEBUG = True
gesture_threshold = 400
GESTURE_COOLDOWN = 1
TWO_UP_DISTANCE_THRESHOLD = 0.05  # İki parmak arası mesafe eşiği 
DRAWING_MODE = False  # İzleme modu başlangıçta kapalı
last_tracked_gesture = None
model = load_model('gesture_recognizer.keras')

# Zoom mode configuration
ZOOM_CONFIG = {
    "toggle_gesture": "ok",  # The gesture that toggles zoom mode
    "active": False            # Zoom mode starts inactive
}

# Named pipe for interprocess communication
PIPE_PATH = "/tmp/gesture_pipe"

both_hands_detected_time = None
both_hands_stable_duration = 0.5  # Require both hands to be present for 0.5 seconds
BOTH_HANDS_REQUIRED_GESTURES = ["take_picture", "timeout", "heart", "heart2"]  # Gestures that need both hands to form

def clear_websocket_file():
    """Clear the websocket.txt file at program startup"""
    if DEBUG:
        try:
            # Create an empty file (or truncate existing file)
            with open('websocket.txt', 'w') as f:
                pass
            print("Websocket debug file cleared")
        except Exception as e:
            print(f"Error clearing websocket file: {e}")

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
        # Stop and close Pi Camera
        if 'picam2' in globals():
            picam2.stop()
            picam2.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

# MediaPipe el tespit edici
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
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
    
def both_hands_present(results):
    """
    Check if both hands are detected in the current frame
    Returns True if exactly 2 hands are detected
    """
    return results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2

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
                        print(f"Command: {command}")
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

# First, update the toggle_zoom_mode function to accept a specific direction parameter
def toggle_zoom_mode(direction=None):
    """
    Toggle between normal mode and zoom mode
    direction: "on" to force zoom mode on, "off" to force zoom mode off, None to toggle
    """
    global ZOOM_CONFIG
    
    if direction == "on":
        
        send_websocket("zoom_in")
        # Only change if not already in zoom mode
        if not ZOOM_CONFIG["active"]:
            ZOOM_CONFIG["active"] = True
            print("Mode switched to: ZOOM MODE")
        
        return True
    
    elif direction == "off":
        # Only change if currently in zoom mode
        if ZOOM_CONFIG["active"]:
            ZOOM_CONFIG["active"] = False
            print("Mode switched to: NORMAL MODE")
            return False
        return ZOOM_CONFIG["active"]  # Already in normal mode
    
    else:
        # Toggle behavior (original functionality)
        ZOOM_CONFIG["active"] = not ZOOM_CONFIG["active"]
        mode_name = "ZOOM" if ZOOM_CONFIG["active"] else "NORMAL"
        print(f"Mode switched to: {mode_name} MODE")
        
        # Only send zoom_in command when activating zoom mode
        if ZOOM_CONFIG["active"]:
            send_websocket("zoom_in")
        
        return ZOOM_CONFIG["active"]

# Now modify the gesture handler to use these specific directions
def handle_gesture(frame, gesture_text, landmarks, is_right_hand=True, both_hands_active=False):
    global last_gesture_time, gesture_threshold, DRAWING_MODE, last_tracked_gesture
    current_time = time.time()
    
    # Only process if we have hand landmarks
    if landmarks:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Check for "ok" gesture to switch TO zoom mode
        if gesture_text == "ok" and current_time - last_gesture_time >= GESTURE_COOLDOWN:
            # Switch to zoom mode
            is_zoom_mode = toggle_zoom_mode(direction="on")
            
            # Display zoom mode indicator on frame
            if DEBUG:
                mode_text = "ZOOM MODE ACTIVE"
                mode_color = (0, 0, 255)
                cv2.putText(frame, mode_text, (frame_width - 300, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2, cv2.LINE_AA)
            
            last_gesture_time = current_time
            return frame

        # Check for "palm" gesture ABOVE threshold to switch BACK to normal mode
        if gesture_text == "palm" and int(landmarks[0].y * frame_height) < gesture_threshold and ZOOM_CONFIG["active"]:
            if current_time - last_gesture_time >= GESTURE_COOLDOWN:
                # Switch back to normal mode
                toggle_zoom_mode(direction="off")
                send_websocket("zoom_reset")
                
                if DEBUG:
                    cv2.putText(frame, "NORMAL MODE", (frame_width - 300, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "ZOOM RESET", (10, frame_height - 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                last_gesture_time = current_time
                return frame

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
            
            # Handle gesture differently in zoom mode
            if ZOOM_CONFIG["active"]:
                # In zoom mode, remap certain gestures
                if gesture_text == "like":
                    send_websocket("up")
                    if DEBUG:
                        cv2.putText(frame, "ZOOM UP", (10, frame_height - 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                elif gesture_text == "dislike":
                    send_websocket("down")
                    if DEBUG:
                        cv2.putText(frame, "ZOOM DOWN", (10, frame_height - 140),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                elif gesture_text == "three_gun":
                    thumb_tip = landmarks[4]  # Thumb tip
                    index_tip = landmarks[8]  # Index finger tip
                    
                    # Calculate direction based on thumb and index finger position
                    if thumb_tip.x < index_tip.x:  # Thumb is to the right of index finger
                        send_websocket("right")
                    else:  # Thumb is to the left of index finger
                        send_websocket("left")
                # Remove the palm gesture handler from here since we handle it above
                # No need for the else clause
            else:
                # Normal mode - original gesture handling
                if gesture_text == "call":
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
                    
                    if DEBUG:
                        print(f"Hand: {'Right' if is_right_hand else 'Left'}")
                        print(f"Landmark 20 x: {landmarks[20].x}, Landmark 17 x: {landmarks[17].x}")
                        print(f"Thumb to index angle: {angle:.2f} degrees")
                    
                    if 75 < angle < 130 and index_closed and middle_closed and ring_closed and pinky_closed:
                        send_websocket("dislike")
                elif gesture_text == "like":
                    # Angle between thumb tip, wrist, and index tip
                    angle = calculate_angle(landmarks[4], landmarks[2], landmarks[5])
                    if DEBUG:
                        print(f"Thumb to index angle: {angle:.2f} degrees")
                    if 75 < angle < 120:  # Thumb is roughly perpendicular to index
                        send_websocket("like")
                # elif gesture_text == "ok":
                #     # gesture_threshold = int(landmarks[8].y * frame_height)
                #     send_websocket("ok")
                elif gesture_text == "rock":
                    send_websocket("rock")
                elif gesture_text == "three":
                    send_websocket("three")
                elif gesture_text == "three2":
                    send_websocket("three2")
                elif gesture_text == "timeout":
                    if both_hands_active:
                        send_websocket("timeout")
                elif gesture_text == "three_gun":
                    thumb_tip = landmarks[4]  # Thumb tip
                    index_tip = landmarks[8]  # Index finger tip
                    
                    # Calculate direction based on thumb and index finger position
                    if thumb_tip.x < index_tip.x:  # Thumb is to the right of index finger
                        send_websocket("inv_three_gun")
                    else:  # Thumb is to the left of index finger
                        send_websocket("three_gun")
                elif gesture_text == "palm" and int(landmarks[1].y * frame_height) < gesture_threshold:
                    send_websocket("palm")
                elif gesture_text == "take_picture":
                    if both_hands_active:
                        send_websocket("take_picture")
                elif gesture_text == "hand_heart":
                    if both_hands_active:
                        send_websocket("heart")
                elif gesture_text == "hand_heart2":
                    if both_hands_active:
                        send_websocket("hand_heart2")
                elif gesture_text == "middle_finger":
                    send_websocket("mid_finger")
                elif gesture_text == "thumb_index":
                    send_websocket("thumb_index")
                elif gesture_text == "holy":
                    send_websocket("holy")
                elif gesture_text == "three_3":
                    send_websocket(gesture_text)
                    
            last_tracked_gesture = gesture_text

            # Update last gesture time
            last_gesture_time = current_time

        # Display zoom mode status on the frame if active
        if ZOOM_CONFIG["active"] and DEBUG:
            cv2.putText(frame, "ZOOM MODE", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame

# Setup the named pipe before starting camera
clear_websocket_file()
setup_pipe()

# Initialize Pi Camera 2
print("Initializing Pi Camera 2...")
picam2 = Picamera2()

# Configure camera - adjust resolution as needed
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

# Optional: Set camera controls for better performance
picam2.set_controls({
    "ExposureTime": 20000,  # 20ms exposure time
    "AnalogueGain": 1.0,
    "FrameRate": 30.0
})

# Start the camera
picam2.start()
print("Pi Camera 2 started successfully")

# Allow camera to warm up
time.sleep(2)

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
        # Capture frame from Pi Camera
        frame = picam2.capture_array()
        
        # Convert RGB to BGR for OpenCV compatibility
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # El tespit et
        results = hands.process(rgb_frame)
        
        # Check if both hands are present
        both_hands_active = both_hands_present(results)
        
        # Track how long both hands have been present
        if both_hands_active:
            if both_hands_detected_time is None:
                both_hands_detected_time = time.time()
        else:
            both_hands_detected_time = None
        
        # Check if both hands have been stable for required duration
        both_hands_stable = (both_hands_detected_time is not None and 
                           time.time() - both_hands_detected_time >= both_hands_stable_duration)
        
        # Varsayılan durum "IDLE"
        gesture_text = "IDLE"
        current_hand_landmarks = None
        is_right_hand = True  # Default assumption
        
        if results.multi_hand_landmarks:
            # Process the primary hand (first detected hand for gesture recognition)
            primary_hand = results.multi_hand_landmarks[0]
            
            # Get hand classification for primary hand
            if results.multi_handedness:
                hand_label = results.multi_handedness[0].classification[0].label
                is_right_hand = hand_label == "Right"

            # ⬇️ Bu çizim işlemi sadece DEBUG modda yapılacak ⬇️
            if DEBUG:
                # Draw landmarks for all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS)

            current_hand_landmarks = primary_hand.landmark
            
            # El özelliklerini çıkar (primary hand için)
            features_df = extract_hand_features(primary_hand)
            
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

            if confidence > 0.70:
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
                    last_output_time = time.time()
            else:
                # El tespit edildi ama yetersiz confidence degere sahip
                if DEBUG:
                    gesture_text = "Uncertain"
                    gesture_text = f"Uncertain ({confidence:.2f})"
                current_gesture = "IDLE"
        else:
            # Hand not detected
            gesture_text = "IDLE"
            current_gesture = "IDLE"
        
        # Handle gesture processing with both hands information
        frame = handle_gesture(frame, gesture_text, current_hand_landmarks, is_right_hand, both_hands_stable)
        
        # Debug information
        if DEBUG:
            # Display number of hands detected
            hands_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            cv2.putText(frame, f"Hands: {hands_count}/2", (30, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show both hands status
            if both_hands_active:
                if both_hands_stable:
                    cv2.putText(frame, "BOTH HANDS READY", (30, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    elapsed = time.time() - both_hands_detected_time if both_hands_detected_time else 0
                    remaining = max(0, both_hands_stable_duration - elapsed)
                    cv2.putText(frame, f"Both hands: {remaining:.1f}s", (30, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Need both hands", (30, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Main gesture text
            cv2.putText(frame, gesture_text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

            if current_hand_landmarks:
                hand_text = "Right Hand" if is_right_hand else "Left Hand"
                cv2.putText(frame, hand_text, (30, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.line(frame, (0, gesture_threshold), (frame.shape[1], gesture_threshold), (0, 0, 255), 2)
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
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    cleanup_handler(None, None)
