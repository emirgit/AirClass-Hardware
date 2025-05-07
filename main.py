import cv2
import mediapipe as mp
import math
import time
import socket

# SOCKET settings
HOST = '172.20.10.3'  # Debian VM IP
PORT = 65432

try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print(f"Connected to hardware server at {HOST}:{PORT}")
except Exception as e:
    print(f"Could not connect to hardware server: {e}")
    exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Flags and tracking
left_pinch_active = False
right_pinch_active = False
prev_lr_distance = None
prev_left_distance = None
locked = False
lock_time = 0
timeout_duration = 3  # slide timeout

zoom_locked = False
zoom_lock_time = 0
zoom_timeout_duration = 0  # zoom timeout

# Calibration for zoom percentage
min_distance = 20
max_distance = 150

def calculate_percentage(distance, min_dist, max_dist):
    percent = (distance - min_dist) / (max_dist - min_dist)
    percent = max(0, min(1, percent))
    return int(percent * 100)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape

        left_pinch = False
        right_pinch = False
        left_center = None
        right_center = None
        left_finger_distance = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                index_point = (int(index_tip.x * w), int(index_tip.y * h))
                thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                distance = math.dist(index_point, thumb_point)
                center_point = ((index_point[0] + thumb_point[0]) // 2, (index_point[1] + thumb_point[1]) // 2)

                if hand_label == 'Left':
                    left_pinch = distance < 30
                    left_center = center_point
                    left_finger_distance = distance
                    # ✅ Left: draw when NOT pinched
                    if not left_pinch:
                        cv2.circle(frame, index_point, 10, (0, 255, 0), 2)
                        cv2.circle(frame, thumb_point, 10, (0, 255, 0), 2)
                        cv2.line(frame, index_point, thumb_point, (255, 0, 0), 2)

                elif hand_label == 'Right':
                    right_pinch = distance < 30
                    right_center = center_point
                    # ✅ Right: always draw
                    cv2.circle(frame, index_point, 10, (0, 255, 0), 2)
                    cv2.circle(frame, thumb_point, 10, (0, 255, 0), 2)
                    cv2.line(frame, index_point, thumb_point, (255, 0, 0), 2)

        if left_pinch:
            left_pinch_active = True

        if right_pinch:
            right_pinch_active = True

        # === SLIDE CONTROL ===
        if left_pinch_active and right_pinch_active and left_center and right_center:
            lr_distance = math.dist(left_center, right_center)

            if prev_lr_distance is not None:
                current_time = time.time()
                if not locked or (current_time - lock_time > timeout_duration):
                    try:
                        if lr_distance - prev_lr_distance > 10:
                            print("next_slide")
                            client_socket.sendall(b'next_slide')
                            locked = True
                            lock_time = current_time
                        elif prev_lr_distance - lr_distance > 10:
                            print("previous_slide")
                            client_socket.sendall(b'previous_slide')
                            locked = True
                            lock_time = current_time
                    except Exception as e:
                        print(f"Error sending slide: {e}")

            prev_lr_distance = lr_distance

        # === ZOOM CONTROL: IF RIGHT PINCH IS ACTIVE, USE LEFT DISTANCE ===
        if right_pinch_active and left_finger_distance is not None:
            zoom_percent = calculate_percentage(left_finger_distance, min_distance, max_distance)

            if prev_left_distance is not None:
                current_time = time.time()
                if not zoom_locked or (current_time - zoom_lock_time > zoom_timeout_duration):
                    try:
                        if left_finger_distance - prev_left_distance > 10:
                            msg = f"zoom_in:{zoom_percent}%"
                            print(msg)
                            client_socket.sendall(msg.encode())
                            zoom_locked = True
                            zoom_lock_time = current_time
                        elif prev_left_distance - left_finger_distance > 10:
                            msg = f"zoom_out:{zoom_percent}%"
                            print(msg)
                            client_socket.sendall(msg.encode())
                            zoom_locked = True
                            zoom_lock_time = current_time
                    except Exception as e:
                        print(f"Error sending zoom: {e}")

            prev_left_distance = left_finger_distance

        # RESET
        if not left_pinch and not right_pinch:
            left_pinch_active = False
            right_pinch_active = False
            prev_lr_distance = None
            prev_left_distance = None
            locked = False
            zoom_locked = False

        cv2.imshow('Hand Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    print("Closing socket and camera...")
    try:
        client_socket.close()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()
