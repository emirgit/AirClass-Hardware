import cv2
import mediapipe as mp
import math

# MediaPipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Kamera başlat
cap = cv2.VideoCapture(0)

# Durum değişkenleri
left_pinch_active = False
right_pinch_active = False
prev_lr_distance = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    h, w, _ = frame.shape

    # Sol ve sağ el pinch mesafeleri ve merkezleri
    left_pinch = False
    right_pinch = False
    left_center = None
    right_center = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            index_point = (int(index_tip.x * w), int(index_tip.y * h))
            thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            # Ekrana çizim
            cv2.circle(frame, index_point, 10, (0, 255, 0), 2)
            cv2.circle(frame, thumb_point, 10, (0, 255, 0), 2)
            cv2.line(frame, index_point, thumb_point, (255, 0, 0), 2)

            # Pinch mesafe hesapla
            distance = math.dist(index_point, thumb_point)
            center_point = ((index_point[0] + thumb_point[0]) // 2, (index_point[1] + thumb_point[1]) // 2)

            if hand_label == 'Left':
                left_pinch = distance < 30
                left_center = center_point
            elif hand_label == 'Right':
                right_pinch = distance < 30
                right_center = center_point

    # Mantık kontrol
    if left_pinch:
        left_pinch_active = True

    if left_pinch_active and right_pinch:
        right_pinch_active = True

    if left_pinch_active and right_pinch_active and left_center and right_center:
        # Sol ve sağ el merkezleri arasındaki mesafe
        lr_distance = math.dist(left_center, right_center)

        if prev_lr_distance is not None:
            if lr_distance - prev_lr_distance > 10:
                print("next_slide")
                left_pinch_active = False
                right_pinch_active = False
            elif prev_lr_distance - lr_distance > 10:
                print("previous_slide")
                left_pinch_active = False
                right_pinch_active = False

        prev_lr_distance = lr_distance

    cv2.imshow('Hand Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
