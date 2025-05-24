import cv2
import mediapipe as mp
import numpy as np
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Finger landmark indices
finger_indices = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

# Base landmark indices for each finger
finger_bases = {
    'thumb': 1,  # Added thumb base as landmark 1 (CMC joint)
    'index': 5,
    'middle': 9,
    'ring': 13,
    'pinky': 17
}

# Initialize camera
cap = cv2.VideoCapture(1)
selected_finger = 'index'  # Default finger

# Function to calculate 3D distance between landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + 
                    (landmark1.y - landmark2.y)**2 + 
                    (landmark1.z - landmark2.z)**2)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from camera")
        break
        
    # Flip the image horizontally for a more intuitive mirror view
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_image)
    
    # Draw the hand annotations on the image
    image_height, image_width, _ = image.shape
    
    # Create a black region at the top for instructions
    instruction_region = np.zeros((80, image_width, 3), dtype=np.uint8)
    cv2.putText(instruction_region, "Press key to select finger: (t)humb, (i)ndex, (m)iddle, (r)ing, (p)inky, (q)uit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(instruction_region, f"Currently tracking: {selected_finger} finger", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Combine instruction region with the main image
    display_image = np.vstack([instruction_region, image])
    
    text_color = (0, 0, 255)  # Red color in BGR
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all hand landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            # Get selected fingertip landmark
            finger_idx = finger_indices[selected_finger]
            landmark = hand_landmarks.landmark[finger_idx]
            
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            
            # Draw a circle on the selected fingertip
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
            
            # Calculate distance between tip and base for the selected finger
            distance_text = ""
            if selected_finger in finger_bases:
                base_idx = finger_bases[selected_finger]
                base_landmark = hand_landmarks.landmark[base_idx]
                distance = calculate_distance(landmark, base_landmark)
                
                # Draw base point in a different color
                base_x, base_y = int(base_landmark.x * image_width), int(base_landmark.y * image_height)
                cv2.circle(image, (base_x, base_y), 8, (255, 0, 0), -1)  # Blue circle
                
                # Draw line between tip and base
                cv2.line(image, (x, y), (base_x, base_y), (255, 0, 255), 2)  # Magenta line
                
                distance_text = f"distance: {distance:.4f}"
                
                # For specific gestures, show additional info in actual pixel coordinates
                if selected_finger == 'index' or selected_finger == 'middle':
                    pixel_text = f"Width: {x}, Height: {y}"
                    cv2.putText(image, pixel_text, (10, image_height - 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
            # Display coordinates
            coord_text = f"X: {landmark.x:.3f}, Y: {landmark.y:.3f}, Z: {landmark.z:.3f}"
            cv2.putText(image, coord_text, (10, image_height - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
            # Display distance if applicable
            if distance_text:
                cv2.putText(image, distance_text, (10, image_height - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            
            # Draw line from wrist to selected fingertip
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * image_width), int(wrist.y * image_height)
            cv2.line(image, (wrist_x, wrist_y), (x, y), (0, 255, 0), 2)
    
    # Combine instruction region with the main image
    display_image = np.vstack([instruction_region, image])
    
    # Display the resulting frame
    cv2.imshow('MediaPipe Hand Tracking', display_image)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        selected_finger = 'thumb'
    elif key == ord('i'):
        selected_finger = 'index'
    elif key == ord('m'):
        selected_finger = 'middle'
    elif key == ord('r'):
        selected_finger = 'ring'
    elif key == ord('p'):
        selected_finger = 'pinky'

# Release resources
cap.release()
cv2.destroyAllWindows()