import configargparse
import cv2 as cv
import time

from utils import CvFpsCalc
from gestures import *
from collections import Counter

last_gesture_time = 0
gesture_cooldown = 1.5

gesture_threshold = 400 # configuration will be done at the start

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    # parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args


def send_websocket(command):
    with open('websocket.txt', 'a') as f:
        f.write(command + '\n')

def handle_gesture(debug_image, gesture_id, gesture_labels, wrist_y):
    global last_gesture_time
    current_time = time.time()

    is_above_threshold = 0

    if wrist_y is not None:
        is_above_threshold = wrist_y < gesture_threshold

    if gesture_id is not None and 0 <= gesture_id < len(gesture_labels):
        hand_gesture_name = gesture_labels[gesture_id]
    else:
        hand_gesture_name = "IDLE"

    if current_time - last_gesture_time >= gesture_cooldown:
        if (hand_gesture_name != "IDLE") & (is_above_threshold == 1):
            if hand_gesture_name == "Forward":
                # print("Detected Forward - Performing Action A")
                # send_websocket("Action A")
                a = 1
            elif hand_gesture_name == "Config":
                # print("Detected Config - Performing Action B")
                # send_websocket("Action B")
                a = 1
            elif hand_gesture_name == "Up":
                # print("Detected Up - Performing Action C")
                # send_websocket("Action C")
                a = 1
            elif hand_gesture_name == "OK":
                # print("Detected OK - Performing Action D")
                # send_websocket("Action D")
                a = 1
            elif hand_gesture_name == "Down":
                # print("Detected Down - Performing Action E")
                # send_websocket("Action E")
                a = 1
            elif hand_gesture_name == "Stop":
                # print("Detected Stop - Performing Action F")
                # send_websocket("Action F")
                a = 1
            elif hand_gesture_name == "Left":
                send_websocket(hand_gesture_name)
            elif hand_gesture_name == "Right":
                send_websocket(hand_gesture_name)
            
            last_gesture_time = current_time

    return debug_image, hand_gesture_name

def main():
    global gesture_buffer
    global gesture_id

    # clear websocket ouput
    with open("websocket.txt", "w") as f:
        f.write("# Gesture Commands Log\n")

    # Argument parsing
    args = get_args()
    WRITE_CONTROL = False

    gesture_labels = ["Forward", "Config", "Up", "OK", "Down", "Stop", "Left", "Right"]

    cap = cv.VideoCapture(1)

    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1

    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            continue

        # interpret image
        debug_image, gesture_id, hand_landmarks_list = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # Red line for testing purposes
        cv.line(debug_image, (0, gesture_threshold), (args.width*2, gesture_threshold), (0, 0, 255), 2)

        wrist_y = None
        if hand_landmarks_list is not None and len(hand_landmarks_list) > 0:
            image_width, image_height = debug_image.shape[1], debug_image.shape[0]
            wrist_positions = []

            for hand_landmarks in hand_landmarks_list[:2]:
                wrist_y_pos = int(hand_landmarks.landmark[1].y * image_height)
                wrist_positions.append(wrist_y_pos)

            wrist_y = min(wrist_positions)

        # handle image
        debug_image, gesture_name = handle_gesture(debug_image, gesture_id, gesture_labels, wrist_y)

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
