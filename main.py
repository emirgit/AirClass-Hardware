import configargparse
import cv2 as cv
import time

from utils import CvFpsCalc
from gestures import *
from collections import Counter

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
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


def load_labels(label_path):
    print("Labels loading")
    labels = []
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]
    i = 1
    for label in labels:
        print("label " + str(i) + ": " + label)
        i += 1
    print("labels loaded")
    return labels

def handle_gesture(debug_image, gesture_id, gesture_labels):
    if gesture_id is not None and 0 <= gesture_id < len(gesture_labels):
        hand_gesture_name = gesture_labels[gesture_id]
        # print("handGesture Name: " + str(hand_gesture_name))
    else:
        hand_gesture_name = "IDLE"


    if hand_gesture_name == "Forward":
        print("Detected Forward - Performing Action A")
        send_websocket("Action A")
    elif hand_gesture_name == "Config":
        print("Detected Config - Performing Action B")
        send_websocket("Action B")
    elif hand_gesture_name == "Up":
        print("Detected Up - Performing Action C")
        send_websocket("Action C")
    elif hand_gesture_name == "OK":
        print("Detected OK - Performing Action D")
        send_websocket("Action D")
    elif hand_gesture_name == "Down":
        print("Detected Down - Performing Action E")
        send_websocket("Action E")
    elif hand_gesture_name == "Stop":
        print("Detected Stop - Performing Action F")
        send_websocket("Action F")
    elif hand_gesture_name == "Left":
        print("Detected Left - Performing Action G")
        send_websocket("Action G")
    elif hand_gesture_name == "Right":
        print("Detected Right - Performing Action H")
        send_websocket("Action H")

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

    # gesture_labels = load_labels('./model/keypoint_classifier/keypoint.csv')
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
        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # handle image
        debug_image, gesture_name = handle_gesture(debug_image, gesture_id, gesture_labels)

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
