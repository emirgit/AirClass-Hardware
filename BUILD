cc_binary(
    name = "airclass_hand_detection",
    srcs = ["airclass_hand_detection_main.cc"],

    data = [
        "//mediapipe/graphs/hand_tracking:hand_tracking_desktop_live.pbtxt",
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],

    deps = [
        # ── MediaPipe core ──
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:parse_text_proto",

        "//mediapipe/calculators/core:constant_side_packet_calculator_cc_proto",


        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",

        # ── OpenCV facades ──
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_videoio",
        "//mediapipe/framework/port:opencv_imgproc",

        "@com_google_protobuf//:protobuf",
    ],
)
