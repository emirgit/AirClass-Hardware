// AirClass – gesture-driven command recogniser.
// Two-hand thumbs-up toggles ACTIVE / PASSIVE.
// While ACTIVE, single-hand gestures emit four textual commands.
// NOW includes video display.

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"  
#include "mediapipe/framework/port/opencv_highgui_inc.h"  

#include <google/protobuf/text_format.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <memory>
#include <vector>
#include <string>

namespace mp = mediapipe;

/* ---------- helpers ---------- */

enum class Gesture { kUnknown, kThumbsUp, kThumbsDown,
                     kOpenPalm, kClosedPalm };

inline bool finger_extended(const mp::NormalizedLandmark& tip,
                            const mp::NormalizedLandmark& pip) {
  return tip.y() < pip.y();
}

Gesture classify(const mp::NormalizedLandmarkList& lm) {
  if (lm.landmark_size() < 21) return Gesture::kUnknown;
  const auto& w  = lm.landmark(0);   // wrist
  const auto& t4 = lm.landmark(4);   // thumb-tip
  const auto& t3 = lm.landmark(3);   // thumb-ip
  int ext =
      finger_extended(lm.landmark( 8), lm.landmark( 6)) +
      finger_extended(lm.landmark(12), lm.landmark(10)) +
      finger_extended(lm.landmark(16), lm.landmark(14)) +
      finger_extended(lm.landmark(20), lm.landmark(18));
  bool thumb_up   = finger_extended(t4, t3) && (t4.y() < w.y());
  bool thumb_down = !finger_extended(t4, t3) && (t4.y() > w.y());
  if (thumb_up   && ext == 0) return Gesture::kThumbsUp;
  if (thumb_down && ext == 0) return Gesture::kThumbsDown;
  if (ext >= 4)               return Gesture::kOpenPalm;
  if (ext == 0)               return Gesture::kClosedPalm;
  return Gesture::kUnknown;
}

/* ---------- main ---------- */

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  /* 1. Load graph config */
  const std::string graph_path =
      "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";
  std::string graph_txt;
  absl::Status st = mp::file::GetContents(graph_path, &graph_txt);
  if (!st.ok()) {
    LOG(ERROR) << "Failed to load graph config from " << graph_path << ": " << st;
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Graph config loaded successfully from: " << graph_path;

  mp::CalculatorGraphConfig cfg;
  if (!google::protobuf::TextFormat::ParseFromString(graph_txt, &cfg)) {
    LOG(ERROR) << "Failed to parse graph config.";
    return EXIT_FAILURE;
  }

  mp::CalculatorGraph graph;
  st = graph.Initialize(cfg);
  if (!st.ok()) {
    LOG(ERROR) << "Failed to initialize graph: " << st;
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Graph initialized successfully.";

  /* 2. Camera */
  LOG(INFO) << "Opening camera device 0...";
  cv::VideoCapture cam(0);
  if (!cam.isOpened()) {
    LOG(ERROR) << "ERROR: Cannot open camera device 0.";
    return EXIT_FAILURE;
  }
  const int cam_width = static_cast<int>(cam.get(cv::CAP_PROP_FRAME_WIDTH));
  const int cam_height = static_cast<int>(cam.get(cv::CAP_PROP_FRAME_HEIGHT));
  LOG(INFO) << "Camera opened successfully: " << cam_width << "x" << cam_height;

  /* 4. Pollers for output streams (BEFORE StartRun) */
  // --- Poller for Landmarks ---
  const std::string landmark_stream_name = "landmarks";
  LOG(INFO) << "Adding output stream poller for '" << landmark_stream_name << "'...";
  auto landmark_poller_or = graph.AddOutputStreamPoller(landmark_stream_name);
  if (!landmark_poller_or.ok()) {
    LOG(ERROR) << "Failed to add poller for stream '" << landmark_stream_name
               << "': " << landmark_poller_or.status();
    return EXIT_FAILURE;
  }
  auto landmark_poller = std::move(landmark_poller_or.value());
  LOG(INFO) << "Landmark poller created successfully.";

  // --- Poller for Video Output ---
  const std::string video_stream_name = "output_video";
  LOG(INFO) << "Adding output stream poller for '" << video_stream_name << "'...";
  auto video_poller_or = graph.AddOutputStreamPoller(video_stream_name);
  if (!video_poller_or.ok()) {
    LOG(ERROR) << "Failed to add poller for stream '" << video_stream_name
               << "': " << video_poller_or.status();
    return EXIT_FAILURE;
  }
  auto video_poller = std::move(video_poller_or.value());
  LOG(INFO) << "Video poller created successfully.";


  /* 3. Start the graph */
  LOG(INFO) << "Starting graph run (no side packets)...";
  st = graph.StartRun({});
  if (!st.ok()) {
    LOG(ERROR) << "Failed to start graph run: " << st;
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Graph run started successfully.";

  /* 5. Main loop */
  bool active = false;
  bool last_both_up = false;
  int64_t frame_timestamp_us = 0;
  const int64_t frame_interval_us = 33333; // ~30 FPS
  const std::string window_name = "AirClass Output";

  cv::Mat frame_bgr; // Input frame from camera
  cv::Mat frame_display; // Frame to display (output from graph)

  LOG(INFO) << "Starting main loop (press ESC to exit)...";

  while (true) {
    // --- Get Frame from Camera ---
    cam >> frame_bgr;
    if (frame_bgr.empty()) {
      LOG(WARNING) << "Camera returned empty frame.";
      if (!cam.isOpened()) { LOG(ERROR) << "Camera disconnected."; break; }
      cv::waitKey(1); continue;
    }

    // --- Prepare Input Packet ---
    auto input_frame = std::make_unique<mp::ImageFrame>(
        mp::ImageFormat::SRGB, frame_bgr.cols, frame_bgr.rows,
        mp::ImageFrame::kDefaultAlignmentBoundary);
    // Create a view into the mp::ImageFrame's data to copy into
    cv::Mat input_frame_mat = mp::formats::MatView(input_frame.get());
    // Copy BGR to RGB into the view
    cv::cvtColor(frame_bgr, input_frame_mat, cv::COLOR_BGR2RGB);

    // --- Send Frame to Graph ---
    st = graph.AddPacketToInputStream(
           "input_video",
           mp::Adopt(input_frame.release()).At(mp::Timestamp(frame_timestamp_us)));
    if (!st.ok()) {
      LOG(ERROR) << "Failed to add packet to input stream 'input_video': " << st;
      break;
    }
    frame_timestamp_us += frame_interval_us;

    // --- Receive Landmarks ---
    mp::Packet landmark_packet;
    std::vector<mp::NormalizedLandmarkList> hands;
    if (landmark_poller.Next(&landmark_packet)) {
        hands = landmark_packet.Get<std::vector<mp::NormalizedLandmarkList>>();
    } // else: No landmarks packet available yet for this timestamp

    // --- Receive Output Video Frame ---
    mp::Packet video_packet;
    if (video_poller.Next(&video_packet)) {
      // Get the ImageFrame
      const auto& output_frame = video_packet.Get<mp::ImageFrame>();

      // Convert to cv::Mat for display. Use MatView to avoid copying if possible.
      // The output from HandRendererSubgraph is likely RGB.
      cv::Mat output_frame_mat = mp::formats::MatView(&output_frame);

      // Convert RGB back to BGR for OpenCV imshow
      cv::cvtColor(output_frame_mat, frame_display, cv::COLOR_RGB2BGR);

    } // else: No video packet available yet. Keep showing previous frame?

    /* --- Gesture Logic (remains the same) --- */
    std::vector<Gesture> current_gestures;
    for (const auto& hand_landmarks : hands) {
      current_gestures.push_back(classify(hand_landmarks));
    }
    bool both_up = current_gestures.size() == 2 &&
                   current_gestures[0] == Gesture::kThumbsUp &&
                   current_gestures[1] == Gesture::kThumbsUp;
    if (both_up && !last_both_up) {
      active = !active;
      std::cout << "\n=== SYSTEM " << (active ? "ACTIVATED" : "PASSIVE") << " ===\n";
    }
    last_both_up = both_up;
    if (active && current_gestures.size() == 1 && !both_up) {
      switch (current_gestures[0]) {
        case Gesture::kThumbsUp:   std::cout << "Command: ACCEPT\n"; break;
        case Gesture::kThumbsDown: std::cout << "Command: REJECT\n"; break;
        case Gesture::kOpenPalm:   std::cout << "Command: RIGHT\n";  break;
        case Gesture::kClosedPalm: std::cout << "Command: LEFT\n";   break;
        case Gesture::kUnknown:    break;
      }
    }

    // --- Display Video Frame ---
    if (!frame_display.empty()) {
        cv::imshow(window_name, frame_display);
    } else {
        // Optionally display the raw camera feed if graph output isn't ready yet
        // cv::imshow(window_name, frame_bgr);
        LOG_EVERY_N(INFO, 300) << "Waiting for initial video output from graph..."; // Log occasionally
    }

    // --- Exit Condition ---
    if (cv::waitKey(5) == 27) { // ESC key
        LOG(INFO) << "ESC pressed, exiting loop.";
        break;
    }
  } // End main while loop

  /* 6. Clean up */
  LOG(INFO) << "Closing input streams...";
  st = graph.CloseAllPacketSources();
  if (!st.ok()) { LOG(ERROR) << "Failed to close input streams: " << st; }

  LOG(INFO) << "Waiting for graph to complete...";
  st = graph.WaitUntilDone();
  if (!st.ok()) { LOG(ERROR) << "Failed while waiting for graph to complete: " << st; }

  cv::destroyWindow(window_name); // Close the OpenCV window
  LOG(INFO) << "AirClass finished.";
  return st.ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}
