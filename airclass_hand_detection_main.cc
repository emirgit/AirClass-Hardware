// AirClass – gesture-driven command recogniser.
// Two-hand thumbs-up toggles ACTIVE / PASSIVE.
// While ACTIVE, single-hand gestures emit four textual commands.
// This version includes video display, an attempt to set the resolution, an FPS counter,
// and a 3-second cooldown after any gesture-driven action.
// Packet timestamps for MediaPipe are now based on a monotonic clock.

// TODO:
// 1. Decide the distinct hand gesture
// 2. Make the code more modular such as dividing it into the files or methods
// 3. Apply the OOP principle and integrate the communication system to allow communication with server

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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace mp = mediapipe;

/* ---------- helpers ---------- */

// The Gesture enum represents the basic hand poses we recognise.
enum class Gesture { kUnknown, kThumbsUp, kThumbsDown, kOpenPalm, kClosedPalm };

// A fingertip is considered extended when its y-coordinate is above the PIP joint.
inline bool finger_extended(const mp::NormalizedLandmark& tip,
                            const mp::NormalizedLandmark& pip) {
  return tip.y() < pip.y();
}

// The classify function interprets a list of 21 hand landmarks into one of the Gesture values.
Gesture classify(const mp::NormalizedLandmarkList& lm) {
  if (lm.landmark_size() < 21) {
    // If there aren’t enough landmarks, the gesture cannot be determined.
    return Gesture::kUnknown;
  }
  const auto& wrist = lm.landmark(0);
  const auto& thumb_tip = lm.landmark(4);
  const auto& thumb_ip  = lm.landmark(3);

  // Count how many of the four fingers (index through pinky) are extended.
  int extended_count =
      finger_extended(lm.landmark(8),  lm.landmark(6))  +
      finger_extended(lm.landmark(12), lm.landmark(10)) +
      finger_extended(lm.landmark(16), lm.landmark(14)) +
      finger_extended(lm.landmark(20), lm.landmark(18));

  bool thumb_up   = finger_extended(thumb_tip, thumb_ip) && (thumb_tip.y() < wrist.y());
  bool thumb_down = !finger_extended(thumb_tip, thumb_ip) && (thumb_tip.y() > wrist.y());

  if (thumb_up   && extended_count == 0) return Gesture::kThumbsUp;
  if (thumb_down && extended_count == 0) return Gesture::kThumbsDown;
  if (extended_count >= 4)               return Gesture::kOpenPalm;
  if (extended_count == 0)               return Gesture::kClosedPalm;
  return Gesture::kUnknown;
}

/* ---------- main ---------- */

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  // The graph configuration text is read from its PBtxt file on disk.
  const std::string graph_path =
      "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt";
  std::string graph_txt;
  auto status = mp::file::GetContents(graph_path, &graph_txt);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to load graph config from " << graph_path << ": " << status;
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Graph configuration loaded.";

  mp::CalculatorGraphConfig cfg;
  if (!google::protobuf::TextFormat::ParseFromString(graph_txt, &cfg)) {
    LOG(ERROR) << "Failed to parse graph configuration.";
    return EXIT_FAILURE;
  }

  mp::CalculatorGraph graph;
  status = graph.Initialize(cfg);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize graph: " << status;
    return EXIT_FAILURE;
  }
  LOG(INFO) << "MediaPipe graph initialized.";

  // The camera is opened, and the driver’s internal buffer is limited to one frame.
  LOG(INFO) << "Opening camera device 0...";
  cv::VideoCapture cam(0);
  if (!cam.isOpened()) {
    LOG(ERROR) << "Cannot open camera device 0.";
    return EXIT_FAILURE;
  }
  if (!cam.set(cv::CAP_PROP_BUFFERSIZE, 1)) {
    LOG(WARNING) << "Could not limit camera buffer size; some queuing may occur.";
  }
  cam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  const int cam_width  = static_cast<int>(cam.get(cv::CAP_PROP_FRAME_WIDTH));
  const int cam_height = static_cast<int>(cam.get(cv::CAP_PROP_FRAME_HEIGHT));
  LOG(INFO) << "Camera ready at resolution: " << cam_width << "x" << cam_height;

  // Pollers are created to pull landmarks and rendered video from the graph.
  auto landmark_poller_or = graph.AddOutputStreamPoller("landmarks");
  if (!landmark_poller_or.ok()) {
    LOG(ERROR) << "Failed to add poller for landmarks: " << landmark_poller_or.status();
    cam.release();
    return EXIT_FAILURE;
  }
  auto landmark_poller = std::move(landmark_poller_or.value());

  auto video_poller_or = graph.AddOutputStreamPoller("output_video");
  if (!video_poller_or.ok()) {
    LOG(ERROR) << "Failed to add poller for output_video: " << video_poller_or.status();
    cam.release();
    return EXIT_FAILURE;
  }
  auto video_poller = std::move(video_poller_or.value());

  // The graph is started, ready to process incoming frames.
  status = graph.StartRun({});
  if (!status.ok()) {
    LOG(ERROR) << "Failed to start graph run: " << status;
    cam.release();
    return EXIT_FAILURE;
  }
  LOG(INFO) << "Graph run started.";

  bool active = false;
  bool last_both_up = false;

  const std::string window_name = "AirClass Output";
  cv::namedWindow(window_name, /*flags=*/0);

  // Variables for FPS computation.
  auto fps_start_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  double display_fps = 0.0;

  // A cooldown timer ensures at least 3 seconds between any two actions.
  auto last_action_time = std::chrono::steady_clock::now() - std::chrono::seconds(10);
  const std::chrono::seconds cooldown(3);

  LOG(INFO) << "Entering main loop (press ESC to exit)...";
  while (true) {
    // A fresh camera frame is grabbed here.
    cv::Mat frame_bgr;
    cam >> frame_bgr;
    if (frame_bgr.empty()) {
      LOG(WARNING) << "Empty frame received from camera.";
      if (!cam.isOpened()) {
        LOG(ERROR) << "Camera appears to be disconnected.";
        break;
      }
      cv::waitKey(1);
      continue;
    }

    // The frame is wrapped into a MediaPipe ImageFrame and sent to the graph.
    auto input_frame = std::make_unique<mp::ImageFrame>(
        mp::ImageFormat::SRGB, frame_bgr.cols, frame_bgr.rows,
        mp::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_mat = mp::formats::MatView(input_frame.get());
    cv::cvtColor(frame_bgr, input_mat, cv::COLOR_BGR2RGB);

    // Packet timestamps use the monotonic steady_clock to avoid any system-time jumps.
    auto now_tp = std::chrono::steady_clock::now();
    int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         now_tp.time_since_epoch())
                         .count();
    mp::Timestamp timestamp(now_us);

    status = graph.AddPacketToInputStream(
        "input_video", mp::Adopt(input_frame.release()).At(timestamp));
    if (!status.ok()) {
      LOG(ERROR) << "Failed to add packet to input stream: " << status;
      break;
    }

    // The landmark poller is drained so only the most recent set is used.
    std::vector<Gesture> gestures;
    if (landmark_poller.QueueSize() > 0) {
      mp::Packet packet;
      int n = landmark_poller.QueueSize();
      for (int i = 0; i < n - 1; ++i) {
        landmark_poller.Next(&packet);
      }
      if (landmark_poller.Next(&packet)) {
        const auto& hand_lists =
            packet.Get<std::vector<mp::NormalizedLandmarkList>>();
        for (const auto& lm : hand_lists) {
          gestures.push_back(classify(lm));
        }
      }
    }

    // The video poller is drained so only the newest rendered frame is shown.
    cv::Mat graph_output_bgr;
    bool got_video = false;
    if (video_poller.QueueSize() > 0) {
      mp::Packet packet;
      int m = video_poller.QueueSize();
      for (int i = 0; i < m - 1; ++i) {
        video_poller.Next(&packet);
      }
      if (video_poller.Next(&packet)) {
        const auto& output_frame = packet.Get<mp::ImageFrame>();
        cv::Mat output_mat = mp::formats::MatView(&output_frame);
        cv::cvtColor(output_mat, graph_output_bgr, cv::COLOR_RGB2BGR);
        got_video = true;
      }
    }

    // Gesture logic runs only if the cooldown has expired.
    bool in_cooldown = std::chrono::steady_clock::now() < (last_action_time + cooldown);
    bool current_both_up = (gestures.size() == 2 &&
                            gestures[0] == Gesture::kThumbsUp &&
                            gestures[1] == Gesture::kThumbsUp);
    if (!in_cooldown) {
      bool did_action = false;

      // A transition into two-thumbs-up toggles the ACTIVE state.
      if (current_both_up && !last_both_up) {
        active = !active;
        std::cout << "\n=== SYSTEM " << (active ? "ACTIVATED" : "PASSIVE") << " ===\n";
        did_action = true;
      }
      // When ACTIVE, a single hand triggers one of four textual commands.
      else if (active && gestures.size() == 1 && !current_both_up) {
        switch (gestures[0]) {
          case Gesture::kThumbsUp:
            std::cout << "Command: ACCEPT\n";
            did_action = true;
            break;
          case Gesture::kThumbsDown:
            std::cout << "Command: REJECT\n";
            did_action = true;
            break;
          case Gesture::kOpenPalm:
            std::cout << "Command: RIGHT\n";
            did_action = true;
            break;
          case Gesture::kClosedPalm:
            std::cout << "Command: LEFT\n";
            did_action = true;
            break;
          default:
            break;
        }
      }
      if (did_action) {
        last_action_time = std::chrono::steady_clock::now();
      }
    }

    // The transition state is updated so we can catch activation toggles next frame.
    last_both_up = current_both_up;

    // Choose the freshest frame for display.
    cv::Mat display_frame = got_video ? graph_output_bgr : frame_bgr;

    // Update and overlay the FPS counter.
    frame_count++;
    auto fps_now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(fps_now - fps_start_time).count();
    if (elapsed >= 1.0) {
      display_fps = frame_count / elapsed;
      fps_start_time = fps_now;
      frame_count = 0;
    }
    if (!display_frame.empty()) {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(1) << display_fps;
      cv::putText(display_frame, "FPS: " + ss.str(), {10, 30},
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
      if (in_cooldown) {
        cv::putText(display_frame, "COOLDOWN", {10, 60},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 0, 255}, 2);
      }
      cv::imshow(window_name, display_frame);
    }

    // Pressing ESC exits the main loop.
    if (cv::waitKey(5) == 27) {
      LOG(INFO) << "ESC pressed, exiting.";
      break;
    }
  }

  // Cleanup of streams and graph shutdown.
  graph.CloseAllPacketSources().IgnoreError();
  graph.WaitUntilDone().IgnoreError();
  cv::destroyWindow(window_name);
  if (cam.isOpened()) {
    cam.release();
  }
  LOG(INFO) << "AirClass terminated.";
  return EXIT_SUCCESS;
}
