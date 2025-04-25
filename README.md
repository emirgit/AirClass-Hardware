# AirClass - Hardware

AirClass is a project developed for the **CSE396 Computer Engineering Course** at **Gebze Technical University**.

This repository contains the **Hardware** part of the system, responsible for capturing and recognizing hand gestures using a **Raspberry Pi** and a camera. Gestures are interpreted using **OpenCV with C++** and sent as commands to the desktop application to control digital slides and other classroom tools. The goal is to allow teachers to interact with lesson content without needing to return to the computer — making classroom experiences more interactive and seamless.

---

## 👥 Team Members

| Student Number  | Name                    | Role               | GitHub | LinkedIn |
|-----------------|-------------------------|--------------------|--------|----------|
| 210104004071    | Muhammed Emir Kara      | Hardware Developer | [GitHub](https://github.com/emirgit) | [LinkedIn](https://www.linkedin.com/in/muhammed-emir-kara-787605251/) |
| 240104004980    | Helin Saygılı           | Hardware Developer | [GitHub](#) | [LinkedIn](#) |
| 210104004092    | Ahmet Mücahit Gündüz    | Hardware Developer | [GitHub](#) | [LinkedIn](#) |
| 200104004015    | Ahmet Sadri Güler       | Hardware Developer | [GitHub](#) | [LinkedIn](#) |
| 220104004921    | Ahmet Can Hatipoğlu     | Hardware Developer | [GitHub](#) | [LinkedIn](#) |
| 220104004923    | Selin Bardakcı          | Hardware Developer | [GitHub](#) | [LinkedIn](#) |
| 200104004068    | Kenan Eren Ayyılmaz     | Hardware Developer | [GitHub](https://github.com/Erenayyilmaz) | [LinkedIn](https://www.linkedin.com/in/kenanerenayyilmaz/) |
| 200104004107    | Veysel Cemaloğlu        | Hardware Developer | [GitHub](https://github.com/veyselcmlgl) | [LinkedIn](https://www.linkedin.com/in/veyselcmlgl/) |
| 210104004074    | Umut Hüseyin Satır      | Hardware Developer | [GitHub](#) | [LinkedIn](#) |

---

## 🛠️ Tech Stack

- **Language:** C++
- **Computer Vision:** OpenCV
- **Platform:** Raspberry Pi
- **Build System:** CMake
- **Communication:** Socket or Serial (with desktop module)

---

## 📄 Execution

> **Prerequisites**
>
> * MediaPipe cloned in `~/mediapipe`
> * Bazel ≥ 5  
> * OpenCV dev files (Ubuntu/Debian `sudo apt install libopencv-dev`)
> * A webcam passed through to the VM as **/dev/video0**  


### 1 . Drop the sources

```bash
# from *inside* the mediapipe repo
cd ~/mediapipe/mediapipe/examples/desktop

# copy or clone your project folder here
# result: mediapipe/examples/desktop/airclass_hand_detection

```
### 2. Build
```bash
cd ~/mediapipe          # repo root (same level as WORKSPACE)

# CPU-only build, disable GPU (not available in most VMs)
bazel build -c opt \
  --define MEDIAPIPE_DISABLE_GPU=1 \
  mediapipe/examples/desktop/airclass_hand_detection:airclass_hand_detection

```

### 2. Run
```bash
cd ~/mediapipe          # stay at repo root

# send MediaPipe logs to the terminal
GLOG_logtostderr=1 \
bazel-bin/mediapipe/examples/desktop/airclass_hand_detection/airclass_hand_detection

```


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
