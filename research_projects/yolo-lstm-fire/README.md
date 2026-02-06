
# 🔥 Fire Detection in CCTV Footage using YOLO + LSTM

📌 Project Description

This project implements fire detection in CCTV surveillance videos using a combination of YOLO (You Only Look Once) for object detection and LSTM (Long Short-Term Memory) for temporal sequence learning.
The system aims to detect fire accurately from images, video streams, and live webcam footage, enabling early fire detection for safety-critical environments.

# 🎯 Objectives
Detect fire in images, recorded videos, and live CCTV streams.

Improve detection accuracy by combining YOLO for spatial features and LSTM for temporal dependencies.

Provide an end-to-end pipeline from dataset preparation → training → inference.

Make the solution easy to run on Windows and MacOS.

# ⚙️ Methodology

1. Dataset Preparation

    Collected fire/non-fire images and videos.

    Performed data augmentation using Albumentations.

    Converted annotations into YOLO format.

2. Model Training

    Trained YOLOv8 on fire dataset.

    Extracted spatio-temporal features for LSTM model.

3. Inference

    Real-time detection on images, videos, and webcam feed.

    Fire classification enhanced by LSTM.


Dataset → Data Augmentation → YOLO Training
         → Feature Extraction → LSTM Training → Detection


# 📂 Project Structure

📂 Project Structure

yolo+lstm/
│── .venv/                     # Virtual environment  
│── models/  
│   ├── best.pt                 # Trained YOLO model  
│   ├── lstm_fire_model.h5       # Trained LSTM model  
│  
│── test/  
│   ├── checkvideo.py           # Video detection  
│   ├── livewebcam.py           # Webcam detection  
│   ├── test.py                 # Image detection  
│   ├── yolo+lstm_test.py       # YOLO + LSTM pipeline  
│  
│── test_images/                # Sample test images  
│── test_videos/                # Sample test videos  
│── README.md                   # Project documentation  
│── LICENSE                     # License file  





# 🛠️ Installation & Setup
🔹 1. Clone Repository


🔹 2. Create Virtual Environment

Mac/Linux:

in root folder after cloning :- 

python3 -m venv .venv

source .venv/bin/activate

Windows (PowerShell):-

python -m venv .venv

.\.venv\Scripts\activate

# ensure that the venv is activated 
🔹 3. Install Dependencies

```bash
pip install ultralytics==8.3.203
pip install opencv-python==4.12.0.88
pip install numpy==2.0.2
pip install pillow==11.3.0
pip install matplotlib==3.9.4
pip install torch==2.8.0
pip install torchvision==0.23.0
pip install tensorflow==2.20.0
pip install h5py==3.14.0
pip install scikit-learn==1.6.1
pip install pandas==2.3.2




# ▶️ Running the Project
make sure you are in correct directory


# to view in image,video, live web cam 
 1.   make sure you are in correct directory.
 2.   make sure that you have created venv and activated.
 3.   make sure you have set correct paths inside the files to access the images , videos, models 
   


# 🚀 Future Enhancements
1. Deploy as a real-time alert system (email/SMS notification).

2. Optimize inference for edge devices (Raspberry Pi, Jetson Nano).

3. Extend dataset with smoke detection.

## 👨‍💻 Author
Developed by k.suresh kumar

📧 Contact: kalavalasureshkumar@gmail.com


## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
