# 👀 Real-Time Face Recognition with OpenCV & Python 🤖

## 🎯 Overview

Welcome to the **Face Recognition** project using **OpenCV** and **Python**! This simple yet powerful system allows you to **capture images**, **train a face recognition model**, and then use it for **real-time face detection**.

This repository includes everything you need to get started with face recognition, from image capture to model training and real-time predictions! 🚀 The best part? You can test it all out using just your **webcam** and **Python**. Whether you want to build a fun project or start learning about computer vision, this is a perfect starting point!

The project is split into three main parts:

1. **Capture Dataset** – Capture images of faces to create your own custom dataset.
2. **Train the Model** – Train the face recognition model using the captured dataset.
3. **Real-Time Face Recognition** – Use your trained model for real-time face recognition via your webcam.

---

## ✨ Features

- **📸 Image Capture**: Capture face images with just a webcam. Store them with unique IDs for later training.
- **🤖 Train Your Own Model**: Use **OpenCV's LBPH (Local Binary Pattern Histogram)** face recognizer to train your model on the captured faces.
- **🔍 Real-Time Recognition**: With a webcam, detect faces in real-time and get predictions with confidence scores.
- **💥 Alien Detection**: If the model isn’t sure about a face, it labels it "ALIEN" for fun! 👽
- **🛠️ Easy Setup**: All you need is a Python environment, OpenCV, and a webcam!

---

## 🏗️ How It Works

The project is divided into **3 main steps**: 

### 1. **Capture Dataset** – `generate_dataset.py`

The **`generate_dataset.py`** script captures images of faces using a webcam. Here’s how it works:

- **Face Detection**: Detects faces in real-time using **OpenCV's Haar Cascade Classifier**.
- **Image Storage**: Captures images of faces and stores them in a folder named `dataset` with unique file names (e.g., `user.1.1.jpg`, `user.1.2.jpg`).
- **Live Preview**: Displays the cropped face with an ID number for each captured image.

Run it using:

```bash
python generate_dataset.py
```

This script will continuously capture face images until you press **Enter** or reach the limit of 1000 images.

---

### 2. **Train the Model** – `train_model.py`

Once you have your face dataset, use the **`train_model.py`** script to train a face recognition model.

- **Dataset Loading**: Loads all the images from the `dataset` folder.
- **Face Recognition Model**: Trains a face recognition model using **OpenCV’s LBPH (Local Binary Pattern Histogram)** algorithm.
- **Model Saving**: Saves the trained model as `detector.xml` for later use.

Run it using:

```bash
python train_model.py
```

After training, you’ll have a model saved as `detector.xml`, which can be used for real-time predictions.

---

### 3. **Real-Time Face Recognition** – `predict_face.py`

The **`predict_face.py`** script uses your trained model for real-time face recognition via your webcam.

- **Face Detection**: Detects faces in the webcam feed using the **Haar Cascade Classifier**.
- **Model Prediction**: Uses the trained **LBPH model** to recognize faces.
- **Confidence Score**: Displays the prediction and a confidence score.
- **“Alien” Detection**: If the model isn’t confident (less than 75% confidence), it labels the face as **"ALIEN"** for fun. 👽

Run it using:

```bash
python predict_face.py
```

You’ll be able to see the webcam feed with a rectangle around the detected faces and the name of the recognized person displayed above their face.

---

## 🎥 Demo

Once everything is set up, you can see your system working in real-time! 🚀 Just follow these steps:

1. **Capture Images**: Run `generate_dataset.py` to collect face images for training.
2. **Train the Model**: Use `train_model.py` to create a custom face recognizer.
3. **Real-Time Recognition**: Finally, run `predict_face.py` and watch your face recognition system in action!

---

## 🛠️ Requirements

Before running the code, make sure you have the following:

- **Python 3.x** installed.
- **OpenCV** for computer vision tasks (for face detection and recognition).
- **NumPy** for handling image data.
- **Pillow** (PIL) for image manipulation.

Install them using:

```bash
pip install opencv-python numpy pillow
```

---

## 🔥 Example Output

Here’s what you can expect when running **`predict_face.py`** in real-time:

- **Face Detection**: The webcam feed will show rectangles around detected faces.
- **Face Recognition**: When a face is recognized, it will display the name and confidence level. For example:

  ```
  NIRBHAY (Confidence: 92.34%)
  ```

- **Alien Detection**: If the system is unsure (below 75% confidence), it will show "ALIEN" on the screen.

  ```
  ALIEN (Confidence: 60.12%)
  ```

---

## 🖥️ Troubleshooting

- **Low Recognition Accuracy**: If the model is not recognizing faces well, ensure the dataset is diverse and contains clear, well-lit images of faces. You can also increase the number of training images.
- **Webcam Issues**: If OpenCV can’t access your webcam, check the camera permissions or settings on your system.
- **Missing Libraries**: Ensure you have all the required dependencies installed (OpenCV, NumPy, Pillow).

---

## 🚀 Next Steps & Enhancements

This project is a great starting point for face recognition! Here are some ideas to enhance it further:

- **🎭 Emotion Detection**: Combine face recognition with emotion detection to make your system even smarter!
- **🧑‍🤝‍🧑 Multi-Person Recognition**: Train the model to recognize more people by adding more data to your dataset.
- **⚡ Transfer Learning**: Use pre-trained models for feature extraction and improve the accuracy of face recognition.
- **📦 Web Deployment**: Deploy this project as a web application using **Flask** or **Django** for online face recognition.

---

## 🎉 License

This project is licensed under the **MIT License**. Feel free to contribute, improve, and build upon it!

---

## 💬 Let’s Connect!

If you have any questions or suggestions, feel free to reach out! Happy coding, and enjoy building your very own **Face Recognition System**! 🎉

---