
# 📝 EMNIST Handwritten Character Recognition

### *A Deep Learning Project with Statistical Validation Interface*

This project implements a Convolutional Neural Network (CNN) to classify 47 different categories of handwritten characters using the EMNIST (Extended MNIST) Balanced dataset. It features a Gradio-based analysis tool to verify model performance across thousands of test samples.

## 🚀 Project Overview

* **Dataset:** EMNIST Balanced (131,600 images).
* **Classes:** 47 (Digits 0-9, Uppercase A-Z, and select lowercase letters).
* **Framework:** TensorFlow / Keras.
* **Accuracy Logic:** Statistical batch testing (100 samples per class).

## 🛠️ Data Pipeline & Preprocessing

To ensure the model learns effectively, the raw EMNIST data underwent several critical transformations:

1. **Orientation Correction:** EMNIST images are natively stored transposed and flipped. We applied `np.rot90(np.fliplr(img))` to restore them to an upright, human-readable position.
2. **Normalization:** Pixel values were scaled from  to a  range to improve training stability and speed.
3. **One-Hot Encoding:** Labels were converted into categorical vectors to facilitate multi-class classification.

## 🧠 Model Architecture

The CNN was designed to extract spatial features from the  grayscale inputs:

* **Input Layer:** .
* **Convolutional Layers:** Two layers with ReLU activation for feature detection.
* **Pooling:** Max-pooling layers to reduce dimensionality.
* **Regularization:** Dropout (0.5) to prevent overfitting.
* **Output:** Dense layer with Softmax activation for 47-class probability distribution.

## 📊 Statistical Evaluation Tool

While live drawing interfaces can be affected by stroke thickness and canvas resolution, this project utilizes a **Statistical Accuracy Tester** for objective validation:

* **Batch Analysis:** Instead of a single image, the tool pulls 100 random samples from the test set () for a chosen character.
* **Real-time Accuracy Reporting:** Calculates the percentage of correct identifications to show exactly where the model is strongest.
* **Error Identification:** Helps identify "confusion pairs" (e.g., why the model might mistake a handwritten '7' for a '6').

## 💻 How to Use

1. **Environment:** Run the notebook in Google Colab or a local environment with a GPU.
2. **Training:** Execute the training cells to fit the model on the EMNIST Balanced data.
3. **Analysis:** Launch the Gradio link at the bottom of the notebook.
4. **Test:** Select a character from the dropdown and click "Run Accuracy Test" to see the statistical breakdown.
