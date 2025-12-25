# CodeAlpha_Handwritten_Character-Recognition


### *End-to-End CNN with Interactive Gradio Interface*

This repository contains a full-stack Machine Learning pipeline that classifies handwritten digits and letters. It solves the common "orientation" problem in the EMNIST dataset and provides a web-based UI for real-time testing.

---

## 🏗️ Technical Architecture

The system is built on a custom Convolutional Neural Network (CNN) designed to balance speed and accuracy.

### 1. The Model "Brain"

* **Convolutional Layers:** Two layers using  filters to detect edges, curves, and loops.
* **Pooling:** Max-pooling layers to make the model invariant to the exact position of the character.
* **Regularization:** Dropout at 50% to prevent the model from overfitting on specific handwriting styles.
* **Output:** A 47-way Softmax layer providing probability distributions for all supported characters.

---

## 📊 Dataset: EMNIST Balanced

We use the `emnist-balanced-train.csv` dataset from Kaggle.

* **Total Classes:** 47
* **Contents:** * **0-9**: Digits.
* **A-Z**: Uppercase letters.
* **a-r**: Specific lowercase letters (where they differ significantly from uppercase).



### The "90-Degree" Solution

Raw EMNIST data is stored in a transposed format (flipped and rotated). This project includes a preprocessing pipeline that restores characters to their natural upright position:

```python
# The Transformation Logic
X = np.rot90(np.fliplr(img))

```

---

## 🌐 Gradio Web Interface

The project concludes with a **Gradio** deployment. This creates a local web server where you can draw on a digital canvas.

* **Input:** 28x28 Sketchpad.
* **Processing:** The sketch is automatically resized, grayscaled, and normalized to match the training data.
* **Output:** The predicted character and a bar chart showing the model's confidence.

---

## 🚀 Step-by-Step Execution Flow

1. **Authentication:** Automated Kaggle login using environment variables.
2. **Acquisition:** Downloading the 47-class balanced dataset via `opendatasets`.
3. **Preprocessing:** * Reshaping to .
* Normalization to  range.
* Categorical one-hot encoding for 47 classes.


4. **Training:** 10 epochs with the Adam optimizer and Categorical Crossentropy loss.
5. **Evaluation:** Generating a **Confusion Matrix** to identify "look-alike" errors (e.g., 'O' vs '0').
6. **Deployment:** Launching the Gradio UI.

---

## 🛠️ Requirements

* `tensorflow` (Deep Learning)
* `gradio` (Web UI)
* `numpy` & `pandas` (Data Math)
* `matplotlib` & `seaborn` (Visualization)
* `opendatasets` (Kaggle Integration)

---

## 📝 Class Mapping Table

The model indices () are mapped as follows:
`0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt`

