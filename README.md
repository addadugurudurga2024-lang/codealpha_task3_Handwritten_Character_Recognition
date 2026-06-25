# 🖊️ Handwritten Character Recognition

> **CodeAlpha Internship — Task 3**  
> A Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9).

---

## 📌 Project Overview

This project implements a **Handwritten Character Recognition** system using a deep learning CNN model built with TensorFlow/Keras. The model is trained on the classic **MNIST dataset** (70,000 grayscale images of handwritten digits) and achieves high accuracy on unseen test data.

---

## 📁 Project Structure

```
codealpha_task3_Handwritten_Character_Recognition/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── dataset/                        # Dataset info (auto-downloaded via Keras)
│
├── source_code/
│   ├── handwritten_character_recognition.py    # Main Python script
│
├── report/
│   └── Handwritten_Character_Recognition_Report.pdf
│
├── images/                         # Output visualizations
│   ├── sample_digits.png
│   ├── digit_distribution.png
│   ├── training_accuracy.png
│   ├── validation_accuracy.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── outputs/
│   ├── accuracy_output.txt
│   ├── classification_report.txt
│   └── sample_predictions.txt
│
└── .gitignore
```

---

## 🧠 Model Architecture

| Layer              | Type         | Details                        |
|--------------------|--------------|--------------------------------|
| Input              | Input Layer  | Shape: (28, 28, 1)             |
| Conv2D (1)         | Convolution  | 32 filters, 3×3, ReLU         |
| MaxPooling2D (1)   | Pooling      | 2×2                            |
| Conv2D (2)         | Convolution  | 64 filters, 3×3, ReLU         |
| MaxPooling2D (2)   | Pooling      | 2×2                            |
| Flatten            | Flatten      | —                              |
| Dense              | Fully Connected | 128 units, ReLU             |
| Dropout            | Regularization  | Rate: 0.3                   |
| Dense (Output)     | Fully Connected | 10 units, Softmax           |

- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Epochs:** 5  
- **Validation Split:** 20%

---

## 📊 Dataset

- **Name:** MNIST (Modified National Institute of Standards and Technology)
- **Training samples:** 60,000
- **Testing samples:** 10,000
- **Image size:** 28 × 28 grayscale
- **Classes:** 10 (digits 0–9)
- **Source:** Automatically downloaded via `tensorflow.keras.datasets.mnist`

No manual dataset download is required.

---

## 📈 Results

| Metric         | Value     |
|----------------|-----------|
| Test Accuracy  | ~99%      |
| Test Loss      | ~0.03     |

> Actual values may vary slightly across runs due to random weight initialization.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/addadugurudurga2024-lang/codealpha_task3_Handwritten_Character_Recognition.git
cd codealpha_task3_Handwritten_Character_Recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script

```bash
python source_code/handwritten_character_recognition.py
```

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for full dependency list.

Main libraries:
- `tensorflow >= 2.16`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**A D S ABHISHEK**  
CodeAlpha Internship — Task 3  
GitHub: [@addadugurudurga2024-lang](https://github.com/addadugurudurga2024-lang)

---

## 🏷️ Tags

`Deep Learning` · `CNN` · `MNIST` · `TensorFlow` · `Keras` · `Python` · `Computer Vision` · `Handwritten Recognition` · `CodeAlpha`
