# 🌲 Forest Cover Type Classification  
### Deep Learning (MLP) vs Ensemble Learning (Random Forest)

## 📌 Project Overview
This project compares a **deep learning Multi-Layer Perceptron (MLP)** with a **Random Forest ensemble model** on the UCI Forest CoverType dataset.  
The objective is to improve neural network performance using architectural tuning and regularization, and to analyze why ensemble models often outperform neural networks on structured/tabular data.

---

## 📊 Dataset
- **Name:** UCI Forest CoverType
- **Source:** sklearn.datasets.fetch_covtype
- **Samples:** ~581,000
- **Features:** 54
- **Classes:** 7 forest cover types
- **Note:** Dataset is not uploaded due to size; loaded directly via sklearn.

---

## 🎯 Objectives
- Improve baseline MLP using:
  - Deeper & wider architectures
  - Batch Normalization
  - Dropout & L2 Regularization
  - Optimizer & Learning Rate tuning
- Achieve ~94% test accuracy (target)
- Compare MLP with RandomForest using GridSearchCV
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix

---

## 🧠 Models Implemented

### 1️⃣ Baseline MLP
- ReLU activation
- Minimal architecture
- Accuracy: ~84%

### 2️⃣ Improved & Regularized MLP
- LeakyReLU + BatchNorm
- Dropout & L2 Regularization
- Adam optimizer + LR scheduling
- Accuracy: ~88%

### 3️⃣ RandomForest (GridSearchCV)
- Hyperparameter tuning
- Excellent generalization
- Accuracy: ~94–95%

---

## 📈 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (Macro & Weighted)
- Confusion Matrix
- Training & Validation Curves

---

## 🔍 Key Insights
- Neural networks require extensive tuning on tabular data
- Ensemble methods naturally handle feature interactions
- RandomForest outperformed MLP despite heavy regularization
- Confirms theoretical expectations taught in class

---

## 🛠️ How to Run
```bash
pip install -r requirements.txt
