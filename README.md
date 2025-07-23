# 😄 Human Emotion Detection Using Deep Learning

This project focuses on **classifying human facial emotions** into three categories — `Angry`, `Happy`, and `Sad` — using a series of progressively advanced deep learning models including CNNs, ResNets, Transformers, and Transfer Learning (EfficientNet & ViT). It is built using **TensorFlow**, **Keras**, and **Hugging Face Transformers**.

> 💡 This portfolio project showcases strong understanding of model building, training pipelines, evaluation, transfer learning, transformers, class imbalance handling, ensembling, and visualization techniques.

---

## 🚀 Project Highlights

- ✅ Custom CNN architecture (LeNet-style)
- ✅ Custom-built **ResNet34** using residual blocks
- ✅ Transfer learning with **EfficientNetB4**
- ✅ Custom **Vision Transformer (ViT)** implementation
- ✅ Pretrained **ViT from HuggingFace**
- ✅ **Model ensembling** (ResNet + EfficientNet)
- ✅ Handling **imbalanced datasets** with class weights
- ✅ **CutMix augmentation** + standard image augmentation
- ✅ Metrics tracking via **Weights & Biases (wandb)**
- ✅ Evaluation with:
  - Accuracy & loss curves
  - Confusion matrix
  - Classification report
  - Prediction visualizations
  - Feature map inspection (VGG-based)

---

## 📁 Dataset

**Source**: [Kaggle - Human Emotions Dataset](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes)

- Classes: `angry`, `happy`, `sad`
- Format: RGB images categorized into training and test folders
- Resized to `256x256`

---

## 🧠 Models Implemented

### 🔹 1. LeNet-Style CNN
- Entry-level CNN with Conv2D → MaxPool → Dense
- Used as a performance baseline

### 🔹 2. Custom ResNet34
- Built from scratch using residual blocks and skip connections
- Demonstrates in-depth architectural knowledge

### 🔹 3. Transfer Learning with EfficientNetB4
- Both frozen and fine-tuned backbones used
- Improved performance with fewer parameters

### 🔹 4. Vision Transformer (ViT)
- Patch encoding
- Transformer encoder blocks (Multi-head attention + FFN)
- Flatten + Dense for classification

### 🔹 5. Hugging Face ViT
- Uses `google/vit-base-patch16-224-in21k`
- Integrated with custom classification head
- Trained using Keras + Transformers

### 🔹 6. Model Ensembling
- Averaged predictions from ResNet34 and EfficientNetB4
- Boosts generalization

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Top-3 Accuracy**
- **Confusion Matrix** (normalized & raw)
- **Precision, Recall, F1-Score** via `classification_report`
- **Prediction Grid Visualization**
- **Feature Maps** from VGG for introspection

---

## 🧪 Sample Results

| Model               | Accuracy | Top-3 Accuracy | Notes                              |
|--------------------|----------|----------------|------------------------------------|
| LeNet CNN          | ~70%     | ~85%           | Baseline model                     |
| ResNet34 (custom)  | ~83%     | ~95%           | Deep, well-regularized             |
| EfficientNetB4     | ~88%     | ~97%           | Frozen + fine-tuned                |
| Vision Transformer | ~85%     | ~94%           | Custom ViT implementation          |
| Hugging Face ViT   | ~89%     | ~98%           | Pretrained ViT fine-tuned          |
| Ensemble Model     | **90%**  | **98%+**       | Combined predictions               |

> ⚠️ *Metrics may vary slightly depending on hardware, random seeds, and batch sizes.*

---

## 📦 Dependencies

Install required packages:
```bash
pip install -r requirements.txt
