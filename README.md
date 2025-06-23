# Emotion Classification from Tweets

This project applies core NLP and deep learning concepts to classify emotions in English tweets using the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

## 📊 Dataset
- Short English tweets labeled with **6 emotion classes**:
  - `0`: sadness
  - `1`: joy
  - `2`: love
  - `3`: anger
  - `4`: fear
  - `5`: surprise
- Predefined train, validation, and test splits.
- Text length ranges:  
  - **Train:** min=2, max=66, mean=19.17  
  - **Validation:** mean=18.87  
  - **Test:** mean=19.15  
- Vocabulary size: **15,214**
- **Baseline (majority class) accuracy**: 0.34

## 🛠️ Models

### 1. Shallow MLP (Feedforward)
- **Architecture:** Embedding → Avg/Max Pooling → Dropout → Linear
- Trained on embedded word vectors with basic aggregation.

### 2. Bag-of-Words + MLP
- **Vectorization:** Sparse BoW (max 5000 features)
- **Architecture:** BoW → 2-layer MLP with ReLU + Dropout

### 3. Transformer-Based Classifier
- Trained on tokenized tweets using a Transformer encoder.
- Leverages contextualized embeddings for higher accuracy.

## 🧠 Interpretability
To understand and explain model predictions, we used:
- **LIME**
- **Integrated Gradients**
- **Interactive Visualization Tools**

## ✅ Project Highlights
- Custom data pipeline for text processing
- Comparison of shallow vs. deep models
- Insightful analysis of model behavior using interpretability tools

