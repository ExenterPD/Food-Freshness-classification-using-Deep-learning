## Research Paper – Detailed Project Summary

### 1. Problem Statement
Food spoilage leads to significant economic loss and health risks. Manual inspection of food freshness is subjective, inconsistent, and time-consuming. This research proposes an automated, vision-based food freshness classification system that combines deep learning feature extraction with classical machine learning classifiers and explainable AI techniques.

The objective is to accurately classify food images into **Fresh (Good)** and **Spoiled (Bad)** categories while ensuring **model interpretability** using SHAP-based explanations.

---

### 2. Proposed System Overview
The project follows a **hybrid learning pipeline** that integrates convolutional–transformer-based deep models with traditional classifiers:

**Execution Flow**  
`model_code.py → Hybrid_code.py → hybrid_model_1.py / hybrid_model_2.py → Hybrid_shap_1_i.py / Hybrid_shap_1_ii.py`

The system consists of four major stages:
1. Feature learning using pretrained deep models
2. Hybrid model construction and training
3. Performance evaluation
4. Explainability using SHAP

---

### 3. Dataset Description
- Image-based food freshness dataset
- Binary labels:
  - **Good (1)** – Fresh food
  - **Bad (0)** – Spoiled food
- Multiple food categories (meat, dairy, grains, fish, baked goods, pickles, etc.)
- Stratified train–test splitting ensures class balance

---

### 4. Feature Extraction and Base Model (model_code.py)
- Uses a **pretrained MobileViT architecture** as the backbone
- Combines CNN-based local feature extraction with transformer-based global representation
- Final fully connected layer removed to extract **high-dimensional feature embeddings**
- Image preprocessing includes resizing, normalization, and tensor conversion

This stage converts raw images into robust numerical feature vectors suitable for downstream machine learning models.

---

### 5. Hybrid Learning Pipeline (Hybrid_code.py)
- Acts as the **orchestration layer** of the system
- Handles:
  - Feature loading
  - Dataset preparation
  - Model selection
  - Training and evaluation control
- Enables seamless experimentation with different hybrid architectures

---

### 6. Hybrid Model Architectures

#### 6.1 Hybrid Model 1 (hybrid_model_1.py)
- Deep feature vectors extracted using MobileViT
- Features fed into classical ML classifiers such as:
  - XGBoost
  - LightGBM
  - CatBoost
  - MLP-based classifiers
- Optimized using hyperparameter tuning

**Advantages:**
- Faster convergence
- Reduced overfitting
- Strong performance on limited datasets

#### 6.2 Hybrid Model 2 (hybrid_model_2.py)
- Enhanced hybrid architecture with additional regularization and feature refinement
- Supports comparative evaluation across multiple classifiers
- Improves generalization by leveraging ensemble-like behavior

---

### 7. Evaluation Metrics
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics ensure balanced performance assessment, especially for real-world food safety applications.

---

### 8. Explainable AI with SHAP

#### 8.1 SHAP Analysis – Part I (Hybrid_shap_1_i.py)
- Applies SHAP (SHapley Additive exPlanations) on hybrid models
- Computes feature-level contribution scores
- Generates global explanations highlighting dominant learned features

#### 8.2 SHAP Analysis – Part II (Hybrid_shap_1_ii.py)
- Focuses on **local explanations**
- Explains individual predictions
- Visualizes how specific features influence classification as Fresh or Spoiled

**Impact:**
- Enhances transparency
- Builds trust in AI-driven food quality assessment
- Suitable for real-world deployment

---

### 9. Key Contributions
- Novel **MobileViT + ML hybrid architecture** for food freshness classification
- Efficient feature reuse instead of full end-to-end retraining
- SHAP-based explainability for vision-derived features
- Scalable and modular design

---

### 10. Conclusion
The proposed hybrid framework demonstrates strong classification performance while maintaining interpretability. By combining deep vision models with classical machine learning and explainable AI, the system provides a practical and trustworthy solution for automated food freshness assessment.

Future extensions include real-time deployment, multi-class freshness grading, and integration with IoT-based food monitoring systems.

---

---

## Updated README.md (Project Ready)

# Hybrid Food Freshness Classification Using Explainable AI

## Overview
This project presents a **hybrid deep learning framework** for food freshness classification using image data. The system combines **MobileViT-based feature extraction**, **classical machine learning classifiers**, and **SHAP-based explainability** to deliver accurate and interpretable predictions.

The model classifies food items into:
- **Fresh (Good)**
- **Spoiled (Bad)**

---

## Project Pipeline
```
model_code.py
   ↓
Hybrid_code.py
   ↓
hybrid_model_1.py / hybrid_model_2.py
   ↓
Hybrid_shap_1_i.py / Hybrid_shap_1_ii.py
```

---

## Key Features
- MobileViT-based deep feature extraction
- Hybrid ML classifiers (XGBoost, LightGBM, CatBoost, MLP)
- High accuracy with reduced training cost
- SHAP-based global and local explainability
- Modular and research-oriented code structure

---

## Dataset Structure
```
dataset/
 ├── Category/
 │   ├── Food Item/
 │   │   ├── Good/
 │   │   └── Bad/
```

**Label Mapping**:
- Good → 1
- Bad → 0

---

## Installation
```bash
git clone <repository-url>
cd Hybrid-Food-Freshness
pip install -r requirements.txt
```

---

## Usage

### Step 1: Feature Extraction
```bash
python model_code.py
```

### Step 2: Hybrid Model Training
```bash
python Hybrid_code.py
```

### Step 3: Train Hybrid Models
```bash
python hybrid_model_1.py
python hybrid_model_2.py
```

### Step 4: Explainability Analysis
```bash
python Hybrid_shap_1_i.py
python Hybrid_shap_1_ii.py
```

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Explainable AI
SHAP is used to:
- Identify important deep features
- Explain individual predictions
- Improve transparency and trust

---

## Applications
- Smart food quality inspection
- Retail and supply-chain monitoring
- Food safety automation

---

## Future Work
- Real-time mobile deployment
- Multi-class freshness grading
- Edge AI and IoT integration

---

## License
MIT License

---

## Citation
If you use this work in academic research, please cite appropriately.

---

**Author**: Tanay Patel

