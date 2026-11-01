# Cat vs Dog Image Feature Extraction Dataset

This repository contains an ARFF (Attribute-Relation File Format) dataset and feature extraction scripts used for image classification and machine learning coursework. The features are extracted from cat and dog images using texture, color, and shape analysis techniques.

## 📊 Dataset Features
The dataset includes the following numerical attributes extracted per image:
* **Texture Features (GLCM):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
* **Color Statistics:** Mean Blue, Mean Green, Mean Red channels.
* **Shape Features:** Hu Moments (Hu 1 through Hu 7).
* **Class Label:** `kedi` (cat) or `kopek` (dog).

## 🛠️ Feature Extraction Script
The extraction script is available in `extract_features.py`.

### Requirements
```bash
pip install opencv-python scikit-image numpy
```

### Usage
```bash
python extract_features.py
```
Outputs an ARFF file (`cat_dog_features.arff`) compatible with Weka and scikit-learn.
