# Cat vs Dog Image Feature Extraction Dataset

This repository contains an ARFF (Attribute-Relation File Format) dataset[cite: 1] used for image classification and machine learning coursework. The features were extracted from cat and dog images using texture and shape analysis techniques[cite: 1].

## Dataset Features
The dataset includes the following numerical attributes extracted per image:
* **Texture Features (GLCM):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM.
* **Color Statistics:** Mean Blue, Mean Green, Mean Red channels.
* **Shape Features:** Hu Moments (1 through 7).
* **Class Label:** `kedi` (cat) or `kopek` (dog).
