# Cat vs Dog Image Feature Extraction Dataset

This repository contains an ARFF (Attribute-Relation File Format) dataset[cite: 1] used for image classification and machine learning coursework. The features were extracted from cat and dog images using texture and shape analysis techniques[cite: 1].

## Dataset Features
The dataset includes the following numerical attributes extracted per image[cite: 1]:
* **Texture Features (GLCM):** Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM[cite: 1].
* **Color Statistics:** Mean Blue, Mean Green, Mean Red channels[cite: 1].
* **Shape Features:** Hu Moments (1 through 7)[cite: 1].
* **Class Label:** `kedi` (cat) or `kopek` (dog)[cite: 1].