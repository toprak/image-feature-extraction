"""
Image Feature Extraction Script for Cat vs Dog Dataset
Extracts GLCM texture features, Color statistics, and Hu Moments,
then exports the extracted features into ARFF format.
"""

import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_image_features(image_path, label):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. Color Statistics (RGB channels)
    b, g, r = cv2.split(img)
    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    # Convert to grayscale for texture & shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Texture Features (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    # 3. Shape Features (Hu Moments)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Log scale transform for Hu moments
    hu_transformed = [-np.sign(h) * np.log10(abs(h)) if h != 0 else 0 for h in hu_moments]
    
    features = [
        contrast, dissimilarity, homogeneity, energy, correlation, asm,
        mean_b, mean_g, mean_r
    ] + hu_transformed + [label]
    
    return features

def export_to_arff(dataset, output_file):
    header = """@relation cat_dog_features

@attribute contrast numeric
@attribute dissimilarity numeric
@attribute homogeneity numeric
@attribute energy numeric
@attribute correlation numeric
@attribute asm numeric
@attribute mean_blue numeric
@attribute mean_green numeric
@attribute mean_red numeric
@attribute hu1 numeric
@attribute hu2 numeric
@attribute hu3 numeric
@attribute hu4 numeric
@attribute hu5 numeric
@attribute hu6 numeric
@attribute hu7 numeric
@attribute class {kedi, kopek}

@data
"""
    with open(output_file, 'w') as f:
        f.write(header)
        for row in dataset:
            f.write(','.join(str(val) for val in row) + '\n')
    print(f"Dataset successfully exported to {output_file}")

if __name__ == '__main__':
    print("Feature extraction utility ready. Point to your image dataset directory to process.")
