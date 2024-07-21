# Skin-Cancer-Classification
# Introduction
Skin cancer is one of the most common forms of cancer, and early detection is crucial for effective treatment. This project aims to develop a deep learning model to classify skin cancer from images, helping in the early diagnosis and management of the disease. The model is capable of identifying multiple types of skin lesions, including melanoma and other benign and malignant types.

# Model Overview
The skin cancer classification model is built using Convolutional Neural Networks (CNNs) and trained on a dataset of skin lesion images. The model is designed to classify the following types of skin lesions:

BCC (Basal Cell Carcinoma)
MEL (Melanoma)
NV (Nevus)
VASC (Vascular lesions)
AKIEC (Actinic Keratoses and Intraepithelial Carcinoma)
BKL (Benign Keratosis-like lesions)
DF (Dermatofibroma)
SCC (Squamous Cell Carcinoma)

# Project Structure
app.py: The main application file to run the Streamlit web app.
model: Directory containing the pre-trained model file (Model2_EffB4_No_meta.hdf5).

# Requirements
Python 3.6 or higher
TensorFlow
Streamlit
Pillow
NumPy
