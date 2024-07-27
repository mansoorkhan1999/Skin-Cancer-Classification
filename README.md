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
# Usage
Open the application in your web browser. By default, it runs on http://localhost:8501.
Fill in the patient details, including Patient ID, Name, Age, Gender, and Anatomical Site.
Upload an image of the skin lesion.
Click the UPLOAD! button to classify the skin lesion.
The model will predict the type of skin lesion and recommend a doctor based on the classification result.
# Our Web Application
Here are some snapshots of the web application:
<img width="547" alt="image" src="https://github.com/user-attachments/assets/29620114-e2fd-4364-a5a4-97d12dd2e99d">
<img width="464" alt="image" src="https://github.com/user-attachments/assets/5cc7426d-7363-4287-8add-c9dbf0fff4ad">
<img width="487" alt="image" src="https://github.com/user-attachments/assets/2a583522-a25b-4f58-b0fc-179f8e262999">

