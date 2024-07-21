
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('Model2.hdf5')

# Class labels
class_labels = ['Nv', 'Mel', 'bcc', 'bkl', 'akiec', 'scc', 'vasc', 'df']

# Mapping of predicted classes to doctors
doctor_recommendations = {
    'Bcc': {
        'name': 'Dr. John Smith',
        'specialization': 'Dermatologist',
        'contact': '123-456-7890',
        'email': 'johnsmith@example.com',
        'hospital': 'Skin Care Clinic',
        'address': '123 Main St, Anytown, USA'
    },
    'Mel': {
        'name': 'Dr. Jane Doe',
        'specialization': 'Oncologist',
        'contact': '987-654-3210',
        'email': 'janedoe@example.com',
        'hospital': 'Cancer Treatment Center',
        'address': '456 Elm St, Othertown, USA'
    },
    # Add other class to doctor mappings here
    'scc': {
        'name': 'Dr. Emily Brown',
        'specialization': 'Dermatologist',
        'contact': '555-123-4567',
        'email': 'emilybrown@example.com',
        'hospital': 'City Hospital',
        'address': '789 Oak St, Somewhere, USA'
    },
    'Nv': {
        'name': 'Dr. Alice Green',
        'specialization': 'General Practitioner',
        'contact': '555-987-6543',
        'email': 'alicegreen@example.com',
        'hospital': 'Downtown Clinic',
        'address': '101 Pine St, Anycity, USA'
    },
    'vasc': {
        'name': 'Dr. Robert Black',
        'specialization': 'Vascular Surgeon',
        'contact': '555-234-5678',
        'email': 'robertblack@example.com',
        'hospital': 'Heart and Vascular Institute',
        'address': '202 Birch St, Sometown, USA'
    },
    'akiec': {
        'name': 'Dr. William White',
        'specialization': 'Dermatopathologist',
        'contact': '555-345-6789',
        'email': 'williamwhite@example.com',
        'hospital': 'Skin Disease Center',
        'address': '303 Cedar St, Anothertown, USA'
    },
    'bkl': {
        'name': 'Dr. Nancy Blue',
        'specialization': 'Pediatric Dermatologist',
        'contact': '555-456-7890',
        'email': 'nancyblue@example.com',
        'hospital': 'Children\'s Hospital',
        'address': '404 Maple St, Elsewhere, USA'
    },
    'df': {
        'name': 'Dr. Kevin Red',
        'specialization': 'Dermatologic Surgeon',
        'contact': '555-567-8901',
        'email': 'kevinred@example.com',
        'hospital': 'Advanced Dermatology Center',
        'address': '505 Walnut St, Thistown, USA'
    }
}

def prepare_image(image):
    image = image.resize((380, 380)) 
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Inject custom CSS
def inject_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #f8f9fa;  /* Light grey background color */
            color: #333333;
        }
        .stApp {
            background: #ffffff;  /* White background for the form */
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .stTextInput > div > div > input, .stSelectbox > div > div > div > input {
            border: 1px solid #d4d4d4;
            border-radius: 5px;
            padding: 10px;
            width: 100%;
        }
        .stButton > button {
            background-color: #ff6347;  /* Tomato background color for buttons */
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #ff4500;  /* Darker tomato on hover */
        }
        .uploadedImage {
            border: 1px solid #d4d4d4;
            border-radius: 10px;
            padding: 10px;
            max-width: 100%;
            margin: 10px 0;
        }
        .form-container {
            background: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Skin Cancer Classification", layout="wide")
    st.title("SKIN CANCER CLASSIFICATION")
    
    inject_custom_css()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Fill Patient Detail")
        with st.form(key='patient_form', clear_on_submit=True):
            patient_id = st.text_input("Patient ID")
            name = st.text_input("Name")
            age = st.text_input("Age")
            gender = st.selectbox("Gender", ["Select gender", "Male", "Female", "Other"])
            anatomical_site = st.selectbox("Anatomical Site", ["Select diagnosis location", "Head/Neck", "Upper Extremities", "Lower Extremities", "Trunk", "Other"])
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            # Create columns for the buttons
            button_col1, button_col2 = st.columns([1, 1])
            with button_col1:
                submit_button = st.form_submit_button(label='UPLOAD!')
            with button_col2:
                reset_button = st.form_submit_button(label='RESET')

    if reset_button:
        st.experimental_rerun()

    if submit_button:
        if not uploaded_file or gender == "Select gender" or anatomical_site == "Select diagnosis location" or not patient_id or not name or not age:
            st.error("Please fill all the details and upload an image.")
        else:
            try:
                image = Image.open(uploaded_file)
                with col2:
                    st.image(image, caption='Uploaded Image.', use_column_width=True, output_format='JPEG')
                    st.write("Classifying...")

                    with st.spinner('Wait for it...'):
                        prepared_image = prepare_image(image)
                        prediction = model.predict(prepared_image)
                        predicted_class = class_labels[np.argmax(prediction)]

                    st.success("The predicted class is: {}".format(predicted_class))
                    

                    # Display the patient details and predicted class
                    st.header("Patient Detail")
                    st.write(f"**Patient ID:** {patient_id}")
                    st.write(f"**Name:** {name}")
                    st.write(f"**Age:** {age}")
                    st.write(f"**Gender:** {gender}")
                    st.write(f"**Anatomical Site:** {anatomical_site}")
                    st.write(f"**Prediction:** {predicted_class}")

                    # Recommend doctor based on predicted class
                    if predicted_class in doctor_recommendations:
                        st.header("Recommended Doctor")
                        doctor = doctor_recommendations[predicted_class]
                        st.write(f"**Doctor Name:** {doctor['name']}")
                        st.write(f"**Specialization:** {doctor['specialization']}")
                        st.write(f"**Contact Number:** {doctor['contact']}")
                        st.write(f"**Email:** {doctor['email']}")
                        st.write(f"**Hospital Name:** {doctor['hospital']}")
                        st.write(f"**Address:** {doctor['address']}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
