import streamlit as st
from PIL import Image
import gdown
import tensorflow as tf
import numpy as np
import os

# Custom metric functions
def dice_coefficient(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + tf.keras.backend.epsilon())

def jaccard_index(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f + y_pred_f) - intersection
    return (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

# Custom weighted binary crossentropy loss function
def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * (weights[1] * y_true + weights[0] * (1 - y_true))
        return tf.keras.backend.mean(weighted_bce)
    return loss

# Define the weights for the custom loss function
weighted_loss = weighted_binary_crossentropy([0.5355597809300489, 7.530414514976497])

# Google Drive link to the model
gdrive_url = 'https://drive.google.com/uc?export=download&id=1MB7DOQq6--oIYF6TWdn7kisjXWnPI1E4'
model_file = '/mount/src/building_footprint_extraction/unet_vgg_14.keras'

@st.cache(allow_output_mutation=True)
def load_model():
    # Download the model from Google Drive
    gdown.download(gdrive_url, model_file, quiet=False)

    # Check if the model file was downloaded
    if os.path.exists(model_file):
        st.write(f"Model configured")
    else:
        st.write(f"Model file not found at {model_file}.")
        return None

    # Load the Keras model with custom metrics and loss function
    model = tf.keras.models.load_model(model_file, custom_objects={
        'dice_coefficient': dice_coefficient, 
        'jaccard_index': jaccard_index, 
        'precision': precision, 
        'specificity': specificity, 
        'sensitivity': sensitivity, 
        'loss': weighted_loss
    })
    return model

# Load the model
model = load_model()

# Title and description
st.title("Building Footprint Extractor")
st.write("Upload Sentinel 2 satellite imagery to automatically extract building footprints through semantic segmentation.")

# Section 1: Imagery Upload
st.subheader("Imagery")
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "png", "tiff"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"{len(uploaded_files)} image(s) uploaded successfully!")
    
    # Section 2: Settings
    st.subheader("Settings")
    
    # Thresholding toggle
    apply_threshold = st.checkbox("Apply Thresholding?")
    
    # Show threshold slider if thresholding is selected
    if apply_threshold:
        threshold_value = st.slider("Select Threshold Value:", 0.0, 0.99, 0.5)
        st.write(f"Thresholding Enabled. Value: {threshold_value}")
    else:
        st.write("Thresholding Disabled.")
    
    # Overlay option
    overlay_option = st.checkbox("Create Additional Overlay Imagery?")

    # Process button
    if st.button("Process Imagery"):
        st.subheader("Results")
        st.write("Processing...")

        # Loop through each uploaded file
        for uploaded_file in uploaded_files:
            # Read the uploaded file as an image
            image = Image.open(uploaded_file)

            # Convert image to numpy array for model input
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Normalize image
            image_array = image_array / 255.0

            # Get model prediction
            prediction = model.predict(image_array)[0]

            # Apply threshold if selected
            if apply_threshold:
                prediction = (prediction > threshold_value).astype(np.uint8)

            # Convert prediction to image format (PIL)
            prediction_image = Image.fromarray((prediction * 255).astype(np.uint8))

            # Display results side by side (original, prediction, and overlay if selected)
            if overlay_option:
                col1, col2, col3 = st.columns(3)
            else:
                col1, col2 = st.columns(2)

            # Display original image
            with col1:
                st.image(image, caption="Original", use_column_width=True)

            # Display prediction image
            with col2:
                st.image(prediction_image, caption="Prediction", use_column_width=True)

            # Display overlay if option is selected
            if overlay_option:
                # Create overlay
                overlay_image = Image.blend(image.convert("RGBA"), prediction_image.convert("RGBA"), alpha=0.5)
                with col3:
                    st.image(overlay_image, caption="Prediction Overlay", use_column_width=True)

        # Download options
        st.subheader("Download Options")
        download_options = st.multiselect(
            "Select what you would like to download:",
            ["Masks (JPG)", "Overlays (JPG)" if overlay_option else None]  # Only show overlay option if selected
        )

        if st.button("Download"):
            st.write(f"Preparing {', '.join(download_options)} for download... (Functionality Coming Soon)")
