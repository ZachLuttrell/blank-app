import streamlit as st
from PIL import Image
import gdown
import tensorflow as tf
import numpy as np
import os

# Processing Functions
# Function to break an image into patches
def image_to_patches(image, patch_size=256, overlap=32):
    patches = []
    step = patch_size - overlap
    x_max = image.shape[0] - patch_size
    y_max = image.shape[1] - patch_size

    for x in range(0, x_max + 1, step):
        for y in range(0, y_max + 1, step):
            patch = image[x:x + patch_size, y:y + patch_size]
            patches.append(patch)

    # Edge patches (right and bottom)
    if x_max % step != 0:
        for y in range(0, y_max + 1, step):
            patch = image[x_max:x_max + patch_size, y:y + patch_size]
            patches.append(patch)
    if y_max % step != 0:
        for x in range(0, x_max + 1, step):
            patch = image[x:x + patch_size, y_max:y_max + patch_size]
            patches.append(patch)
    if x_max % step != 0 or y_max % step != 0:
        patch = image[x_max:x_max + patch_size, y_max:y_max + patch_size]
        patches.append(patch)

    return patches

# Function to predict on patches
def predict_patches(model, patches):
    predictions = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, verbose=0)
        predictions.append(prediction.squeeze())  # Remove batch dimension
    return predictions

# Function to reassemble patches into the full image
def reassemble_patches(patches, original_shape, patch_size=256, overlap=32):
    step = patch_size - overlap
    image_height, image_width = original_shape[:2]
    reassembled_image = np.zeros(original_shape[:2])

    # Create an accumulator image to count the number of predictions per pixel
    count = np.zeros(original_shape[:2])

    patch_idx = 0
    for x in range(0, image_height - patch_size + 1, step):
        for y in range(0, image_width - patch_size + 1, step):
            reassembled_image[x:x + patch_size, y:y + patch_size] += patches[patch_idx]
            count[x:x + patch_size, y:y + patch_size] += 1
            patch_idx += 1

    # Normalize by the number of patches overlapping at each pixel
    count[count == 0] = 1  # Avoid division by zero
    reassembled_image /= count

    return reassembled_image

# CUSTOM METRICS AND LOSS FUNCTIONS
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
        st.write(f"Model found.")
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

# Function to process each image
def process_image(image):
    image_array = np.array(image) / 255.0  # Normalize the image
    patches = image_to_patches(image_array)  # Convert image into patches

    # Predict on the patches
    predictions = predict_patches(model, patches)

    # Reassemble the patches into a full prediction mask
    full_mask = reassemble_patches(predictions, image_array.shape)
    
    return full_mask

if uploaded_files:
    st.write(f"{len(uploaded_files)} image(s) uploaded successfully!")

    # Section 2: Settings
    st.subheader("Settings")
    
    apply_threshold = st.checkbox("Apply Thresholding?")
    threshold_value = st.slider("Select Threshold Value:", 0.0, 0.99, 0.5) if apply_threshold else None
    overlay_option = st.checkbox("Create Additional Overlay Imagery?")
    
    if st.button("Process Imagery"):
        st.subheader("Results")

        progress_bar = st.progress(0)
        total_images = len(uploaded_files)

        # Loop through each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)

            # Process the image and get the prediction
            st.write(f"Processing {uploaded_file.name}...")
            prediction_mask = process_image(image)

            # Apply threshold if selected
            if apply_threshold:
                prediction_mask = (prediction_mask > threshold_value).astype(np.uint8)

            # Convert prediction mask to image format (PIL)
            prediction_image = Image.fromarray((prediction_mask * 255).astype(np.uint8))

            # Display results (original, prediction, and overlay if selected)
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

            # Display overlay if selected
            if overlay_option:
                overlay_image = Image.blend(image.convert("RGBA"), prediction_image.convert("RGBA"), alpha=0.5)
                with col3:
                    st.image(overlay_image, caption="Prediction Overlay", use_column_width=True)

            # Update progress bar
            progress_bar.progress((idx + 1) / total_images)

        # Download options
        st.subheader("Download Options")
        download_options = st.multiselect(
            "Select what you would like to download:",
            ["Masks (JPG)", "Overlays (JPG)" if overlay_option else None]
        )

        if st.button("Download"):
            st.write(f"Preparing {', '.join(download_options)} for download... (Functionality Coming Soon)")
