import streamlit as st
from PIL import Image
import gdown
import tensorflow as tf
import numpy as np
import os
from io import BytesIO

# Processing Functions
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
            patch = image[x:x + patch_size, y:y + patch_size]
            patches.append(patch)
    if x_max % step != 0 or y_max % step != 0:
        patch = image[x_max:x_max + patch_size, y_max:y_max + patch_size]
        patches.append(patch)

    return patches

def predict_patches(model, patches):
    predictions = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, verbose=0)
        predictions.append(prediction.squeeze())  # Remove batch dimension
    return predictions

def reassemble_patches(patches, original_shape, patch_size=256, overlap=32):
    step = patch_size - overlap
    image_height, image_width = original_shape[:2]
    reassembled_image = np.zeros(original_shape[:2])
    count = np.zeros(original_shape[:2])

    patch_idx = 0
    for x in range(0, image_height - patch_size + 1, step):
        for y in range(0, image_width - patch_size + 1, step):
            reassembled_image[x:x + patch_size, y:y + patch_size] += patches[patch_idx]
            count[x:x + patch_size, y:y + patch_size] += 1
            patch_idx += 1

    count[count == 0] = 1  # Avoid division by zero
    reassembled_image /= count
    return reassembled_image

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")  # Save as JPEG or another format
    img_str = buffered.getvalue()
    st.download_button(
        label=text,
        data=img_str,
        file_name=filename,
        mime="image/jpeg"
    )

def create_download_options(prediction_image, overlay_image, image_filename, overlay_option):
    st.subheader("Download Options")
    download_options = st.multiselect(
        "Select what you would like to download:",
        ["Masks (JPG)", "Overlays (JPG)" if overlay_option else None]
    )
    
    if "Masks (JPG)" in download_options:
        get_image_download_link(prediction_image, f"prediction_{image_filename}.jpg", "Download Prediction Mask")

    if "Overlays (JPG)" in download_options and overlay_option:
        get_image_download_link(overlay_image, f"overlay_{image_filename}.jpg", "Download Overlay Image")

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

def weighted_binary_crossentropy(weights):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weighted_bce = bce * (weights[1] * y_true + weights[0] * (1 - y_true))
        return tf.keras.backend.mean(weighted_bce)
    return loss

weighted_loss = weighted_binary_crossentropy([0.5355597809300489, 7.530414514976497])

gdrive_url = 'https://drive.google.com/uc?export=download&id=1MB7DOQq6--oIYF6TWdn7kisjXWnPI1E4'
model_file = '/mount/src/building_footprint_extraction/unet_vgg_14.keras'

@st.cache(allow_output_mutation=True)
def load_model():
    gdown.download(gdrive_url, model_file, quiet=False)
    model = tf.keras.models.load_model(model_file, custom_objects={
        'dice_coefficient': dice_coefficient, 
        'jaccard_index': jaccard_index, 
        'precision': precision, 
        'specificity': specificity, 
        'sensitivity': sensitivity, 
        'loss': weighted_loss
    })
    return model

model = load_model()

# Title and description
st.title("Building Footprint Extractor")
st.write("Upload Sentinel 2 imagery to extract building footprints.")

st.markdown("### Step 1: Upload Your Imagery")
uploaded_files = st.file_uploader("Choose image(s)", type=["jpg", "png", "tiff"], accept_multiple_files=True)

# Variables to store raw predictions and images for post-processing
raw_predictions = []
original_images = []
image_filenames = []

if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded successfully!**")

    # Process Imagery Button
    if st.button("Process Imagery"):
        st.markdown("### Step 2: Initial Results")
        progress_bar = st.progress(0)
        total_images = len(uploaded_files)

        # Process each image with no thresholding or overlays initially
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            image_filename = os.path.splitext(uploaded_file.name)[0]
            image_array = np.array(image) / 255.0
            if image_array.shape[-1] == 4:
                image_array = image_array[..., :3]

            # Run predictions without threshold or overlays
            patches = image_to_patches(image_array)
            predictions = predict_patches(model, patches)
            full_mask = reassemble_patches(predictions, image_array.shape)

            raw_predictions.append(full_mask)  # Store raw predictions
            original_images.append(image)
            image_filenames.append(image_filename)

            # Display raw prediction as is (no threshold applied)
            prediction_image = Image.fromarray((full_mask * 255).astype(np.uint8))

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(prediction_image, caption="Raw Prediction", use_column_width=True)

            progress_bar.progress((idx + 1) / total_images)

        # After showing initial results, show the post-processing expander
        st.divider()
        st.markdown("### Step 3: Refine Results")
        with st.expander("Adjust Post-Processing Settings"):
            apply_threshold = st.checkbox("Apply Thresholding?")
            threshold_value = st.slider("Select Threshold Value:", 0.0, 0.99, 0.5) if apply_threshold else None
            overlay_option = st.checkbox("Create Overlay Imagery?")

            if st.button("Apply Refinements"):
                st.subheader("Refined Results")
                processed_images = []
                processed_overlays = []

                for idx, (full_mask, image, image_filename) in enumerate(zip(raw_predictions, original_images, image_filenames)):
                    # Apply threshold if selected
                    if apply_threshold:
                        mask = (full_mask > threshold_value).astype(np.uint8)
                    else:
                        mask = (full_mask * 255).astype(np.uint8) / 255.0
                        mask = (mask > 0.5).astype(np.uint8)  # simple binarization if needed
                        
                    prediction_image = Image.fromarray((mask * 255).astype(np.uint8))
                    if overlay_option:
                        overlay_image = Image.blend(image.convert("RGBA"), prediction_image.convert("RGBA"), alpha=0.5)
                    else:
                        overlay_image = None

                    col1, col2 = st.columns(2) if not overlay_option else st.columns(3)
                    with col1:
                        st.image(image, caption=f"{image_filename} - Original", use_column_width=True)
                    with col2:
                        st.image(prediction_image, caption="Refined Mask", use_column_width=True)
                    if overlay_option and overlay_image:
                        with col2 if not overlay_option else col3:  # Adjust if you have a third column
                            st.image(overlay_image, caption="Refined Overlay", use_column_width=True)

                    processed_images.append((prediction_image, f"prediction_{image_filename}.jpg"))
                    if overlay_option and overlay_image:
                        processed_overlays.append((overlay_image, f"overlay_{image_filename}.jpg"))

                # Download options after refinements
                st.divider()
                st.subheader("Download Options")
                download_options = st.multiselect(
                    "Select what you would like to download:",
                    ["Masks (JPG)", "Overlays (JPG)" if overlay_option else None]
                )

                if st.button("Download"):
                    if "Masks (JPG)" in download_options:
                        for image, filename in processed_images:
                            get_image_download_link(image, filename, f"Download {filename}")

                    if "Overlays (JPG)" in download_options and overlay_option:
                        for overlay, filename in processed_overlays:
                            get_image_download_link(overlay, filename, f"Download {filename}")
