import streamlit as st
from PIL import Image
import gdown
import tensorflow as tf
import numpy as np
import os
from io import BytesIO
import zipfile

# ------------------------------------------------------
# 1. Global Page Configuration and Custom CSS
# ------------------------------------------------------
st.set_page_config(
    page_title="Building Footprint Extractor",
    page_icon="🏗️",
    layout="wide"
)

# Inject custom CSS for a polished look.
st.markdown(
    """
    <style>
    /* Custom background for the main area and sidebar */
    .main {
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    /* Header styles */
    h1, h2, h3, h4, h5 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------
# 2. Utility Functions
# ------------------------------------------------------
def image_to_patches(image, patch_size=256, overlap=32):
    """
    Splits the input image into overlapping patches, ensuring full coverage.
    Returns:
      patches: A list of image patches.
      positions: A list of (x, y) coordinates corresponding to the patch's top-left corner.
    """
    patches = []
    positions = []
    step = patch_size - overlap
    height, width = image.shape[:2]

    # Compute the starting x positions.
    x_positions = list(range(0, height - patch_size + 1, step))
    if not x_positions or x_positions[-1] != height - patch_size:
        x_positions.append(height - patch_size)

    # Compute the starting y positions.
    y_positions = list(range(0, width - patch_size + 1, step))
    if not y_positions or y_positions[-1] != width - patch_size:
        y_positions.append(width - patch_size)

    # Extract patches for each combination of x and y positions.
    for x in x_positions:
        for y in y_positions:
            patch = image[x : x + patch_size, y : y + patch_size]
            patches.append(patch)
            positions.append((x, y))
    return patches, positions

def predict_patches(model, patches):
    predictions = []
    for patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, verbose=0)
        predictions.append(prediction.squeeze())  # Remove batch dimension
    return predictions

def reassemble_patches(patches, positions, original_shape, patch_size=256):
    """
    Reassembles a full image from its patches.
    - patches: List of patch predictions.
    - positions: List of (x, y) coordinates for each patch.
    - original_shape: The shape of the original image.
    - patch_size: The size of each patch.
    
    Returns:
      A full prediction map with overlapping regions averaged.
    """
    height, width = original_shape[:2]
    output = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for patch, (x, y) in zip(patches, positions):
        output[x : x + patch_size, y : y + patch_size] += patch
        count[x : x + patch_size, y : y + patch_size] += 1

    count[count == 0] = 1
    output /= count
    return output

def create_zip_download(images_and_filenames, zip_filename="download.zip"):
    """
    Packages a list of (PIL Image, filename) tuples into a ZIP archive and creates
    a download button for it.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img, fname in images_and_filenames:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            zf.writestr(fname, buffered.getvalue())
    zip_buffer.seek(0)
    st.download_button(
        label="Download ZIP",
        data=zip_buffer,
        file_name=zip_filename,
        mime="application/zip"
    )

# ------------------------------------------------------
# 3. Custom Metrics and Loss Functions 
# ------------------------------------------------------
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

# ------------------------------------------------------
# 4. Model Loading (Using st.cache_resource)
# ------------------------------------------------------
gdrive_url = 'https://drive.google.com/uc?export=download&id=1MB7DOQq6--oIYF6TWdn7kisjXWnPI1E4'
model_file = '/mount/src/building_footprint_extraction/unet_vgg_14.keras'

@st.cache_resource
def load_model():
    gdown.download(gdrive_url, model_file, quiet=False)
    model = tf.keras.models.load_model(
        model_file,
        custom_objects={
            'dice_coefficient': dice_coefficient, 
            'jaccard_index': jaccard_index, 
            'precision': precision, 
            'specificity': specificity, 
            'sensitivity': sensitivity, 
            'loss': weighted_loss
        }
    )
    return model

model = load_model()

# ------------------------------------------------------
# 5. Session State Setup
# ------------------------------------------------------
if 'raw_predictions' not in st.session_state:
    st.session_state.raw_predictions = []
if 'original_images' not in st.session_state:
    st.session_state.original_images = []
if 'image_filenames' not in st.session_state:
    st.session_state.image_filenames = []

# ------------------------------------------------------
# 6. Sidebar Configuration
# ------------------------------------------------------
st.sidebar.header("Configuration Options")

# File uploader in the sidebar
uploaded_files = st.sidebar.file_uploader(
    "Choose image(s)", 
    type=["jpg", "png", "tiff"], 
    accept_multiple_files=True
)

# Post-processing settings moved to the sidebar
threshold_value = st.sidebar.slider("Select Threshold Value:", 0.0, 0.99, 0.5)
overlay_option = st.sidebar.checkbox("Create Overlay Imagery?")

# ------------------------------------------------------
# 7. App Title and Description (Main Area)
# ------------------------------------------------------
st.title("Building Footprint Extractor")
st.write("Upload Sentinel-2 imagery to extract building footprints.")

st.markdown(
    """
    **Instructions:**
    
    1. **Upload Imagery:** Use the file uploader in the sidebar to choose your image(s).
    2. **Run Processing:** Click the **Run Segmentation** button to process the uploaded images.
    3. **Adjust Results:** Use the sidebar slider to select a threshold value and the checkbox to enable overlay imagery.
    4. **Download Results:** Once processing is complete, select the desired outputs and click the **Download Selected Files** button to download your results as a ZIP archive.
    """
)

# ------------------------------------------------------
# 8. Step 1: Run Segmentation
# ------------------------------------------------------
if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded successfully!**")
    
    # Using a spinner to indicate a long-running process.
    if st.button("Run Segmentation"):
        with st.spinner("Running segmentation..."):
            # Clear previous session state predictions (if any)
            st.session_state.raw_predictions = []
            st.session_state.original_images = []
            st.session_state.image_filenames = []
            
            progress_bar = st.progress(0)
            total_images = len(uploaded_files)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file).convert("RGB")
                image_filename = os.path.splitext(uploaded_file.name)[0]
                image_array = np.array(image) / 255.0
                if image_array.shape[-1] == 4:
                    image_array = image_array[..., :3]
                    
                patches, positions = image_to_patches(image_array, patch_size=256, overlap=32)
                predictions = predict_patches(model, patches)
                full_mask = reassemble_patches(predictions, positions, image_array.shape, patch_size=256)
                
                st.session_state.raw_predictions.append(full_mask)
                st.session_state.original_images.append(image)
                st.session_state.image_filenames.append(image_filename)
                
                # Display the raw prediction alongside the original.
                prediction_image = Image.fromarray((full_mask * 255).astype(np.uint8))
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_container_width=True)
                with col2:
                    st.image(prediction_image, caption="Raw Prediction", use_container_width=True)
                
                progress_bar.progress((idx + 1) / total_images)
            
            st.success("Segmentation complete!")

# ------------------------------------------------------
# 9. Step 2: Post-Processing (Interactive Refinement)
# ------------------------------------------------------
if st.session_state.raw_predictions:
    st.markdown("### Refined Results")
    
    refined_processed_images = []  # (mask image, filename) tuples
    refined_overlays = []          # (overlay image, filename) tuples
    
    for idx, (raw_mask, orig_img, image_filename) in enumerate(zip(
            st.session_state.raw_predictions,
            st.session_state.original_images,
            st.session_state.image_filenames)):
        
        # Apply threshold to the raw prediction.
        refined_mask_array = (raw_mask > threshold_value).astype(np.uint8)
        refined_mask_image = Image.fromarray((refined_mask_array * 255).astype(np.uint8))
        
        # Create overlay if requested.
        if overlay_option:
            orig_rgba = orig_img.convert("RGBA")
            mask_rgba = refined_mask_image.convert("RGBA")
            overlay_image = Image.blend(orig_rgba, mask_rgba, alpha=0.5)
        else:
            overlay_image = None
        
        # Display the results.
        if overlay_option:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(orig_img, caption=f"{image_filename} - Original", use_container_width=True)
            with col2:
                st.image(refined_mask_image, caption="Thresholded Mask", use_container_width=True)
            with col3:
                st.image(overlay_image, caption="Overlay", use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(orig_img, caption=f"{image_filename} - Original", use_container_width=True)
            with col2:
                st.image(refined_mask_image, caption="Thresholded Mask", use_container_width=True)
        
        refined_processed_images.append((refined_mask_image, f"prediction_{image_filename}.jpg"))
        if overlay_option and overlay_image:
            refined_overlays.append((overlay_image, f"overlay_{image_filename}.jpg"))
    
    # ------------------------------------------------------
    # 10. Step 3: Download Options (ZIP File Download)
    # ------------------------------------------------------
    st.markdown("### Download Options")
    download_choices = st.multiselect(
        "Select what you would like to download:",
        ["Masks (JPG)", "Overlays (JPG)"] if overlay_option else ["Masks (JPG)"]
    )
    
    if st.button("Download Selected Files"):
        files_to_zip = []
        if "Masks (JPG)" in download_choices:
            files_to_zip.extend(refined_processed_images)
        if "Overlays (JPG)" in download_choices and overlay_option:
            files_to_zip.extend(refined_overlays)
        
        if files_to_zip:
            create_zip_download(files_to_zip, zip_filename="refined_results.zip")
        else:
            st.warning("No files selected for download.")
