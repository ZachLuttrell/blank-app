# Building Footprint Extraction Through Semantic Segmentation

This repository contains the code for my semantic segmentation application designed to extract building footprints from Sentinel-2 imagery using a UNet-based model with a VGG16 backbone. This application allows users to upload satellite images, process them using the trained model, and obtain prediction masks, along with optional overlays of the detected building footprints.
Application Link

The live application is hosted on Streamlit and can be accessed here: [Building Footprint Extraction App](https://building-footprint-extraction.streamlit.app/)

## Features

* Upload satellite images (in JPG, PNG, or TIFF format) for processing
* Obtain prediction masks for building footprints via semantic segmentation
* Option to threshold the predictions and generate overlays with the original image
* Batch processing of multiple images
* Download options for prediction masks and overlay images

How to run it on your own machine
## 1. Clone the repository:

    $ git clone https://github.com/ZachLuttrell/building-footprint-extraction.git
    $ cd building-footprint-extraction

## 2. Install dependencies

You can install all necessary Python packages by running:

    $ pip install -r requirements.txt

## 3. Run the Streamlit app

After installing the requirements, you can run the app locally with:

    $ streamlit run streamlit_app.py

The app will now be accessible in your browser at http://localhost:8501/ by default.

## 4. Downloading and loading the model

This app automatically downloads the trained UNet+VGG16 model from Google Drive during runtime. The model file is stored as unet_vgg_14.keras and is required for generating predictions.

If you want to manually download the model file, you can access it here: [Download Model.](https://drive.google.com/file/d/1MB7DOQq6--oIYF6TWdn7kisjXWnPI1E4/view?usp=drive_link)
Files in the Repository

    streamlit_app.py: Main application file for running the Streamlit interface.
    requirements.txt: List of dependencies required to run the application.
    model/: Directory where the model is stored once downloaded from Google Drive.
    README.md: This file, containing instructions on how to run the application.

## Model Information

The model used in this application is a UNet architecture with a VGG16 backbone, specifically trained for building footprint extraction from Sentinel-2 imagery. It includes custom metrics like Dice Coefficient and Jaccard Index, along with a weighted binary cross-entropy loss function to handle class imbalance in the training data.

## Deployment Instructions

To deploy this application yourself (e.g., on Streamlit Cloud or another cloud platform), follow these steps:

    Clone the repository and set it up on your local machine.
    Push the repository to a cloud-hosted GitHub repository (if not already done).
    Deploy to Streamlit Cloud or another platform. For Streamlit Cloud:
        Sign in to Streamlit and create a new app.
        Connect your GitHub repository and select the streamlit_app.py file as the entry point.
        Streamlit Cloud will handle the rest, and your app will be accessible via a public URL.

## Future Improvements

    Expand support for additional imagery formats.
    Further optimization for performance and speed, especially for large image batches.

Contact

Feel free to reach out if you have any questions or suggestions:

Author: Zach
Email: zacharywluttrell@gmail.com
