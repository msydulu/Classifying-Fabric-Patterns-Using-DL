
# Classifying-Fabric-Patterns-Using-DL #
DATA SET LINK
# https://drive.google.com/drive/folders/1DX-Xv8K0zAkghXpUjsMTriWEQRI73wIT #

**SUMMARY**

This project builds a CNN that learns to recognize fabric patterns from images and saves the trained model.
Fabric Pattern Classification - Python Code Explanation
Project: Fabric Pattern Classification using Deep Learning
Goal:
Train a model to identify different fabric patterns (like dotted, striped, chequered) from images. Step-by-step Explanation:

1. Importing Libraries:
We use TensorFlow for deep learning and Matplotlib for plotting graphs.

2. Loading the Dataset:
Images are loaded from a folder named 'Fabric Dataset', where each subfolder is a class label.

3. Preprocessing Images:
Images are resized to 180x180 pixels. Batch size is 32, meaning 32 images are loaded at once.

4. Creating Training and Validation Sets:
We split the data - 80% for training, 20% for validation, using image_dataset_from_directory.

5. Getting Class Names:
Prints the names of fabric pattern classes (e.g., 'chequered', 'dotted').

6. Performance Boost (Caching & Prefetching):
Speeds up data loading during training.

7. Building the Model (CNN):
We create a sequential CNN model:

Rescaling (normalize pixel values)
Conv2D & MaxPooling2D layers for feature extraction
Flatten & Dense layers for classification
8. Compiling the Model:
We use Adam optimizer and sparse categorical crossentropy loss. Metrics used: accuracy

9. Training the Model:
The model is trained for 10 epochs using train and validation datasets.

10. Plotting Accuracy & Loss:
We visualize training vs validation accuracy and loss to understand performance.

11. Saving the Model:
We save the trained model as 'fabric_model.h5' for future deployment.

