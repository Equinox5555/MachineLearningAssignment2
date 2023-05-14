import matplotlib.pyplot  as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import platform
import cv2
import tensorflow as tf
import IPython.display as display
import os

from skimage import io, feature, util, color
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from tensorflow import keras
from PIL import Image
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def split_dataset(main_data, train_ratio, val_ratio):
    """
    Split the main_data DataFrame into training, validation, and testing sets based on the patientID column.

    Parameters:
    main_data (DataFrame): the DataFrame containing the image data
    train_ratio (float): the proportion of data to allocate to the training set (e.g., 0.7 for 70%)
    val_ratio (float): the proportion of data to allocate to the validation set (e.g., 0.15 for 15%)

    Returns:
    train_data (DataFrame): the subset of main_data containing the training data
    val_data (DataFrame): the subset of main_data containing the validation data
    test_data (DataFrame): the subset of main_data containing the testing data
    """

    # Set the random seed for reproducibility to student id
    np.random.seed(88)
    
    # Extract the unique patient IDs from your dataset
    unique_patients = np.unique(main_data['patientID'])

    # Shuffle the patient IDs
    np.random.shuffle(unique_patients)

    # Split the list of patient IDs into three disjoint sets
    train_patients, val_test_patients = np.split(unique_patients, [int(len(unique_patients)*train_ratio)])

    val_patients, test_patients = np.split(val_test_patients, [int(len(val_test_patients)*val_ratio)])

    # Finally, use the patient ID sets to filter the original dataset into training, validation, and testing sets
    train_data = main_data[main_data['patientID'].isin(train_patients)]
    val_data = main_data[main_data['patientID'].isin(val_patients)]
    test_data = main_data[main_data['patientID'].isin(test_patients)]

    return train_data, val_data, test_data




def preprocess_data(image_folder, train_data, val_data, test_data, target_size, batch_size):
    """
    Preprocesses the image data and creates data generators for training, validation, and testing sets.

    Parameters:
    image_folder (str): the directory containing the image data
    train_data (DataFrame): the DataFrame containing the training data
    val_data (DataFrame): the DataFrame containing the validation data
    test_data (DataFrame): the DataFrame containing the testing data
    target_size (tuple): a tuple specifying the target size of the input images
    batch_size (int): the batch size for the data generators

    Returns:
    output (dict): a dictionary containing the data generators and the number of samples for each set
    """

    # Get the list of image paths for each set
    train_image_paths = [os.path.join(image_folder, filename) for filename in train_data['ImageName'].values]
    val_image_paths = [os.path.join(image_folder, filename) for filename in val_data['ImageName'].values]
    test_image_paths = [os.path.join(image_folder, filename) for filename in test_data['ImageName'].values]

    # Create data generators for each set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        train_data,
        directory=image_folder,
        x_col='ImageName',
        y_col='cellType',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_data,
        directory=image_folder,
        x_col='ImageName',
        y_col='cellType',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_data,
        directory=image_folder,
        x_col='ImageName',
        y_col='cellType',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Define the output dictionary
    output = {
        'train_generator': train_generator,
        'val_generator': val_generator,
        'test_generator': test_generator,
        'num_train_samples': len(train_data),
        'num_val_samples': len(val_data),
        'num_test_samples': len(test_data)
    }

    return output