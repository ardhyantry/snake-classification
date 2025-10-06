import tensorflow as tf
from transformers import ViTImageProcessor, ViTForImageClassification, TFViTForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from pathlib import Path
import keras_tuner as kt
from transformers import TFViTForImageClassification, ViTImageProcessor

#load model
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

#load dataset
data_dir = Path("/Users/ardhyantry/Documents/GitHub/snake-classification/coba/dataset ular copy")

#preprocessing data
batch_size = 32
img_height = 224
img_width = 224
img_class =2 

#split dataset 70% training, 15% validation, 15% testing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    classes=['Venomous', 'Non Venomous']
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    classes=['Venomous', 'Non Venomous']
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary',
    classes=['Venomous', 'Non Venomous']
)



