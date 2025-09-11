import tensorflow as tf
from transformers import ViTImageProcessor, ViTForImageClassification, TFViTForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2  # Venomous and Non-Venomous
EPOCHS = 10
LEARNING_RATE = 2e-5

# Define paths
DATASET_PATH = 'C:\\Users\\Ardhyan\\Documents\\Code TA\\Dataset'  # Update if needed
LOCAL_MODEL_PATH = 'C:\\Users\\Ardhyan\\Documents\\Code TA\\vit_base_patch16_224'  # Where model will be saved

# Step 1: Download and save the pretrained model
print("Downloading pretrained model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", num_labels=NUM_CLASSES,ignore_mismatched_sizes=True)
processor.save_pretrained(LOCAL_MODEL_PATH)
model_pytorch = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=NUM_CLASSES,ignore_mismatched_sizes=True)
model_pytorch.save_pretrained(LOCAL_MODEL_PATH)
print("Model downloaded and saved to:", LOCAL_MODEL_PATH)

# Step 2: Data augmentation and generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=[0.7, 1.0],
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.3  # 70% train, 30% split for validation+test
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

def preprocess_batch(batch):
    # Ensure images are RGB and shape (batch_size, 224, 224, 3)
    images, labels = batch
    if images.shape[-1] != 3:
        images = tf.image.grayscale_to_rgb(images)
    return images, labels

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='validation'
)

test_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='validation',
    shuffle=False
)

# Wrap generators with preprocessing
train_generator = map(preprocess_batch, train_generator)
val_generator = map(preprocess_batch, val_generator)
test_generator = map(preprocess_batch, test_generator)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='validation'
)

test_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    classes=['venomous', 'Non Venomous'],
    subset='validation',
    shuffle=False
)

# Step 3: Load the model for TensorFlow
processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
model = TFViTForImageClassification.from_pretrained(
    LOCAL_MODEL_PATH,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
    from_pt=True  # Convert PyTorch weights to TensorFlow
)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Callback for learning rate adjustment
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Step 4: Train the model
print("Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[lr_scheduler]
)

# Step 5: Evaluate the model
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
print("Raw Predictions shape:", predictions.logits.shape)  # Debug raw shape
if len(predictions.logits.shape) != 2 or predictions.logits.shape[1] != NUM_CLASSES:
    raise ValueError(f"Unexpected predictions shape: {predictions.logits.shape}. Expected (n_samples, {NUM_CLASSES})")
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

# Get all test labels
true_labels = test_generator.classes[test_generator.index_array[:len(predicted_labels)]]
print("True labels shape:", true_labels.shape)  # Debug
print("Predicted labels shape:", predicted_labels.shape)  # Debug

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Step 6: Save the model
model.save('snake_vit_model.keras')

# Convert to TFLite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('snake_vit_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved as snake_vit_model.tflite for mobile deployment")