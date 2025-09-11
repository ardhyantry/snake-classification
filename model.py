import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import TFViTForImageClassification, ViTImageProcessor

# Disable oneDNN warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Parameter Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
NUM_CLASSES = 2  # e.g., Class A and Class B
MODEL_HF_NAME = "google/vit-base-patch16-224"
LOCAL_MODEL_DIR = "C:\\Users\\Ardhyan\\Documents\\Code TA\\vit_base_patch16_224"

# Prepare Dataset with Augmentation
def prepare_dataset(data_dir):
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        rotation_range=20,
        zoom_range=[0.8, 1.0],
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    return train_generator, validation_generator

def create_vit_classifier():
    from transformers import TFViTForImageClassification, ViTImageProcessor

    # Load base model without specifying num_labels
    base_model = TFViTForImageClassification.from_pretrained(
        LOCAL_MODEL_DIR,
        ignore_mismatched_sizes=True,
        from_pt=True
    )

    processor = ViTImageProcessor.from_pretrained(LOCAL_MODEL_DIR)

    def preprocess_images_np(images):
        # images: numpy array of shape (batch, 224, 224, 3)
        # Convert to list of PIL images
        pil_images = [tf.keras.preprocessing.image.array_to_img(img) for img in images]
        processed = processor(images=pil_images, return_tensors="np")
        return processed['pixel_values']

    # Keras Lambda layer for preprocessing using numpy
    def preprocess_layer(x):
        # x: tf.Tensor of shape (batch, 224, 224, 3), dtype uint8
        # Convert to numpy, process, then back to tf.Tensor
        def _preprocess(images):
            images = images.numpy()
            return preprocess_images_np(images)
        pixel_values = tf.py_function(_preprocess, [x], tf.float32)
        pixel_values.set_shape([None, 3, IMG_HEIGHT, IMG_WIDTH])
        return pixel_values

    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.uint8)
    pixel_values = tf.keras.layers.Lambda(preprocess_layer)(inputs)
    # Get the base model outputs (logits for original head)
    features = base_model.vit(pixel_values)[0]  # shape: (batch, seq_len, hidden)
    pooled = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(features)  # CLS token

    # New classification head for your number of classes
    logits = tf.keras.layers.Dense(NUM_CLASSES)(pooled)
    outputs = tf.keras.layers.Activation('sigmoid')(logits)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Train the model
def train_model(data_dir):
    train_generator, validation_generator = prepare_dataset(data_dir)
    model = create_vit_classifier()

    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-7)
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=callbacks
    )

    # Save the model using TensorFlow's native method
    model.save("vit_b16_snake_classifier.keras")
    print("Model saved in Keras format at: vit_b16_snake_classifier.keras")

    return model, history

# Plot training results
def plot_training(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    DATA_DIR = "C:\\Users\\Ardhyan\\Documents\\Code TA\\DATASET"  

    print(f"Training on dataset: {DATA_DIR}")

    model, history = train_model(DATA_DIR)

    # Plot training performance
    plot_training(history)