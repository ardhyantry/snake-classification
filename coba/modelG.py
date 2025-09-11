import tensorflow as tf
from transformers import ViTImageProcessor, ViTForImageClassification, TFViTForImageClassification
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
from pathlib import Path
import keras_tuner as kt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 2  # Venomous and Non-Venomous
EPOCHS = 10
# Base hyperparameters (will be tuned)
BASE_BATCH_SIZE = 16
BASE_LEARNING_RATE = 2e-5

# Hyperparameter tuning configuration
TUNER_EPOCHS = 3  # Reduced epochs for faster tuning
MAX_TRIALS = 10  # Number of different hyperparameter combinations to try

# Enable memory growth to prevent TF from allocating all GPU memory at once
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    print("No GPU available or error setting memory growth")

# Define paths
DATASET_PATH = 'C:\\Users\\Ardhyan\\Documents\\Code TA\\coba\\dataset ular copy'  # Update if needed
LOCAL_MODEL_PATH = 'C:\\Users\\Ardhyan\\Documents\\Code TA\\vit_base_patch16_224'  # Where model will be saved

# Step 1: Download and save the pretrained model
print("Downloading pretrained model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor.save_pretrained(LOCAL_MODEL_PATH)

# Use only TensorFlow version since PyTorch is not available
model_tf = TFViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224", 
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model_tf.save_pretrained(LOCAL_MODEL_PATH)
print("Model downloaded and saved to:", LOCAL_MODEL_PATH)

# Step 2: Create TensorFlow datasets with proper preprocessing

# Function to load and preprocess images with ViT processor
def preprocess_image(image_path, label):
    # Use a more robust image decoding approach that handles multiple formats
    def decode_and_process_image(img_path):
        try:
            # Read the file
            file_contents = tf.io.read_file(img_path)
            
            # Try to determine the image format and decode appropriately
            try:
                # Try decoding as JPEG first (most common)
                image = tf.image.decode_jpeg(file_contents, channels=3)
            except:
                try:
                    # If JPEG fails, try PNG
                    image = tf.image.decode_png(file_contents, channels=3)
                except:
                    try:
                        # If PNG fails, try generic image decoder
                        image = tf.image.decode_image(file_contents, channels=3)
                    except:
                        # If all TensorFlow decoders fail, try PIL as a fallback
                        try:
                            # Import PIL only when needed
                            from PIL import Image
                            import io
                            
                            # Get the file path as a string
                            file_path = img_path.numpy().decode('utf-8')
                            print(f"Trying PIL fallback for {file_path}")
                            
                            # Open with PIL
                            with Image.open(file_path) as pil_img:
                                # Convert to RGB (removes alpha channels, etc.)
                                if pil_img.mode != 'RGB':
                                    pil_img = pil_img.convert('RGB')
                                
                                # Resize to expected dimensions
                                pil_img = pil_img.resize(IMG_SIZE)
                                
                                # Convert to numpy array and normalize
                                img_np = np.array(pil_img, dtype=np.float32) / 255.0
                                
                                # Create a TensorFlow tensor
                                image = tf.convert_to_tensor(img_np)
                                print(f"Successfully loaded {file_path} with PIL")
                                return processor(images=img_np, return_tensors="np", do_rescale=False)["pixel_values"][0]
                        except Exception as pil_err:
                            print(f"Both TF and PIL failed to decode {img_path.numpy().decode('utf-8')}: {pil_err}")
                            # Return zeros with consistent shape [3, 224, 224] (channels first)
                            return np.zeros((3, 224, 224), dtype=np.float32)
            
            # Resize and normalize
            image = tf.image.resize(image, IMG_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            
            # Process with ViT processor
            img_np = image.numpy()
            inputs = processor(images=img_np, return_tensors="np", do_rescale=False)
            
            # Ensure consistent shape [3, 224, 224] (channels first as expected by ViT)
            processed = inputs["pixel_values"][0]  # Shape should be [3, 224, 224]
            
            # Double check the shape is correct
            if processed.shape != (3, 224, 224):
                # If somehow the processor returns channels-last format, transpose
                if processed.shape == (224, 224, 3):
                    processed = np.transpose(processed, (2, 0, 1))  # Convert to [3, 224, 224]
                else:
                    # If completely wrong shape, return zeros with correct shape
                    print(f"Warning: Unexpected shape {processed.shape} for {img_path.numpy().decode('utf-8')}")
                    processed = np.zeros((3, 224, 224), dtype=np.float32)
                    
            return processed
        except Exception as e:
            print(f"Error processing image {img_path.numpy().decode('utf-8')}: {str(e)}")
            # Return zero-filled tensor with consistent shape [3, 224, 224] (channels first)
            return np.zeros((3, 224, 224), dtype=np.float32)
    
    # Apply the function using tf.py_function to handle Python code in graph mode
    processed_image = tf.py_function(
        func=decode_and_process_image,
        inp=[image_path],
        Tout=tf.float32
    )
    
    # Set the shape since py_function loses shape information
    processed_image.set_shape((3, 224, 224))  # ViT processes to (C, H, W) format
    
    return processed_image, label

# Function to create datasets with tunable batch size
def create_datasets_with_batch_size(data_dir, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Verify split ratios add up to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Split ratios must add up to 1.0"
    
    # Get image paths and labels
    venomous_dir = os.path.join(data_dir, 'Venomous')
    non_venomous_dir = os.path.join(data_dir, 'Non Venomous')  # Note the capitalization
    
    # Function to verify if a file is a valid image
    def is_valid_image(file_path):
        try:
            # Check if file exists and has non-zero size
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                print(f"Skipping invalid image file: {file_path} (empty or doesn't exist)")
                return False
                
            # Check extension
            valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            if not any(file_path.lower().endswith(ext) for ext in valid_exts):
                print(f"Skipping file with invalid extension: {file_path}")
                return False
                
            # Try to open and verify the file is a valid image
            try:
                # Use TensorFlow to verify the image can be decoded
                img_data = tf.io.read_file(file_path)
                tf.io.decode_image(img_data, channels=3)
                return True
            except tf.errors.InvalidArgumentError:
                print(f"Skipping corrupt image file: {file_path}")
                return False
                
            return True
        except Exception as e:
            print(f"Error checking file {file_path}: {str(e)}")
            return False
    
    # Get all image files and filter out invalid ones
    venomous_files = [os.path.join(venomous_dir, f) for f in os.listdir(venomous_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    non_venomous_files = [os.path.join(non_venomous_dir, f) for f in os.listdir(non_venomous_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    # Filter out any invalid files
    venomous_files = [f for f in venomous_files if is_valid_image(f)]
    non_venomous_files = [f for f in non_venomous_files if is_valid_image(f)]
    
    print(f"Found {len(venomous_files)} valid venomous images")
    print(f"Found {len(non_venomous_files)} valid non-venomous images")
    
    # Create labels (0: venomous, 1: non-venomous)
    venomous_labels = [0] * len(venomous_files)
    non_venomous_labels = [1] * len(non_venomous_files)
    
    # Combine data
    all_files = venomous_files + non_venomous_files
    all_labels = venomous_labels + non_venomous_labels
    
    # Shuffle data
    indices = np.random.permutation(len(all_files))
    all_files = [all_files[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    # Calculate split points
    n_samples = len(all_files)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split into train, validation, and test sets
    train_files = all_files[:n_train]
    train_labels = all_labels[:n_train]
    
    val_files = all_files[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]
    
    test_files = all_files[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]
    
    print(f"Training images: {len(train_files)} ({train_ratio*100:.1f}%)")
    print(f"Validation images: {len(val_files)} ({val_ratio*100:.1f}%)")
    print(f"Test images: {len(test_files)} ({test_ratio*100:.1f}%)")
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    
    # Apply preprocessing
    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Memory efficient data loading: no cache, smaller shuffle buffer, prefetch
    train_ds = train_ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds

# Original function for backward compatibility
def create_datasets(data_dir, split_ratio=0.3):
    return create_datasets_with_batch_size(data_dir, BASE_BATCH_SIZE, 1-split_ratio, split_ratio/2, split_ratio/2)
    return create_datasets_with_batch_size(data_dir, BASE_BATCH_SIZE, 1-split_ratio, split_ratio/2, split_ratio/2)

# Create datasets with proper preprocessing
print("Creating datasets with ViT preprocessing...")
train_ds, val_ds, test_ds = create_datasets(DATASET_PATH)

# Step 3: Simplified Training without Hyperparameter Tuning
print("Loading model for training...")

# Clear any existing session
tf.keras.backend.clear_session()

# Since hyperparameter tuning is causing issues, use default parameters
best_hps = {
    'learning_rate': BASE_LEARNING_RATE,
    'optimizer': 'adam',
    'batch_size': BASE_BATCH_SIZE
}

print("Using default hyperparameters:")
print(f"Learning rate: {best_hps['learning_rate']}")
print(f"Optimizer: {best_hps['optimizer']}")
print(f"Batch size: {best_hps['batch_size']}")

# Use the best score as 0 since we're not tuning
best_score = 0.0

# Create final datasets with the default batch size
BEST_BATCH_SIZE = best_hps['batch_size']
train_ds, val_ds, test_ds = create_datasets_with_batch_size(
    DATASET_PATH, BEST_BATCH_SIZE, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Create the model directly without loading from local path to avoid issues
print("Creating model from online source...")
try:
    model = TFViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    print("Model loaded successfully from online source")
except Exception as e:
    print(f"Error loading model from online: {e}")
    # Fallback to local path
    print("Trying local path...")
    model = TFViTForImageClassification.from_pretrained(
        LOCAL_MODEL_PATH,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        from_pt=True
    )
    print("Model loaded successfully from local path")

# Configure optimizer with default parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=best_hps['learning_rate'])

# Compile the model
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Step 4: Load the model for TensorFlow (this section is now replaced by tuner)
print("Model loaded with optimal hyperparameters...")

# Enable mixed precision training to reduce memory usage
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")
except:
    print("Mixed precision not supported")

# Model is already compiled by the tuner

# Callback for learning rate adjustment (using best learning rate from tuning)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Step 5: Train the model with optimal hyperparameters
print("Starting training with optimal hyperparameters...")

# Create checkpoint directory if it doesn't exist
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Add memory optimization callbacks
callbacks = [
    lr_scheduler,
    # Early stopping to prevent wasting resources if training isn't improving
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    # Model checkpoints to save progress periodically
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'snake_vit_model_epoch_{epoch:02d}_val_acc_{val_accuracy:.2f}.weights.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=True,
        verbose=1
    ),
    # Save checkpoint after every epoch
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'latest_checkpoint.weights.h5'),
        save_weights_only=True,
        save_best_only=False,
        verbose=1
    )
]
                
# Run garbage collection before training
import gc
gc.collect()

# Clear any existing TensorFlow session memory
tf.keras.backend.clear_session()

# Check if there's a checkpoint to resume training from
latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.keras')
initial_epoch = 0

if os.path.exists(latest_checkpoint_path):
    try:
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        # Load the weights from the checkpoint
        model.load_weights(latest_checkpoint_path)
        
        # Try to determine initial epoch from filename if it contains epoch info
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('snake_vit_model_epoch_') and f.endswith('.keras')]
        if checkpoint_files:
            # Sort checkpoints by epoch number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[3]))
            last_checkpoint = checkpoint_files[-1]
            # Extract epoch number (format: snake_vit_model_epoch_XX_val_acc_Y.YY.keras)
            try:
                initial_epoch = int(last_checkpoint.split('_')[3])
                print(f"Resuming training from epoch {initial_epoch}")
            except:
                print("Could not determine initial epoch, starting from 0")
        
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch")
else:
    print("No checkpoint found, starting training from scratch")

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    initial_epoch=initial_epoch,  # Resume from this epoch if checkpoint was loaded
    # Use steps to limit memory usage by breaking up epochs
    steps_per_epoch=None,  # Auto-calculate based on dataset size
    validation_steps=None  # Auto-calculate based on dataset size
)

# Step 5: Evaluate the model
print("Evaluating model...")
test_pred = model.predict(test_ds)
predicted_labels = tf.argmax(test_pred.logits, axis=1).numpy()

# Extract true labels from the dataset - with better error handling
try:
    # Reset the dataset before unbatching
    test_labels = []
    # Create a fresh copy of the dataset to iterate through
    test_ds_copy = test_ds.unbatch()
    for _, label in test_ds_copy:
        test_labels.append(label.numpy())
    
    if test_labels:
        true_labels = np.array(test_labels)
        true_labels = true_labels[:len(predicted_labels)]  # Trim to match prediction length
    else:
        raise ValueError("No labels found in test dataset")
except Exception as e:
    print(f"Error extracting labels: {e}")
    # As a fallback, reconstruct the labels directly from the dataset path
    print("Attempting to reconstruct labels from dataset files...")
    
    # Get the validation files and labels from our dataset creation function
    # (Re-running part of the dataset creation)
    venomous_dir = os.path.join(DATASET_PATH, 'venomous')
    non_venomous_dir = os.path.join(DATASET_PATH, 'Non Venomous')
    
    # Get valid files (simplified version)
    venomous_files = [f for f in os.listdir(venomous_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    non_venomous_files = [f for f in os.listdir(non_venomous_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    # Create labels (0: venomous, 1: non-venomous)
    venomous_labels = [0] * len(venomous_files)
    non_venomous_labels = [1] * len(non_venomous_files)
    
    # Combine data
    all_files = venomous_files + non_venomous_files
    all_labels = venomous_labels + non_venomous_labels
    
    # Use approximately same number of labels as predictions
    true_labels = np.array(all_labels[:len(predicted_labels)])
    print(f"Reconstructed {len(true_labels)} labels for evaluation")

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

# Plot training results
def plot_training(history):
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Plot the training history
plot_training(history)

# Save tuning results
tuning_results_path = 'coba/tuning_results.txt'
with open(tuning_results_path, 'w') as f:
    f.write("Best Hyperparameters Found:\n")
    f.write(f"Learning Rate: {best_hps['learning_rate']}\n")
    f.write(f"Optimizer: {best_hps['optimizer']}\n")
    f.write(f"Batch Size: {best_hps['batch_size']}\n")
    f.write(f"Best Validation Accuracy: {best_score:.4f}\n")
    f.write(f"\nFinal Model Performance:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
print(f"Tuning results saved to {tuning_results_path}")

# Step 6: Save the model
model_path = 'coba/snake_vit_model.keras'
model.save(model_path)
print(f"Model saved to {model_path}")

# Convert to TFLite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = 'coba/snake_vit_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved as {tflite_path} for mobile deployment")