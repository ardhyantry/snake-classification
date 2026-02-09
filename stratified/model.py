#pake env vit_snake 
#di sini pake TFVITModel dari transformers 
#DENGAN AUGMENTASI - untuk perbandingan dengan model tanpa augmentasi
import tensorflow as tf
from transformers import ViTImageProcessor, TFViTModel
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from pathlib import Path
import keras_tuner as kt
from PIL import Image
import sys
import seaborn as sns

sys.modules['keras'] = tf.keras
# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = (224, 224)
NUM_CLASSES = 2
EPOCHS = 100

BASE_BATCH_SIZE = 16
BASE_LEARNING_RATE = 2e-5

# Hyperparameter tuning configuration
TUNER_EPOCHS = 10
MAX_TRIALS = 20

# DATASET_PATH = 'C:\\Users\\Ardhyan\\Documents\\GitHub\\snake-classification\\dataset ular'  
DATASET_PATH = '/Users/ardhyantry/Documents/GitHub/snake-classification/dataset ularBaru'  
LOCAL_MODEL_PATH = './vit_base_patch16_224'

print("Downloading pretrained model...")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
processor.save_pretrained(LOCAL_MODEL_PATH)

# Save backbone only (for functional model)
backbone = TFViTModel.from_pretrained("google/vit-base-patch16-224")
backbone.save_pretrained(LOCAL_MODEL_PATH)
print("Model downloaded and saved to:", LOCAL_MODEL_PATH)

# Preprocessing function (simplified and robust)
def preprocess_image(image_path, label):
    def decode_and_process_image(img_path):
        file_path = img_path.numpy().decode('utf-8')
        try:
            image = Image.open(file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="np")
            pixel_values = inputs["pixel_values"][0]
            return pixel_values.astype(np.float32)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((3, 224, 224), dtype=np.float32)

    processed_image = tf.py_function(
        func=decode_and_process_image,
        inp=[image_path],
        Tout=tf.float32
    )
    processed_image.set_shape((3, 224, 224))
    return processed_image, label

def is_valid_image(file_path):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    if not os.path.isfile(file_path):
        return False
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_exts:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def create_datasets_with_batch_size(data_dir, batch_size, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10

    venomous_dir = os.path.join(data_dir, 'venomous')
    non_venomous_dir = os.path.join(data_dir, 'non_venomous')

    venomous_files = [os.path.join(venomous_dir, f) for f in os.listdir(venomous_dir)]
    non_venomous_files = [os.path.join(non_venomous_dir, f) for f in os.listdir(non_venomous_dir)]

    venomous_files = [f for f in venomous_files if is_valid_image(f)]
    non_venomous_files = [f for f in non_venomous_files if is_valid_image(f)]

    print(f"Found {len(venomous_files)} valid venomous images")
    print(f"Found {len(non_venomous_files)} valid non-venomous images")

    all_files = venomous_files + non_venomous_files
    all_labels = [0] * len(venomous_files) + [1] * len(non_venomous_files)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files,
        all_labels,
        test_size=(val_ratio + test_ratio),
        stratify=all_labels,  
        random_state=42
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,  
        random_state=42
    )

    print(f"\n{'='*60}")
    print("STRATIFIED DATASET SPLIT - WITH AUGMENTATION")
    print(f"{'='*60}")
    print(f"Training set:   {len(train_files)} images")
    print(f"  - Venomous:     {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
    print(f"  - Non-Venomous: {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
    print(f"\nValidation set: {len(val_files)} images")
    print(f"  - Venomous:     {val_labels.count(0)} ({val_labels.count(0)/len(val_labels)*100:.1f}%)")
    print(f"  - Non-Venomous: {val_labels.count(1)} ({val_labels.count(1)/len(val_labels)*100:.1f}%)")
    print(f"\nTest set:       {len(test_files)} images")
    print(f"  - Venomous:     {test_labels.count(0)} ({test_labels.count(0)/len(test_labels)*100:.1f}%)")
    print(f"  - Non-Venomous: {test_labels.count(1)} ({test_labels.count(1)/len(test_labels)*100:.1f}%)")
    print(f"{'='*60}\n")

    # Create tf.data.Dataset objects from pre-stratified subsets
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # ========== AUGMENTATION FUNCTION ==========
    # Augmentasi untuk ViT dengan format channels-first (3, 224, 224)
    def augment(image, label):
        # ViT menggunakan format channels-first (3, 224, 224)
        # Transpose ke channels-last untuk augmentasi
        image = tf.transpose(image, [1, 2, 0])  # (3, 224, 224) -> (224, 224, 3)
        
        # Random zoom between 0.7 and 1.0
        zoom_factor = tf.random.uniform([], 0.7, 1.0)
        h, w = IMG_SIZE[0], IMG_SIZE[1]
        new_h = tf.cast(tf.cast(h, tf.float32) * zoom_factor, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) * zoom_factor, tf.int32)
        
        # Resize to zoomed size then back to original size (crops center)
        image = tf.image.resize(image, [new_h, new_w])
        image = tf.image.resize_with_crop_or_pad(image, h, w)
        
        # Random rotation between -20 and +20 degrees
        def rotate_image(img):
            from scipy import ndimage
            angle = np.random.uniform(-20.0, 20.0)
            return ndimage.rotate(img, angle, reshape=False, mode='nearest')
        
        image = tf.py_function(
            func=rotate_image,
            inp=[image],
            Tout=tf.float32
        )
        image.set_shape((224, 224, 3))
        
        # Other augmentations
        image = tf.image.random_flip_left_right(image)
        
        # Transpose back ke channels-first untuk ViT
        image = tf.transpose(image, [2, 0, 1])  # (224, 224, 3) -> (3, 224, 224)
        
        return image, label

    # DENGAN AUGMENTASI - apply augmentasi pada training set
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle dengan buffer lebih besar untuk randomness yang lebih baik
    train_ds = train_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Return split statistics for saving to txt
    split_info = {
        'total_images': len(all_files),
        'total_venomous': len(venomous_files),
        'total_non_venomous': len(non_venomous_files),
        'train_total': len(train_files),
        'train_venomous': train_labels.count(0),
        'train_non_venomous': train_labels.count(1),
        'val_total': len(val_files),
        'val_venomous': val_labels.count(0),
        'val_non_venomous': val_labels.count(1),
        'test_total': len(test_files),
        'test_venomous': test_labels.count(0),
        'test_non_venomous': test_labels.count(1),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }

    return train_ds, val_ds, test_ds, split_info

def create_datasets(data_dir, split_ratio=0.3):
    return create_datasets_with_batch_size(
        data_dir, BASE_BATCH_SIZE,
        train_ratio=1-split_ratio,
        val_ratio=split_ratio/2,
        test_ratio=split_ratio/2
    )

# Global variable to store split info
split_info = None

# Build model function for KerasTuner (returns real tf.keras.Model)
def build_model(hp):
    inputs = tf.keras.Input(shape=(3, 224, 224), dtype=tf.float32)
    
    # Load ViT backbone
    vit = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    vit.trainable = False 

    # Get pooled features
    outputs = vit(inputs).pooler_output  # (batch, 768)

    # Tune dropout
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    x = tf.keras.layers.Dropout(dropout)(outputs)

    # Classification head
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits)

    # Tune learning rate & optimizer
    lr = hp.Choice("learning_rate", [1e-5, 2e-5, 5e-5, 1e-4])
    opt_type = hp.Choice("optimizer", ["adam", "adamw"])

    if opt_type == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return model

# Create datasets
print("Creating datasets...")
train_ds, val_ds, test_ds, split_info = create_datasets(DATASET_PATH)

# Setup tuner
tuner_dir = "./stratified/tuner_results"
os.makedirs(tuner_dir, exist_ok=True)

tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=MAX_TRIALS,
    directory=tuner_dir,
    project_name="snake_vit_tuning_with_aug",
    overwrite=False
)

print("\nStarting hyperparameter search with RandomSearch...")
tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=TUNER_EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ],
    verbose=1
)

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_score = tuner.oracle.get_best_trials(1)[0].score

print(f"\nBest validation loss: {best_score:.4f}")
print(f"Best LR: {best_hps.get('learning_rate')}, Optimizer: {best_hps.get('optimizer')}, Dropout: {best_hps.get('dropout')}")

# Final model with best hps
def create_final_model(hp):
    inputs = tf.keras.Input(shape=(3, 224, 224), dtype=tf.float32)
    vit = TFViTModel.from_pretrained("google/vit-base-patch16-224")
    vit.trainable = False
    x = vit(inputs).pooler_output
    x = tf.keras.layers.Dropout(hp.get("dropout"))(x)
    # Dengan softmax karena menggunakan from_logits=False
    logits = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=logits)

    if hp.get("optimizer") == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=hp.get("learning_rate"))
    else:
        opt = tf.keras.optimizers.AdamW(learning_rate=hp.get("learning_rate"))

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    return model

model = create_final_model(best_hps)

# Train final model
checkpoint_dir = "./stratified/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate
print("Evaluating on test set...")
logits = model.predict(test_ds)
predicted_labels = tf.argmax(logits, axis=1).numpy()

# Extract true labels
true_labels = []
for _, label in test_ds.unbatch():
    true_labels.append(label.numpy())
true_labels = np.array(true_labels)[:len(predicted_labels)]

# Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Classification report
class_names = ['Venomous', 'Non-Venomous']
report = classification_report(true_labels, predicted_labels, target_names=class_names, digits=4)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\n" + "="*60)
print("SUMMARY METRICS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("="*60)

#plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Venomous', 'Non-Venomous'],
            yticklabels=['Venomous', 'Non-Venomous'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (With Augmentation)')
plt.savefig('stratified/confusion_matrix.png')
plt.show()

# Plot training history
def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy (With Augmentation)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss (With Augmentation)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('stratified/training_history.png')
    plt.show()

plot_training(history)

# Save results
os.makedirs("stratified", exist_ok=True)
with open("stratified/tuning_results.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("VISION TRANSFORMER (ViT) - SNAKE CLASSIFICATION RESULTS\n")
    f.write("DENGAN AUGMENTASI\n")
    f.write("="*60 + "\n\n")
    
    f.write("===== BEST HYPERPARAMETERS =====\n")
    f.write(f"Best Validation Loss: {best_score:.4f}\n")
    f.write(f"Learning Rate: {best_hps.get('learning_rate')}\n")
    f.write(f"Optimizer: {best_hps.get('optimizer')}\n")
    f.write(f"Dropout: {best_hps.get('dropout')}\n\n")
    
    f.write("===== DATASET INFORMATION =====\n")
    f.write(f"Dataset Path: {DATASET_PATH}\n")
    f.write(f"Total Images: {split_info['total_images']}\n")
    f.write(f"  - Total Venomous: {split_info['total_venomous']}\n")
    f.write(f"  - Total Non-Venomous: {split_info['total_non_venomous']}\n\n")
    
    f.write("===== STRATIFIED DATA SPLIT =====\n")
    f.write(f"Split Ratio: Train={split_info['train_ratio']*100:.0f}% / Val={split_info['val_ratio']*100:.0f}% / Test={split_info['test_ratio']*100:.0f}%\n\n")
    
    f.write(f"Training Set: {split_info['train_total']} images\n")
    f.write(f"  - Venomous:     {split_info['train_venomous']} ({split_info['train_venomous']/split_info['train_total']*100:.1f}%)\n")
    f.write(f"  - Non-Venomous: {split_info['train_non_venomous']} ({split_info['train_non_venomous']/split_info['train_total']*100:.1f}%)\n\n")
    
    f.write(f"Validation Set: {split_info['val_total']} images\n")
    f.write(f"  - Venomous:     {split_info['val_venomous']} ({split_info['val_venomous']/split_info['val_total']*100:.1f}%)\n")
    f.write(f"  - Non-Venomous: {split_info['val_non_venomous']} ({split_info['val_non_venomous']/split_info['val_total']*100:.1f}%)\n\n")
    
    f.write(f"Test Set: {split_info['test_total']} images\n")
    f.write(f"  - Venomous:     {split_info['test_venomous']} ({split_info['test_venomous']/split_info['test_total']*100:.1f}%)\n")
    f.write(f"  - Non-Venomous: {split_info['test_non_venomous']} ({split_info['test_non_venomous']/split_info['test_total']*100:.1f}%)\n\n")
    
    f.write("===== AUGMENTATION TECHNIQUES =====\n")
    f.write("1. Random Zoom: 0.7-1.0 scale\n")
    f.write("2. Random Rotation: -20° to +20°\n")
    f.write("3. Random Horizontal Flip\n\n")

    f.write("===== FINAL EVALUATION ON TEST SET =====\n")
    f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Test Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Test Recall: {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"Test F1-Score: {f1:.4f} ({f1*100:.2f}%)\n\n")
    
    f.write("="*60 + "\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("="*60 + "\n")
    f.write(report + "\n")

    f.write("\n" + "="*60 + "\n")
    f.write("CONFUSION MATRIX\n")
    f.write("="*60 + "\n")
    f.write(f"                  Predicted\n")
    f.write(f"              Venomous  Non-Venomous\n")
    f.write(f"Actual\n")
    f.write(f"Venomous         {conf_matrix[0][0]:3d}        {conf_matrix[0][1]:3d}\n")
    f.write(f"Non-Venomous     {conf_matrix[1][0]:3d}        {conf_matrix[1][1]:3d}\n\n")
    
    f.write("="*60 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*60 + "\n")
    f.write(f"True Positives (Venomous correctly identified): {conf_matrix[0][0]}\n")
    f.write(f"True Negatives (Non-Venomous correctly identified): {conf_matrix[1][1]}\n")
    f.write(f"False Positives (Non-Venomous misclassified as Venomous): {conf_matrix[1][0]}\n")
    f.write(f"False Negatives (Venomous misclassified as Non-Venomous): {conf_matrix[0][1]}\n\n")
    f.write(f"Total Correct Predictions: {conf_matrix[0][0] + conf_matrix[1][1]}/{len(true_labels)}\n")
    f.write(f"Total Incorrect Predictions: {conf_matrix[0][1] + conf_matrix[1][0]}/{len(true_labels)}\n\n")

    f.write("="*60 + "\n")
    f.write("TRAINING HISTORY\n")
    f.write("="*60 + "\n")
    epochs_count = len(history.history['accuracy'])
    for i in range(epochs_count):
        f.write(
            f"Epoch {i+1:03d}: "
            f"Train Acc={history.history['accuracy'][i]:.4f}, "
            f"Val Acc={history.history['val_accuracy'][i]:.4f}, "
            f"Train Loss={history.history['loss'][i]:.4f}, "
            f"Val Loss={history.history['val_loss'][i]:.4f}\n"
        )

    f.write("\n" + "="*60 + "\n")
    f.write("MODEL SUMMARY\n")
    f.write("="*60 + "\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print(f"\n✅ Results saved to 'stratified/tuning_results.txt'")

# Save model
model.save("stratified/snake_vit_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("stratified/snake_vit_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ All done! Model saved and TFLite exported.")
