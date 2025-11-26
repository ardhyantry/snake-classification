# pake env vit_snake -> sekarang resNet
import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path
from PIL import Image
import keras_tuner as kt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Keras compatibility hack if you used previously
sys.modules['keras'] = tf.keras

# Repro
tf.random.set_seed(42)
np.random.seed(42)

# Config
IMG_SIZE = (224, 224)
NUM_CLASSES = 2
EPOCHS = 100

BASE_BATCH_SIZE = 16
BASE_LEARNING_RATE = 2e-5

# Hyperparameter tuning configuration
TUNER_EPOCHS = 10
MAX_TRIALS = 20

# Paths - adjust DATASET_PATH sesuai lingkunganmu
DATASET_PATH = '/Users/ardhyantry/Documents/GitHub/snake-classification/dataset ularBaru'
RESNET_DIR = './resNet'
LOCAL_MODEL_PATH = os.path.join(RESNET_DIR, 'resnet_backbone')
TUNER_DIR = os.path.join(RESNET_DIR, 'tuner_results')
CHECKPOINT_DIR = os.path.join(RESNET_DIR, 'checkpoints')
os.makedirs(RESNET_DIR, exist_ok=True)
os.makedirs(TUNER_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Using ResNet50V2 (ImageNet weights) as backbone. Saving outputs to:", RESNET_DIR)

# Helper: valid image
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

# Preprocess single image returning (224,224,3) float32 preprocessed for ResNetV2
def preprocess_image(image_path, label):
    def _load(path):
        p = path.numpy().decode('utf-8')
        try:
            img = Image.open(p).convert('RGB')
            img = img.resize(IMG_SIZE, Image.BICUBIC)
            arr = np.array(img).astype(np.float32)
            # use resnet_v2 preprocess_input
            arr = tf.keras.applications.resnet_v2.preprocess_input(arr)
            return arr
        except Exception as e:
            print(f"Error loading {p}: {e}")
            return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

    img = tf.py_function(func=_load, inp=[image_path], Tout=tf.float32)
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img, label

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

    indices = np.random.permutation(len(all_files))
    all_files = [all_files[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    n = len(all_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = all_files[:n_train]
    train_labels = all_labels[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    val_labels = all_labels[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    test_labels = all_labels[n_train + n_val:]

    print(f"Training: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))

    train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation (only on training)
    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image, label

    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

def create_datasets(data_dir, split_ratio=0.3):
    return create_datasets_with_batch_size(
        data_dir, BASE_BATCH_SIZE,
        train_ratio=1-split_ratio,
        val_ratio=split_ratio/2,
        test_ratio=split_ratio/2
    )

# Build model for KerasTuner
def build_model(hp):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)

    # Hyperparams
    base_trainable = hp.Choice("backbone_trainable", [False, True])  # frozen or fine-tune
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    lr = hp.Choice("learning_rate", [1e-5, 2e-5, 5e-5, 1e-4])
    opt_type = hp.Choice("optimizer", ["adam", "adamw"])

    # ResNet50V2 backbone
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling='avg'
    )
    base_model.trainable = bool(base_trainable)

    x = base_model.output  # (batch, features)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='classifier')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
train_ds, val_ds, test_ds = create_datasets(DATASET_PATH)

# KerasTuner setup
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=MAX_TRIALS,
    directory=TUNER_DIR,
    project_name="snake_resnet_tuning",
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

# Best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_trial = tuner.oracle.get_best_trials(1)[0]
best_score = best_trial.score if hasattr(best_trial, 'score') else None

print(f"\nBest validation accuracy (tuner reported): {best_score}")
print(f"Best LR: {best_hps.get('learning_rate')}, Optimizer: {best_hps.get('optimizer')}, Dropout: {best_hps.get('dropout')}, Backbone trainable: {best_hps.get('backbone_trainable')}")

# Create final model with best hps
def create_final_model(hp):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling='avg'
    )
    base_model.trainable = bool(hp.get("backbone_trainable"))
    x = base_model.output
    x = tf.keras.layers.Dropout(hp.get("dropout"))(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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

# Callbacks (save weights under resNet)
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.weights.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max"
    )
]

# Train final model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Evaluate
print("Evaluating on test set...")
probs = model.predict(test_ds)
predicted_labels = tf.argmax(probs, axis=1).numpy()

# Extract true labels from test_ds
true_labels = []
for batch in test_ds.unbatch():
    lbl = batch[1].numpy()
    true_labels.append(lbl)
true_labels = np.array(true_labels)[:len(predicted_labels)]

# Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, zero_division=0)
recall = recall_score(true_labels, predicted_labels, zero_division=0)
f1 = f1_score(true_labels, predicted_labels, zero_division=0)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Venomous', 'Non-Venomous'],
            yticklabels=['Venomous', 'Non-Venomous'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
cm_path = os.path.join(RESNET_DIR, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.show()

# Plot training history
def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(RESNET_DIR, 'training_history.png')
    plt.savefig(hist_path)
    plt.show()

plot_training(history)

# Save results summary
results_path = os.path.join(RESNET_DIR, 'tuning_results.txt')
with open(results_path, "w") as f:
    f.write("===== BEST HYPERPARAMETERS =====\n")
    f.write(f"Best Validation Accuracy (tuner): {best_score}\n")
    f.write(f"Learning Rate: {best_hps.get('learning_rate')}\n")
    f.write(f"Optimizer: {best_hps.get('optimizer')}\n")
    f.write(f"Dropout: {best_hps.get('dropout')}\n")
    f.write(f"Backbone trainable: {best_hps.get('backbone_trainable')}\n\n")

    f.write("===== FINAL EVALUATION ON TEST SET =====\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n")
    f.write(f"Test F1-Score: {f1:.4f}\n\n")

    f.write("===== CONFUSION MATRIX =====\n")
    f.write(np.array2string(conf_matrix, separator=', ') + "\n\n")

    f.write("===== TRAINING HISTORY =====\n")
    epochs_done = len(history.history['accuracy'])
    for i in range(epochs_done):
        f.write(
            f"Epoch {i+1:03d}: "
            f"Train Acc={history.history['accuracy'][i]:.4f}, "
            f"Val Acc={history.history['val_accuracy'][i]:.4f}, "
            f"Train Loss={history.history['loss'][i]:.4f}, "
            f"Val Loss={history.history['val_loss'][i]:.4f}\n"
        )

    f.write("\n===== MODEL SUMMARY =====\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# Save model (Keras)
model_save_path = os.path.join(RESNET_DIR, "snake_resnet_model.keras")
model.save(model_save_path)

# Convert to TFLite
tflite_save_path = os.path.join(RESNET_DIR, "snake_resnet_model.tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(tflite_save_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Done! All outputs saved under: {RESNET_DIR}")
