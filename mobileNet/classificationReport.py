import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 2
DATASET_PATH = '/Users/ardhyantry/Documents/GitHub/snake-classification/dataset ularBaru'
WEIGHTS_PATH = '/Users/ardhyantry/Documents/GitHub/snake-classification/mobileNet/checkpoints/best_model.weights.h5'

# Class names
class_names = ['Venomous', 'Non-Venomous']

# Preprocessing function for MobileNetV3
def preprocess_image(image_path, label):
    def decode_and_process_image(img_path):
        file_path = img_path.numpy().decode('utf-8')
        try:
            image = Image.open(file_path).convert("RGB")
            image = image.resize(IMG_SIZE, Image.BICUBIC)
            img_array = np.array(image).astype(np.float32)
            # MobileNet V3 preprocessing
            img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
            return img_array
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

    processed_image = tf.py_function(
        func=decode_and_process_image,
        inp=[image_path],
        Tout=tf.float32
    )
    processed_image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
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

# Load test dataset
print("Loading test dataset...")
venomous_dir = os.path.join(DATASET_PATH, 'venomous')
non_venomous_dir = os.path.join(DATASET_PATH, 'non_venomous')

venomous_files = [os.path.join(venomous_dir, f) for f in os.listdir(venomous_dir)]
non_venomous_files = [os.path.join(non_venomous_dir, f) for f in os.listdir(non_venomous_dir)]

venomous_files = [f for f in venomous_files if is_valid_image(f)]
non_venomous_files = [f for f in non_venomous_files if is_valid_image(f)]

all_files = venomous_files + non_venomous_files
all_labels = [0] * len(venomous_files) + [1] * len(non_venomous_files)

# Shuffle with seed for reproducibility
np.random.seed(42)
indices = np.random.permutation(len(all_files))
all_files = [all_files[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# Split (using same ratios as training: 70/15/15)
n = len(all_files)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

test_files = all_files[n_train + n_val:]
test_labels = all_labels[n_train + n_val:]

print(f"Test set size: {len(test_files)}")

# Create test dataset
test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(16).prefetch(tf.data.AUTOTUNE)

# Rebuild MobileNetV3 model architecture
print("Rebuilding MobileNetV3Large model architecture...")
inputs = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)

# Load MobileNetV3Large backbone
base_model = tf.keras.applications.MobileNetV3Large(
    include_top=False,
    weights='imagenet',
    input_tensor=inputs,
    pooling='avg'
)

# Use the best hyperparameters from tuning_results.txt
base_model.trainable = True  # backbone trainable = 1
x = base_model.output

# Dropout from best hyperparameters (0.0 - no dropout)
# Since dropout is 0.0, we skip the dropout layer
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Load weights
print(f"Loading weights from {WEIGHTS_PATH}...")
model.load_weights(WEIGHTS_PATH)

print("Compiling model...")
# Use the optimizer from best hyperparameters (adam with lr=0.0001)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

# Make predictions
print("Making predictions on test set...")
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Extract true labels
y_test = []
for _, label in test_ds.unbatch():
    y_test.append(label.numpy())
y_test = np.array(y_test)[:len(y_pred)]

# Generate classification report
report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Additional metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print to console
print("\n" + "="*60)
print("MOBILENETV3LARGE - CLASSIFICATION REPORT")
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

# Save to text file
output_file = 'mobileNet/classification_report.txt'
with open(output_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("MOBILENETV3LARGE - SNAKE CLASSIFICATION TEST SET EVALUATION\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Dataset: {DATASET_PATH}\n")
    f.write(f"Model Weights: {WEIGHTS_PATH}\n")
    f.write(f"Model Architecture: MobileNetV3Large (ImageNet pretrained)\n")
    f.write(f"Test Set Size: {len(y_test)} images\n")
    f.write(f"  - Venomous: {np.sum(y_test == 0)}\n")
    f.write(f"  - Non-Venomous: {np.sum(y_test == 1)}\n\n")
    
    f.write("Best Hyperparameters:\n")
    f.write("  - Learning Rate: 0.0001\n")
    f.write("  - Optimizer: Adam\n")
    f.write("  - Dropout: 0.0 (No dropout)\n")
    f.write("  - Backbone Trainable: Yes\n\n")
    
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
    f.write("SUMMARY METRICS\n")
    f.write("="*60 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    f.write("="*60 + "\n\n")
    
    f.write("="*60 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*60 + "\n")
    f.write(f"True Positives (Venomous correctly identified): {conf_matrix[0][0]}\n")
    f.write(f"True Negatives (Non-Venomous correctly identified): {conf_matrix[1][1]}\n")
    f.write(f"False Positives (Non-Venomous misclassified as Venomous): {conf_matrix[1][0]}\n")
    f.write(f"False Negatives (Venomous misclassified as Non-Venomous): {conf_matrix[0][1]}\n\n")
    
    f.write(f"Total Correct Predictions: {conf_matrix[0][0] + conf_matrix[1][1]}/{len(y_test)}\n")
    f.write(f"Total Incorrect Predictions: {conf_matrix[0][1] + conf_matrix[1][0]}/{len(y_test)}\n")

print(f"\n✅ Classification report saved to '{output_file}'")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('MobileNetV3Large - Confusion Matrix (Test Set)')
plt.tight_layout()
plt.savefig('mobileNet/test_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved as 'mobileNet/test_confusion_matrix.png'")
plt.show()

print("\n" + "="*60)
print("✅ MobileNetV3Large evaluation completed successfully!")
print("="*60)
