# Script to visualize data augmentation output for stratified model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import os
import glob

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE = (224, 224)
DATASET_PATH = '/Users/ardhyantry/Documents/GitHub/snake-classification/dataset ularBaru'

def load_sample_images(data_dir, num_samples=2):
    """Load sample images from each class"""
    venomous_dir = os.path.join(data_dir, 'venomous')
    non_venomous_dir = os.path.join(data_dir, 'non_venomous')
    
    venomous_files = glob.glob(os.path.join(venomous_dir, '*.*'))[:num_samples]
    non_venomous_files = glob.glob(os.path.join(non_venomous_dir, '*.*'))[:num_samples]
    
    return venomous_files + non_venomous_files

def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(IMG_SIZE, Image.BICUBIC)
    img_array = np.array(image).astype(np.float32) / 255.0
    return img_array

def apply_zoom(image, zoom_factor):
    """Apply zoom augmentation (same as in model.py)"""
    h, w = IMG_SIZE[0], IMG_SIZE[1]
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)
    
    # Resize to zoomed size
    img_tensor = tf.image.resize(image, [new_h, new_w])
    # Crop or pad back to original size
    img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, h, w)
    return img_tensor.numpy()

def apply_rotation(image, angle):
    """Apply rotation augmentation (same as in model.py: -20 to +20 degrees)"""
    rotated = ndimage.rotate(image, angle, reshape=False, mode='nearest')
    return rotated

def apply_flip(image):
    """Apply horizontal flip (same as in model.py)"""
    return np.fliplr(image)

def visualize_augmentations(image_path, save_path='augmentation_output.jpg'):
    """Visualize all augmentations applied to a single image
    Based on augment() function in model.py:
    - Random zoom between 0.7 and 1.0
    - Random rotation between -20 and +20 degrees
    - Random horizontal flip
    """
    
    # Load original image
    original = load_image(image_path)
    
    # Create figure with 2 rows x 4 columns (8 panels, 1 empty)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original, Zoom 0.7, Zoom 0.8, Zoom 0.9
    axes[0, 0].imshow(original)
    axes[0, 0].set_xlabel('(a) Original', fontsize=20)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    
    zoomed_70 = apply_zoom(original, 0.7)
    axes[0, 1].imshow(zoomed_70)
    axes[0, 1].set_xlabel('(b) Zoom 0.7', fontsize=20)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    
    zoomed_80 = apply_zoom(original, 0.8)
    axes[0, 2].imshow(zoomed_80)
    axes[0, 2].set_xlabel('(c) Zoom 0.8', fontsize=20)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    
    zoomed_90 = apply_zoom(original, 0.9)
    axes[0, 3].imshow(zoomed_90)
    axes[0, 3].set_xlabel('(d) Zoom 0.9', fontsize=20)
    axes[0, 3].set_xticks([])
    axes[0, 3].set_yticks([])
    
    # Row 2: Rotation -20°, Rotation +20°, Horizontal Flip, empty
    rotated_neg = apply_rotation(original, -20)
    axes[1, 0].imshow(np.clip(rotated_neg, 0, 1))
    axes[1, 0].set_xlabel('(e) Rotation -20°', fontsize=20)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    rotated_pos = apply_rotation(original, 20)
    axes[1, 1].imshow(np.clip(rotated_pos, 0, 1))
    axes[1, 1].set_xlabel('(f) Rotation +20°', fontsize=20)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    
    flipped = apply_flip(original)
    axes[1, 2].imshow(flipped)
    axes[1, 2].set_xlabel('(g) Horizontal Flip', fontsize=20)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    
    # Hide the last empty panel
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', format='jpg')
    plt.show()
    print(f"Saved augmentation visualization to: {save_path}")

def visualize_multiple_samples(data_dir, num_samples=2, save_path='augmentation_samples.jpg'):
    """Visualize augmentations for multiple sample images (venomous and non-venomous)"""
    
    sample_files = load_sample_images(data_dir, num_samples)
    
    if len(sample_files) == 0:
        print("No images found in dataset directory!")
        return
    
    num_images = len(sample_files)
    num_augs = 6  # Original + 5 augmentations
    
    fig, axes = plt.subplots(num_images, num_augs, figsize=(18, 3 * num_images))
    fig.suptitle('Data Augmentation Samples (Stratified Model)', fontsize=16, y=1.02)
    
    aug_names = ['Original', 'Zoom 70%', 'Rotation +15°', 'H-Flip', 'Zoom+Rot', 'Combined']
    
    for i, img_path in enumerate(sample_files):
        # Load original
        original = load_image(img_path)
        
        # Get class name from path
        class_name = 'Venomous' if 'venomous' in img_path.lower() and 'non' not in img_path.lower() else 'Non-Venomous'
        
        # Original
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f'{class_name}\n{aug_names[0]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Zoom
        zoomed = apply_zoom(original, 0.7)
        axes[i, 1].imshow(zoomed)
        axes[i, 1].set_title(aug_names[1], fontsize=10)
        axes[i, 1].axis('off')
        
        # Rotation
        rotated = apply_rotation(original, 15)
        axes[i, 2].imshow(np.clip(rotated, 0, 1))
        axes[i, 2].set_title(aug_names[2], fontsize=10)
        axes[i, 2].axis('off')
        
        # Horizontal Flip
        flipped = apply_flip(original)
        axes[i, 3].imshow(flipped)
        axes[i, 3].set_title(aug_names[3], fontsize=10)
        axes[i, 3].axis('off')
        
        # Zoom + Rotation
        zoom_rot = apply_zoom(original, 0.85)
        zoom_rot = apply_rotation(zoom_rot, 10)
        axes[i, 4].imshow(np.clip(zoom_rot, 0, 1))
        axes[i, 4].set_title(aug_names[4], fontsize=10)
        axes[i, 4].axis('off')
        
        # Combined augmentation
        combined = apply_zoom(original, 0.8)
        combined = apply_rotation(combined, -12)
        combined = apply_flip(combined)
        axes[i, 5].imshow(np.clip(combined, 0, 1))
        axes[i, 5].set_title(aug_names[5], fontsize=10)
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', format='jpg')
    plt.show()
    print(f"Saved multi-sample augmentation to: {save_path}")

def main():
    print("=" * 60)
    print("Data Augmentation Visualization")
    print("=" * 60)
    print("\nAugmentations used in training:")
    print("  - Random Zoom: 70% to 100%")
    print("  - Random Rotation: -20° to +20°")
    print("  - Random Horizontal Flip")
    print("")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at: {DATASET_PATH}")
        print("Please update DATASET_PATH variable.")
        return
    
    # Get sample image (only 1)
    sample_files = load_sample_images(DATASET_PATH, num_samples=1)
    
    if len(sample_files) == 0:
        print("No images found!")
        return
    
    print(f"Using image: {sample_files[0]}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Show single image with all augmentations
    print("\nVisualizing augmentations...")
    visualize_augmentations(
        sample_files[0], 
        save_path=os.path.join(script_dir, 'augmentation_output.jpg')
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
