import tensorflow as tf
import numpy as np
from transformers import ViTImageProcessor

# Load kembali processor dan model
processor = ViTImageProcessor.from_pretrained("C:\\Users\\Ardhyan\\Documents\\Code TA\\vit_base_patch16_224")
model = tf.keras.models.load_model("snake_vit_model.keras", compile=False)

# Fungsi prediksi dengan output probabilitas
def predict_image(image_path):
    # Load & decode image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0

    # Convert to numpy for processor
    image_np = image.numpy()

    # Preprocess with ViT processor
    inputs = processor(images=image_np, return_tensors="np", do_rescale=False)
    pixel_values = inputs["pixel_values"]  # shape: (1, 3, 224, 224)

    # Run prediction
    logits = model(pixel_values, training=False).logits.numpy()  # shape: (1, num_classes)

    # Convert logits to probabilities using softmax
    probs = tf.nn.softmax(logits, axis=1).numpy()

    # Ambil kelas prediksi dan confidence
    predicted_class = np.argmax(probs, axis=1)[0]
    confidence = probs[0][predicted_class]

    class_names = ['Venomous', 'Non-Venomous']
    print(f"🧪 Prediksi: **{class_names[predicted_class]}**")
    print(f"📊 Probabilitas: {probs}")
    print(f"✅ Confidence: {confidence:.4f}")

predict_image("C:\\Users\\Ardhyan\\Documents\\Code TA\\DATASET\\VENOMOUS\\ophiophagus_100.jpg")
