from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf

model = load_model("coba/snake_vit_model.keras")
logits = model.predict(test_ds)
predicted_labels = tf.argmax(logits, axis=1).numpy()

true_labels = []
for _, label in test_ds.unbatch():
    true_labels.append(label.numpy())
true_labels = np.array(true_labels)[:len(predicted_labels)]

accuracy = accuracy_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
