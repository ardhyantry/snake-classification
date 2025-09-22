import tensorflow as tf

#load model tflite model
interpreter = tf.lite.Interpreter(model_path="/Users/ardhyantry/Documents/GitHub/snake-classification/snake_vit_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

#load image