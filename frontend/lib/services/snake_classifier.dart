import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

// Conditional import untuk TensorFlow Lite
// Hanya import tflite_flutter jika tidak di web
// ignore: avoid_web_libraries_in_flutter
import 'package:tflite_flutter/tflite_flutter.dart' if (dart.library.html) 'stub_tflite.dart';

class SnakeClassifier {
  Interpreter? _interpreter;
  late List<String> _labels;
  bool _isModelLoaded = false;

  // Model input shape (adjust based on your model)
  static const int inputSize = 224;
  static const int numChannels = 3;
  
  // ViT Image Processor configuration
  static const double rescaleFactor = 0.00392156862745098; // 1/255
  static const List<double> imageMean = [0.5, 0.5, 0.5];
  static const List<double> imageStd = [0.5, 0.5, 0.5];

  Future<void> loadModel() async {
    try {
      // Check if running on web platform
      if (kIsWeb) {
        print('Running on web platform - TensorFlow Lite not supported');
        print('Using simulated model for demonstration');
        _labels = await _loadLabels();
        _isModelLoaded = true;
        return;
      }
      
      // Load the TFLite model for non-web platforms
      _interpreter = await Interpreter.fromAsset('assets/models/snake_vit_model.tflite');
      
      // Load labels
      _labels = await _loadLabels();
      
      _isModelLoaded = true;
      print('Snake classifier model loaded successfully');
      if (_interpreter != null) {
        print('Input shape: ${_interpreter!.getInputTensor(0).shape}');
        print('Output shape: ${_interpreter!.getOutputTensor(0).shape}');
      }
    } catch (e) {
      print('Error loading TFLite model: $e');
      print('Using fallback simulated model for demonstration');
      
      // Load labels even if model fails
      _labels = await _loadLabels();
      _isModelLoaded = true; // Allow app to continue with simulation
    }
  }

  Future<List<String>> _loadLabels() async {
    try {
      final String labelData = await rootBundle.loadString('assets/labels/snake_labels.txt');
      return labelData.split('\n').where((label) => label.isNotEmpty).toList();
    } catch (e) {
      print('Error loading labels: $e');
      // Return default labels if file doesn't exist
      return ['Venomous Snake', 'Non-Venomous Snake'];
    }
  }

  Future<Map<String, dynamic>> classifyImage(File imageFile) async {
    if (!_isModelLoaded) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    try {
      // Read and preprocess the image
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Resize image to model input size
      final img.Image resizedImage = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
      );

      // Try to use real TensorFlow Lite model if available and not on web
      if (_interpreter != null && !kIsWeb) {
        return await _runTFLiteInference(resizedImage);
      } else {
        // Fallback to simulation for web platform or when TFLite model is not available
        String reason = kIsWeb ? 'Web platform' : 'TFLite model not available';
        print('Using simulated classification ($reason)');
        return _simulateInference(resizedImage);
      }
    } catch (e) {
      print('Error during image classification: $e');
      throw Exception('Classification failed: $e');
    }
  }

  Future<Map<String, dynamic>> classifyImageFromBytes(Uint8List imageBytes) async {
    if (!_isModelLoaded) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    try {
      // Decode image from bytes
      final img.Image? image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Resize image to model input size
      final img.Image resizedImage = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
      );

      // Try to use real TensorFlow Lite model if available and not on web
      if (_interpreter != null && !kIsWeb) {
        return await _runTFLiteInference(resizedImage);
      } else {
        // Fallback to simulation for web platform or when TFLite model is not available
        String reason = kIsWeb ? 'Web platform' : 'TFLite model not available';
        print('Using simulated classification ($reason)');
        return _simulateInference(resizedImage);
      }
    } catch (e) {
      print('Error during image classification: $e');
      throw Exception('Classification failed: $e');
    }
  }

  Future<Map<String, dynamic>> _runTFLiteInference(img.Image image) async {
    try {
      // Convert image to input tensor
      final Float32List input = _imageToByteList(image);

      // Prepare output tensor - create 2D list for TensorFlow Lite
      final output = List.generate(1, (index) => List.filled(_labels.length, 0.0));

      // Run inference with proper tensor shapes
      _interpreter!.run([input], output);

      // Get prediction results from first batch
      final List<double> probabilities = List<double>.from(output[0]);
      
      // Find the class with highest probability
      int maxIndex = 0;
      double maxProbability = probabilities[0];
      
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          maxIndex = i;
        }
      }

      final String predictedLabel = maxIndex < _labels.length 
          ? _labels[maxIndex] 
          : 'Unknown';

      return {
        'label': predictedLabel,
        'confidence': maxProbability,
        'probabilities': probabilities,
        'all_results': List.generate(_labels.length, (index) => {
          'label': _labels[index],
          'confidence': probabilities[index],
        }),
        'method': 'TensorFlow Lite',
      };
    } catch (e) {
      print('TFLite inference failed: $e, falling back to simulation');
      return _simulateInference(image);
    }
  }

  Float32List _imageToByteList(img.Image image) {
    final Float32List convertedBytes = Float32List(inputSize * inputSize * numChannels);
    final buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    print('\n=== ViT PREPROCESSING ===');
    print('Rescale factor: $rescaleFactor');
    print('Image mean: $imageMean');
    print('Image std: $imageStd');
    print('========================\n');

    for (int i = 0; i < inputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        final pixel = image.getPixel(j, i);
        
        // Step 1: Rescale pixel values (divide by 255)
        double r = pixel.r * rescaleFactor;
        double g = pixel.g * rescaleFactor;
        double b = pixel.b * rescaleFactor;
        
        // Step 2: Normalize with mean and std: (pixel - mean) / std
        buffer[pixelIndex++] = (r - imageMean[0]) / imageStd[0];
        buffer[pixelIndex++] = (g - imageMean[1]) / imageStd[1];
        buffer[pixelIndex++] = (b - imageMean[2]) / imageStd[2];
      }
    }

    return convertedBytes;
  }

  Map<String, dynamic> _simulateInference(img.Image image) {
    // This is a fallback simulation when TFLite model is not available
    // In production, always use the real TensorFlow Lite model
    
    // Calculate some basic image features for simulation
    int totalRed = 0, totalGreen = 0, totalBlue = 0;
    int pixelCount = 0;
    
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        totalRed += pixel.r.toInt();
        totalGreen += pixel.g.toInt();
        totalBlue += pixel.b.toInt();
        pixelCount++;
      }
    }
    
    // Calculate average colors
    double avgRed = totalRed / pixelCount;
    double avgGreen = totalGreen / pixelCount;
    double avgBlue = totalBlue / pixelCount;
    
    // Simple heuristic for demonstration (not actual snake detection)
    // This is just for UI testing - replace with real model
    double brightness = (avgRed + avgGreen + avgBlue) / 3;
    double colorVariance = ((avgRed - brightness).abs() + 
                           (avgGreen - brightness).abs() + 
                           (avgBlue - brightness).abs()) / 3;
    
    // Simulate probability based on image characteristics
    Random random = Random();
    double baseProb = (brightness + colorVariance) / 500.0;
    baseProb = baseProb.clamp(0.1, 0.9);
    
    // Add some randomness for simulation
    double venomousProb = baseProb + (random.nextDouble() - 0.5) * 0.3;
    venomousProb = venomousProb.clamp(0.1, 0.9);
    double nonVenomousProb = 1.0 - venomousProb;

    List<double> probabilities = [venomousProb, nonVenomousProb];
    
    // Find the class with highest probability
    int maxIndex = venomousProb > nonVenomousProb ? 0 : 1;
    double maxProbability = probabilities[maxIndex];

    final String predictedLabel = maxIndex < _labels.length 
        ? _labels[maxIndex] 
        : 'Unknown';

    return {
      'label': predictedLabel,
      'confidence': maxProbability,
      'probabilities': probabilities,
      'all_results': List.generate(_labels.length, (index) => {
        'label': _labels[index],
        'confidence': probabilities[index],
      }),
      'note': 'This is a simulated result for demonstration. Replace with actual TensorFlow Lite model.',
    };
  }

  void dispose() {
    if (_interpreter != null) {
      _interpreter!.close();
      _interpreter = null;
    }
    _isModelLoaded = false;
  }

  bool get isModelLoaded => _isModelLoaded;
  List<String> get labels => _labels;
}
