import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

// Import kondisional untuk TensorFlow Lite
// Hanya import tflite_flutter jika tidak di web
// ignore: avoid_web_libraries_in_flutter
import 'package:tflite_flutter/tflite_flutter.dart' if (dart.library.html) 'stub_tflite.dart';

class SnakeClassifier {
  Interpreter? _interpreter;
  late List<String> _labels;
  bool _isModelLoaded = false;

  // Ukuran input model (sesuaikan dengan model Anda)
  static const int inputSize = 224;
  static const int numChannels = 3;
  
  // Konfigurasi ViT Image Processor
  static const double rescaleFactor = 0.00392156862745098; // 1/255
  static const List<double> imageMean = [0.5, 0.5, 0.5];
  static const List<double> imageStd = [0.5, 0.5, 0.5];

  Future<void> loadModel() async {
    try {
      // Periksa apakah berjalan di platform web
      if (kIsWeb) {
        print('Running on web platform - TensorFlow Lite not supported');
        print('Using simulated model for demonstration');
        _labels = await _loadLabels();
        _isModelLoaded = true;
        return;
      }
      
      // Muat model TFLite untuk platform non-web
      _interpreter = await Interpreter.fromAsset('assets/models/snake_vit_model.tflite');
      
      // Muat label
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
      
      // Muat label meskipun model gagal
      _labels = await _loadLabels();
      _isModelLoaded = true; // Izinkan aplikasi melanjutkan dengan simulasi
    }
  }

  Future<List<String>> _loadLabels() async {
    try {
      final String labelData = await rootBundle.loadString('assets/labels/snake_labels.txt');
      return labelData.split('\n').where((label) => label.isNotEmpty).toList();
    } catch (e) {
      print('Error loading labels: $e');
      // Kembalikan label default jika file tidak ada
      return ['Venomous Snake', 'Non-Venomous Snake'];
    }
  }

  Future<Map<String, dynamic>> classifyImage(File imageFile) async {
    if (!_isModelLoaded) {
      throw Exception('Model not loaded. Call loadModel() first.');
    }

    try {
      // Baca dan preproses gambar
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Ubah ukuran gambar ke ukuran input model
      final img.Image resizedImage = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
      );

      // Gunakan model TensorFlow Lite
      if (_interpreter != null && !kIsWeb) {
        return await _runTFLiteInference(resizedImage);
      } else {
        throw Exception('TFLite model not available on this platform');
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
      // Dekode gambar dari bytes
      final img.Image? image = img.decodeImage(imageBytes);
      
      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Ubah ukuran gambar ke ukuran input model
      final img.Image resizedImage = img.copyResize(
        image,
        width: inputSize,
        height: inputSize,
      );

      // Gunakan model TensorFlow Lite
      if (_interpreter != null && !kIsWeb) {
        return await _runTFLiteInference(resizedImage);
      } else {
        throw Exception('TFLite model not available on this platform');
      }
    } catch (e) {
      print('Error during image classification: $e');
      throw Exception('Classification failed: $e');
    }
  }

  Future<Map<String, dynamic>> _runTFLiteInference(img.Image image) async {
    try {
      // Konversi gambar ke tensor input dengan bentuk [1, 3, 224, 224] (channels first untuk ViT)
      final input = _imageToChannelsFirst(image);

      // Siapkan tensor output - buat list 2D untuk TensorFlow Lite
      final output = List.generate(1, (index) => List.filled(_labels.length, 0.0));

      // Jalankan inferensi dengan bentuk tensor yang benar
      _interpreter!.run(input, output);

      // Dapatkan hasil prediksi dari batch pertama
      final List<double> probabilities = List<double>.from(output[0]);
      
      // Temukan kelas dengan probabilitas tertinggi
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
      print('TFLite inference failed: $e');
      rethrow;
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

  /// Konversi gambar ke format channels-first [1, 3, 224, 224] untuk model ViT
  List<List<List<List<double>>>> _imageToChannelsFirst(img.Image image) {
    // Buat tensor 4D dengan bentuk [1, 3, 224, 224]
    final tensor = List.generate(
      1, // ukuran batch
      (_) => List.generate(
        numChannels, // channel (R, G, B)
        (_) => List.generate(
          inputSize, // tinggi
          (_) => List.filled(inputSize, 0.0), // lebar
        ),
      ),
    );

    print('\n=== PREPROCESSING ViT (Channels First) ===');
    print('Bentuk input: [1, 3, 224, 224]');
    print('Faktor rescale: $rescaleFactor');
    print('Mean gambar: $imageMean');
    print('Std gambar: $imageStd');
    print('==========================================\n');

    for (int h = 0; h < inputSize; h++) {
      for (int w = 0; w < inputSize; w++) {
        final pixel = image.getPixel(w, h);
        
        // Langkah 1: Rescale nilai pixel (bagi dengan 255)
        double r = pixel.r * rescaleFactor;
        double g = pixel.g * rescaleFactor;
        double b = pixel.b * rescaleFactor;
        
        // Langkah 2: Normalisasi dengan mean dan std: (pixel - mean) / std
        // Format channels first: tensor[batch][channel][tinggi][lebar]
        tensor[0][0][h][w] = (r - imageMean[0]) / imageStd[0]; // Channel merah
        tensor[0][1][h][w] = (g - imageMean[1]) / imageStd[1]; // Channel hijau
        tensor[0][2][h][w] = (b - imageMean[2]) / imageStd[2]; // Channel biru
      }
    }

    return tensor;
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
