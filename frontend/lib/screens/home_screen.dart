import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import '../services/snake_classifier.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  Uint8List? _imageBytes; // For web platform
  String? _prediction;
  double? _confidence;
  bool _isLoading = false;
  
  final ImagePicker _picker = ImagePicker();
  final SnakeClassifier _classifier = SnakeClassifier();

  @override
  void initState() {
    super.initState();
    _initializeClassifier();
  }

  Future<void> _initializeClassifier() async {
    try {
      await _classifier.loadModel();
      print('Model loaded successfully');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<void> _pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxHeight: 224,
        maxWidth: 224,
        imageQuality: 85,
      );
      
      if (image != null) {
        if (kIsWeb) {
          final Uint8List bytes = await image.readAsBytes();
          setState(() {
            _imageBytes = bytes;
            _image = null;
            _prediction = null;
            _confidence = null;
          });
        } else {
          setState(() {
            _image = File(image.path);
            _imageBytes = null;
            _prediction = null;
            _confidence = null;
          });
        }
        await _classifyImage();
      }
    } catch (e) {
      _showErrorDialog('Error picking image from gallery: $e');
    }
  }

  Future<void> _pickImageFromCamera() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        maxHeight: 224,
        maxWidth: 224,
        imageQuality: 85,
      );
      
      if (image != null) {
        if (kIsWeb) {
          final Uint8List bytes = await image.readAsBytes();
          setState(() {
            _imageBytes = bytes;
            _image = null;
            _prediction = null;
            _confidence = null;
          });
        } else {
          setState(() {
            _image = File(image.path);
            _imageBytes = null;
            _prediction = null;
            _confidence = null;
          });
        }
        await _classifyImage();
      }
    } catch (e) {
      _showErrorDialog('Error taking photo: $e');
    }
  }

  Future<void> _classifyImage() async {
    if (_image == null && _imageBytes == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      Map<String, dynamic> result;
      if (kIsWeb && _imageBytes != null) {
        // For web platform, use bytes
        result = await _classifier.classifyImageFromBytes(_imageBytes!);
      } else if (_image != null) {
        // For mobile/desktop platforms, use file
        result = await _classifier.classifyImage(_image!);
      } else {
        throw Exception('No image available for classification');
      }
      
      setState(() {
        _prediction = result['label'];
        _confidence = result['confidence'];
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showErrorDialog('Error classifying image: $e');
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Error'),
          content: Text(message),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        );
      },
    );
  }

  Color _getResultColor() {
    if (_prediction == null) return Colors.grey;
    return _prediction!.toLowerCase().contains('venomous') 
        ? Colors.red 
        : Colors.green;
  }

  IconData _getResultIcon() {
    if (_prediction == null) return Icons.help_outline;
    return _prediction!.toLowerCase().contains('venomous') 
        ? Icons.warning 
        : Icons.check_circle;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Snake Identifier',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green[600],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.green[600]!, Colors.green[50]!],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Header Card
                Card(
                  elevation: 4,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Icon(
                          Icons.camera_alt,
                          size: 48,
                          color: Colors.green[600],
                        ),
                        const SizedBox(height: 8),
                        const Text(
                          'Identifikasi Ular',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Ambil foto atau pilih dari galeri untuk mengidentifikasi jenis ular',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // Image Display Area
                Expanded(
                  flex: 3,
                  child: Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: Colors.grey[100],
                      ),
                      child: _image == null && _imageBytes == null
                          ? const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(
                                    Icons.image_outlined,
                                    size: 64,
                                    color: Colors.grey,
                                  ),
                                  SizedBox(height: 16),
                                  Text(
                                    'Belum ada gambar dipilih',
                                    style: TextStyle(
                                      fontSize: 16,
                                      color: Colors.grey,
                                    ),
                                  ),
                                  SizedBox(height: 8),
                                  Text(
                                    'Pilih gambar untuk memulai identifikasi',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: Colors.grey,
                                    ),
                                  ),
                                ],
                              ),
                            )
                          : ClipRRect(
                              borderRadius: BorderRadius.circular(12),
                              child: kIsWeb && _imageBytes != null
                                  ? Image.memory(
                                      _imageBytes!,
                                      fit: BoxFit.cover,
                                      width: double.infinity,
                                      height: double.infinity,
                                    )
                                  : _image != null
                                      ? Image.file(
                                          _image!,
                                          fit: BoxFit.cover,
                                          width: double.infinity,
                                          height: double.infinity,
                                        )
                                      : Container(
                                          color: Colors.grey[300],
                                          child: Center(
                                            child: Text('Image loading error'),
                                          ),
                                        ),
                            ),
                    ),
                  ),
                ),
                
                const SizedBox(height: 20),
                
                // Results Area
                if (_isLoading || _prediction != null)
                  Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: _isLoading
                          ? const Column(
                              children: [
                                CircularProgressIndicator(),
                                SizedBox(height: 16),
                                Text(
                                  'Menganalisis gambar...',
                                  style: TextStyle(fontSize: 16),
                                ),
                              ],
                            )
                          : Column(
                              children: [
                                Icon(
                                  _getResultIcon(),
                                  size: 48,
                                  color: _getResultColor(),
                                ),
                                const SizedBox(height: 12),
                                Text(
                                  'Hasil Identifikasi:',
                                  style: TextStyle(
                                    fontSize: 16,
                                    color: Colors.grey[600],
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  _prediction ?? '',
                                  style: TextStyle(
                                    fontSize: 20,
                                    fontWeight: FontWeight.bold,
                                    color: _getResultColor(),
                                  ),
                                  textAlign: TextAlign.center,
                                ),
                                if (_confidence != null) ...[
                                  const SizedBox(height: 8),
                                  Text(
                                    'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ],
                            ),
                    ),
                  ),
                
                const SizedBox(height: 20),
                
                // Action Buttons
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _pickImageFromCamera,
                        icon: const Icon(Icons.camera_alt),
                        label: const Text('Kamera'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green[600],
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _pickImageFromGallery,
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Galeri'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green[600],
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _classifier.dispose();
    super.dispose();
  }
}
