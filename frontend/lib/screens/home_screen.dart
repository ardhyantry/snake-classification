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
  Uint8List? _imageBytes; 
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

  Future<void> _pickImageFromCamera() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        // Tanpa batasan ukuran - pertahankan kualitas asli untuk tampilan UI
        // Preprocessing ke 224x224 dilakukan secara internal oleh classifier
      );
      
      if (image != null) {
        await _processImage(image);
      }
    } catch (e) {
      _showError('Error taking photo: $e');
    }
  }

  Future<void> _pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        // Tanpa batasan ukuran - pertahankan kualitas asli untuk tampilan UI
        // Preprocessing ke 224x224 dilakukan secara internal oleh classifier
      );
      
      if (image != null) {
        await _processImage(image);
      }
    } catch (e) {
      _showError('Error picking image: $e');
    }
  }

  Future<void> _processImage(XFile image) async {
    setState(() {
      _isLoading = true;
      _prediction = null;
      _confidence = null;
    });

    try {
      if (kIsWeb) {
        final Uint8List bytes = await image.readAsBytes();
        setState(() {
          _imageBytes = bytes;
          _image = null;
        });
      } else {
        setState(() {
          _image = File(image.path);
          _imageBytes = null;
        });
      }
      await _classifyImage();
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      _showError('Error processing image: $e');
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
        result = await _classifier.classifyImageFromBytes(_imageBytes!);
      } else if (_image != null) {
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
      _showError('Error classifying image: $e');
    }
  }

  void _showError(String message) {
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Bagian Header - Latar Belakang Hijau
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(24, 60, 24, 40),
              decoration: const BoxDecoration(
                color: Color(0xFFDDEEDC),
              ),
              child: Column(
                children: [
                  const Text(
                    'Snake Identifier',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF0F4D2C),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Identify venomous and non-venomous snakes using AI-powered image recognition',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 16,
                      color: Color(0xFF4A5F4A),
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),

            // Kontainer Konten Utama
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                children: [
                  // Kontainer Aksi Utama
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: const Color(0xFFE4E4E4),
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      children: [
                          const Flexible(
                            child: Text(
                              'Choose an option to get started',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w500,
                                color: Color(0xFF333333),
                              ),
                            ),
                          ),
                        const SizedBox(height: 32),

                        // Tombol Ambil Foto
                        SizedBox(
                          width: double.infinity,
                          height: 58,
                          child: ElevatedButton(
                            onPressed: _isLoading ? null : _pickImageFromCamera,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF3D8B4E),
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 2,
                              disabledBackgroundColor: Colors.grey,
                            ),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.camera_alt, size: 24),
                                SizedBox(width: 12),
                                Text(
                                  'Take a photo',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                        const SizedBox(height: 20),

                        // Teks ATAU
                        const Text(
                          'OR',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFF3D8B4E),
                          ),
                        ),

                        const SizedBox(height: 20),

                        // Tombol Unggah dari Galeri
                        SizedBox(
                          width: double.infinity,
                          height: 58,
                          child: ElevatedButton(
                            onPressed: _isLoading ? null : _pickImageFromGallery,
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF607065),
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 2,
                              disabledBackgroundColor: Colors.grey,
                            ),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.upload_file, size: 24),
                                SizedBox(width: 12),
                                Flexible(
                                  child: Text(
                                    'Upload from Gallery',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w600,
                                    ),
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 32),

                  // Pratinjau Gambar (jika tersedia)
                  if (_image != null || _imageBytes != null) ...[
                    ConstrainedBox(
                      constraints: BoxConstraints(
                        maxHeight: MediaQuery.of(context).size.height * 0.4,
                      ),
                      child: Container(
                        width: double.infinity,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(16),
                          color: Colors.grey[200],
                          boxShadow: [
                            BoxShadow(
                              color: Colors.grey.withOpacity(0.3),
                              spreadRadius: 2,
                              blurRadius: 8,
                              offset: const Offset(0, 2),
                            ),
                          ],
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(16),
                          child: kIsWeb && _imageBytes != null
                              ? Image.memory(_imageBytes!, fit: BoxFit.contain)
                              : _image != null
                                  ? Image.file(_image!, fit: BoxFit.contain)
                                  : const SizedBox(),
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                  ],

                  // Memuat atau Hasil
                  if (_isLoading)
                    Container(
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.grey.withOpacity(0.2),
                            spreadRadius: 2,
                            blurRadius: 8,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: const Column(
                        children: [
                          CircularProgressIndicator(
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Color(0xFF3D8B4E),
                            ),
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Analyzing image...',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),

                  if (_prediction != null && !_isLoading) ...[
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.grey.withOpacity(0.2),
                            spreadRadius: 2,
                            blurRadius: 8,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          Icon(
                            _prediction!.toLowerCase().contains('non')
                                ? Icons.check_circle_rounded
                                : Icons.warning_rounded,
                            size: 64,
                            color: _prediction!.toLowerCase().contains('non')
                                ? const Color(0xFF3D8B4E)
                                : Colors.red,
                          ),
                          const SizedBox(height: 16),
                          const Text(
                            'Result:',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Flexible(
                            child: Text(
                              _prediction!,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: _prediction!.toLowerCase().contains('non')
                                    ? const Color(0xFF3D8B4E)
                                    : Colors.red,
                              ),
                              overflow: TextOverflow.ellipsis,
                              maxLines: 2,
                            ),
                          ),
                          if (_confidence != null) ...[
                            const SizedBox(height: 12),
                            Text(
                              'Confidence: ${(_confidence! * 100).toStringAsFixed(1)}%',
                              style: const TextStyle(
                                fontSize: 16,
                                color: Colors.grey,
                              ),
                            ),
                          ],
                        ],
                      ),
                    ),
                    const SizedBox(height: 24),
                  ],

                  // Kotak Info Bawah
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 16,
                    ),
                    decoration: BoxDecoration(
                      color: const Color(0xFFDDDDDD),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Text(
                      'For best results, ensure the snake is clearly visible and well-lit',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 13,
                        color: Color(0xFF555555),
                        height: 1.4,
                      ),
                    ),
                  ),

                  const SizedBox(height: 24),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
