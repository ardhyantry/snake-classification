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

  Widget _buildFirstAidItem({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(icon, color: color, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                description,
                style: const TextStyle(
                  fontSize: 13,
                  color: Color(0xFF555555),
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _showError(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Kesalahan'),
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
                    'Identifikasi Ular',
                    style: TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Color(0xFF0F4D2C),
                    ),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Identifikasi ular berbisa dan tidak berbisa menggunakan teknologi AI',
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
                          const Text(
                            'Pilih opsi untuk memulai',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w500,
                              color: Color(0xFF333333),
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
                            child: const FittedBox(
                              fit: BoxFit.scaleDown,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.camera_alt, size: 24),
                                  SizedBox(width: 12),
                                  Text(
                                    'Ambil Foto',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 20),

                        // Teks ATAU
                        const Text(
                          'ATAU',
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
                            child: const FittedBox(
                              fit: BoxFit.scaleDown,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Icon(Icons.upload_file, size: 24),
                                  SizedBox(width: 12),
                                  Text(
                                    'Unggah dari Galeri',
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ],
                              ),
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
                            'Menganalisis gambar...',
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
                            'Hasil:',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.grey,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            _prediction!.toLowerCase().contains('non') ? 'Tidak Berbisa' : 'Berbisa',
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
                          if (_confidence != null) ...[
                            const SizedBox(height: 12),
                            Text(
                              'Akurasi: ${(_confidence! * 100).toStringAsFixed(1)}%',
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

                    // Panduan Pertolongan Pertama
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: _prediction!.toLowerCase().contains('non')
                            ? const Color(0xFFE8F5E9)
                            : const Color(0xFFFFEBEE),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: _prediction!.toLowerCase().contains('non')
                              ? const Color(0xFF4CAF50)
                              : const Color(0xFFE53935),
                          width: 1.5,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                Icons.medical_services_rounded,
                                color: _prediction!.toLowerCase().contains('non')
                                    ? const Color(0xFF2E7D32)
                                    : const Color(0xFFC62828),
                                size: 28,
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  'Panduan Pertolongan Pertama',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: _prediction!.toLowerCase().contains('non')
                                        ? const Color(0xFF2E7D32)
                                        : const Color(0xFFC62828),
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          
                          if (_prediction!.toLowerCase().contains('non')) ...[
                            // Panduan untuk ular tidak berbisa
                            _buildFirstAidItem(
                              icon: Icons.water_drop,
                              title: 'Cuci Luka',
                              description: 'Cuci luka dengan air mengalir dan sabun untuk mengurangi kuman dari mulut ular.',
                              color: const Color(0xFF2E7D32),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.healing,
                              title: 'Hentikan Perdarahan',
                              description: 'Tekan dengan kain bersih atau kasa steril untuk menghentikan perdarahan ringan.',
                              color: const Color(0xFF2E7D32),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.medical_services,
                              title: 'Tutup Luka',
                              description: 'Tutup luka dengan perban bersih. Jaga luka tetap kering dan bersih.',
                              color: const Color(0xFF2E7D32),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.visibility,
                              title: 'Amati Tanda Infeksi',
                              description: 'Perhatikan kemerahan meluas, nyeri bertambah, bengkak, atau keluar nanah.',
                              color: const Color(0xFF2E7D32),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.local_hospital,
                              title: 'Periksa ke Faskes',
                              description: 'Kunjungi fasilitas kesehatan untuk evaluasi vaksin tetanus dan antibiotik bila ada infeksi.',
                              color: const Color(0xFF2E7D32),
                            ),
                          ] else ...[
                            // Panduan untuk ular berbisa - DARURAT
                            Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: const Color(0xFFC62828),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Row(
                                children: [
                                  Icon(Icons.warning_amber_rounded, color: Colors.white, size: 24),
                                  SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      'SEGERA BAWA KE IGD TERDEKAT!',
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: 16),
                            
                            // Langkah yang Dianjurkan
                            const Text(
                              'Yang HARUS Dilakukan:',
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFFC62828),
                              ),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.self_improvement,
                              title: 'Tenangkan Korban',
                              description: 'Kepanikan meningkatkan denyut jantung dan mempercepat penyebaran bisa melalui aliran darah.',
                              color: const Color(0xFFC62828),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.accessibility_new,
                              title: 'Imobilisasi',
                              description: 'Posisikan anggota tubuh yang tergigit sejajar atau lebih rendah dari jantung. Gunakan bidai sederhana. Jangan menggerakkan area gigitan.',
                              color: const Color(0xFFC62828),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.watch_off,
                              title: 'Lepas Benda Melilit',
                              description: 'Lepas cincin, gelang, sepatu, atau jam karena pembengkakan dapat terjadi cepat.',
                              color: const Color(0xFFC62828),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.water_drop,
                              title: 'Bersihkan Luka Ringan',
                              description: 'Cukup dengan air bersih, tanpa digosok atau ditekan.',
                              color: const Color(0xFFC62828),
                            ),
                            const SizedBox(height: 12),
                            _buildFirstAidItem(
                              icon: Icons.local_hospital,
                              title: 'Transportasi Segera',
                              description: 'Bawa ke fasilitas kesehatan untuk observasi klinis dan pemberian antivenom.',
                              color: const Color(0xFFC62828),
                            ),
                            
                            const SizedBox(height: 20),
                            
                            // Tindakan yang DILARANG
                            Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.grey[100],
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: const Color(0xFFC62828)),
                              ),
                              child: const Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Yang DILARANG:',
                                    style: TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                      color: Color(0xFFC62828),
                                    ),
                                  ),
                                  SizedBox(height: 8),
                                  Text('❌ Menghisap luka', style: TextStyle(fontSize: 13)),
                                  SizedBox(height: 4),
                                  Text('❌ Menyayat atau membakar luka', style: TextStyle(fontSize: 13)),
                                  SizedBox(height: 4),
                                  Text('❌ Mengikat terlalu kencang (tourniquet)', style: TextStyle(fontSize: 13)),
                                  SizedBox(height: 4),
                                  Text('❌ Memberi ramuan tradisional, alkohol, atau es', style: TextStyle(fontSize: 13)),
                                ],
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
                      'Untuk hasil terbaik, pastikan ular terlihat jelas dan pencahayaan cukup',
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
