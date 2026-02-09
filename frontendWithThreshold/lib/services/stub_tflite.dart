// Stub implementation untuk TensorFlow Lite di platform web
// Karena tflite_flutter menggunakan dart:ffi yang tidak tersedia di web

class Interpreter {
  static Future<Interpreter> fromAsset(String assetName) async {
    throw UnsupportedError('TensorFlow Lite tidak didukung di platform web');
  }
  
  void run(input, output) {
    throw UnsupportedError('TensorFlow Lite tidak didukung di platform web');
  }
  
  void close() {
    // No-op untuk web
  }
  
  dynamic getInputTensor(int index) {
    throw UnsupportedError('TensorFlow Lite tidak didukung di platform web');
  }
  
  dynamic getOutputTensor(int index) {
    throw UnsupportedError('TensorFlow Lite tidak didukung di platform web');
  }
}
