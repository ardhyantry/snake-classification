# TensorFlow Lite Model Placeholder

## Untuk menggunakan model TensorFlow Lite yang sebenarnya:

1. **Letakkan file model Anda di sini:**
   - Ganti file ini dengan `snake_vit_model.tflite` yang sebenarnya
   - Model harus dilatih untuk klasifikasi ular (venomous vs non-venomous)

2. **Format model yang diperlukan:**
   - Input: [1, 224, 224, 3] (batch, height, width, channels)
   - Output: [1, 2] (batch, jumlah kelas)
   - Tipe: Float32

3. **Struktur model:**
   - Model Vision Transformer (ViT) atau CNN
   - Normalisasi input: [0, 1] range
   - Output: probabilitas untuk setiap kelas

4. **Jika model tidak tersedia:**
   - Aplikasi akan menggunakan simulasi
   - Semua fitur UI tetap berfungsi
   - Hasil simulasi ditampilkan dengan jelas

## Contoh pelatihan model:
- Dataset: Gambar ular dengan label venomous/non-venomous
- Preprocessing: Resize ke 224x224, normalisasi [0,1]
- Model: Vision Transformer atau MobileNet
- Export: TensorFlow Lite format (.tflite)

Letakkan file `snake_vit_model.tflite` di direktori ini untuk mengaktifkan klasifikasi AI yang sebenarnya.
