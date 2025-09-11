# Snake Identifier App

A Flutter mobile application for identifying snakes from images using a pre-trained TensorFlow Lite model. The app can classify images captured from the camera or selected from the gallery to determine if a snake is venomous or non-venomous.

## Features

- **Camera Integration**: Take photos directly within the app
- **Gallery Selection**: Choose existing images from device gallery
- **AI Classification**: Uses TensorFlow Lite model for snake identification
- **Real-time Results**: Displays classification results with confidence scores
- **User-friendly UI**: Clean, intuitive interface in Indonesian language
- **Visual Feedback**: Color-coded results (red for venomous, green for non-venomous)

## Prerequisites

- Flutter SDK (3.8.1 or later)
- Dart SDK
- Android Studio / Xcode for mobile development
- TensorFlow Lite model file (`snake_vit_model.tflite`)

## Installation

1. Clone the repository or extract the project files
2. Navigate to the project directory:
   ```bash
   cd frontend
   ```

3. Install dependencies:
   ```bash
   flutter pub get
   ```

4. **Important**: Add your TensorFlow Lite model file:
   - Place your `snake_vit_model.tflite` file in `assets/models/`
   - The model should be trained to classify snake images into venomous/non-venomous categories

5. Update the labels file if needed:
   - Edit `assets/labels/snake_labels.txt` to match your model's output classes

## Running the App

### For Development
```bash
flutter run
```

### For Release
```bash
flutter build apk
# or for iOS
flutter build ios
```

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── screens/
│   └── home_screen.dart      # Main UI screen
└── services/
    └── snake_classifier.dart # TensorFlow Lite model handler

assets/
├── models/
│   └── snake_vit_model.tflite # TensorFlow Lite model
└── labels/
    └── snake_labels.txt       # Classification labels
```

## Model Requirements

The TensorFlow Lite model should:
- Accept 224x224x3 RGB images as input
- Output probabilities for each class
- Be compatible with the TensorFlow Lite Flutter plugin

### Expected Model Input/Output:
- **Input**: Float32[1, 224, 224, 3] (normalized to 0-1 range)
- **Output**: Float32[1, 2] (probabilities for each class)

## Dependencies

Key packages used in this project:
- `image_picker`: For camera and gallery access
- `camera`: Camera functionality
- `tflite_flutter`: TensorFlow Lite integration
- `image`: Image processing
- `path_provider`: File system access

## Usage

1. Launch the app
2. Tap "Kamera" to take a new photo or "Galeri" to select an existing image
3. The app will automatically process the image and display results
4. Results show the classification (Venomous/Non-Venomous) with confidence percentage
5. Color coding: Red for venomous, Green for non-venomous

## Troubleshooting

### Common Issues:

1. **Model loading errors**: Ensure the `.tflite` file is properly placed in `assets/models/`
2. **Camera permissions**: Grant camera and storage permissions when prompted
3. **Build errors**: Run `flutter clean && flutter pub get` to refresh dependencies

### Developer Mode (Windows):
If you encounter symlink issues during build, enable Developer Mode:
1. Open Windows Settings
2. Go to Update & Security → For developers
3. Enable Developer Mode

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Disclaimer

This app is for educational purposes. Always consult with experts for professional snake identification and safety advice.
