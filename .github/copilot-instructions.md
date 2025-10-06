# GitHub Copilot Instructions for Snake Classification Project

## Project Overview

This repository contains a snake classification system that identifies whether a snake is venomous or non-venomous using computer vision and machine learning.

### Project Structure

```
snake-classification/
├── frontend/              # Flutter mobile application
│   ├── lib/              # Dart source code
│   │   ├── main.dart
│   │   ├── screens/      # UI screens
│   │   └── services/     # Business logic (ML inference)
│   ├── assets/           # App resources
│   │   ├── models/       # TensorFlow Lite models (.tflite)
│   │   └── labels/       # Classification labels
│   └── test/             # Flutter tests
├── coba/                 # ML model training experiments
├── DATASET/              # Training dataset
├── *.py                  # Python scripts for ML training and data collection
└── snake_vit_model.tflite # Trained TensorFlow Lite model
```

## Technology Stack

### Frontend (Flutter/Dart)
- **Framework**: Flutter SDK 3.8.1+
- **Language**: Dart SDK
- **ML Framework**: TensorFlow Lite (`tflite_flutter`)
- **Key Dependencies**:
  - `image_picker`: Camera and gallery access
  - `camera`: Camera functionality
  - `image`: Image processing
  - `path_provider`: File system access

### Backend/ML (Python)
- **Framework**: TensorFlow 2.15.0
- **Model**: Vision Transformer (ViT) from Hugging Face Transformers
- **Key Libraries**:
  - `transformers`: Pre-trained ViT model
  - `keras`: Model training
  - `scikit-learn`: Evaluation metrics
  - `keras-tuner`: Hyperparameter tuning
  - `opencv-python`, `pillow`: Image processing

## Development Guidelines

### Flutter App Development

#### Running the App
```bash
cd frontend
flutter pub get
flutter run              # Run on connected device
flutter run -d windows   # Run on Windows desktop
flutter run -d chrome    # Run on web
```

#### Building the App
```bash
flutter build apk        # Android release
flutter build ios        # iOS release (macOS only)
flutter clean            # Clean build artifacts
```

#### Code Organization
- Place UI screens in `lib/screens/`
- Place business logic and services in `lib/services/`
- Keep widgets small and reusable
- Use `async`/`await` for asynchronous operations
- Follow Flutter's Material Design guidelines

#### ML Model Integration
- The app expects a TensorFlow Lite model at `assets/models/snake_vit_model.tflite`
- Model input: 224x224x3 RGB images (normalized 0-1)
- Model output: Float32[1, 2] probabilities for [Non-Venomous, Venomous]
- The `SnakeClassifier` service handles model loading and inference

### Python ML Development

#### Setting Up Environment
```bash
pip install -r requirements.txt
```

#### Training Models
- Main training script: `coba/modelG.py`
- Uses Vision Transformer (ViT) from `google/vit-base-patch16-224`
- Performs hyperparameter tuning with `keras-tuner`
- Outputs TensorFlow Lite model for mobile deployment

#### Model Training Best Practices
- Use `IMG_SIZE = (224, 224)` to match ViT input requirements
- Set random seeds for reproducibility (`tf.random.set_seed(42)`)
- Enable GPU memory growth to prevent OOM errors
- Validate image integrity before training (check for corrupt files)
- Use train/val/test split: 70%/15%/15%

#### Data Collection
- Scripts: `scrap.py`, `bingCrawler.py`, `imageCrawler.py`
- Use `icrawler` for web scraping
- Check for duplicate images with `duplicate.py`
- Store images in organized dataset folders

## Code Style and Best Practices

### Dart/Flutter
- Follow [Effective Dart](https://dart.dev/guides/language/effective-dart) guidelines
- Use `flutter_lints` for code analysis
- Run `flutter analyze` before committing
- Use meaningful variable names in English (UI text can be in Indonesian)
- Add error handling with try-catch blocks
- Use `const` constructors where possible for performance

### Python
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for functions and classes
- Handle exceptions gracefully
- Log important steps during model training
- Use f-strings for string formatting

## Testing

### Flutter Tests
```bash
cd frontend
flutter test             # Run all tests
flutter test --coverage  # Generate coverage report
```

### Python Model Validation
- Evaluate models with confusion matrix, accuracy, precision, recall, F1-score
- Use `score.py` for model evaluation
- Test model inference with `test.py` or `loadmodel.py`

## Common Pitfalls and Troubleshooting

### Flutter Issues

1. **TensorFlow Lite Compatibility**
   - Current versions may have compatibility issues
   - The app currently uses simulated classification as a fallback
   - Ensure `tflite_flutter` version matches Flutter/Dart SDK

2. **Build Errors**
   - Run `flutter clean && flutter pub get` to refresh dependencies
   - On Windows, enable Developer Mode for symlink support
   - Check that all assets are properly declared in `pubspec.yaml`

3. **Model Loading Errors**
   - Verify `.tflite` file is in `assets/models/`
   - Check file size (should be ~87MB for ViT model)
   - Ensure model is included in `pubspec.yaml` assets

4. **Camera Permissions**
   - Add camera/storage permissions to `AndroidManifest.xml` (Android)
   - Add camera usage description to `Info.plist` (iOS)

### Python/ML Issues

1. **Memory Issues**
   - Enable GPU memory growth: `tf.config.experimental.set_memory_growth(gpu, True)`
   - Reduce batch size if OOM occurs
   - Close model and clear session after training

2. **Model Loading**
   - Download from Hugging Face: `TFViTForImageClassification.from_pretrained()`
   - Set `ignore_mismatched_sizes=True` when changing number of classes
   - Use local cache if online loading fails

3. **Corrupt Images**
   - Implement `is_valid_image()` check before loading
   - Skip corrupt files and log them
   - Clean dataset with `duplicate.py`

## Project-Specific Conventions

### Language Usage
- **Code**: English (variable names, comments, documentation)
- **UI Text**: Indonesian (for end-user facing text in the Flutter app)
- **Model Labels**: English class names in code, can use Indonesian in UI

### File Naming
- Python: `snake_case.py`
- Dart: `snake_case.dart`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` (Python), `camelCase` (Dart)

### Git Workflow
- Create feature branches for new work
- Test thoroughly before committing
- Write clear commit messages
- Update documentation when changing APIs or behavior

## Model Information

### Current Model
- **Architecture**: Vision Transformer (ViT) base, patch size 16, 224x224 input
- **Classes**: 2 (Non-Venomous, Venomous)
- **Input Size**: 224x224x3 RGB
- **Preprocessing**: Normalize pixel values to [0, 1]
- **Output**: Softmax probabilities for each class

### Snake Species in Dataset
The dataset includes various snake species:
- Venomous: Naja sputatrix (Java Cobra), Ophiophagus hannah (King Cobra), Bungarus candidus (Krait)
- Non-Venomous: Python reticulatus (Reticulated Python), Coelognathus radiatus (Rat Snake), Dendrelaphis pictus (Painted Bronzeback)

## Additional Resources

- **Flutter Documentation**: https://flutter.dev/docs
- **TensorFlow Lite**: https://www.tensorflow.org/lite
- **Vision Transformer**: https://huggingface.co/docs/transformers/model_doc/vit
- **Dart Language**: https://dart.dev/guides

## Notes for Copilot

When suggesting code changes:
1. **Maintain consistency** with existing code style and patterns
2. **Consider mobile constraints**: Memory, battery, storage when working on Flutter app
3. **Preserve model compatibility**: Don't change input/output dimensions without retraining
4. **Test edge cases**: Corrupt images, missing files, permission denials
5. **Keep UI responsive**: Use async operations for heavy tasks
6. **Follow Material Design**: When adding/modifying Flutter UI components
7. **Validate data**: Always check image validity before processing
8. **Handle errors gracefully**: Provide user-friendly error messages
9. **Document model changes**: If modifying ML pipeline, update comments
10. **Respect Indonesian UI**: Keep end-user text in Indonesian language
