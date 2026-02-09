# Flutter Snake Identifier - Quick Start Guide

## 🚀 Running the App

### Option 1: Enable Developer Mode (Recommended for full features)
1. Run this command to open Developer Mode settings:
   ```
   start ms-settings:developers
   ```
2. Toggle "Developer Mode" to ON
3. Restart your computer if prompted
4. Then run: `flutter run -d windows`

### Option 2: Run in Web Browser (No Developer Mode needed)
```bash
flutter run -d chrome
```

### Option 3: Run in Edge Browser
```bash
flutter run -d edge
```

## 📱 App Features
- Camera integration (web browser will ask for camera permission)
- Gallery image selection
- Snake classification simulation
- Beautiful Indonesian UI
- Real-time results with confidence scores

## 🔧 Troubleshooting

### If Developer Mode can't be enabled:
- Use web browser option: `flutter run -d chrome`
- Web version supports all features except native camera (browser camera works)

### If build fails:
```bash
flutter clean
flutter pub get
flutter run -d chrome
```

## 🎯 Testing the App
1. Click "Kamera" or "Galeri" to select an image
2. View the simulated classification results
3. Results show confidence percentage and color coding

Your app is ready to run! 🐍📱
