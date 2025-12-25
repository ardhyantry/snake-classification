# TensorFlow Lite GPU Delegate keep rules
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.nnapi.** { *; }

# Keep all TensorFlow Lite classes
-keep class org.tensorflow.lite.** { *; }

# Suppress warnings for missing TensorFlow Lite GPU classes
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options
-dontwarn org.tensorflow.lite.gpu.**

# Preserve line numbers for debugging
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile
