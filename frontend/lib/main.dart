import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const SnakeIdentifierApp());
}

class SnakeIdentifierApp extends StatelessWidget {
  const SnakeIdentifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Klasifikasi Ular',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
