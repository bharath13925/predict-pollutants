// Add to pubspec.yaml:
// dependencies:
//   flutter:
//     sdk: flutter
//   animate_do: ^3.0.2
//   http: ^1.1.0
//   flutter_map: ^6.1.0
//   latlong2: ^0.9.0
//   fl_chart: ^0.68.0
//   url_launcher: ^6.2.2
//
// Then run: flutter pub get

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:fl_chart/fl_chart.dart';

// IMPORT THE HEALTH TIPS FILE
import 'health_tips_data.dart';

void main() {
  runApp(const AirAwareApp());
}

class AirAwareApp extends StatelessWidget {
  const AirAwareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Air Aware',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.teal, fontFamily: 'Poppins'),
      home: const AirAwareHome(),
    );
  }
}

class AirAwareHome extends StatefulWidget {
  const AirAwareHome({super.key});

  @override
  State<AirAwareHome> createState() => _AirAwareHomeState();
}

class _AirAwareHomeState extends State<AirAwareHome> {
  final TextEditingController _locationController = TextEditingController();
  bool _loading = false;
  bool _analysisInProgress = false;
  String _analysisStep = "";
  Map<String, dynamic>? _aqiData;
  List<dynamic>? _predictions;
  Map<String, dynamic>? _modelInfo;
  Map<String, dynamic>? _historicalData;
  Map<String, dynamic>? _trainingResult;

  // CHANGE THIS TO YOUR SERVER IP
  final String baseUrl = "https://air-quality-api-etwm.onrender.com/api";

  double _normalizeValue(String pollutant, double value, String unit) {
    if (pollutant.toUpperCase().contains('CO')) {
      if (unit.toLowerCase().contains('ppm')) {
        return value * 1145; // ppm → µg/m³
      } else if (unit.toLowerCase().contains('mg')) {
        return value * 1000; // mg/m³ → µg/m³
      }
    }
    // You can extend this for SO2, NO2 etc. if needed
    return value;
  }

  /// ✅ COMPREHENSIVE ACCURACY CALCULATION
  /// Based on MAE (Mean Absolute Error), Test Loss, and Model Confidence
  String _calculateModelAccuracy(Map<String, dynamic>? modelData) {
    if (modelData == null) return 'N/A';

    final testMae = _safeGetNumber(modelData['test_mae']);
    final testLoss = _safeGetNumber(modelData['test_loss']);
    final trainMae = _safeGetNumber(modelData['train_mae']);

    // If no valid metrics, return N/A
    if (testMae <= 0 || testMae.isNaN || testMae.isInfinite) {
      return 'N/A (No Test Data)';
    }

    // ✅ PRIMARY METRIC: MAE-based accuracy
    // MAE represents average prediction error in µg/m³
    // Lower MAE = Higher Accuracy
    double maeAccuracy = 0;

    if (testMae < 2) {
      maeAccuracy = 98.0; // Exceptional
    } else if (testMae < 5) {
      maeAccuracy = 95.0; // Excellent
    } else if (testMae < 8) {
      maeAccuracy = 92.0; // Very Good
    } else if (testMae < 12) {
      maeAccuracy = 88.0; // Good
    } else if (testMae < 15) {
      maeAccuracy = 83.0; // Acceptable
    } else if (testMae < 20) {
      maeAccuracy = 78.0; // Fair
    } else if (testMae < 25) {
      maeAccuracy = 72.0; // Moderate
    } else if (testMae < 30) {
      maeAccuracy = 65.0; // Below Average
    } else if (testMae < 40) {
      maeAccuracy = 55.0; // Poor
    } else {
      maeAccuracy = 45.0; // Very Poor
    }

    // ✅ SECONDARY METRIC: Overfitting check (Train vs Test MAE)
    double overfittingPenalty = 0;
    if (trainMae > 0 && testMae > trainMae) {
      double overfitRatio = testMae / trainMae;
      if (overfitRatio > 2.0) {
        overfittingPenalty = 10; // Severe overfitting
      } else if (overfitRatio > 1.5) {
        overfittingPenalty = 5; // Moderate overfitting
      } else if (overfitRatio > 1.2) {
        overfittingPenalty = 2; // Slight overfitting
      }
    }

    // ✅ TERTIARY METRIC: Loss-based confidence
    double lossConfidence = 0;
    if (testLoss > 0 && testLoss.isFinite) {
      if (testLoss < 100) {
        lossConfidence = 2; // Excellent loss
      } else if (testLoss < 300) {
        lossConfidence = 1; // Good loss
      } else if (testLoss > 1000) {
        lossConfidence = -3; // Poor loss
      }
    }

    // ✅ FINAL ACCURACY CALCULATION
    double finalAccuracy = maeAccuracy - overfittingPenalty + lossConfidence;
    finalAccuracy = finalAccuracy.clamp(0, 100);

    // ✅ CATEGORIZATION
    String category;
    String emoji;

    if (finalAccuracy >= 95) {
      category = "Exceptional";
      emoji = "🌟";
    } else if (finalAccuracy >= 90) {
      category = "Excellent";
      emoji = "✨";
    } else if (finalAccuracy >= 85) {
      category = "Very Good";
      emoji = "🎯";
    } else if (finalAccuracy >= 80) {
      category = "Good";
      emoji = "👍";
    } else if (finalAccuracy >= 75) {
      category = "Acceptable";
      emoji = "✓";
    } else if (finalAccuracy >= 70) {
      category = "Fair";
      emoji = "⚡";
    } else if (finalAccuracy >= 60) {
      category = "Moderate";
      emoji = "⚠️";
    } else if (finalAccuracy >= 50) {
      category = "Below Average";
      emoji = "⚠️";
    } else {
      category = "Poor - Retrain";
      emoji = "🔄";
    }

    return "$emoji ${finalAccuracy.toStringAsFixed(1)}% ($category)";
  }

  /// ✅ DETAILED ACCURACY BREAKDOWN FOR INFO DISPLAY
  String _getAccuracyDetails(Map<String, dynamic>? modelData) {
    if (modelData == null) return 'No model data available';

    final testMae = _safeGetNumber(modelData['test_mae']);
    final testLoss = _safeGetNumber(modelData['test_loss']);
    final trainMae = _safeGetNumber(modelData['train_mae']);

    if (testMae <= 0) return 'Insufficient test data';

    String details = '';
    details += 'Test MAE: ${testMae.toStringAsFixed(2)} µg/m³\n';
    details += 'Train MAE: ${trainMae.toStringAsFixed(2)} µg/m³\n';
    details += 'Test Loss: ${testLoss.toStringAsFixed(2)}\n';

    if (trainMae > 0) {
      double ratio = testMae / trainMae;
      details += '\n';
      if (ratio > 2.0) {
        details +=
            '⚠️ Model overfitting detected (Test/Train: ${ratio.toStringAsFixed(2)}x)';
      } else if (ratio > 1.5) {
        details +=
            '⚡ Moderate generalization gap (${ratio.toStringAsFixed(2)}x)';
      } else {
        details += '✓ Good generalization (${ratio.toStringAsFixed(2)}x)';
      }
    }

    details += '\n\nAvg Prediction Error: ±${testMae.toStringAsFixed(1)} µg/m³';

    return details;
  }

  Future<void> _runFullAnalysis(String city) async {
    setState(() {
      _analysisInProgress = true;
      _analysisStep = "Initializing full analysis...";
      _predictions = null;
      _modelInfo = null;
      _historicalData = null;
      _trainingResult = null;
    });

    try {
      _updateStep("📥 Collecting historical data...");

      final uri = Uri.parse("$baseUrl/full-analysis");
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'city': city}),
          )
          .timeout(const Duration(seconds: 6000));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        setState(() {
          _historicalData = data['historical_data'];
          _trainingResult = data['model_training'];

          if (data['current_data'] != null) {
            _aqiData = _processAQIData(data['current_data'], city);
          }

          if (data['predictions'] != null &&
              data['predictions']['predictions'] != null) {
            _predictions = data['predictions']['predictions'];
            _modelInfo = data['predictions']['model_info'];
          }

          final trainingResult = _trainingResult;
          if (_modelInfo == null && trainingResult != null) {
            _modelInfo = {
              'model_type': trainingResult['model_type'] ?? 'LSTM',
              'trained_at': trainingResult['trained_at'] ?? 'Unknown',
              'test_mae': trainingResult['test_mae'],
              'test_loss': trainingResult['test_loss'],
              'train_mae': trainingResult['train_mae'],
              'train_loss': trainingResult['train_loss'],
              'lookback_days': trainingResult['lookback_days'],
              'training_samples': trainingResult['training_samples'],
              'test_samples': trainingResult['test_samples'],
            };
          }

          _analysisInProgress = false;
        });

        if (mounted) {
          final historicalData = _historicalData;
          final trainingResult = _trainingResult;

          final daysCollected =
              historicalData?['records_stored'] ??
              historicalData?['records'] ??
              historicalData?['days_collected'] ??
              0;

          final accuracy = _calculateModelAccuracy(
            _modelInfo ?? trainingResult,
          );

          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '✅ Full analysis complete!\n'
                'Data collected: $daysCollected days\n'
                'Model accuracy: $accuracy',
              ),
              backgroundColor: Colors.green,
              duration: const Duration(seconds: 5),
            ),
          );
        }
      } else {
        throw Exception('HTTP ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      debugPrint("❌ Error: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.redAccent,
            duration: const Duration(seconds: 5),
          ),
        );
      }
      setState(() => _analysisInProgress = false);
    }
  }

  void _updateStep(String step) {
    setState(() => _analysisStep = step);
    debugPrint(step);
  }

  Future<void> _quickFetch(String city) async {
    setState(() {
      _loading = true;
      _aqiData = null;
      _predictions = null;
      _modelInfo = null;
    });

    try {
      final uri = Uri.parse("$baseUrl/air-quality?city=$city");
      debugPrint("🌐 Fetching from: $uri");

      final response = await http.get(uri).timeout(const Duration(seconds: 30));

      debugPrint("📡 Response status: ${response.statusCode}");

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        debugPrint("✅ JSON decoded successfully");

        if (data["error"] != null) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(data["error"]),
                backgroundColor: Colors.orangeAccent,
              ),
            );
          }
          setState(() => _loading = false);
          return;
        }

        setState(() {
          _aqiData = _processAQIData(data, city);
          _loading = false;
        });
      } else {
        throw Exception('HTTP ${response.statusCode}');
      }
    } catch (e) {
      debugPrint("❌ Error: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
      setState(() => _loading = false);
    }
  }

  Map<String, dynamic>? _processAQIData(
    Map<String, dynamic> data,
    String city,
  ) {
    if (data["pollutants"] != null && data["pollutants"] is Map) {
      Map<String, Map<String, dynamic>> pollutantData = {};

      (data["pollutants"] as Map).forEach((key, pollutantInfo) {
        if (pollutantInfo is Map) {
          dynamic rawValue = pollutantInfo["value"];
          String unit = pollutantInfo["unit"]?.toString() ?? "";
          double? value;

          if (rawValue != null) {
            if (rawValue is int) {
              value = rawValue.toDouble();
            } else if (rawValue is double) {
              value = rawValue;
            } else if (rawValue is String) {
              value = double.tryParse(rawValue);
            }
          }

          if (value != null) {
            // ✅ Normalize CO and similar pollutants before using
            value = _normalizeValue(key, value, unit);

            pollutantData[key] = {
              "value": value,
              "unit": "µg/m³", // Standardized unit for display
              "parameter": pollutantInfo["parameter"]?.toString() ?? key,
              "source": pollutantInfo["source"]?.toString() ?? "GEE/OpenAQ",
            };
          }
        }
      });

      if (pollutantData.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text("No valid pollutant values found."),
              backgroundColor: Colors.orangeAccent,
            ),
          );
        }
        return null;
      }

      final overallAQI = _calculateOverallAQI(pollutantData);

      Map<String, dynamic>? mapTiles;
      if (data["map_tiles"] != null && data["map_tiles"] is Map) {
        mapTiles = Map<String, dynamic>.from(data["map_tiles"]);
      }

      return {
        "city": data["city"] ?? city,
        "pollutants": pollutantData,
        "overallAQI": overallAQI,
        "map_tiles": mapTiles,
        "coordinates": data["coordinates"],
        "timestamp": data["timestamp"],
        "note": data["note"] ?? "",
      };
    }
    return null;
  }

  Future<void> _loadPredictions(String city) async {
    try {
      final uri = Uri.parse("$baseUrl/predict?city=$city");
      final response = await http.get(uri).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        if (data['status'] == 'success') {
          setState(() {
            _predictions = data['predictions'];
            _modelInfo = data['model_info'];
          });

          if (mounted) {
            final accuracy = _calculateModelAccuracy(_modelInfo);

            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  '✅ Predictions loaded!\n'
                  'Model: ${_modelInfo?['model_type'] ?? 'Unknown'}\n'
                  'Accuracy: $accuracy',
                ),
                backgroundColor: Colors.green,
              ),
            );
          }
        } else {
          throw Exception(data['error'] ?? 'Unknown error');
        }
      } else {
        final errorData = jsonDecode(response.body);
        throw Exception(errorData['error'] ?? 'Model not trained yet');
      }
    } catch (e) {
      debugPrint("Could not load predictions: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Could not load predictions: $e\n\nTip: Run "Generate Predictions" first to train the model',
            ),
            backgroundColor: Colors.orange,
            duration: const Duration(seconds: 5),
          ),
        );
      }
    }
  }

  Map<String, dynamic> _calculateOverallAQI(
    Map<String, Map<String, dynamic>> pollutants,
  ) {
    int maxLevel = 0;
    String worstPollutant = "";
    double maxValue = 0.0;

    pollutants.forEach((parameter, data) {
      double value = data["value"];
      int level = _getAQILevel(parameter, value);

      // ✅ Find worst pollutant based on AQI level first, then value
      if (level > maxLevel || (level == maxLevel && value > maxValue)) {
        maxLevel = level;
        maxValue = value;
        worstPollutant = parameter;
      }
    });

    return {
      "level": maxLevel,
      "worstPollutant": worstPollutant,
      "category": _getAQICategoryFromLevel(maxLevel),
      "color": _getColorFromLevel(maxLevel),
    };
  }

  int _getAQILevel(String parameter, double value) {
    String paramNormalized = parameter
        .toUpperCase()
        .replaceAll('₂', '2')
        .replaceAll('₃', '3')
        .replaceAll('₄', '4');

    // ✅ All values are in µg/m³ (except CH4 in ppb)
    if (paramNormalized.contains("PM2.5") || paramNormalized == "PM25") {
      if (value <= 30) return 0;
      if (value <= 60) return 1;
      if (value <= 90) return 2;
      if (value <= 120) return 3;
      if (value <= 250) return 4;
      return 5;
    }

    if (paramNormalized.contains("PM10")) {
      if (value <= 50) return 0;
      if (value <= 100) return 1;
      if (value <= 250) return 2;
      if (value <= 350) return 3;
      if (value <= 430) return 4;
      return 5;
    }

    if (paramNormalized.contains("NO2")) {
      if (value <= 40) return 0;
      if (value <= 80) return 1;
      if (value <= 180) return 2;
      if (value <= 280) return 3;
      if (value <= 400) return 4;
      return 5;
    }

    if (paramNormalized.contains("SO2")) {
      if (value <= 40) return 0;
      if (value <= 80) return 1;
      if (value <= 380) return 2;
      if (value <= 800) return 3;
      if (value <= 1600) return 4;
      return 5;
    }

    if (paramNormalized.contains("CO")) {
      if (value <= 1000) return 0;
      if (value <= 2000) return 1;
      if (value <= 10000) return 2;
      if (value <= 17000) return 3;
      if (value <= 34000) return 4;
      return 5;
    }

    if (paramNormalized.contains("CH4")) {
      // CH4 is in ppb
      if (value <= 1800) return 0;
      if (value <= 1850) return 1;
      if (value <= 1900) return 2;
      if (value <= 1950) return 3;
      if (value <= 2000) return 4;
      return 5;
    }

    return 0;
  }

  String _getAQICategoryFromLevel(int level) {
    switch (level) {
      case 0:
        return "Good";
      case 1:
        return "Satisfactory";
      case 2:
        return "Moderate";
      case 3:
        return "Poor";
      case 4:
        return "Very Poor";
      case 5:
        return "Severe";
      default:
        return "Unknown";
    }
  }

  Color _getColorFromLevel(int level) {
    switch (level) {
      case 0:
        return Colors.green;
      case 1:
        return Colors.yellow[700]!;
      case 2:
        return Colors.orange;
      case 3:
        return Colors.red;
      case 4:
        return Colors.purple;
      case 5:
        return Colors.brown[900]!;
      default:
        return Colors.grey;
    }
  }

  Color _getCategoryColor(String pollutant, double value) {
    if (pollutant.toLowerCase().contains('pm2.5') || pollutant == 'pm25') {
      if (value <= 30) return Colors.green;
      if (value <= 60) return Colors.yellow[700]!;
      if (value <= 90) return Colors.orange;
      return Colors.red;
    }
    if (pollutant.toLowerCase().contains('pm10')) {
      if (value <= 50) return Colors.green;
      if (value <= 100) return Colors.yellow[700]!;
      if (value <= 250) return Colors.orange;
      return Colors.red;
    }
    if (pollutant.toLowerCase().contains('ch4')) {
      if (value <= 1800) return Colors.green;
      if (value <= 1850) return Colors.yellow[700]!;
      if (value <= 1900) return Colors.orange;
      return Colors.red;
    }
    return Colors.grey;
  }

  double _safeGetNumber(dynamic value) {
    if (value == null) return 0.0;
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? 0.0;
    return 0.0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF56CCF2), Color(0xFF2F80ED)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                FadeInDown(
                  child: const Text(
                    "🌫️ Air Aware",
                    style: TextStyle(
                      fontSize: 38,
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                FadeInUp(
                  child: const Text(
                    "Real-time OpenAQ + AI predictions",
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70, fontSize: 16),
                  ),
                ),
                const SizedBox(height: 30),

                FadeInUp(
                  delay: const Duration(milliseconds: 300),
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.95),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        TextField(
                          controller: _locationController,
                          decoration: InputDecoration(
                            labelText: 'Enter city name (e.g., Delhi)',
                            prefixIcon: const Icon(
                              Icons.location_on,
                              color: Colors.blueAccent,
                            ),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                          ),
                        ),
                        const SizedBox(height: 15),
                        Row(
                          children: [
                            Expanded(
                              child: ElevatedButton.icon(
                                onPressed: _loading || _analysisInProgress
                                    ? null
                                    : () {
                                        String city = _locationController.text
                                            .trim();
                                        if (city.isNotEmpty) {
                                          _quickFetch(city);
                                        } else {
                                          ScaffoldMessenger.of(
                                            context,
                                          ).showSnackBar(
                                            const SnackBar(
                                              content: Text(
                                                'Please enter a city name',
                                              ),
                                              backgroundColor: Colors.redAccent,
                                            ),
                                          );
                                        }
                                      },
                                icon: const Icon(Icons.search, size: 20),
                                label: const Text('Quick Check'),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.teal,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 15,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: ElevatedButton.icon(
                                onPressed: _loading || _analysisInProgress
                                    ? null
                                    : () {
                                        String city = _locationController.text
                                            .trim();
                                        if (city.isNotEmpty) {
                                          _runFullAnalysis(city);
                                        } else {
                                          ScaffoldMessenger.of(
                                            context,
                                          ).showSnackBar(
                                            const SnackBar(
                                              content: Text(
                                                'Please enter a city name',
                                              ),
                                              backgroundColor: Colors.redAccent,
                                            ),
                                          );
                                        }
                                      },
                                icon: const Icon(Icons.psychology, size: 18),
                                label: const Text(
                                  'Generate\nPredictions',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(fontSize: 13, height: 1.1),
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.purple,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 12,
                                    horizontal: 8,
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
                const SizedBox(height: 30),

                if (_loading && _aqiData == null)
                  const CircularProgressIndicator(color: Colors.white),

                if (_analysisInProgress)
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.9),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Column(
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 15),
                        Text(
                          _analysisStep,
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 14),
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          "This may take 2-5 minutes...",
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey,
                            fontStyle: FontStyle.italic,
                          ),
                        ),
                      ],
                    ),
                  ),

                if (_aqiData != null && !_analysisInProgress)
                  Column(
                    children: [
                      FadeInUp(
                        child: Container(
                          padding: const EdgeInsets.all(20),
                          margin: const EdgeInsets.symmetric(vertical: 10),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                _aqiData!["overallAQI"]["color"],
                                _aqiData!["overallAQI"]["color"].withOpacity(
                                  0.7,
                                ),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: _aqiData!["overallAQI"]["color"]
                                    .withOpacity(0.4),
                                blurRadius: 15,
                                offset: const Offset(0, 5),
                              ),
                            ],
                          ),
                          child: Column(
                            children: [
                              Text(
                                "📍 ${_aqiData!["city"]}",
                                style: const TextStyle(
                                  fontSize: 28,
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                "Overall: ${_aqiData!["overallAQI"]["category"]}",
                                style: const TextStyle(
                                  fontSize: 18,
                                  color: Colors.white,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 5),
                              Text(
                                "Worst: ${_aqiData!["overallAQI"]["worstPollutant"]}",
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.white.withOpacity(0.9),
                                ),
                              ),
                              if (_aqiData!["note"] != null &&
                                  _aqiData!["note"].toString().isNotEmpty)
                                Padding(
                                  padding: const EdgeInsets.only(top: 8),
                                  child: Text(
                                    _aqiData!["note"],
                                    style: TextStyle(
                                      fontSize: 11,
                                      color: Colors.white.withOpacity(0.8),
                                      fontStyle: FontStyle.italic,
                                    ),
                                    textAlign: TextAlign.center,
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      ElevatedButton.icon(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => HealthTipsScreen(
                                aqiLevel: _aqiData!["overallAQI"]["level"],
                                worstPollutant:
                                    _aqiData!["overallAQI"]["worstPollutant"],
                                predictions: _predictions,
                              ),
                            ),
                          );
                        },
                        icon: const Icon(Icons.health_and_safety, size: 24),
                        label: const Text('View Health Tips & Precautions'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.redAccent,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(
                            horizontal: 30,
                            vertical: 18,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                          elevation: 5,
                        ),
                      ),

                      const SizedBox(height: 20),

                      if (_aqiData!["map_tiles"] != null)
                        Column(
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  "🗺️ Interactive Air Quality Map",
                                  style: TextStyle(
                                    fontSize: 18,
                                    color: Colors.white.withOpacity(0.95),
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                IconButton(
                                  icon: const Icon(
                                    Icons.fullscreen,
                                    color: Colors.white,
                                    size: 28,
                                  ),
                                  onPressed: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => FullScreenMapView(
                                          mapTiles: _aqiData!["map_tiles"],
                                          cityName: _aqiData!["city"],
                                        ),
                                      ),
                                    );
                                  },
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            _buildInteractiveMapCard(_aqiData!["map_tiles"]),
                            const SizedBox(height: 25),
                          ],
                        ),

                      Text(
                        "📊 Pollutant Measurements",
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.white.withOpacity(0.9),
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 15),

                      ...(_aqiData!["pollutants"]
                              as Map<String, Map<String, dynamic>>)
                          .entries
                          .map(
                            (entry) => _pollutantCard(
                              entry.key,
                              entry.value["value"],
                              entry.value["unit"],
                              entry.value["source"],
                            ),
                          )
                          .toList(),

                      const SizedBox(height: 20),

                      if (_predictions == null && !_analysisInProgress)
                        ElevatedButton.icon(
                          onPressed: () => _loadPredictions(_aqiData!['city']),
                          icon: const Icon(Icons.trending_up),
                          label: const Text('Load 7-Day Forecast'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.deepPurple,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 30,
                              vertical: 15,
                            ),
                          ),
                        ),

                      if (_aqiData!["timestamp"] != null)
                        Container(
                          padding: const EdgeInsets.all(15),
                          margin: const EdgeInsets.only(top: 20),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              const Icon(
                                Icons.access_time,
                                color: Colors.white,
                                size: 16,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                "Last updated: ${_aqiData!['timestamp']}",
                                style: const TextStyle(
                                  fontSize: 12,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),

                if (_historicalData != null)
                  Container(
                    margin: const EdgeInsets.only(top: 20),
                    padding: const EdgeInsets.all(15),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "📊 Historical Data Collected",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Days: ${_historicalData!['days_collected'] ?? _historicalData!['records'] ?? _historicalData!['records_stored'] ?? 0}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        if (_historicalData!['date_range'] != null)
                          Text(
                            "Range: ${_historicalData!['date_range']['start']} to ${_historicalData!['date_range']['end']}",
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                  ),

                if (_trainingResult != null)
                  Container(
                    margin: const EdgeInsets.only(top: 10),
                    padding: const EdgeInsets.all(15),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          "🤖 Model Training Results",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Model: ${_trainingResult!['model_type']?.toString().toUpperCase() ?? 'LSTM'}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        Text(
                          "Accuracy: ${_calculateModelAccuracy(_trainingResult)}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        if (_trainingResult!['training_samples'] != null)
                          Text(
                            "Training samples: ${_trainingResult!['training_samples']}",
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                        const SizedBox(height: 8),
                        GestureDetector(
                          onTap: () {
                            showDialog(
                              context: context,
                              builder: (context) => AlertDialog(
                                title: const Text('Model Performance Details'),
                                content: SingleChildScrollView(
                                  child: Text(
                                    _getAccuracyDetails(_trainingResult),
                                    style: const TextStyle(fontSize: 13),
                                  ),
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(context),
                                    child: const Text('Close'),
                                  ),
                                ],
                              ),
                            );
                          },
                          child: Text(
                            "📊 Tap for detailed metrics",
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.8),
                              fontSize: 11,
                              fontStyle: FontStyle.italic,
                              decoration: TextDecoration.underline,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                if (_predictions != null) ...[
                  const SizedBox(height: 30),
                  _buildPredictionsCard(),
                  const SizedBox(height: 20),
                  _buildPredictionCharts(),
                ],

                if (_modelInfo != null || _trainingResult != null)
                  Container(
                    margin: const EdgeInsets.only(top: 20),
                    padding: const EdgeInsets.all(15),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Builder(
                      builder: (context) {
                        final combinedModelInfo = {
                          'model_type':
                              _modelInfo?['model_type'] ??
                              _trainingResult?['model_type'] ??
                              'LSTM',
                          'trained_at':
                              _modelInfo?['trained_at'] ??
                              _trainingResult?['trained_at'] ??
                              'Unknown',
                          'lookback_days':
                              _modelInfo?['lookback_days'] ??
                              _trainingResult?['lookback_days'],
                          'test_mae':
                              _trainingResult?['test_mae'] ??
                              _modelInfo?['test_mae'],
                          'test_loss':
                              _trainingResult?['test_loss'] ??
                              _modelInfo?['test_loss'],
                          'train_mae':
                              _trainingResult?['train_mae'] ??
                              _modelInfo?['train_mae'],
                          'train_loss':
                              _trainingResult?['train_loss'] ??
                              _modelInfo?['train_loss'],
                          'training_samples':
                              _trainingResult?['training_samples'] ??
                              _modelInfo?['training_samples'],
                          'test_samples':
                              _trainingResult?['test_samples'] ??
                              _modelInfo?['test_samples'],
                        };

                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              "🤖 ML Model Info",
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              "Type: ${combinedModelInfo['model_type'].toString().toUpperCase()}",
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 12,
                              ),
                            ),
                            Text(
                              "Trained: ${combinedModelInfo['trained_at']}",
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 12,
                              ),
                            ),
                            Text(
                              "Accuracy: ${_calculateModelAccuracy(combinedModelInfo)}",
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 12,
                              ),
                            ),
                            if (combinedModelInfo['lookback_days'] != null)
                              Text(
                                "Lookback: ${combinedModelInfo['lookback_days']} days",
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            const SizedBox(height: 8),
                            GestureDetector(
                              onTap: () {
                                showDialog(
                                  context: context,
                                  builder: (context) => AlertDialog(
                                    title: const Text(
                                      'Model Performance Details',
                                    ),
                                    content: SingleChildScrollView(
                                      child: Text(
                                        _getAccuracyDetails(combinedModelInfo),
                                        style: const TextStyle(fontSize: 13),
                                      ),
                                    ),
                                    actions: [
                                      TextButton(
                                        onPressed: () => Navigator.pop(context),
                                        child: const Text('Close'),
                                      ),
                                    ],
                                  ),
                                );
                              },
                              child: Text(
                                "📊 View detailed metrics",
                                style: TextStyle(
                                  color: Colors.white.withOpacity(0.8),
                                  fontSize: 11,
                                  fontStyle: FontStyle.italic,
                                  decoration: TextDecoration.underline,
                                ),
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildColorLegendItem(Color color, String colorName, String meaning) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          Container(
            width: 20,
            height: 20,
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(4),
              border: Border.all(color: Colors.grey[400]!),
            ),
          ),
          const SizedBox(width: 10),
          Text(
            "$colorName:",
            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
          ),
          const SizedBox(width: 5),
          Text(
            meaning,
            style: TextStyle(fontSize: 12, color: Colors.grey[700]),
          ),
        ],
      ),
    );
  }

  Widget _buildInteractiveMapCard(Map<String, dynamic> mapTiles) {
    final center = mapTiles['center'];
    final lat = center['lat'];
    final lon = center['lon'];
    final tileUrl = mapTiles['tile_url'];

    return Card(
      color: Colors.white.withOpacity(0.98),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 15),
      elevation: 8,
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 60,
                  height: 60,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Colors.blue, Colors.cyan],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: const Icon(Icons.map, color: Colors.white, size: 35),
                ),
                const SizedBox(width: 15),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "Live Satellite Map",
                        style: TextStyle(
                          fontSize: 19,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        "Pan & zoom to explore",
                        style: TextStyle(fontSize: 13, color: Colors.grey),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),

            ClipRRect(
              borderRadius: BorderRadius.circular(15),
              child: SizedBox(
                height: 350,
                child: FlutterMap(
                  options: MapOptions(
                    initialCenter: LatLng(lat, lon),
                    initialZoom: 8.0,
                    minZoom: 5.0,
                    maxZoom: 13.0,
                  ),
                  children: [
                    TileLayer(
                      urlTemplate:
                          'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                      userAgentPackageName: 'com.example.airaware',
                    ),
                    TileLayer(
                      urlTemplate: tileUrl,
                      userAgentPackageName: 'com.example.airaware',
                      tileBuilder: (context, widget, tile) {
                        return Opacity(opacity: 0.7, child: widget);
                      },
                    ),
                    MarkerLayer(
                      markers: [
                        Marker(
                          point: LatLng(lat, lon),
                          width: 40,
                          height: 40,
                          child: const Icon(
                            Icons.location_pin,
                            color: Colors.red,
                            size: 40,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 15),

            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "🎨 Air Quality Scale:",
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 8),
                  _buildColorLegendItem(Colors.green, "Green", "Good"),
                  _buildColorLegendItem(
                    Colors.yellow[700]!,
                    "Yellow",
                    "Satisfactory",
                  ),
                  _buildColorLegendItem(Colors.orange, "Orange", "Moderate"),
                  _buildColorLegendItem(Colors.red, "Red", "Poor/Hazardous"),
                ],
              ),
            ),

            const SizedBox(height: 12),

            if (mapTiles['date_range'] != null)
              Text(
                "📅 ${mapTiles['date_range']['start']} to ${mapTiles['date_range']['end']}",
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
            if (mapTiles['pollutants_included'] != null)
              Text(
                "🛰️ Pollutants: ${(mapTiles['pollutants_included'] as List).join(', ')}",
                style: const TextStyle(fontSize: 11, color: Colors.grey),
              ),
            const SizedBox(height: 8),
            Row(
              children: [
                const Icon(Icons.touch_app, size: 16, color: Colors.grey),
                const SizedBox(width: 5),
                Text(
                  "Tap fullscreen icon to explore in detail",
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[700],
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _pollutantCard(
    String parameter,
    double value,
    String unit,
    String? source,
  ) {
    final level = _getAQILevel(parameter, value);
    final color = _getColorFromLevel(level);
    final category = _getAQICategoryFromLevel(level);

    IconData icon = Icons.air;
    if (parameter.toUpperCase().contains('PM')) {
      icon = Icons.grain;
    } else if (parameter.toUpperCase().contains('NO') ||
        parameter.toUpperCase().contains('SO')) {
      icon = Icons.cloud;
    } else if (parameter.toUpperCase().contains('CO')) {
      icon = Icons.smoke_free;
    } else if (parameter.toUpperCase().contains('CH4')) {
      icon = Icons.local_fire_department;
    }

    return Card(
      color: Colors.white.withOpacity(0.95),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 15),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  width: 50,
                  height: 50,
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(icon, color: color, size: 30),
                ),
                const SizedBox(width: 15),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        parameter,
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Text(
                        category,
                        style: TextStyle(
                          fontSize: 14,
                          color: color,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      value.toStringAsFixed(2),
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      unit,
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                  ],
                ),
              ],
            ),
            if (source != null && source.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Row(
                  children: [
                    const Icon(Icons.source, size: 12, color: Colors.grey),
                    const SizedBox(width: 4),
                    Text(
                      source,
                      style: const TextStyle(
                        fontSize: 10,
                        color: Colors.grey,
                        fontStyle: FontStyle.italic,
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionsCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.95),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Colors.purple, Colors.deepPurple],
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(
                  Icons.auto_graph,
                  color: Colors.white,
                  size: 30,
                ),
              ),
              const SizedBox(width: 15),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      "7-Day Forecast",
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      "AI Accuracy: ${_calculateModelAccuracy(_modelInfo)}",
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),

          ...(_predictions as List).map((pred) {
            return Container(
              margin: const EdgeInsets.only(bottom: 15),
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        pred['date'],
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: _getCategoryColor(
                            'pm25',
                            _safeGetNumber(pred['pm25']),
                          ).withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _getAQICategory(_safeGetNumber(pred['pm25'])),
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: _getCategoryColor(
                              'pm25',
                              _safeGetNumber(pred['pm25']),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 10,
                    runSpacing: 8,
                    children: [
                      _buildPollutantChip(
                        'PM2.5',
                        _safeGetNumber(pred['pm25']),
                        'µg/m³',
                      ),
                      _buildPollutantChip(
                        'PM10',
                        _safeGetNumber(pred['pm10']),
                        'µg/m³',
                      ),
                      _buildPollutantChip(
                        'NO₂',
                        _safeGetNumber(pred['no2']),
                        'µg/m³',
                      ),
                      _buildPollutantChip(
                        'SO₂',
                        _safeGetNumber(pred['so2']),
                        'µg/m³',
                      ),
                      _buildPollutantChip(
                        'CO',
                        _safeGetNumber(pred['co']),
                        'µg/m³',
                      ),
                      _buildPollutantChip(
                        'CH₄',
                        _safeGetNumber(pred['ch4']),
                        'ppb',
                      ),
                    ],
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }

  Widget _buildPollutantChip(String label, double value, String unit) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.grey[300]!),
      ),
      child: Text(
        "$label: ${value.toStringAsFixed(1)} $unit",
        style: const TextStyle(fontSize: 11),
      ),
    );
  }

  String _getAQICategory(double pm25) {
    if (pm25 <= 30) return "Good";
    if (pm25 <= 60) return "Moderate";
    if (pm25 <= 90) return "Unhealthy";
    return "Severe";
  }

  Widget _buildPredictionCharts() {
    List<FlSpot> pm25Spots = [];
    List<FlSpot> pm10Spots = [];
    List<FlSpot> ch4Spots = [];

    for (int i = 0; i < _predictions!.length; i++) {
      pm25Spots.add(
        FlSpot(i.toDouble(), _safeGetNumber(_predictions![i]['pm25'])),
      );
      pm10Spots.add(
        FlSpot(i.toDouble(), _safeGetNumber(_predictions![i]['pm10'])),
      );
      ch4Spots.add(
        FlSpot(i.toDouble(), _safeGetNumber(_predictions![i]['ch4'])),
      );
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.95),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                "📈 7-Day Pollutant Trends",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _calculateModelAccuracy(_modelInfo),
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 25),

          _buildChartSection(
            "PM2.5 Concentration",
            pm25Spots,
            Colors.red,
            'µg/m³',
          ),

          const SizedBox(height: 30),

          _buildChartSection(
            "PM10 Concentration",
            pm10Spots,
            Colors.orange,
            'µg/m³',
          ),

          const SizedBox(height: 30),

          _buildChartSection(
            "CH₄ (Methane) Concentration",
            ch4Spots,
            Colors.green,
            'µg/m³',
          ),
        ],
      ),
    );
  }

  Widget _buildChartSection(
    String title,
    List<FlSpot> spots,
    Color color,
    String unit,
  ) {
    double minY = spots.map((e) => e.y).reduce((a, b) => a < b ? a : b);
    double maxY = spots.map((e) => e.y).reduce((a, b) => a > b ? a : b);
    double adjustedMinY = (minY * 0.8).floorToDouble();
    double adjustedMaxY = (maxY * 1.2).ceilToDouble();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
        const SizedBox(height: 15),
        Container(
          height: 280,
          decoration: BoxDecoration(
            color: Colors.grey[50],
            borderRadius: BorderRadius.circular(15),
            border: Border.all(color: Colors.grey[300]!),
          ),
          padding: const EdgeInsets.all(16),
          child: LineChart(
            LineChartData(
              minY: adjustedMinY,
              maxY: adjustedMaxY,
              gridData: FlGridData(
                show: true,
                drawVerticalLine: true,
                horizontalInterval: (adjustedMaxY - adjustedMinY) / 5,
                getDrawingHorizontalLine: (value) {
                  return FlLine(
                    color: Colors.grey[300],
                    strokeWidth: 1,
                    dashArray: [5, 5],
                  );
                },
                getDrawingVerticalLine: (value) {
                  return FlLine(
                    color: Colors.grey[300],
                    strokeWidth: 1,
                    dashArray: [5, 5],
                  );
                },
              ),
              titlesData: FlTitlesData(
                leftTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 45,
                    interval: (adjustedMaxY - adjustedMinY) / 5,
                    getTitlesWidget: (value, meta) {
                      return Text(
                        '${value.toInt()} $unit',
                        style: const TextStyle(
                          fontSize: 10,
                          fontWeight: FontWeight.w500,
                        ),
                      );
                    },
                  ),
                ),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    getTitlesWidget: (value, meta) {
                      if (value.toInt() >= 0 &&
                          value.toInt() < _predictions!.length) {
                        final date = _predictions![value.toInt()]['date'];
                        final day = DateTime.parse(date).day;
                        final month = DateTime.parse(date).month;
                        return Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(
                            '$day/$month',
                            style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        );
                      }
                      return const Text('');
                    },
                  ),
                ),
                topTitles: const AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
                rightTitles: const AxisTitles(
                  sideTitles: SideTitles(showTitles: false),
                ),
              ),
              borderData: FlBorderData(
                show: true,
                border: Border.all(color: Colors.grey[400]!, width: 1),
              ),
              lineBarsData: [
                LineChartBarData(
                  spots: spots,
                  isCurved: true,
                  color: color,
                  barWidth: 4,
                  isStrokeCapRound: true,
                  dotData: FlDotData(
                    show: true,
                    getDotPainter: (spot, percent, barData, index) {
                      return FlDotCirclePainter(
                        radius: 6,
                        color: color,
                        strokeWidth: 2,
                        strokeColor: Colors.white,
                      );
                    },
                  ),
                  belowBarData: BarAreaData(
                    show: true,
                    gradient: LinearGradient(
                      colors: [color.withOpacity(0.3), color.withOpacity(0.05)],
                      begin: Alignment.topCenter,
                      end: Alignment.bottomCenter,
                    ),
                  ),
                ),
              ],
              lineTouchData: LineTouchData(
                enabled: true,
                touchTooltipData: LineTouchTooltipData(
                  getTooltipItems: (touchedSpots) {
                    return touchedSpots.map((spot) {
                      final date = _predictions![spot.x.toInt()]['date'];
                      return LineTooltipItem(
                        '$date\n${spot.y.toStringAsFixed(1)} $unit',
                        const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      );
                    }).toList();
                  },
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

// ============ HEALTH TIPS SCREEN ============
class HealthTipsScreen extends StatelessWidget {
  final int aqiLevel;
  final String worstPollutant;
  final List<dynamic>? predictions;

  const HealthTipsScreen({
    super.key,
    required this.aqiLevel,
    required this.worstPollutant,
    this.predictions,
  });

  String _normalizePollutantName(String name) {
    String normalized = name
        .toUpperCase()
        .replaceAll('₂', '2')
        .replaceAll('₃', '3')
        .replaceAll('₄', '4')
        .replaceAll(' ', '');

    if (normalized.contains('PM2.5') || normalized == 'PM25') return 'PM2.5';
    if (normalized.contains('PM10')) return 'PM10';
    if (normalized.contains('NO2')) return 'NO2';
    if (normalized.contains('SO2')) return 'SO2';
    if (normalized.contains('CO') && !normalized.contains('CO2')) return 'CO';
    if (normalized.contains('CH4')) return 'CH4';

    return name;
  }

  double _safeGetNumber(dynamic value) {
    if (value == null) return 0.0;
    if (value is num) return value.toDouble();
    if (value is String) return double.tryParse(value) ?? 0.0;
    return 0.0;
  }

  @override
  Widget build(BuildContext context) {
    final generalAdvice = HealthTipsData.getGeneralAdviceByLevel()[aqiLevel];
    final normalizedPollutant = _normalizePollutantName(worstPollutant);
    final pollutantInfo = HealthTipsData.pollutantInfo[normalizedPollutant];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Health Tips & Precautions'),
        backgroundColor: Colors.teal,
        elevation: 0,
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF56CCF2), Color(0xFF2F80ED)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.1),
                        blurRadius: 10,
                        offset: const Offset(0, 5),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Text(
                        '${generalAdvice?.emoji ?? "😊"} Air Quality: ${generalAdvice?.category ?? "Unknown"}',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 10),
                      Text(
                        generalAdvice?.advice ?? "No advice available",
                        style: const TextStyle(
                          fontSize: 16,
                          color: Colors.black54,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 25),

                if (pollutantInfo != null) ...[
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 10,
                          offset: const Offset(0, 5),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              pollutantInfo.emoji,
                              style: const TextStyle(fontSize: 32),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    'Primary Concern',
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey,
                                    ),
                                  ),
                                  Text(
                                    pollutantInfo.name,
                                    style: const TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.black87,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 15),
                        Text(
                          pollutantInfo.description,
                          style: const TextStyle(
                            fontSize: 14,
                            color: Colors.black54,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 20),

                  _buildSectionCard(
                    title: '🫁 Health Effects',
                    items: pollutantInfo.healthEffects,
                    color: Colors.red.shade50,
                  ),

                  const SizedBox(height: 20),

                  _buildSectionCard(
                    title: '✅ Recommended Precautions',
                    items: pollutantInfo.precautions,
                    color: Colors.green.shade50,
                  ),

                  const SizedBox(height: 20),

                  _buildSectionCard(
                    title: '⚠️ Vulnerable Groups',
                    items: pollutantInfo.vulnerableGroups,
                    color: Colors.orange.shade50,
                  ),
                ],

                const SizedBox(height: 25),

                if (generalAdvice != null)
                  _buildSectionCard(
                    title: '📋 General Actions to Take',
                    items: generalAdvice.actions,
                    color: Colors.blue.shade50,
                  ),

                const SizedBox(height: 20),

                _buildSectionCard(
                  title: '🏠 Indoor Air Quality Tips',
                  items: HealthTipsData.getIndoorAirQualityTips(),
                  color: Colors.purple.shade50,
                ),

                const SizedBox(height: 20),

                _buildSectionCard(
                  title: '🍎 Dietary Recommendations',
                  items: HealthTipsData.getDietaryRecommendations(),
                  color: Colors.teal.shade50,
                ),

                const SizedBox(height: 20),

                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.red.shade200, width: 2),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Row(
                        children: [
                          Icon(
                            Icons.phone_in_talk,
                            color: Colors.red,
                            size: 28,
                          ),
                          SizedBox(width: 10),
                          Text(
                            'Emergency Contacts',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.black87,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 15),
                      ...HealthTipsData.getEmergencyContacts().map((contact) {
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 5),
                          child: Text(
                            contact,
                            style: const TextStyle(
                              fontSize: 16,
                              color: Colors.black87,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        );
                      }),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                if (predictions != null && predictions!.isNotEmpty) ...[
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.orange.shade50,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: Colors.orange.shade200,
                        width: 2,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(
                              Icons.calendar_today,
                              color: Colors.orange,
                              size: 28,
                            ),
                            SizedBox(width: 10),
                            Text(
                              '7-Day Forecast Alert',
                              style: TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.black87,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 15),
                        _buildForecastWarnings(),
                      ],
                    ),
                  ),
                ],

                const SizedBox(height: 30),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSectionCard({
    required String title,
    required List<String> items,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Colors.black87,
            ),
          ),
          const SizedBox(height: 15),
          ...items.asMap().entries.map((entry) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    margin: const EdgeInsets.only(top: 5, right: 10),
                    width: 6,
                    height: 6,
                    decoration: const BoxDecoration(
                      color: Colors.black54,
                      shape: BoxShape.circle,
                    ),
                  ),
                  Expanded(
                    child: Text(
                      entry.value,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.black87,
                        height: 1.5,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildForecastWarnings() {
    if (predictions == null || predictions!.isEmpty) {
      return const Text(
        'No forecast data available',
        style: TextStyle(color: Colors.black54),
      );
    }

    List<Widget> warnings = [];
    int poorDaysCount = 0;

    for (var pred in predictions!) {
      double pm25 = _safeGetNumber(pred['pm25']);
      if (pm25 > 60) {
        poorDaysCount++;
      }
    }

    if (poorDaysCount > 0) {
      warnings.add(
        Text(
          '⚠️ Expected $poorDaysCount day(s) with unhealthy air quality in the next week',
          style: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            color: Colors.black87,
          ),
        ),
      );
      warnings.add(const SizedBox(height: 10));
      warnings.add(
        const Text(
          'Plan indoor activities for those days. Keep medications handy.',
          style: TextStyle(fontSize: 14, color: Colors.black54),
        ),
      );
    } else {
      warnings.add(
        const Text(
          '✅ Air quality expected to remain acceptable throughout the week',
          style: TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.bold,
            color: Colors.green,
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: warnings,
    );
  }
}

// ============ FULL SCREEN MAP VIEW ============
class FullScreenMapView extends StatelessWidget {
  final Map<String, dynamic> mapTiles;
  final String cityName;

  const FullScreenMapView({
    super.key,
    required this.mapTiles,
    required this.cityName,
  });

  @override
  Widget build(BuildContext context) {
    final center = mapTiles['center'];
    final lat = center['lat'];
    final lon = center['lon'];
    final tileUrl = mapTiles['tile_url'];

    return Scaffold(
      appBar: AppBar(
        title: Text('Air Quality Map - $cityName'),
        backgroundColor: Colors.teal,
        elevation: 0,
      ),
      body: Stack(
        children: [
          FlutterMap(
            options: MapOptions(
              initialCenter: LatLng(lat, lon),
              initialZoom: 9.0,
              minZoom: 5.0,
              maxZoom: 13.0,
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'com.example.airaware',
              ),
              TileLayer(
                urlTemplate: tileUrl,
                userAgentPackageName: 'com.example.airaware',
                tileBuilder: (context, widget, tile) {
                  return Opacity(opacity: 0.7, child: widget);
                },
              ),
              MarkerLayer(
                markers: [
                  Marker(
                    point: LatLng(lat, lon),
                    width: 50,
                    height: 50,
                    child: const Icon(
                      Icons.location_pin,
                      color: Colors.red,
                      size: 50,
                    ),
                  ),
                ],
              ),
            ],
          ),
          Positioned(
            bottom: 20,
            left: 20,
            right: 20,
            child: Card(
              color: Colors.white.withOpacity(0.95),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(15),
              ),
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      "🎨 Air Quality Scale",
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildLegendItem(Colors.green, "Good"),
                        _buildLegendItem(Colors.yellow[700]!, "Moderate"),
                        _buildLegendItem(Colors.orange, "Poor"),
                        _buildLegendItem(Colors.red, "Severe"),
                      ],
                    ),
                    const SizedBox(height: 8),
                    if (mapTiles['date_range'] != null)
                      Text(
                        "📅 ${mapTiles['date_range']['start']} to ${mapTiles['date_range']['end']}",
                        style: const TextStyle(
                          fontSize: 10,
                          color: Colors.grey,
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLegendItem(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 16,
          height: 16,
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 11)),
      ],
    );
  }
}
