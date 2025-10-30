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

void main() {
  runApp(const AirAwareApp());
}

class AirAwareApp extends StatelessWidget {
  const AirAwareApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Air Aware - Complete',
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
  final String baseUrl = "http://10.117.36.104:5000/api";

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
      _updateStep("📥 Collecting 2 months of historical data...");

      final uri = Uri.parse("$baseUrl/full-analysis");
      final response = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'city': city}),
          )
          .timeout(
            const Duration(seconds: 60000), // 10 minutes for full analysis
          );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);

        setState(() {
          // Store all components
          _historicalData = data['historical_data'];
          _trainingResult = data['model_training'];

          // Process current AQI data
          if (data['current_data'] != null) {
            _aqiData = _processAQIData(data['current_data'], city);
          }

          // Store predictions
          if (data['predictions'] != null &&
              data['predictions']['predictions'] != null) {
            _predictions = data['predictions']['predictions'];
            _modelInfo = data['predictions']['model_info'];
          }

          _analysisInProgress = false;
        });

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                '✅ Full analysis complete!\n'
                'Data collected: ${_historicalData?['days_collected'] ?? 0} days\n'
                'Model accuracy: ${(_trainingResult?['test_score'] ?? 0 * 100).toStringAsFixed(1)}%',
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
      print("❌ Error: $e");
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
    print(step);
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
      print("🌐 Fetching from: $uri");

      final response = await http
          .get(uri)
          .timeout(const Duration(seconds: 300));

      print("📡 Response status: ${response.statusCode}");

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        print("✅ JSON decoded successfully");

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
      print("❌ Error: $e");
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
            pollutantData[key] = {
              "value": value,
              "unit": pollutantInfo["unit"]?.toString() ?? "",
              "parameter": pollutantInfo["parameter"]?.toString() ?? key,
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
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  '✅ Predictions loaded!\n'
                  'Model: ${_modelInfo?['model_type'] ?? 'Unknown'}\n'
                  'Accuracy: ${(_modelInfo?['test_score'] ?? 0 * 100).toStringAsFixed(1)}%',
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
      print("Could not load predictions: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Could not load predictions: $e\n\nTip: Run "Full AI Analysis" first to train the model',
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

    pollutants.forEach((parameter, data) {
      int level = _getAQILevel(parameter, data["value"], data["unit"]);
      if (level > maxLevel) {
        maxLevel = level;
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

  int _getAQILevel(String parameter, double value, String unit) {
    String paramNormalized = parameter
        .toUpperCase()
        .replaceAll('₂', '2')
        .replaceAll('₃', '3');

    double valueInUgM3 = value;

    if (unit.toLowerCase() == "ppb") {
      if (paramNormalized.contains("NO2")) {
        valueInUgM3 = value * 1.88;
      } else if (paramNormalized.contains("SO2")) {
        valueInUgM3 = value * 2.62;
      } else if (paramNormalized.contains("CO")) {
        valueInUgM3 = value * 1.145;
      } else if (paramNormalized.contains("O3")) {
        valueInUgM3 = value * 1.96;
      }
    }

    if (paramNormalized.contains("PM2.5") || paramNormalized == "PM25") {
      if (valueInUgM3 <= 30) return 0;
      if (valueInUgM3 <= 60) return 1;
      if (valueInUgM3 <= 90) return 2;
      if (valueInUgM3 <= 120) return 3;
      if (valueInUgM3 <= 250) return 4;
      return 5;
    }

    if (paramNormalized.contains("PM10")) {
      if (valueInUgM3 <= 50) return 0;
      if (valueInUgM3 <= 100) return 1;
      if (valueInUgM3 <= 250) return 2;
      if (valueInUgM3 <= 350) return 3;
      if (valueInUgM3 <= 430) return 4;
      return 5;
    }

    if (paramNormalized.contains("NO2")) {
      if (valueInUgM3 <= 40) return 0;
      if (valueInUgM3 <= 80) return 1;
      if (valueInUgM3 <= 180) return 2;
      if (valueInUgM3 <= 280) return 3;
      if (valueInUgM3 <= 400) return 4;
      return 5;
    }

    if (paramNormalized.contains("SO2")) {
      if (valueInUgM3 <= 40) return 0;
      if (valueInUgM3 <= 80) return 1;
      if (valueInUgM3 <= 380) return 2;
      if (valueInUgM3 <= 800) return 3;
      if (valueInUgM3 <= 1600) return 4;
      return 5;
    }

    if (paramNormalized.contains("CO")) {
      if (valueInUgM3 <= 1000) return 0;
      if (valueInUgM3 <= 2000) return 1;
      if (valueInUgM3 <= 10000) return 2;
      if (valueInUgM3 <= 17000) return 3;
      if (valueInUgM3 <= 34000) return 4;
      return 5;
    }

    if (paramNormalized.contains("O3")) {
      if (valueInUgM3 <= 50) return 0;
      if (valueInUgM3 <= 100) return 1;
      if (valueInUgM3 <= 168) return 2;
      if (valueInUgM3 <= 208) return 3;
      if (valueInUgM3 <= 748) return 4;
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
    return Colors.grey;
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
                // Header
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
                    "Real-time data + AI-powered predictions",
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white70, fontSize: 16),
                  ),
                ),
                const SizedBox(height: 30),

                // Input Card
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
                                icon: const Icon(Icons.psychology, size: 20),
                                label: const Text('Full AI'),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: Colors.purple,
                                  foregroundColor: Colors.white,
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 15,
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

                // Loading indicator
                if (_loading && _aqiData == null)
                  const CircularProgressIndicator(color: Colors.white),

                // Analysis Progress
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

                // Current Data Section
                if (_aqiData != null && !_analysisInProgress)
                  Column(
                    children: [
                      // Overall AQI Card
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
                            ],
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      // Interactive Map Section
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

                      // Ground Station Data
                      Text(
                        "📊 Ground Station Measurements",
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
                            ),
                          )
                          .toList(),

                      const SizedBox(height: 20),

                      // Load predictions button if not already loaded
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

                      // Timestamp
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

                // Historical Data Summary
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
                          "Days: ${_historicalData!['days_collected']}",
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

                // Training Result Summary
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
                          "Model: ${_trainingResult!['model_type']?.toString().toUpperCase() ?? 'Unknown'}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        Text(
                          "Accuracy: ${(_trainingResult!['test_score'] * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        if (_trainingResult!['cv_score'] != null)
                          Text(
                            "Cross-validation: ${(_trainingResult!['cv_score'] * 100).toStringAsFixed(1)}%",
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                  ),

                // Predictions Section
                if (_predictions != null) ...[
                  const SizedBox(height: 30),
                  _buildPredictionsCard(),
                  const SizedBox(height: 20),
                  _buildPredictionCharts(),
                ],

                // Model Info
                if (_modelInfo != null)
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
                          "🤖 ML Model Info",
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Type: ${_modelInfo!['model_type']?.toString().toUpperCase() ?? 'Unknown'}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        Text(
                          "Trained: ${_modelInfo!['trained_at'] ?? 'Unknown'}",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        Text(
                          "Accuracy: ${(_modelInfo!['test_score'] * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                        if (_modelInfo!['test_mae'] != null)
                          Text(
                            "MAE: ${_modelInfo!['test_mae'].toStringAsFixed(2)}",
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                      ],
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

            // Interactive Map
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

            // Color Legend
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

  Widget _pollutantCard(String parameter, double value, String unit) {
    final level = _getAQILevel(parameter, value, unit);
    final color = _getColorFromLevel(level);
    final category = _getAQICategoryFromLevel(level);

    return Card(
      color: Colors.white.withOpacity(0.95),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 15),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: color.withOpacity(0.2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(Icons.air, color: color, size: 30),
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
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      "7-Day Forecast",
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    Text(
                      "AI-powered predictions",
                      style: TextStyle(fontSize: 14, color: Colors.grey),
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
                            pred['pm25'],
                          ).withOpacity(0.2),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          _getAQICategory(pred['pm25']),
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                            color: _getCategoryColor('pm25', pred['pm25']),
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
                      _buildPollutantChip('PM2.5', pred['pm25'], 'µg/m³'),
                      _buildPollutantChip('PM10', pred['pm10'], 'µg/m³'),
                      _buildPollutantChip('NO₂', pred['no2'], 'µg/m³'),
                      _buildPollutantChip('SO₂', pred['so2'], 'µg/m³'),
                      _buildPollutantChip('CO', pred['co'], 'µg/m³'),
                      _buildPollutantChip('O₃', pred['o3'], 'µg/m³'),
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

    for (int i = 0; i < _predictions!.length; i++) {
      pm25Spots.add(FlSpot(i.toDouble(), _predictions![i]['pm25']));
      pm10Spots.add(FlSpot(i.toDouble(), _predictions![i]['pm10']));
    }

    // Calculate min/max for better scaling
    double minPM25 = pm25Spots.map((e) => e.y).reduce((a, b) => a < b ? a : b);
    double maxPM25 = pm25Spots.map((e) => e.y).reduce((a, b) => a > b ? a : b);
    double minY = (minPM25 * 0.8).floorToDouble();
    double maxY = (maxPM25 * 1.2).ceilToDouble();

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
                child: const Text(
                  "AI Predicted",
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 25),

          // PM2.5 Chart
          const Text(
            "PM2.5 Concentration",
            style: TextStyle(
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
                minY: minY,
                maxY: maxY,
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: true,
                  horizontalInterval: (maxY - minY) / 5,
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
                      interval: (maxY - minY) / 5,
                      getTitlesWidget: (value, meta) {
                        return Text(
                          '${value.toInt()} µg/m³',
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
                  topTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                ),
                borderData: FlBorderData(
                  show: true,
                  border: Border.all(color: Colors.grey[400]!, width: 1),
                ),
                lineBarsData: [
                  LineChartBarData(
                    spots: pm25Spots,
                    isCurved: true,
                    color: Colors.red,
                    barWidth: 4,
                    isStrokeCapRound: true,
                    dotData: FlDotData(
                      show: true,
                      getDotPainter: (spot, percent, barData, index) {
                        return FlDotCirclePainter(
                          radius: 6,
                          color: Colors.red,
                          strokeWidth: 2,
                          strokeColor: Colors.white,
                        );
                      },
                    ),
                    belowBarData: BarAreaData(
                      show: true,
                      gradient: LinearGradient(
                        colors: [
                          Colors.red.withOpacity(0.3),
                          Colors.red.withOpacity(0.05),
                        ],
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
                          '$date\n${spot.y.toStringAsFixed(1)} µg/m³',
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
          const SizedBox(height: 30),

          // PM10 Chart
          const Text(
            "PM10 Concentration",
            style: TextStyle(
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
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: true,
                  getDrawingHorizontalLine: (value) {
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
                      getTitlesWidget: (value, meta) {
                        return Text(
                          '${value.toInt()} µg/m³',
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
                  topTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                  rightTitles: AxisTitles(
                    sideTitles: SideTitles(showTitles: false),
                  ),
                ),
                borderData: FlBorderData(
                  show: true,
                  border: Border.all(color: Colors.grey[400]!, width: 1),
                ),
                lineBarsData: [
                  LineChartBarData(
                    spots: pm10Spots,
                    isCurved: true,
                    color: Colors.orange,
                    barWidth: 4,
                    isStrokeCapRound: true,
                    dotData: FlDotData(
                      show: true,
                      getDotPainter: (spot, percent, barData, index) {
                        return FlDotCirclePainter(
                          radius: 6,
                          color: Colors.orange,
                          strokeWidth: 2,
                          strokeColor: Colors.white,
                        );
                      },
                    ),
                    belowBarData: BarAreaData(
                      show: true,
                      gradient: LinearGradient(
                        colors: [
                          Colors.orange.withOpacity(0.3),
                          Colors.orange.withOpacity(0.05),
                        ],
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
                          '$date\n${spot.y.toStringAsFixed(1)} µg/m³',
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
      ),
    );
  }
}

// FULL SCREEN MAP VIEW
class FullScreenMapView extends StatelessWidget {
  final Map<String, dynamic> mapTiles;
  final String cityName;
  final List<dynamic>? stations;

  const FullScreenMapView({
    super.key,
    required this.mapTiles,
    required this.cityName,
    this.stations,
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
                  if (stations != null)
                    ...stations!
                        .map((station) {
                          if (station["coordinates"] != null) {
                            return Marker(
                              point: LatLng(
                                station["coordinates"]["latitude"],
                                station["coordinates"]["longitude"],
                              ),
                              width: 35,
                              height: 35,
                              child: GestureDetector(
                                onTap: () {
                                  _showStationInfo(context, station);
                                },
                                child: const Icon(
                                  Icons.sensors,
                                  color: Colors.blue,
                                  size: 30,
                                ),
                              ),
                            );
                          }
                          return null;
                        })
                        .whereType<Marker>()
                        .toList(),
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
                    Text(
                      "📅 ${mapTiles['date_range']['start']} to ${mapTiles['date_range']['end']}",
                      style: const TextStyle(fontSize: 10, color: Colors.grey),
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

  void _showStationInfo(BuildContext context, Map<String, dynamic> station) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(station["name"] ?? "Station"),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              if (station["coordinates"] != null)
                Text(
                  "📍 ${station["coordinates"]["latitude"].toStringAsFixed(4)}, ${station["coordinates"]["longitude"].toStringAsFixed(4)}",
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
              const SizedBox(height: 10),
              const Text(
                "Measurements:",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              if (station["measurements"] != null)
                ...(station["measurements"] as List).map((m) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(m["parameter"] ?? "Unknown"),
                        Text(
                          "${m["value"]?.toStringAsFixed(2) ?? 'N/A'} ${m["unit"] ?? ''}",
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ],
                    ),
                  );
                }).toList(),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text("Close"),
          ),
        ],
      ),
    );
  }
}
