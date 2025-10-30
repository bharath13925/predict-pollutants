// health_tips_data.dart
// Place this file in the same directory as main.dart (lib folder)

class PollutantInfo {
  final String name;
  final String description;
  final String emoji;
  final List<String> healthEffects;
  final List<String> precautions;
  final List<String> vulnerableGroups;

  PollutantInfo({
    required this.name,
    required this.description,
    required this.emoji,
    required this.healthEffects,
    required this.precautions,
    required this.vulnerableGroups,
  });
}

class HealthTipsData {
  static final Map<String, PollutantInfo> pollutantInfo = {
    'PM2.5': PollutantInfo(
      name: 'PM2.5 (Fine Particulate Matter)',
      description:
          'Tiny particles less than 2.5 micrometers that can penetrate deep into lungs and bloodstream',
      emoji: '🔴',
      healthEffects: [
        'Respiratory irritation and breathing difficulties',
        'Aggravation of asthma and other lung diseases',
        'Increased risk of heart attacks and strokes',
        'Long-term exposure linked to lung cancer',
        'Reduced lung function in children',
      ],
      precautions: [
        'Stay indoors and keep windows closed during high pollution',
        'Use air purifiers with HEPA filters at home',
        'Wear N95 or N99 masks when going outside',
        'Avoid outdoor exercise, especially in mornings',
        'Monitor air quality before planning outdoor activities',
        'Keep indoor plants to improve air quality',
        'Use exhaust fans while cooking',
      ],
      vulnerableGroups: [
        'Children and elderly people',
        'People with asthma or respiratory conditions',
        'Heart disease patients',
        'Pregnant women',
      ],
    ),
    'PM10': PollutantInfo(
      name: 'PM10 (Coarse Particulate Matter)',
      description:
          'Particles less than 10 micrometers including dust, pollen, and mold',
      emoji: '🟠',
      healthEffects: [
        'Eye, nose, and throat irritation',
        'Coughing and difficulty breathing',
        'Worsening of asthma symptoms',
        'Reduced lung function',
        'Increased respiratory infections',
      ],
      precautions: [
        'Wear dust masks (N95) during dusty conditions',
        'Keep doors and windows closed on windy days',
        'Use wet mopping instead of dry sweeping',
        'Avoid construction sites and dusty areas',
        'Rinse eyes with clean water if irritated',
        'Stay hydrated to help body flush out toxins',
        'Use saline nasal spray to clear nasal passages',
      ],
      vulnerableGroups: [
        'People with allergies',
        'Asthma patients',
        'Children and elderly',
        'Outdoor workers',
      ],
    ),
    'NO2': PollutantInfo(
      name: 'NO₂ (Nitrogen Dioxide)',
      description:
          'Reddish-brown gas produced from vehicle emissions and industrial processes',
      emoji: '🟤',
      healthEffects: [
        'Inflammation of airways',
        'Reduced immunity to lung infections',
        'Increased asthma attacks',
        'Chronic bronchitis in long-term exposure',
        'Reduced lung development in children',
      ],
      precautions: [
        'Avoid busy roads during peak traffic hours',
        'Keep distance from vehicle exhaust',
        'Ventilate your home during low traffic periods',
        'Use public transport instead of personal vehicles',
        'Plant trees and shrubs around your home',
        'Service vehicles regularly to reduce emissions',
        'Avoid idling your vehicle',
      ],
      vulnerableGroups: [
        'Children and teenagers',
        'Asthma and COPD patients',
        'People near highways',
        'Traffic police and street vendors',
      ],
    ),
    'SO2': PollutantInfo(
      name: 'SO₂ (Sulfur Dioxide)',
      description:
          'Colorless gas with a pungent odor from burning fossil fuels',
      emoji: '🟡',
      healthEffects: [
        'Breathing problems and throat irritation',
        'Asthma attacks and wheezing',
        'Chest tightness and shortness of breath',
        'Cardiovascular problems',
        'Eye irritation and tearing',
      ],
      precautions: [
        'Avoid areas near industrial facilities',
        'Stay indoors when SO₂ levels are high',
        'Use air conditioning with proper filters',
        'Keep rescue inhalers handy if asthmatic',
        'Avoid vigorous outdoor activities',
        'Report industrial emissions to authorities',
        'Install carbon filters in ventilation systems',
      ],
      vulnerableGroups: [
        'Asthma patients',
        'People with heart conditions',
        'Elderly population',
        'Industrial area residents',
      ],
    ),
    'CO': PollutantInfo(
      name: 'CO (Carbon Monoxide)',
      description:
          'Odorless, colorless gas that reduces oxygen delivery to organs',
      emoji: '⚫',
      healthEffects: [
        'Headaches and dizziness',
        'Reduced oxygen to vital organs',
        'Chest pain in heart patients',
        'Fatigue and confusion',
        'Death at very high levels',
      ],
      precautions: [
        'Install CO detectors in your home',
        'Never leave car engine running in garage',
        'Ensure proper ventilation when using heaters',
        'Avoid enclosed parking areas with running engines',
        'Service gas appliances regularly',
        'Never use BBQ grills indoors',
        'Seek fresh air immediately if you feel dizzy',
      ],
      vulnerableGroups: [
        'Heart disease patients',
        'Pregnant women and fetuses',
        'Anemia patients',
        'People in enclosed spaces',
      ],
    ),
    'O3': PollutantInfo(
      name: 'O₃ (Ground-level Ozone)',
      description:
          'Formed when pollutants react with sunlight; different from protective upper atmosphere ozone',
      emoji: '🔵',
      healthEffects: [
        'Chest pain and coughing',
        'Throat irritation and congestion',
        'Worsening of bronchitis and emphysema',
        'Reduced lung function',
        'Increased susceptibility to respiratory infections',
      ],
      precautions: [
        'Avoid outdoor activities during afternoon (peak O₃ hours)',
        'Exercise early morning or evening when ozone is lower',
        'Stay in air-conditioned spaces on high ozone days',
        'Reduce car trips and combine errands',
        'Refuel vehicles in evening to reduce ozone formation',
        'Keep asthma action plan updated',
        'Stay hydrated and breathe through nose',
      ],
      vulnerableGroups: [
        'Children and active adults',
        'Asthma and COPD patients',
        'Outdoor workers and athletes',
        'People with respiratory conditions',
      ],
    ),
  };

  static Map<int, GeneralAdvice> getGeneralAdviceByLevel() {
    return {
      0: GeneralAdvice(
        category: 'Good',
        emoji: '😊',
        color: 'green',
        advice: 'Air quality is satisfactory. Enjoy outdoor activities!',
        actions: [
          'Perfect time for outdoor exercise and activities',
          'Open windows for natural ventilation',
          'Great day for children to play outside',
          'Maintain regular outdoor routine',
        ],
      ),
      1: GeneralAdvice(
        category: 'Satisfactory',
        emoji: '🙂',
        color: 'yellow',
        advice: 'Air quality is acceptable for most people.',
        actions: [
          'Unusually sensitive people should consider reducing prolonged outdoor exertion',
          'Generally safe for outdoor activities',
          'Monitor if you have respiratory sensitivity',
          'Good time for moderate outdoor exercise',
        ],
      ),
      2: GeneralAdvice(
        category: 'Moderate',
        emoji: '😐',
        color: 'orange',
        advice: 'Sensitive groups should limit prolonged outdoor exposure.',
        actions: [
          'Children and elderly should reduce outdoor exertion',
          'People with respiratory conditions should limit prolonged activities',
          'Consider wearing masks during extended outdoor time',
          'Keep windows closed during peak pollution hours',
          'Reschedule strenuous outdoor activities if possible',
        ],
      ),
      3: GeneralAdvice(
        category: 'Poor',
        emoji: '😷',
        color: 'red',
        advice: 'Everyone should reduce outdoor activities.',
        actions: [
          'Wear N95 masks when going outside',
          'Avoid prolonged outdoor exertion for everyone',
          'Keep windows and doors closed',
          'Use air purifiers indoors',
          'Sensitive groups should stay indoors',
          'Reschedule outdoor events if possible',
          'Keep emergency medications handy',
        ],
      ),
      4: GeneralAdvice(
        category: 'Very Poor',
        emoji: '🚨',
        color: 'purple',
        advice: 'Stay indoors and avoid all outdoor activities.',
        actions: [
          'Everyone should avoid outdoor activities',
          'Stay indoors with windows and doors closed',
          'Use air purifiers continuously',
          'Wear N95/N99 masks if you must go outside',
          'Avoid all physical exertion',
          'Check on elderly neighbors and vulnerable people',
          'Consider relocating sensitive individuals temporarily',
          'Keep emergency contact numbers ready',
        ],
      ),
      5: GeneralAdvice(
        category: 'Severe',
        emoji: '☠️',
        color: 'brown',
        advice: 'Health emergency! Minimize all exposure.',
        actions: [
          'Do not go outside unless absolutely necessary',
          'Seal all windows and doors',
          'Run air purifiers on maximum',
          'Wear N99 masks with proper fit if you must go out',
          'Seek medical attention if experiencing symptoms',
          'Consider evacuating if possible',
          'Keep medications and emergency supplies ready',
          'Monitor vulnerable family members closely',
          'Follow official health advisories',
        ],
      ),
    };
  }

  static List<String> getEmergencyContacts() {
    return [
      '🚨 Emergency Medical: 108 / 102',
      '🏥 Ambulance: 108',
      '🫁 Poison Control: 1066',
      '👮 Police: 100',
      '🚒 Fire: 101',
    ];
  }

  static List<String> getIndoorAirQualityTips() {
    return [
      'Use exhaust fans in kitchen and bathrooms',
      'Avoid smoking indoors',
      'Keep indoor plants (Snake plant, Spider plant, Peace lily)',
      'Vacuum regularly with HEPA filter',
      'Control humidity levels (30-50%)',
      'Avoid using harsh chemicals for cleaning',
      'Let new furniture off-gas in ventilated area',
      'Use natural air fresheners like essential oils',
      'Regular maintenance of HVAC systems',
      'Keep indoor spaces clutter-free for better air circulation',
    ];
  }

  static List<String> getDietaryRecommendations() {
    return [
      '🥦 Eat antioxidant-rich foods (berries, green tea, dark chocolate)',
      '🥕 Increase vitamin C intake (citrus fruits, bell peppers)',
      '🥬 Consume leafy greens (spinach, kale, broccoli)',
      '🐟 Include omega-3 fatty acids (fish, walnuts, flaxseeds)',
      '🧄 Add turmeric and ginger to your diet',
      '💧 Stay well-hydrated (8-10 glasses of water daily)',
      '🍯 Consume honey to soothe throat irritation',
      '🫖 Drink herbal teas (tulsi, ginger, green tea)',
      '🥜 Include nuts and seeds for vitamin E',
      '🍊 Boost immunity with zinc-rich foods',
    ];
  }
}

class GeneralAdvice {
  final String category;
  final String emoji;
  final String color;
  final String advice;
  final List<String> actions;

  GeneralAdvice({
    required this.category,
    required this.emoji,
    required this.color,
    required this.advice,
    required this.actions,
  });
}
