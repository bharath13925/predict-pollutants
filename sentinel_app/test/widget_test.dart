import 'package:flutter_test/flutter_test.dart';
import 'package:sentinel_app/main.dart'; // 👈 adjust if your project name differs

void main() {
  testWidgets('Check AirAware app loads with main buttons', (
    WidgetTester tester,
  ) async {
    // Build the app
    await tester.pumpWidget(const AirAwareApp());

    // Verify AppBar title
    expect(find.text('AirAware (IQAir + GEE)'), findsOneWidget);

    // Verify key buttons exist
    expect(find.text('Use GPS'), findsOneWidget);
    expect(find.text('Enter City'), findsOneWidget);
    expect(find.text('IQAir Data'), findsOneWidget);
    expect(find.text('Fetch from GEE'), findsOneWidget);
  });
}
