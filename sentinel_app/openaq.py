import openaq
from geopy.geocoders import Nominatim
from openaq import AuthError, BadRequestError
from flask import Flask, request, jsonify
import os
from datetime import datetime, timedelta

# --- OpenAQ Configuration ---
API_KEY = os.environ.get("OPENAQ_API_KEY", "2518a7f3080acce0c735b03f23005d33cff8718480d07a9e8ae904dece78564a")

# Initialize OpenAQ client
client = openaq.OpenAQ(api_key=API_KEY)
# Initialize geolocator for converting city names to coordinates
geolocator = Nominatim(user_agent="air-quality-checker")

# --- Flask App Configuration ---
app = Flask(__name__)

def get_coordinates(city_name):
    """Get (latitude, longitude) of a city using Nominatim."""
    try:
        location = geolocator.geocode(city_name)
        if location:
            print(f"Found coordinates for {city_name}: {location.latitude}, {location.longitude}")
            return (location.latitude, location.longitude)
        else:
            print(f"No coordinates found for {city_name}")
            return None
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None

def fetch_air_quality_data(city_name):
    """Fetch air quality data for a city and format it for the response."""
    coords = get_coordinates(city_name)
    if not coords:
        return {"error": f"Could not find coordinates for {city_name}"}, 404

    try:
        # Step 1: Get nearby air quality stations
        print(f"Fetching locations near {coords}...")
        locations_response = client.locations.list(
            coordinates=coords,
            radius=25000,  # 25 km
            limit=100
        )
        
        locations = locations_response.results if hasattr(locations_response, 'results') else []
        print(f"Found {len(locations)} locations")
        
        if not locations:
            return {"message": f"No air quality stations found near {city_name}"}, 404

        # Target pollutants we want to fetch
        target_pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        
        all_station_data = []
        city_pollutants = {}  # Store latest pollutant data for the city
        
        for loc in locations:
            print(f"\nProcessing location: {loc.name} (ID: {loc.id})")
            
            station_data = {
                "id": loc.id,
                "name": loc.name,
                "coordinates": {
                    "latitude": loc.coordinates.latitude,
                    "longitude": loc.coordinates.longitude
                } if hasattr(loc, 'coordinates') else None,
                "measurements": []
            }
            
            # Get sensors from the location object
            if hasattr(loc, 'sensors') and loc.sensors:
                print(f"  Found {len(loc.sensors)} sensors")
                
                # Track which parameters we've already added
                added_parameters = set()
                
                for sensor in loc.sensors:
                    try:
                        # Get parameter info from sensor
                        parameter = sensor.parameter.name if hasattr(sensor.parameter, 'name') else 'unknown'
                        unit = sensor.parameter.units if hasattr(sensor.parameter, 'units') else 'unknown'
                        display_name = sensor.parameter.display_name if hasattr(sensor.parameter, 'display_name') else parameter
                        sensor_id = sensor.id
                        
                        parameter_lower = parameter.lower()
                        
                        # Only include target pollutants and only once per parameter per station
                        if display_name not in added_parameters and any(pollutant in parameter_lower for pollutant in target_pollutants):
                            print(f"    Fetching data for sensor {sensor_id} ({display_name})...")
                            
                            try:
                                # Get latest measurement for this specific sensor
                                # Use measurements.list with sensors_id and limit to 1
                                sensor_measurements = client.measurements.list(
                                    sensors_id=sensor_id,
                                    limit=1
                                )
                                
                                if sensor_measurements and hasattr(sensor_measurements, 'results') and sensor_measurements.results:
                                    latest = sensor_measurements.results[0]
                                    
                                    # Extract datetime properly
                                    datetime_info = None
                                    if hasattr(latest, 'datetime'):
                                        if isinstance(latest.datetime, dict):
                                            datetime_info = latest.datetime
                                        else:
                                            datetime_info = str(latest.datetime)
                                    
                                    measurement_data = {
                                        "parameter": display_name,
                                        "value": latest.value if hasattr(latest, 'value') else None,
                                        "unit": unit,
                                        "datetime": datetime_info
                                    }
                                    
                                    station_data["measurements"].append(measurement_data)
                                    added_parameters.add(display_name)
                                    
                                    # Update city-level data
                                    if display_name not in city_pollutants:
                                        city_pollutants[display_name] = measurement_data
                                    else:
                                        # Compare timestamps and keep the most recent
                                        if measurement_data['datetime'] and city_pollutants[display_name]['datetime']:
                                            try:
                                                current_time = datetime.fromisoformat(
                                                    measurement_data['datetime']['utc'].replace('Z', '+00:00')
                                                )
                                                stored_time = datetime.fromisoformat(
                                                    city_pollutants[display_name]['datetime']['utc'].replace('Z', '+00:00')
                                                )
                                                if current_time > stored_time:
                                                    city_pollutants[display_name] = measurement_data
                                            except:
                                                pass
                                    
                                    print(f"      ✓ {display_name}: {measurement_data['value']} {unit}")
                                else:
                                    print(f"      ✗ No data available")
                                    
                            except Exception as e:
                                print(f"      Error fetching sensor data: {e}")
                                # Don't stop, continue with next sensor
                                continue
                                
                    except Exception as e:
                        print(f"    Error processing sensor: {e}")
                        continue
            
            # Only add stations that have measurements
            if station_data["measurements"]:
                all_station_data.append(station_data)
                print(f"  ✓ Added station with {len(station_data['measurements'])} measurements")
            else:
                print(f"  ✗ No measurements found for this station")
        
        print(f"\nTotal stations with data: {len(all_station_data)}")
        print(f"City-level pollutants: {list(city_pollutants.keys())}")
        
        return {
            "city": city_name,
            "coordinates": {"latitude": coords[0], "longitude": coords[1]},
            "pollutants": city_pollutants,
            "stations": all_station_data,
            "debug": {
                "total_locations_found": len(locations),
                "locations_with_data": len(all_station_data)
            }
        }, 200

    except AuthError as e:
        print(f"Auth error: {e}")
        return {"error": "Authentication Error: Please check your API key."}, 401
    except BadRequestError as e:
        print(f"Bad request error: {e}")
        return {"error": f"Bad Request Error: {e.detail if hasattr(e, 'detail') else str(e)}"}, 400
    except Exception as e:
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred: {str(e)}"}, 500


# --- Flask API Endpoint ---
@app.route('/api/air-quality', methods=['GET'])
def air_quality_endpoint():
    """
    API endpoint to get air quality data for a city.
    Usage: GET /api/air-quality?city=Delhi
    """
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400

    data, status_code = fetch_air_quality_data(city_name)
    return jsonify(data), status_code

# --- Run the Flask App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)