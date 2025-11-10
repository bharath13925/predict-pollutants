from flask import Flask, request, jsonify
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import pickle
import json
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping 
import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
import gridfs
import tempfile
import openaq
from datetime import timezone


load_dotenv()

# --- MongoDB Configuration ---
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("❌ MONGODB_URI environment variable not set!")

try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    mongo_client.admin.command('ping')
    db = mongo_client['air_quality_db']
    historical_collection = db['historical_data']
    models_collection = db['models']
    fs = gridfs.GridFS(db)
    print("✅ MongoDB Atlas connected successfully")
    
    # Create indexes
    historical_collection.create_index([("city", ASCENDING), ("date", ASCENDING)], unique=True)
    historical_collection.create_index([("city", ASCENDING)])
    historical_collection.create_index([("date", ASCENDING)])
    models_collection.create_index([("city", ASCENDING)], unique=True)
    
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    raise

# --- OpenAQ Configuration ---
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "6f1d18cea1904724d3604e5cd1fa201c6404de8c78401391820d33652983615f")
openaq_client = openaq.OpenAQ(api_key=OPENAQ_API_KEY)
print("✅ OpenAQ client initialized")

# --- Google Earth Engine Setup ---
service_key_json = os.getenv("GEE_SERVICE_KEY_JSON")
GEE_AVAILABLE = False  

if service_key_json:
    try:
        service_key_dict = json.loads(service_key_json)
        credentials = ee.ServiceAccountCredentials(service_key_dict["client_email"], key_data=service_key_json)
        ee.Initialize(credentials)
        print("✅ Google Earth Engine initialized successfully")
        GEE_AVAILABLE = True
    except Exception as e:
        print(f"❌ Failed to initialize GEE: {e}")
else:
    print("⚠️ Environment variable GEE_SERVICE_KEY_JSON not found")

print(f"GEE Status: {'✅ Available' if GEE_AVAILABLE else '❌ Not Available'}")

# --- Flask App Configuration ---
app = Flask(__name__)

# --- Configuration ---
DATA_LAG_DAYS = 30
MERRA_LAG_DAYS = 25  # MERRA-2 typical lag is 2-4 weeks
DAYS_BACK = 1455
LOOKBACK_DAYS = 90
MODEL_RETRAINING_DAYS = 7
CHUNK_SIZE = 30

geolocator = Nominatim(user_agent="air-quality-gee")

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

def fetch_openaq_pm_data(lat, lon, start_date, end_date, radius_km=25):
    """
    Fetch PM2.5 and PM10 data from OpenAQ for the specified date range.
    This is used to fill in the gap when MERRA-2 data is not available (recent 2-4 weeks).
    """
    try:
        print(f"  🌐 Fetching OpenAQ PM data from {start_date.date()} to {end_date.date()}...")
        
        # Get nearby locations
        locations_response = openaq_client.locations.list(
            coordinates=(lat, lon),
            radius=radius_km * 1000,  # Convert km to meters
            limit=100
        )
        
        locations = locations_response.results if hasattr(locations_response, 'results') else []
        print(f"    Found {len(locations)} OpenAQ stations")
        
        if not locations:
            print("    ⚠️ No OpenAQ stations found nearby")
            return None
        
        pm25_values = []
        pm10_values = []
        
        # Convert dates to ISO format for OpenAQ API
        date_from = start_date.strftime('%Y-%m-%dT00:00:00Z')
        date_to = end_date.strftime('%Y-%m-%dT23:59:59Z')
        
        for loc in locations:
            if not hasattr(loc, 'sensors') or not loc.sensors:
                continue
                
            for sensor in loc.sensors:
                try:
                    parameter = sensor.parameter.name if hasattr(sensor.parameter, 'name') else 'unknown'
                    parameter_lower = parameter.lower()
                    sensor_id = sensor.id
                    
                    # Fetch PM2.5
                    if 'pm25' in parameter_lower or 'pm2.5' in parameter_lower:
                        try:
                            measurements = openaq_client.measurements.list(
                                sensors_id=sensor_id,
                                date_from=date_from,
                                date_to=date_to,
                                limit=1000
                            )
                            
                            if measurements and hasattr(measurements, 'results'):
                                for m in measurements.results:
                                    if hasattr(m, 'value') and m.value is not None:
                                        pm25_values.append(m.value)
                        except Exception as e:
                            print(f"      Error fetching PM2.5 from sensor {sensor_id}: {e}")
                    
                    # Fetch PM10
                    elif 'pm10' in parameter_lower:
                        try:
                            measurements = openaq_client.measurements.list(
                                sensors_id=sensor_id,
                                date_from=date_from,
                                date_to=date_to,
                                limit=1000
                            )
                            
                            if measurements and hasattr(measurements, 'results'):
                                for m in measurements.results:
                                    if hasattr(m, 'value') and m.value is not None:
                                        pm10_values.append(m.value)
                        except Exception as e:
                            print(f"      Error fetching PM10 from sensor {sensor_id}: {e}")
                            
                except Exception as e:
                    print(f"    Error processing sensor: {e}")
                    continue
        
        result = {}
        if pm25_values:
            result['pm25'] = np.mean(pm25_values)
            print(f"    ✓ PM2.5 from OpenAQ: {result['pm25']:.2f} µg/m³ (avg of {len(pm25_values)} measurements)")
        
        if pm10_values:
            result['pm10'] = np.mean(pm10_values)
            print(f"    ✓ PM10 from OpenAQ: {result['pm10']:.2f} µg/m³ (avg of {len(pm10_values)} measurements)")
        
        # If we have PM2.5 but not PM10, estimate PM10
        if 'pm25' in result and 'pm10' not in result:
            result['pm10'] = result['pm25'] * 1.8
            print(f"    ℹ️ PM10 estimated from PM2.5: {result['pm10']:.2f} µg/m³")
        
        return result if result else None
    
    except Exception as e:
        print(f"    ❌ OpenAQ error: {e}")
        traceback.print_exc()
        return None

def fetch_openaq_current_data(lat, lon, radius_km=25):
    """
    Fetch current day air quality data from OpenAQ (real-time).
    Returns latest measurements for PM2.5, PM10, SO2, NO2, CO from TODAY ONLY.
    All pollutants normalized to µg/m³.
    Compatible with OpenAQ API v3+.
    """
    try:
        print(f"  🌐 Fetching current OpenAQ data...")

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        date_from = today_start.strftime('%Y-%m-%dT%H:%M:%SZ')
        date_to = now.strftime('%Y-%m-%dT%H:%M:%SZ')

        print(f"    📅 Fetching data from {date_from} to {date_to}")

        # Get nearby stations
        locations_response = openaq_client.locations.list(
            coordinates=(lat, lon),
            radius=radius_km * 1000,
            limit=100
        )

        locations = getattr(locations_response, "results", [])
        print(f"    Found {len(locations)} OpenAQ stations nearby")

        if not locations:
            print("    ⚠️ No OpenAQ stations found nearby")
            return None

        pollutant_values = {}
        pollutant_timestamps = {}
        target_params = ["pm25", "pm10", "so2", "no2", "co"]

        for loc in locations:
            if not hasattr(loc, "sensors") or not loc.sensors:
                continue

            for sensor in loc.sensors:
                try:
                    parameter = getattr(sensor.parameter, "name", "unknown").lower()
                    sensor_id = getattr(sensor, "id", None)
                    if not sensor_id:
                        continue

                    for target in target_params:
                        if target in parameter or target.replace("2", "") in parameter:
                            # Try both new and old parameter names for backward compatibility
                            try:
                                measurements = None
                                try:
                                    # ✅ For OpenAQ v3+ SDK
                                    measurements = openaq_client.measurements.list(
                                        sensors_id=sensor_id,
                                        datetime_from=date_from,
                                        datetime_to=date_to,
                                        limit=10
                                    )
                                except TypeError:
                                    # 🔁 For older SDK versions
                                    measurements = openaq_client.measurements.list(
                                        sensors_id=sensor_id,
                                        date_from=date_from,
                                        date_to=date_to,
                                        limit=10
                                    )

                                if measurements and hasattr(measurements, "results") and measurements.results:
                                    for m in measurements.results:
                                        if hasattr(m, "value") and m.value is not None:
                                            val = m.value
                                            unit = getattr(m, "unit", "").lower()
                                            timestamp = getattr(getattr(m, "date", {}), "utc", now.isoformat())

                                            # --- Normalize Units ---
                                            if target in ["pm25", "pm10"]:
                                                if "mg" in unit:
                                                    val *= 1000
                                                unit = "µg/m³"

                                            elif target in ["so2", "no2"]:
                                                if "ppb" in unit:
                                                    if target == "no2":
                                                        val *= 1.91
                                                    elif target == "so2":
                                                        val *= 2.62
                                                elif "ppm" in unit:
                                                    if target == "no2":
                                                        val *= 1910
                                                    elif target == "so2":
                                                        val *= 2620
                                                unit = "µg/m³"

                                            elif target == "co":
                                                if "ppm" in unit:
                                                    val *= 1.145 * 1000
                                                elif "mg" in unit:
                                                    val *= 1000
                                                unit = "µg/m³"

                                            pollutant_values.setdefault(target, []).append(val)
                                            pollutant_timestamps.setdefault(target, []).append(timestamp)

                            except Exception as e:
                                print(f"      Error fetching {target} from sensor {sensor_id}: {e}")
                            break

                except Exception:
                    continue

        # --- Average across all stations ---
        pollutant_data = {}
        for k, v in pollutant_values.items():
            pollutant_data[k] = {
                "value": float(np.mean(v)),
                "count": len(v),
                "latest_timestamp": max(pollutant_timestamps[k]) if pollutant_timestamps[k] else None
            }

        if not pollutant_data:
            print("    ⚠️ No pollutant data retrieved from OpenAQ for today")
            return None

        print("  ✅ Final averaged pollutant values (today only):")
        for k, data in pollutant_data.items():
            print(f"     • {k.upper()}: {data['value']:.2f} µg/m³ (avg of {data['count']} measurements)")
            if data['latest_timestamp']:
                print(f"       Latest: {data['latest_timestamp']}")

        return pollutant_data

    except Exception as e:
        print(f"    ❌ OpenAQ current data error: {e}")
        traceback.print_exc()
        return None


def fetch_gee_pollutant_data(lat, lon, start_date, end_date, radius_km=50):
    """
    Fetch pollutant data from Google Earth Engine with OpenAQ fallback for PM data.
    All pollutants converted to µg/m³ for consistency before storage.
    """

    if not GEE_AVAILABLE:
        print("❌ GEE is not available or initialized.")
        return None

    try:
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius_km * 1000)

        start = ee.Date(start_date.strftime('%Y-%m-%d'))
        end = ee.Date(end_date.strftime('%Y-%m-%d'))

        pollutant_data = {}
        TROPOSPHERIC_HEIGHT_M = 8000  # assumed column height

        # --- PM2.5 / PM10 ---
        days_ago = (datetime.now() - end_date).days
        use_openaq_for_pm = days_ago < MERRA_LAG_DAYS

        if use_openaq_for_pm:
            print(f"ℹ️ Using OpenAQ for PM data (within {MERRA_LAG_DAYS} days)")
            openaq_pm = fetch_openaq_pm_data(lat, lon, start_date, end_date)
            if openaq_pm:
                pollutant_data.update(openaq_pm)
            else:
                print("⚠️ OpenAQ PM data unavailable, using MERRA-2 fallback")
                use_openaq_for_pm = False

        if not use_openaq_for_pm:
            try:
                aerosol_collection = ee.ImageCollection('NASA/GSFC/MERRA/aer/2') \
                    .filterDate(start, end) \
                    .filterBounds(aoi) \
                    .select(['BCSMASS', 'OCSMASS', 'SO4SMASS', 'DUSMASS25', 'SSSMASS25'])

                if aerosol_collection.size().getInfo() > 0:
                    pm25_image_kg_m3 = aerosol_collection.mean().reduce(ee.Reducer.sum())
                    pm25_image_ug_m3 = pm25_image_kg_m3.multiply(1e9)

                    pm25_stats = pm25_image_ug_m3.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=aoi,
                        scale=50000,
                        maxPixels=1e9
                    ).getInfo()

                    if 'sum' in pm25_stats and pm25_stats['sum'] is not None:
                        pollutant_data['pm25'] = pm25_stats['sum']
                        pollutant_data['pm10'] = pollutant_data['pm25'] * 1.8
                        print(f"  ✓ PM2.5 (MERRA-2): {pollutant_data['pm25']:.2f} µg/m³")
                        print(f"  ✓ PM10 (MERRA-2): {pollutant_data['pm10']:.2f} µg/m³")
            except Exception as e:
                print(f"⚠️ MERRA-2 PM error: {e}")

        # --- NO2 ---
        try:
            no2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterDate(start, end).filterBounds(aoi) \
                .select('tropospheric_NO2_column_number_density')
            if no2.size().getInfo() > 0:
                val = no2.mean().reduceRegion(ee.Reducer.mean(), aoi, 1000).getInfo()
                if val and 'tropospheric_NO2_column_number_density' in val:
                    pollutant_data['no2'] = val['tropospheric_NO2_column_number_density'] * 46.0055 * 1e6 / TROPOSPHERIC_HEIGHT_M
                    print(f"  ✓ NO2: {pollutant_data['no2']:.2f} µg/m³")
        except Exception as e:
            print(f"⚠️ NO2 error: {e}")

        # --- SO2 ---
        try:
            so2 = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterDate(start, end).filterBounds(aoi) \
                .select('SO2_column_number_density')
            if so2.size().getInfo() > 0:
                val = so2.mean().reduceRegion(ee.Reducer.mean(), aoi, 1000).getInfo()
                if val and 'SO2_column_number_density' in val:
                    pollutant_data['so2'] = val['SO2_column_number_density'] * 64.066 * 1e6 / TROPOSPHERIC_HEIGHT_M
                    print(f"  ✓ SO2: {pollutant_data['so2']:.2f} µg/m³")
        except Exception as e:
            print(f"⚠️ SO2 error: {e}")

        # --- CO ---
        try:
            co = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterDate(start, end).filterBounds(aoi) \
                .select('CO_column_number_density')
            if co.size().getInfo() > 0:
                val = co.mean().reduceRegion(ee.Reducer.mean(), aoi, 1000).getInfo()
                if val and 'CO_column_number_density' in val:
                    pollutant_data['co'] = val['CO_column_number_density'] * 28.01 * 1e6 / TROPOSPHERIC_HEIGHT_M
                    print(f"  ✓ CO: {pollutant_data['co']:.2f} µg/m³")
        except Exception as e:
            print(f"⚠️ CO error: {e}")

        # --- CH4 ---
        try:
            ch4_collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4') \
                .filterDate(start, end).filterBounds(aoi) \
                .select('CH4_column_volume_mixing_ratio_dry_air')
            if ch4_collection.size().getInfo() > 0:
                ch4_mean_image = ch4_collection.mean()
                val = ch4_mean_image.reduceRegion(ee.Reducer.mean(), aoi, 7000).getInfo()
                if val and 'CH4_column_volume_mixing_ratio_dry_air' in val:
                    raw_value = val['CH4_column_volume_mixing_ratio_dry_air']
                    # Handle scale issues — if value > 1, assume already scaled ×1e6
                    if raw_value > 1e-3:  # e.g. 1.8 instead of 1.8e-6
                        raw_value = raw_value / 1e6
                    pollutant_data['ch4'] = raw_value * 1e9  # convert mol/mol → ppb
                    print(f"  ✓ CH4: {pollutant_data['ch4']:.2f} ppb")
        except Exception as e:
            print(f"⚠️ CH4 error: {e}")

        return pollutant_data

    except Exception as e:
        print(f"❌ Error fetching GEE pollutant data: {e}")
        traceback.print_exc()
        return None

def fetch_gee_weather_data(lat, lon, start_date, end_date):
    """Fetch weather data from Google Earth Engine (ERA5 reanalysis data)"""
    if not GEE_AVAILABLE:
        return None
    
    try:
        point = ee.Geometry.Point([lon, lat])
        
        start = ee.Date(start_date.strftime('%Y-%m-%d'))
        end = ee.Date(end_date.strftime('%Y-%m-%d'))
        
        weather_data = {}
        
        try:
            era5_collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                .filterDate(start, end) \
                .filterBounds(point)
            
            era5_count = era5_collection.size().getInfo()
            
            if era5_count > 0:
                weather_stats = era5_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point.buffer(5000),
                    scale=11132,
                    maxPixels=1e9
                ).getInfo()
                
                if 'temperature_2m' in weather_stats and weather_stats['temperature_2m'] is not None:
                    weather_data['temperature'] = weather_stats['temperature_2m'] - 273.15
                    print(f"  ✓ Temperature: {weather_data['temperature']:.2f}°C")
                
                if 'dewpoint_temperature_2m' in weather_stats and weather_stats['dewpoint_temperature_2m'] is not None:
                    T = weather_data.get('temperature', 20)
                    Td = weather_stats['dewpoint_temperature_2m'] - 273.15
                    
                    try:
                        rh = 100 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
                        weather_data['humidity'] = max(0, min(100, rh))
                        print(f"  ✓ Humidity: {weather_data['humidity']:.2f}%")
                    except:
                        pass
                
                u = weather_stats.get('u_component_of_wind_10m', None)
                v = weather_stats.get('v_component_of_wind_10m', None)
                
                if u is not None and v is not None:
                    weather_data['wind_speed'] = np.sqrt(u**2 + v**2)
                    print(f"  ✓ Wind Speed: {weather_data['wind_speed']:.2f} m/s")
                
                if 'total_precipitation_sum' in weather_stats and weather_stats['total_precipitation_sum'] is not None:
                    weather_data['precipitation'] = weather_stats['total_precipitation_sum'] * 1000
                    print(f"  ✓ Precipitation: {weather_data['precipitation']:.2f} mm")
                
                return weather_data if weather_data else None
        
        except Exception as e:
            print(f"  ⚠️ ERA5-Land error: {e}")
        
        try:
            print("  ℹ️ Trying ERA5 hourly as fallback...")
            era5_hourly = ee.ImageCollection('ECMWF/ERA5/DAILY') \
                .filterDate(start, end) \
                .filterBounds(point)
            
            if era5_hourly.size().getInfo() > 0:
                hourly_stats = era5_hourly.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point.buffer(5000),
                    scale=27830,
                    maxPixels=1e9
                ).getInfo()
                
                if 'mean_2m_air_temperature' in hourly_stats and hourly_stats['mean_2m_air_temperature'] is not None:
                    weather_data['temperature'] = hourly_stats['mean_2m_air_temperature'] - 273.15
                
                if 'mean_2m_air_temperature' in hourly_stats and 'dewpoint_2m_temperature' in hourly_stats:
                    T = weather_data.get('temperature', 20)
                    Td = hourly_stats['dewpoint_2m_temperature'] - 273.15
                    rh = 100 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
                    weather_data['humidity'] = max(0, min(100, rh))
                
                u = hourly_stats.get('u_component_of_wind_10m', None)
                v = hourly_stats.get('v_component_of_wind_10m', None)
                if u is not None and v is not None:
                    weather_data['wind_speed'] = np.sqrt(u**2 + v**2)
                
                if 'total_precipitation' in hourly_stats and hourly_stats['total_precipitation'] is not None:
                    weather_data['precipitation'] = hourly_stats['total_precipitation'] * 1000
                
                print(f"  ✓ Retrieved weather data from ERA5 hourly")
                return weather_data if weather_data else None
                
        except Exception as e:
            print(f"  ⚠️ ERA5 hourly error: {e}")
        
        return None
        
    except Exception as e:
        print(f"  Error fetching weather data: {e}")
        traceback.print_exc()
        return None

def fetch_historical_gee_data(city_name, days_back=DAYS_BACK):
    """Fetch historical pollutant and weather data from GEE and OpenAQ, store in MongoDB (all pollutants in µg/m³)."""
    coords = get_coordinates(city_name)
    if not coords:
        return {"error": f"Could not find coordinates for {city_name}"}, 404

    latitude, longitude = coords
    print(f"\n📊 Fetching {days_back} days of data for {city_name} (accounting for {DATA_LAG_DAYS}-day lag)...")

    try:
        safe_end_date = datetime.now() - timedelta(days=DATA_LAG_DAYS)
        start_date = safe_end_date - timedelta(days=days_back)

        stored_count = 0
        current_start = start_date

        while current_start < safe_end_date:
            current_end = min(current_start + timedelta(days=CHUNK_SIZE), safe_end_date)
            print(f"  Processing {current_start.date()} to {current_end.date()}...")

            pollutant_data = fetch_gee_pollutant_data(latitude, longitude, current_start, current_end)
            weather_data = fetch_gee_weather_data(latitude, longitude, current_start, current_end)

            if pollutant_data or weather_data:
                chunk_days = (current_end - current_start).days
                for i in range(chunk_days):
                    d = current_start + timedelta(days=i)
                    doc = {
                        "city": city_name,
                        "latitude": latitude,
                        "longitude": longitude,
                        "date": d.date().isoformat(),
                        **{k: pollutant_data.get(k) for k in ['pm25', 'pm10', 'so2', 'no2', 'co', 'ch4']},
                        **{k: weather_data.get(k) for k in ['temperature', 'humidity', 'wind_speed', 'precipitation'] if weather_data},
                        "created_at": datetime.now(timezone.utc)
                    }
                    try:
                        historical_collection.insert_one(doc)
                        stored_count += 1
                    except DuplicateKeyError:
                        pass

            current_start = current_end

        print(f"✅ Stored {stored_count} daily records for {city_name}")
        return {
            "status": "success",
            "city": city_name,
            "records_stored": stored_count,
            "unit": "µg/m³ (all pollutants)",
            "data_lag_days": DATA_LAG_DAYS
        }, 200

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, 500


def prepare_lstm_data(city_name):
    """Prepare data for LSTM training from MongoDB"""
    try:
        cursor = historical_collection.find(
            {'city': city_name},
            {'_id': 0, 'date': 1, 'pm25': 1, 'pm10': 1, 'so2': 1, 'no2': 1, 'co': 1, 
            'ch4': 1, 'temperature': 1, 'humidity': 1, 'wind_speed': 1, 'precipitation': 1}
        ).sort('date', ASCENDING)
        
        data = list(cursor)
        
        if not data:
            print(f"  ⚠️ No data found for {city_name}")
            return None
        
        df = pd.DataFrame(data)
        
        min_records_needed = LOOKBACK_DAYS + 20
        
        if len(df) < min_records_needed:
            print(f"  ⚠️ Insufficient data: Found {len(df)} rows, need at least {min_records_needed}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        numeric_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'ch4', 
                       'temperature', 'humidity', 'wind_speed', 'precipitation']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(df[col].median())
        
        print(f"  ✓ Prepared {len(df)} days of training data from MongoDB")
        return df
        
    except Exception as e:
        print(f"Error preparing LSTM data: {e}")
        traceback.print_exc()
        return None

def create_lstm_sequences(data, lookback=LOOKBACK_DAYS):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, :7])
    
    return np.array(X), np.array(y)

def save_model_to_mongodb(city_name, model, scaler, metadata):
    """Save trained model and scaler to MongoDB GridFS"""
    try:
        temp_dir = tempfile.gettempdir()
        temp_model_path = os.path.join(temp_dir, f"model_{city_name}.keras")
        
        model.save(temp_model_path)
        scaler_bytes = pickle.dumps(scaler)
        
        with open(temp_model_path, 'rb') as f:
            model_bytes = f.read()
        
        models_collection.delete_one({'city': city_name})
        
        model_id = fs.put(model_bytes, filename=f"model_{city_name}.keras")
        scaler_id = fs.put(scaler_bytes, filename=f"scaler_{city_name}.pkl")
        
        model_doc = {
            'city': city_name,
            'model_id': model_id,
            'scaler_id': scaler_id,
            'metadata': metadata,
            'trained_at': datetime.utcnow()
        }
        
        models_collection.insert_one(model_doc)
        
        os.remove(temp_model_path)
        
        print(f"  💾 Model and scaler saved to MongoDB GridFS")
        return True
        
    except Exception as e:
        print(f"Error saving model to MongoDB: {e}")
        traceback.print_exc()
        return False

def load_model_from_mongodb(city_name):
    """Load trained model and scaler from MongoDB GridFS"""
    try:
        model_doc = models_collection.find_one({'city': city_name})
        
        if not model_doc:
            return None, None, None
        
        temp_dir = tempfile.gettempdir()
        temp_model_path = os.path.join(temp_dir, f"model_{city_name}.keras")
        
        model_bytes = fs.get(model_doc['model_id']).read()
        
        with open(temp_model_path, 'wb') as f:
            f.write(model_bytes)
        
        model = load_model(temp_model_path, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        scaler_bytes = fs.get(model_doc['scaler_id']).read()
        scaler = pickle.loads(scaler_bytes)
        
        os.remove(temp_model_path)
        
        print(f"  ✓ Model loaded from MongoDB GridFS")
        return model, scaler, model_doc['metadata']
        
    except Exception as e:
        print(f"Error loading model from MongoDB: {e}")
        traceback.print_exc()
        return None, None, None

def train_lstm_model(city_name, force_retrain=False):
    """Train LSTM model for air quality prediction"""
    print(f"\n🤖 Training LSTM model for {city_name}...")
    
    if not force_retrain:
        model_doc = models_collection.find_one({'city': city_name})
        
        if model_doc:
            model_time = model_doc['trained_at']
            age_days = (datetime.utcnow() - model_time).days
            
            if age_days <= MODEL_RETRAINING_DAYS:
                print(f"  ✅ Using existing model (only {age_days} days old)")
                metadata = model_doc['metadata']
                
                return {
                    "status": "success",
                    "model_type": "LSTM",
                    "trained_at": model_time.isoformat(),
                    "age_days": age_days,
                    **metadata,
                    "message": "Using existing trained model from MongoDB"
                }, 200
    
    df = prepare_lstm_data(city_name)
    if df is None:
        return {"error": "Insufficient historical data for training"}, 400
    
    feature_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'ch4',
                   'temperature', 'humidity', 'wind_speed', 'precipitation']
    
    data = df[feature_cols].values
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = create_lstm_sequences(data_scaled, LOOKBACK_DAYS)
    
    if len(X) == 0:
        return {"error": "Not enough data points for training"}, 400
    
    print(f"  Training samples: {len(X)}")
    print(f"  Features shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(7)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("  ⏳ Training LSTM model...")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n  ✅ LSTM Performance:")
    print(f"     Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
    print(f"     Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    metadata = {
        'train_loss': float(train_loss),
        'train_mae': float(train_mae),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'lookback_days': LOOKBACK_DAYS
    }
    
    save_model_to_mongodb(city_name, model, scaler, metadata)
    
    return {
        "status": "success",
        "model_type": "LSTM",
        "trained_at": datetime.utcnow().isoformat(),
        "age_days": 0,
        **metadata
    }, 200

def predict_next_week(city_name):
    """Predict pollutant levels for next 7 days starting from TOMORROW"""
    print(f"\n🔮 Generating predictions for {city_name}...")
    
    try:
        model, scaler, metadata = load_model_from_mongodb(city_name)
        
        if model is None:
            return {"error": f"Model not trained yet for {city_name}. Please train the model first."}, 400
        
        # ✅ Get model document for trained_at timestamp
        model_doc = models_collection.find_one({'city': city_name})
        trained_at = model_doc['trained_at'].isoformat() if model_doc else None
            
    except Exception as e:
        print(f"  Error loading model: {e}")
        return {"error": f"Error loading model: {str(e)}"}, 500
    
    df = prepare_lstm_data(city_name)
    if df is None or len(df) < LOOKBACK_DAYS:
        return {"error": f"Insufficient data for prediction (need {LOOKBACK_DAYS} days)"}, 400
    
    feature_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'ch4',
                   'temperature', 'humidity', 'wind_speed', 'precipitation']
    
    recent_data = df[feature_cols].iloc[-LOOKBACK_DAYS:].values
    recent_data_scaled = scaler.transform(recent_data)
    
    predictions = []
    current_sequence = recent_data_scaled.copy()
    
    today = datetime.now()
    
    print(f"  📅 Today's date: {today.date()}")
    print(f"  🔮 Predicting from: {(today + timedelta(days=1)).date()} to {(today + timedelta(days=7)).date()}")
    
    for day in range(7):
        X_input = current_sequence[-LOOKBACK_DAYS:].reshape(1, LOOKBACK_DAYS, len(feature_cols))
        
        prediction_scaled = model.predict(X_input, verbose=0)[0]
        
        full_prediction = np.zeros(len(feature_cols))
        full_prediction[:7] = prediction_scaled
        full_prediction[7:] = current_sequence[-1, 7:]
        
        prediction = scaler.inverse_transform(full_prediction.reshape(1, -1))[0]
        
        pred_date = today + timedelta(days=day+1)
        predictions.append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'day': pred_date.strftime('%A'),
            'day_offset': day + 1,
            'pm25': float(np.clip(prediction[0], 0, 500)),
            'pm10': float(np.clip(prediction[1], 0, 600)),
            'so2': float(np.clip(prediction[2], 0, 100)),
            'no2': float(np.clip(prediction[3], 0, 200)),
            'co': float(np.clip(prediction[4], 0, 50)),
            'ch4': float(np.clip(prediction[6], 1700, 2000))
        })
        
        current_sequence = np.vstack([current_sequence[1:], full_prediction])
    
    print(f"  ✅ Generated {len(predictions)} days of predictions (Tomorrow through Next Week)")
    
    # ✅ UPDATED: Include all model training metrics in response
    return {
        "status": "success",
        "city": city_name,
        "current_date": today.strftime('%Y-%m-%d'),
        "prediction_start_date": (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        "prediction_end_date": (today + timedelta(days=7)).strftime('%Y-%m-%d'),
        "predictions": predictions,
        "model_info": {
            "model_type": "lstm",
            "lookback_days": LOOKBACK_DAYS,
            # ✅ NEW: Include training metrics from metadata
            "test_mae": metadata.get("test_mae") if metadata else None,
            "test_loss": metadata.get("test_loss") if metadata else None,
            "train_mae": metadata.get("train_mae") if metadata else None,
            "train_loss": metadata.get("train_loss") if metadata else None,
            "training_samples": metadata.get("training_samples") if metadata else None,
            "test_samples": metadata.get("test_samples") if metadata else None,
            "trained_at": trained_at
        },
        "note": "Predictions start from tomorrow and extend 7 days into the future"
    }, 200

def model_status_check(city_name):
    """Check model status for a city"""
    try:
        model_doc = models_collection.find_one({'city': city_name})
        
        if not model_doc:
            return {
                "status": "not_found",
                "city": city_name,
                "message": "No trained model exists for this city"
            }, 404
        
        model_time = model_doc['trained_at']
        age_days = (datetime.utcnow() - model_time).days
        
        return {
            "status": "found",
            "city": city_name,
            "model_type": "LSTM",
            "trained_at": model_time.isoformat(),
            "age_days": age_days,
            "needs_retraining": age_days > MODEL_RETRAINING_DAYS,
            "lookback_days": LOOKBACK_DAYS,
            "metadata": model_doc['metadata']
        }, 200
        
    except Exception as e:
        return {"error": str(e)}, 500

def generate_map_tiles(lat, lon, radius_km=100):
    """Generate interactive map tile URL"""
    if not GEE_AVAILABLE:
        return None
    
    try:
        print(f"🗺️ Generating map tiles for ({lat}, {lon})...")
        
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius_km * 1000)
        
        safe_end_date = datetime.now()
        safe_start_date = safe_end_date - timedelta(days=7)
        
        end_date = ee.Date(safe_end_date)
        start_date = ee.Date(safe_start_date)
        
        print(f"  📅 Map date range: {safe_start_date.date()} to {safe_end_date.date()}")
        
        pollutants_data = {}
        
        # NO2
        try:
            no2_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .select('tropospheric_NO2_column_number_density')
            no2_count = no2_collection.size().getInfo()
            if no2_count > 0:
                pollutants_data['NO2'] = no2_collection.mean()
                print(f"  ✓ NO2: {no2_count} images")
        except Exception as e:
            print(f"  ✗ NO2 error: {e}")
        
        # SO2
        try:
            so2_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .select('SO2_column_number_density')
            so2_count = so2_collection.size().getInfo()
            if so2_count > 0:
                pollutants_data['SO2'] = so2_collection.mean()
                print(f"  ✓ SO2: {so2_count} images")
        except Exception as e:
            print(f"  ✗ SO2 error: {e}")
        
        # CO
        try:
            co_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .select('CO_column_number_density')
            co_count = co_collection.size().getInfo()
            if co_count > 0:
                pollutants_data['CO'] = co_collection.mean()
                print(f"  ✓ CO: {co_count} images")
        except Exception as e:
            print(f"  ✗ CO error: {e}")
        
        # Aerosol Index
        try:
            aer_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .select('absorbing_aerosol_index')
            aer_count = aer_collection.size().getInfo()
            if aer_count > 0:
                pollutants_data['AER_AI'] = aer_collection.mean()
                print(f"  ✓ Aerosol: {aer_count} images")
        except Exception as e:
            print(f"  ✗ Aerosol error: {e}")
        
        if not pollutants_data:
            print("  ✗ No pollutant data available")
            return None
        
        normalized_layers = []
        
        if 'NO2' in pollutants_data:
            no2_normalized = pollutants_data['NO2'].subtract(0).divide(0.0003).clamp(0, 1)
            normalized_layers.append(no2_normalized)
        
        if 'SO2' in pollutants_data:
            so2_normalized = pollutants_data['SO2'].subtract(0).divide(0.0012).clamp(0, 1)
            normalized_layers.append(so2_normalized)
        
        if 'CO' in pollutants_data:
            co_normalized = pollutants_data['CO'].subtract(0.03).divide(0.03).clamp(0, 1)
            normalized_layers.append(co_normalized)
        
        if 'AER_AI' in pollutants_data:
            aer_normalized = pollutants_data['AER_AI'].subtract(-1).divide(3).clamp(0, 1)
            normalized_layers.append(aer_normalized)
        
        combined_aqi = normalized_layers[0]
        for layer in normalized_layers[1:]:
            combined_aqi = combined_aqi.max(layer)
        
        vis_params = {
            'min': 0,
            'max': 1,
            'palette': [
                '#00FF00',
                '#90EE90',
                '#FFFF00',
                '#FFA500',
                '#FF4500',
                '#FF0000',
                '#8B0000'
            ]
        }
        
        map_id_dict = combined_aqi.getMapId(vis_params)
        
        print(f"  ✅ Generated map tile URL")
        
        return {
            'tile_url': map_id_dict['tile_fetcher'].url_format,
            'map_id': map_id_dict['mapid'],
            'token': map_id_dict['token'],
            'type': 'combined_aqi',
            'center': {'lat': lat, 'lon': lon},
            'pollutants_included': list(pollutants_data.keys()),
            'date_range': {
                'start': safe_start_date.strftime('%Y-%m-%d'),
                'end': safe_end_date.strftime('%Y-%m-%d')
            },
            'data_lag_days': DATA_LAG_DAYS
        }
        
    except Exception as e:
        print(f"  ✗ Error generating map tiles: {e}")
        traceback.print_exc()
        return None

def fetch_current_air_quality(city_name):
    """Fetch ONLY current day air quality data from OpenAQ (no GEE fallback for pollutants)"""
    coords = get_coordinates(city_name)
    if not coords:
        return {"error": f"Could not find coordinates for {city_name}"}, 404

    try:
        latitude, longitude = coords

        print(f"\n📊 Fetching current air quality for {city_name}...")

        # ✅ Fetch ONLY today's AQ data from OpenAQ
        print(f"  🌐 Fetching current day data from OpenAQ...")
        openaq_current = fetch_openaq_current_data(latitude, longitude)

        if not openaq_current:
            return {"error": "No real-time OpenAQ data available for this location"}, 404

        combined_pollutants = {}
        for key, data in openaq_current.items():
            unit = "µg/m³" if key != "ch4" else "ppb"
            combined_pollutants[key.upper()] = {
                "parameter": key.upper(),
                "value": data["value"],
                "unit": unit,
                "source": "OpenAQ (Real-time)",
                "measurement_count": data["count"],
                "latest_timestamp": data["latest_timestamp"],
                "data_freshness": "Today only"
            }

        # ✅ Still fetch weather & map (non-AQ) from GEE
        safe_end_date = datetime.now() - timedelta(days=DATA_LAG_DAYS)
        start_date = safe_end_date - timedelta(days=7)

        print(f"  🌦️ Fetching recent week weather & satellite maps...")
        weather_data = fetch_gee_weather_data(latitude, longitude, start_date, safe_end_date)
        map_tiles = generate_map_tiles(latitude, longitude, radius_km=100)

        result = {
            "city": city_name,
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "pollutants": combined_pollutants,
            "weather": weather_data,
            "map_tiles": map_tiles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "✅ Real-time pollution from OpenAQ (today only). Weather & satellite tiles from GEE."
        }

        return result, 200

    except Exception as e:
        print(f"❌ Error fetching current data: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500


# --- Flask API Endpoints ---

@app.route('/api/air-quality', methods=['GET'])
def air_quality_endpoint():
    """Get current air quality data from OpenAQ and GEE"""
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    data, status_code = fetch_current_air_quality(city_name)
    return jsonify(data), status_code

@app.route('/api/collect-historical', methods=['POST'])
def collect_historical():
    """Collect historical data from GEE and OpenAQ, store in MongoDB"""
    data = request.get_json()
    city_name = data.get('city')
    days_back = data.get('days_back', DAYS_BACK)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = fetch_historical_gee_data(city_name, days_back=days_back)
    return jsonify(result), status_code

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train LSTM prediction model and store in MongoDB"""
    data = request.get_json()
    city_name = data.get('city')
    force_retrain = data.get('force_retrain', False)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = train_lstm_model(city_name, force_retrain=force_retrain)
    return jsonify(result), status_code

@app.route('/api/predict', methods=['GET'])
def predict():
    """Get predictions for next 7 days (starting from tomorrow)"""
    city_name = request.args.get('city')
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = predict_next_week(city_name)
    return jsonify(result), status_code

@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    """Complete workflow: collect data, train model, and predict"""
    data = request.get_json()
    city_name = data.get('city')
    force_retrain = data.get('force_retrain', False)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    city_name = city_name.strip().title()
    
    print(f"\n🚀 Starting full analysis for {city_name}")
    
    print("\n📊 Step 1: Checking historical data in MongoDB...")
    existing_records = historical_collection.count_documents({'city': city_name})
    
    min_records = LOOKBACK_DAYS + 20
    
    if existing_records < min_records:
        print(f"  ⚠️ Insufficient data ({existing_records} records), need {min_records}. Collecting {DAYS_BACK} days...")
        hist_result, hist_status = fetch_historical_gee_data(city_name, days_back=DAYS_BACK)
        if hist_status != 200:
            return jsonify(hist_result), hist_status
    else:
        print(f"  ✅ Sufficient data exists ({existing_records} records)")
        hist_result = {
            "status": "skipped", 
            "message": "Sufficient data already exists in MongoDB", 
            "records": existing_records,
            "records_stored": existing_records
        }
    
    print("\n🤖 Step 2: Training LSTM model...")
    train_result, train_status = train_lstm_model(city_name, force_retrain=force_retrain)
    if train_status != 200:
        return jsonify(train_result), train_status
    
    print("\n🔮 Step 3: Generating predictions...")
    pred_result, pred_status = predict_next_week(city_name)
    if pred_status != 200:
        return jsonify(pred_result), pred_status
    
    print("\n📊 Step 4: Fetching current data...")
    current_result, current_status = fetch_current_air_quality(city_name)
    
    response_data = {
        "status": "success",
        "city": city_name,
        "historical_data": hist_result,
        "model_training": {
            "status": train_result.get("status"),
            "model_type": train_result.get("model_type", "LSTM"),
            "trained_at": train_result.get("trained_at"),
            "age_days": train_result.get("age_days", 0),
            "train_loss": train_result.get("train_loss"),
            "train_mae": train_result.get("train_mae"),
            "test_loss": train_result.get("test_loss"),
            "test_mae": train_result.get("test_mae"),
            "training_samples": train_result.get("training_samples"),
            "test_samples": train_result.get("test_samples"),
            "lookback_days": train_result.get("lookback_days", LOOKBACK_DAYS),
            "message": train_result.get("message", "Model training completed")
        },
        "predictions": pred_result,
        "current_data": current_result,
        "configuration": {
            "days_back": DAYS_BACK,
            "lookback_period": LOOKBACK_DAYS,
            "data_lag_days": DATA_LAG_DAYS,
            "merra_lag_days": MERRA_LAG_DAYS,
            "model_type": "LSTM",
            "data_sources": {
                "current_pm": "OpenAQ (Real-time, today only)",
                "historical_pm": "OpenAQ (recent) + MERRA-2 (older)",
                "other_pollutants": "Sentinel-5P",
                "weather": "ERA5"
            },
            "database": "MongoDB Atlas",
            "prediction_range": "Tomorrow through next 7 days",
            "note": "PM2.5/PM10 from OpenAQ for current (today only) and recent data, MERRA-2 for historical. Other pollutants from Sentinel-5P."
        }
    }
    
    return jsonify(response_data), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        mongo_client.admin.command('ping')
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return jsonify({
        "status": "ok",
        "gee_available": GEE_AVAILABLE,
        "openaq_available": True,
        "database": "MongoDB Atlas",
        "database_status": db_status,
        "service": "Air Quality API with LSTM Predictions (GEE + OpenAQ + MongoDB)",
        "configuration": {
            "days_back": DAYS_BACK,
            "data_lag_days": DATA_LAG_DAYS,
            "merra_lag_days": MERRA_LAG_DAYS,
            "lookback_period": LOOKBACK_DAYS,
            "chunk_size": CHUNK_SIZE,
            "model_retraining_days": MODEL_RETRAINING_DAYS,
            "data_sources": {
                "current_data": "OpenAQ (Real-time, today only)",
                "pm_historical": "OpenAQ (recent 2-4 weeks) + MERRA-2 (older)",
                "other_pollutants": "Sentinel-5P",
                "weather": "ERA5"
            },
            "storage": "MongoDB Atlas with GridFS",
            "prediction_range": "Tomorrow through next 7 days"
        }
    }), 200

@app.route('/api/database-stats', methods=['GET'])
def database_stats():
    """Get MongoDB database statistics"""
    try:
        total_records = historical_collection.count_documents({})
        
        pipeline = [
            {"$group": {"_id": "$city"}},
            {"$project": {"city": "$_id", "_id": 0}}
        ]
        cities = [doc['city'] for doc in historical_collection.aggregate(pipeline)]
        
        city_stats = {}
        for city in cities:
            pipeline = [
                {"$match": {"city": city}},
                {"$group": {
                    "_id": None,
                    "min_date": {"$min": "$date"},
                    "max_date": {"$max": "$date"},
                    "count": {"$sum": 1}
                }}
            ]
            result = list(historical_collection.aggregate(pipeline))
            
            if result:
                city_stats[city] = {
                    "records": result[0]['count'],
                    "date_range": {
                        "start": result[0]['min_date'],
                        "end": result[0]['max_date']
                    }
                }
        
        total_models = models_collection.count_documents({})
        
        return jsonify({
            "status": "success",
            "database": "MongoDB Atlas",
            "total_records": total_records,
            "total_models": total_models,
            "cities": city_stats,
            "data_sources": "OpenAQ (real-time, today only) + GEE (MERRA-2 + Sentinel-5P)",
            "data_lag_days": DATA_LAG_DAYS,
            "merra_lag_days": MERRA_LAG_DAYS
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check model status for a city"""
    city_name = request.args.get('city')
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = model_status_check(city_name)
    return jsonify(result), status_code

@app.route('/')
def home():
    return jsonify({
        "status": "success",
        "message": "✅ Flask backend with MongoDB Atlas and OpenAQ is running successfully!",
        "database": "MongoDB Atlas",
        "storage": "Persistent (Cloud Database)",
        "data_sources": {
            "real_time": "OpenAQ API (Today only)",
            "historical_pm": "OpenAQ (recent) + MERRA-2 (older)",
            "satellite": "Google Earth Engine (Sentinel-5P)",
            "weather": "ERA5"
        },
        "data_freshness": {
            "current_pollutants": "Real-time from today only",
            "timestamps_included": "Yes - shows when measurements were taken"
        }
    })

# --- Run Flask App ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 Air Quality API with LSTM Predictions (GEE + OpenAQ + MongoDB)")
    print("="*70)
    print(f"GEE Status: {'✅ Available' if GEE_AVAILABLE else '❌ Not Available'}")
    print(f"OpenAQ Status: ✅ Available")
    print(f"Database: MongoDB Atlas")
    print(f"\n📈 CONFIGURATION:")
    print(f"   • Satellite Data Lag: {DATA_LAG_DAYS} days")
    print(f"   • MERRA-2 Data Lag: {MERRA_LAG_DAYS} days (2-4 weeks)")
    print(f"   • Historical Data Collection: {DAYS_BACK} days (adjusted for lag)")
    print(f"   • Model Type: LSTM (Long Short-Term Memory)")
    print(f"   • Model Lookback Period: {LOOKBACK_DAYS} days")
    print(f"   • Data Storage: MongoDB Atlas (Persistent Cloud Database)")
    print(f"   • Model Storage: GridFS (MongoDB Binary Storage)")
    print(f"\n🌐 DATA SOURCES:")
    print(f"   • Current Day PM2.5/PM10: OpenAQ (Real-time, TODAY ONLY)")
    print(f"   • Recent PM2.5/PM10 (0-{MERRA_LAG_DAYS} days): OpenAQ")
    print(f"   • Historical PM2.5/PM10 (>{MERRA_LAG_DAYS} days): MERRA-2 Satellite")
    print(f"   • NO2, SO2, CO, CH4: Sentinel-5P Satellite")
    print(f"   • Weather Data: ERA5 Reanalysis")
    print(f"\n🔮 PREDICTION CONFIGURATION:")
    print(f"   • Prediction Start: TOMORROW (Day +1 from today)")
    print(f"   • Prediction End: 7 days from today (Day +7)")
    print(f"   • Training uses: OpenAQ + GEE integrated data")
    print(f"\n✅ IMPROVEMENTS:")
    print(f"   • Date filtering: Only fetches TODAY's OpenAQ measurements")
    print(f"   • Timestamp tracking: Shows when each measurement was taken")
    print(f"   • Data freshness: Includes measurement count and latest timestamp")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)