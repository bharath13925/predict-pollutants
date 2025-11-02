from flask import Flask, request, jsonify
import ee
import sqlite3
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
load_dotenv()

import os, json, ee
from dotenv import load_dotenv

load_dotenv()

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

# --- Database Setup ---
DB_NAME = 'air_quality_gee.db'

# --- Configuration ---
DATA_LAG_DAYS = 6  # Copernicus and ERA5 have 6-day lag
DAYS_BACK = 1455  # Adjusted from 730 to account for lag (2 years minus lag)
LOOKBACK_DAYS = 90  # LSTM lookback period
MODEL_RETRAINING_DAYS = 7
CHUNK_SIZE = 7  # Collect data in weekly chunks

geolocator = Nominatim(user_agent="air-quality-gee")

def init_database():
    """Initialize SQLite database for historical data storage"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            date DATE NOT NULL,
            pm25 REAL,
            pm10 REAL,
            so2 REAL,
            no2 REAL,
            co REAL,
            o3 REAL,
            ch4 REAL,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            precipitation REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, date)
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_city_date 
        ON historical_data(city, date)
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

init_database()

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

def fetch_gee_pollutant_data(lat, lon, start_date, end_date, radius_km=50):
    """
    Fetch pollutant data from Google Earth Engine Sentinel-5P satellite data
    """
    if not GEE_AVAILABLE:
        return None
    
    try:
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius_km * 1000)
        
        start = ee.Date(start_date.strftime('%Y-%m-%d'))
        end = ee.Date(end_date.strftime('%Y-%m-%d'))
        
        pollutant_data = {}
        
        # PM2.5 - Using Aerosol Optical Depth as proxy
        try:
            aer_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_AER_AI') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('absorbing_aerosol_index')
            
            if aer_collection.size().getInfo() > 0:
                aer_stats = aer_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=1000
                ).getInfo()
                if 'absorbing_aerosol_index' in aer_stats and aer_stats['absorbing_aerosol_index'] is not None:
                    pollutant_data['pm25'] = abs(aer_stats['absorbing_aerosol_index']) * 50
                    print(f"  ✓ PM25: {pollutant_data['pm25']:.2f} ug")
        except Exception as e:
            print(f"  ⚠️ PM2.5 error: {e}")
        
        # NO2
        try:
            no2_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('tropospheric_NO2_column_number_density')
            
            if no2_collection.size().getInfo() > 0:
                no2_stats = no2_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=1000
                ).getInfo()
                if 'tropospheric_NO2_column_number_density' in no2_stats and no2_stats['tropospheric_NO2_column_number_density'] is not None:
                    pollutant_data['no2'] = no2_stats['tropospheric_NO2_column_number_density'] * 46.0055 * 1e6
                    print(f"  ✓ NO2: {pollutant_data['no2']:.2f} ug")
        except Exception as e:
            print(f"  ⚠️ NO2 error: {e}")
        
        # SO2
        try:
            so2_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_SO2') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('SO2_column_number_density')
            
            if so2_collection.size().getInfo() > 0:
                so2_stats = so2_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=1000
                ).getInfo()
                if 'SO2_column_number_density' in so2_stats and so2_stats['SO2_column_number_density'] is not None:
                    pollutant_data['so2'] = so2_stats['SO2_column_number_density'] * 64.066 * 1e6
                    print(f"  ✓ SO2: {pollutant_data['so2']:.2f} ug")
        except Exception as e:
            print(f"  ⚠️ SO2 error: {e}")
        
        # CO
        try:
            co_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CO') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('CO_column_number_density')
            
            if co_collection.size().getInfo() > 0:
                co_stats = co_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=1000
                ).getInfo()
                if 'CO_column_number_density' in co_stats and co_stats['CO_column_number_density'] is not None:
                    pollutant_data['co'] = co_stats['CO_column_number_density'] * 28.01 * 1e3
                    print(f"  ✓ CO: {pollutant_data['co']:.2f} ug")
        except Exception as e:
            print(f"  ⚠️ CO error: {e}")
        
        # O3
        try:
            o3_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('O3_column_number_density')
            
            if o3_collection.size().getInfo() > 0:
                o3_stats = o3_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=1000
                ).getInfo()
                if 'O3_column_number_density' in o3_stats and o3_stats['O3_column_number_density'] is not None:
                    pollutant_data['o3'] = o3_stats['O3_column_number_density'] * 47.9982 * 1e6
                    print(f"  ✓ O3: {pollutant_data['o3']:.2f} ug")
        except Exception as e:
            print(f"  ⚠️ O3 error: {e}")
        
        # CH4
        try:
            ch4_collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CH4') \
                .filterDate(start, end) \
                .filterBounds(aoi) \
                .select('CH4_column_volume_mixing_ratio_dry_air')
            
            ch4_count = ch4_collection.size().getInfo()
            
            if ch4_count == 0:
                print("  ℹ️ CH4 OFFL empty, trying NRTI...")
                ch4_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_CH4') \
                    .filterDate(start, end) \
                    .filterBounds(aoi) \
                    .select('CH4_column_volume_mixing_ratio_dry_air')
                ch4_count = ch4_collection.size().getInfo()
            
            if ch4_count > 0:
                ch4_stats = ch4_collection.mean().reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=7000
                ).getInfo()
                
                if 'CH4_column_volume_mixing_ratio_dry_air' in ch4_stats and ch4_stats['CH4_column_volume_mixing_ratio_dry_air'] is not None:
                    pollutant_data['ch4'] = ch4_stats['CH4_column_volume_mixing_ratio_dry_air']
                    print(f"  ✓ CH4: {pollutant_data['ch4']:.2f} ppb")
            else:
                print("  ⚠️ No CH4 data available for this period")
        except Exception as e:
            print(f"  ⚠️ CH4 error: {e}")
        
        # PM10 (estimated from PM2.5 using typical ratio)
        if 'pm25' in pollutant_data:
            pollutant_data['pm10'] = pollutant_data['pm25'] * 1.8
            print(f"  ✓ PM10: {pollutant_data['pm10']:.2f} ug")
        
        return pollutant_data
        
    except Exception as e:
        print(f"  Error fetching GEE pollutant data: {e}")
        traceback.print_exc()
        return None

def fetch_gee_weather_data(lat, lon, start_date, end_date):
    """
    Fetch weather data from Google Earth Engine (ERA5 reanalysis data)
    """
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
    """
    Fetch historical pollutant and weather data from GEE
    ✅ Collects data from (2 years ago) to (6 days before today)
    """
    coords = get_coordinates(city_name)
    if not coords:
        return {"error": f"Could not find coordinates for {city_name}"}, 404
    
    latitude, longitude = coords
    print(f"\n📊 Fetching {days_back} days of GEE data for {city_name} (accounting for {DATA_LAG_DAYS}-day lag)...")
    
    try:
        # ✅ Stop 6 days before today to avoid incomplete data
        safe_end_date = datetime.now() - timedelta(days=DATA_LAG_DAYS)
        start_date = safe_end_date - timedelta(days=days_back)
        
        print(f"  📅 Historical data range: {start_date.date()} to {safe_end_date.date()}")
        print(f"  ⏰ Data lag accounted for: {DATA_LAG_DAYS} days")
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        stored_count = 0
        
        current_start = start_date
        while current_start < safe_end_date:
            current_end = min(current_start + timedelta(days=CHUNK_SIZE), safe_end_date)
            
            print(f"  Processing {current_start.date()} to {current_end.date()}...")
            
            pollutant_data = fetch_gee_pollutant_data(latitude, longitude, current_start, current_end)
            weather_data = fetch_gee_weather_data(latitude, longitude, current_start, current_end)
            
            if pollutant_data or weather_data:
                chunk_days = (current_end - current_start).days
                
                for day_offset in range(chunk_days):
                    record_date = current_start + timedelta(days=day_offset)
                    
                    cursor.execute(
                        "SELECT id FROM historical_data WHERE city = ? AND date = ?",
                        (city_name, record_date.date())
                    )
                    
                    if cursor.fetchone() is None:
                        cursor.execute('''
                            INSERT INTO historical_data 
                            (city, latitude, longitude, date, pm25, pm10, so2, no2, co, o3, ch4,
                             temperature, humidity, wind_speed, precipitation)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            city_name,
                            latitude,
                            longitude,
                            record_date.date(),
                            pollutant_data.get('pm25') if pollutant_data else None,
                            pollutant_data.get('pm10') if pollutant_data else None,
                            pollutant_data.get('so2') if pollutant_data else None,
                            pollutant_data.get('no2') if pollutant_data else None,
                            pollutant_data.get('co') if pollutant_data else None,
                            pollutant_data.get('o3') if pollutant_data else None,
                            pollutant_data.get('ch4') if pollutant_data else None,
                            weather_data.get('temperature') if weather_data else None,
                            weather_data.get('humidity') if weather_data else None,
                            weather_data.get('wind_speed') if weather_data else None,
                            weather_data.get('precipitation') if weather_data else None
                        ))
                        stored_count += 1
            
            current_start = current_end
            
            if stored_count % 50 == 0 and stored_count > 0:
                conn.commit()
                print(f"  ✓ Committed {stored_count} records so far...")
        
        conn.commit()
        conn.close()
        
        print(f"✅ Stored {stored_count} daily records of historical data")
        
        return {
            "status": "success",
            "city": city_name,
            "records_stored": stored_count,
            "date_range": {
                "start": start_date.strftime('%Y-%m-%d'),
                "end": safe_end_date.strftime('%Y-%m-%d')
            },
            "data_lag_days": DATA_LAG_DAYS,
            "note": f"Historical data collected up to {DATA_LAG_DAYS} days before current date"
        }, 200
        
    except Exception as e:
        print(f"Error fetching historical GEE data: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500

def prepare_lstm_data(city_name):
    """Prepare data for LSTM training"""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = '''
            SELECT date, pm25, pm10, so2, no2, co, o3, ch4,
                   temperature, humidity, wind_speed, precipitation
            FROM historical_data
            WHERE city = ?
            ORDER BY date
        '''
        df = pd.read_sql_query(query, conn, params=(city_name,))
        conn.close()
        
        min_records_needed = LOOKBACK_DAYS + 20
        
        if df.empty or len(df) < min_records_needed:
            print(f"  ⚠️ Insufficient data: Found {len(df)} rows, need at least {min_records_needed}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        numeric_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'ch4', 
                       'temperature', 'humidity', 'wind_speed', 'precipitation']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].fillna(df[col].median())
        
        print(f"  ✓ Prepared {len(df)} days of training data")
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
        y.append(data[i, :7])  # Predict only pollutants (first 7 features)
    
    return np.array(X), np.array(y)

def train_lstm_model(city_name, force_retrain=False):
    """
    Train LSTM model for air quality prediction
    """
    print(f"\n🤖 Training LSTM model for {city_name}...")
    
    model_filename = f"lstm_model_{city_name.replace(' ', '_')}.keras"
    scaler_filename = f"scaler_{city_name.replace(' ', '_')}.pkl"
    
    if not force_retrain and os.path.exists(model_filename):
        model_time = datetime.fromtimestamp(os.path.getmtime(model_filename))
        age_days = (datetime.now() - model_time).days
        
        if age_days <= MODEL_RETRAINING_DAYS:
            print(f"  ✅ Using existing model (only {age_days} days old)")
            
            metadata_file = f"model_metadata_{city_name.replace(' ', '_')}.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                return {
                    "status": "success",
                    "model_type": "LSTM",
                    "trained_at": model_time.isoformat(),
                    "age_days": age_days,
                    "train_loss": metadata.get('train_loss'),
                    "train_mae": metadata.get('train_mae'),
                    "test_loss": metadata.get('test_loss'),
                    "test_mae": metadata.get('test_mae'),
                    "training_samples": metadata.get('training_samples'),
                    "test_samples": metadata.get('test_samples'),
                    "model_file": model_filename,
                    "lookback_days": LOOKBACK_DAYS,
                    "message": "Using existing trained model"
                }, 200
    
    df = prepare_lstm_data(city_name)
    if df is None:
        return {"error": "Insufficient historical data for training"}, 400
    
    feature_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'ch4',
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
    
    model.save(model_filename)
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'train_loss': float(train_loss),
        'train_mae': float(train_mae),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_at': datetime.now().isoformat(),
        'lookback_days': LOOKBACK_DAYS
    }
    
    metadata_file = f"model_metadata_{city_name.replace(' ', '_')}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    print(f"  💾 Model saved to {model_filename}")
    
    return {
        "status": "success",
        "model_type": "LSTM",
        "trained_at": datetime.now().isoformat(),
        "age_days": 0,
        "train_loss": float(train_loss),
        "train_mae": float(train_mae),
        "test_loss": float(test_loss),
        "test_mae": float(test_mae),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "model_file": model_filename,
        "lookback_days": LOOKBACK_DAYS
    }, 200


def predict_next_week(city_name):
    """
    ✅ FIXED: Predict pollutant levels for next 7 days starting from TOMORROW
    Predictions: Tomorrow (Day +1) to Day +7 from current date
    """
    print(f"\n🔮 Generating predictions for {city_name}...")
    
    model_filename = f"lstm_model_{city_name.replace(' ', '_')}.keras"
    model_filename_legacy = f"lstm_model_{city_name.replace(' ', '_')}.h5"
    scaler_filename = f"scaler_{city_name.replace(' ', '_')}.pkl"
    
    try:
        if os.path.exists(model_filename):
            model = load_model(model_filename, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print(f"  ✓ Model loaded: LSTM (.keras format)")
        elif os.path.exists(model_filename_legacy):
            model = load_model(model_filename_legacy, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print(f"  ✓ Model loaded: LSTM (.h5 format - legacy)")
        else:
            return {"error": f"Model not trained yet for {city_name}. Please train the model first."}, 400
            
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
            
    except Exception as e:
        print(f"  Error loading model: {e}")
        return {"error": f"Error loading model: {str(e)}"}, 500
    
    # Get recent data
    df = prepare_lstm_data(city_name)
    if df is None or len(df) < LOOKBACK_DAYS:
        return {"error": f"Insufficient data for prediction (need {LOOKBACK_DAYS} days)"}, 400
    
    feature_cols = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3', 'ch4',
                   'temperature', 'humidity', 'wind_speed', 'precipitation']
    
    recent_data = df[feature_cols].iloc[-LOOKBACK_DAYS:].values
    recent_data_scaled = scaler.transform(recent_data)
    
    predictions = []
    current_sequence = recent_data_scaled.copy()
    
    # ✅ FIXED: Start predictions from TOMORROW (not accounting for data lag)
    today = datetime.now()
    
    print(f"  📅 Today's date: {today.date()}")
    print(f"  🔮 Predicting from: {(today + timedelta(days=1)).date()} to {(today + timedelta(days=7)).date()}")
    
    for day in range(7):
        # ✅ Predict for tomorrow (day+1) through next week (day+7)
        X_input = current_sequence[-LOOKBACK_DAYS:].reshape(1, LOOKBACK_DAYS, len(feature_cols))
        
        prediction_scaled = model.predict(X_input, verbose=0)[0]
        
        full_prediction = np.zeros(len(feature_cols))
        full_prediction[:7] = prediction_scaled
        full_prediction[7:] = current_sequence[-1, 7:]  # Keep weather data same
        
        prediction = scaler.inverse_transform(full_prediction.reshape(1, -1))[0]
        
        # ✅ FIXED: Start from tomorrow (day + 1)
        pred_date = today + timedelta(days=day+1)
        predictions.append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'day': pred_date.strftime('%A'),
            'day_offset': day + 1,  # Tomorrow is +1, day after is +2, etc.
            'pm25': float(np.clip(prediction[0], 0, 500)),
            'pm10': float(np.clip(prediction[1], 0, 600)),
            'so2': float(np.clip(prediction[2], 0, 100)),
            'no2': float(np.clip(prediction[3], 0, 200)),
            'co': float(np.clip(prediction[4], 0, 50)),
            'o3': float(np.clip(prediction[5], 0, 300)),
            'ch4': float(np.clip(prediction[6], 1700, 2000))
        })
        
        current_sequence = np.vstack([current_sequence[1:], full_prediction])
    
    print(f"  ✅ Generated {len(predictions)} days of predictions (Tomorrow through Next Week)")
    
    return {
        "status": "success",
        "city": city_name,
        "current_date": today.strftime('%Y-%m-%d'),
        "prediction_start_date": (today + timedelta(days=1)).strftime('%Y-%m-%d'),
        "prediction_end_date": (today + timedelta(days=7)).strftime('%Y-%m-%d'),
        "predictions": predictions,
        "model_info": {
            "model_type": "lstm",
            "lookback_days": LOOKBACK_DAYS
        },
        "note": "Predictions start from tomorrow and extend 7 days into the future"
    }, 200


def model_status_check(city_name):
    """Check model status for a city"""
    model_filename = f"lstm_model_{city_name.replace(' ', '_')}.keras"
    model_filename_legacy = f"lstm_model_{city_name.replace(' ', '_')}.h5"
    
    if os.path.exists(model_filename):
        model_file = model_filename
        model_format = "keras (native)"
    elif os.path.exists(model_filename_legacy):
        model_file = model_filename_legacy
        model_format = "h5 (legacy)"
    else:
        return {
            "status": "not_found",
            "city": city_name,
            "message": "No trained model exists for this city"
        }, 404
    
    model_time = datetime.fromtimestamp(os.path.getmtime(model_file))
    age_days = (datetime.now() - model_time).days
    
    return {
        "status": "found",
        "city": city_name,
        "model_type": "LSTM",
        "model_format": model_format,
        "trained_at": model_time.isoformat(),
        "age_days": age_days,
        "needs_retraining": age_days > MODEL_RETRAINING_DAYS,
        "lookback_days": LOOKBACK_DAYS,
        "model_file": model_file
    }, 200

def generate_map_tiles(lat, lon, radius_km=100):
    """
    Generate interactive map tile URL
    Uses data from safe date range (accounting for lag)
    """
    if not GEE_AVAILABLE:
        return None
    
    try:
        print(f"🗺️ Generating map tiles for ({lat}, {lon})...")
        
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius_km * 1000)
        
        safe_end_date = datetime.now() - timedelta(days=DATA_LAG_DAYS)
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
        
        # O3
        try:
            o3_collection = ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_O3') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .select('O3_column_number_density')
            o3_count = o3_collection.size().getInfo()
            if o3_count > 0:
                pollutants_data['O3'] = o3_collection.mean()
                print(f"  ✓ O3: {o3_count} images")
        except Exception as e:
            print(f"  ✗ O3 error: {e}")
        
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
        
        if 'O3' in pollutants_data:
            o3_normalized = pollutants_data['O3'].subtract(0.12).divide(0.04).clamp(0, 1)
            normalized_layers.append(o3_normalized)
        
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
    """
    Fetch current air quality data from GEE
    Uses safe date range (accounting for 6-day lag)
    """
    coords = get_coordinates(city_name)
    if not coords:
        return {"error": f"Could not find coordinates for {city_name}"}, 404
    
    try:
        latitude, longitude = coords
        
        safe_end_date = datetime.now() - timedelta(days=DATA_LAG_DAYS)
        start_date = safe_end_date - timedelta(days=7)
        
        print(f"  📅 Fetching current data from {start_date.date()} to {safe_end_date.date()}")
        
        pollutant_data = fetch_gee_pollutant_data(latitude, longitude, start_date, safe_end_date)
        weather_data = fetch_gee_weather_data(latitude, longitude, start_date, safe_end_date)
        
        if not pollutant_data:
            return {"error": "No current data available from satellites"}, 404
        
        map_tiles = generate_map_tiles(latitude, longitude, radius_km=100)
        
        combined_data = {}
        if pollutant_data:
            for key, value in pollutant_data.items():
                unit = "µg/m³"
                if key == 'ch4':
                    unit = "ppb"
                combined_data[key.upper()] = {
                    "parameter": key.upper(),
                    "value": value,
                    "unit": unit
                }
        
        result = {
            "city": city_name,
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "pollutants": combined_data,
            "weather": weather_data,
            "map_tiles": map_tiles,
            "timestamp": safe_end_date.isoformat(),
            "data_date": safe_end_date.strftime('%Y-%m-%d'),
            "data_lag_days": DATA_LAG_DAYS,
            "note": f"Data represents average from {start_date.date()} to {safe_end_date.date()} (accounting for {DATA_LAG_DAYS}-day satellite data lag)"
        }
        
        return result, 200
        
    except Exception as e:
        print(f"Error fetching current data: {e}")
        traceback.print_exc()
        return {"error": str(e)}, 500

# --- Flask API Endpoints ---

@app.route('/api/air-quality', methods=['GET'])
def air_quality_endpoint():
    """Get current air quality data from GEE"""
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    data, status_code = fetch_current_air_quality(city_name)
    return jsonify(data), status_code

@app.route('/api/collect-historical', methods=['POST'])
def collect_historical():
    """Collect historical data from GEE"""
    data = request.get_json()
    city_name = data.get('city')
    days_back = data.get('days_back', DAYS_BACK)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = fetch_historical_gee_data(city_name, days_back=days_back)
    return jsonify(result), status_code

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train LSTM prediction model"""
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
    
    print(f"\n🚀 Starting full analysis for {city_name}")
    
    print("\n📊 Step 1: Checking historical data...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM historical_data WHERE city = ?", (city_name,))
    existing_records = cursor.fetchone()[0]
    conn.close()
    
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
            "message": "Sufficient data already exists", 
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
            "model_file": train_result.get("model_file"),
            "message": train_result.get("message", "Model training completed")
        },
        "predictions": pred_result,
        "current_data": current_result,
        "configuration": {
            "days_back": DAYS_BACK,
            "lookback_period": LOOKBACK_DAYS,
            "data_lag_days": DATA_LAG_DAYS,
            "model_type": "LSTM",
            "data_source": "Google Earth Engine",
            "prediction_range": "Tomorrow through next 7 days",
            "note": f"Historical data accounts for {DATA_LAG_DAYS}-day satellite lag; predictions start from tomorrow"
        }
    }
    
    return jsonify(response_data), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "gee_available": GEE_AVAILABLE,
        "database": DB_NAME,
        "service": "Air Quality API with LSTM Predictions (GEE-based)",
        "configuration": {
            "days_back": DAYS_BACK,
            "data_lag_days": DATA_LAG_DAYS,
            "lookback_period": LOOKBACK_DAYS,
            "chunk_size": CHUNK_SIZE,
            "model_retraining_days": MODEL_RETRAINING_DAYS,
            "data_source": "Google Earth Engine Sentinel-5P & ERA5",
            "prediction_range": "Tomorrow through next 7 days",
            "note": f"Historical data collection accounts for {DATA_LAG_DAYS}-day lag; predictions are for future dates"
        }
    }), 200

@app.route('/api/database-stats', methods=['GET'])
def database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT DISTINCT city FROM historical_data")
        cities = [row[0] for row in cursor.fetchall()]
        
        city_stats = {}
        for city in cities:
            cursor.execute("""
                SELECT MIN(date), MAX(date), COUNT(*) 
                FROM historical_data 
                WHERE city = ?
            """, (city,))
            min_date, max_date, count = cursor.fetchone()
            city_stats[city] = {
                "records": count,
                "date_range": {
                    "start": min_date,
                    "end": max_date
                }
            }
        
        conn.close()
        
        return jsonify({
            "status": "success",
            "total_records": total_records,
            "cities": city_stats,
            "data_lag_days": DATA_LAG_DAYS
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

# --- Run Flask App ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 Air Quality API with LSTM Predictions (GEE-BASED)")
    print("="*70)
    print(f"GEE Status: {'✅ Available' if GEE_AVAILABLE else '❌ Not Available'}")
    print(f"Database: {DB_NAME}")
    print(f"\n📈 CONFIGURATION:")
    print(f"   • Satellite Data Lag: {DATA_LAG_DAYS} days")
    print(f"   • Historical Data Collection: {DAYS_BACK} days (adjusted for lag)")
    print(f"   • Historical Data Range: ~4 years ago to {DATA_LAG_DAYS} days ago")
    print(f"   • Model Type: LSTM (Long Short-Term Memory)")
    print(f"   • Model Lookback Period: {LOOKBACK_DAYS} days")
    print(f"   • Data Collection Chunk Size: {CHUNK_SIZE} days")
    print(f"   • Model Cache Duration: {MODEL_RETRAINING_DAYS} days")
    print(f"   • Data Source: Google Earth Engine (Sentinel-5P + ERA5)")
    print(f"   • Pollutants: PM2.5, PM10, SO2, NO2, CO, O3, CH4")
    print(f"   • Weather: Temperature, Humidity, Wind Speed, Precipitation")
    print(f"\n🔮 PREDICTION CONFIGURATION:")
    print(f"   • Prediction Start: TOMORROW (Day +1 from today)")
    print(f"   • Prediction End: 7 days from today (Day +7)")
    print(f"   • Prediction Duration: 7 days into the future")
    print(f"\n⚠️  DATA FLOW:")
    print(f"   • Historical Collection: Stops {DATA_LAG_DAYS} days before today (satellite lag)")
    print(f"   • Model Training: Uses historical data up to {DATA_LAG_DAYS} days ago")
    print(f"   • Predictions: Start from TOMORROW, no lag applied (future forecast)")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)