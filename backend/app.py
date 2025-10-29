import openaq
from geopy.geocoders import Nominatim
from openaq import AuthError, BadRequestError, RateLimitError, OpenAQ
from flask import Flask, request, jsonify
import os
from datetime import datetime, timedelta
from httpx import ReadTimeout
import ee
import traceback
import sqlite3
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from openaq.shared.exceptions import RateLimitError
import pickle
import json

# --- OpenAQ Configuration ---
API_KEY = os.environ.get("OPENAQ_API_KEY", "1589e5d71be416b7bdf72bf6894def048b9521ad2ed9d513c7e6d9143cdc5072")

# Initialize OpenAQ client
local_client = openaq.OpenAQ(api_key=API_KEY)
geolocator = Nominatim(user_agent="air-quality-checker")

# --- Google Earth Engine Configuration ---
SERVICE_ACCOUNT = 'gee-service-account@street-view-videos-with-maps.iam.gserviceaccount.com'
SERVICE_KEY_FILE = 'gee-service-key.json'

try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, SERVICE_KEY_FILE)
    ee.Initialize(credentials)
    print("✅ Google Earth Engine initialized successfully")
    GEE_AVAILABLE = True
except Exception as e:
    print(f"❌ GEE initialization error: {e}")
    GEE_AVAILABLE = False

# --- Flask App Configuration ---
app = Flask(__name__)

# --- Database Setup ---
DB_NAME = 'air_quality.db'

# --- OPTIMIZATION CONSTANTS ---
DAYS_BACK = 70  # Increased from 60 for more training data
LOOKBACK_PERIOD = 40  # Increased from 30 for better pattern recognition
MODEL_RETRAINING_DAYS = 7  # Retrain model if older than this many days
MAX_SENSORS = 100  # Limit sensors to avoid API overload

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
            timestamp DATETIME NOT NULL,
            pm25 REAL,
            pm10 REAL,
            so2 REAL,
            no2 REAL,
            co REAL,
            o3 REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_city_timestamp 
        ON historical_data(city, timestamp)
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# Initialize database on startup
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

def store_historical_data(city, coords, pollutants_data, timestamp):
    """Store pollutant data in SQLite database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO historical_data 
            (city, latitude, longitude, timestamp, pm25, pm10, so2, no2, co, o3)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            city,
            coords[0],
            coords[1],
            timestamp,
            pollutants_data.get('pm25'),
            pollutants_data.get('pm10'),
            pollutants_data.get('so2'),
            pollutants_data.get('no2'),
            pollutants_data.get('co'),
            pollutants_data.get('o3')
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing data: {e}")
        return False

def check_existing_model(city_name, max_age_days=MODEL_RETRAINING_DAYS):
    """
    Check if a trained model exists and is recent enough.
    Returns (exists: bool, model_data: dict or None, age_days: int)
    """
    model_filename = f"model_{city_name.replace(' ', '_')}.pkl"
    
    if not os.path.exists(model_filename):
        print(f"  ℹ️  No existing model found for {city_name}")
        return False, None, None
    
    try:
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        trained_at = datetime.fromisoformat(model_data['trained_at'])
        age_days = (datetime.now() - trained_at).days
        
        print(f"  📦 Found existing model for {city_name}")
        print(f"     - Trained: {trained_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"     - Age: {age_days} days")
        print(f"     - Model type: {model_data.get('model_type', 'unknown').upper()}")
        print(f"     - Test R²: {model_data.get('test_score', 0):.4f}")
        
        if age_days <= max_age_days:
            print(f"  ✅ Model is recent enough (< {max_age_days} days old)")
            return True, model_data, age_days
        else:
            print(f"  ⚠️  Model is too old (> {max_age_days} days), retraining recommended")
            return True, model_data, age_days
            
    except Exception as e:
        print(f"  ❌ Error loading existing model: {e}")
        return False, None, None

def fetch_historical_openaq_data(city_name, days_back=DAYS_BACK, test_mode=False):
    """
    Fetches historical data from OpenAQ by first finding locations near a city
    and then fetching measurements for each sensor (limited to max_sensors).
    """
    local_client = OpenAQ(api_key=API_KEY)
    max_sensors = MAX_SENSORS

    coords = get_coordinates(city_name)
    if not coords:
        local_client.close()
        return {"error": f"Could not find coordinates for {city_name}"}, 404

    latitude, longitude = coords
    print(f"\n📊 Fetching {days_back} days of historical data for {city_name} (coords: {latitude}, {longitude})...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        target_pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        all_measurements = []

        # Step 1: Find relevant locations/sensors
        print("Step 1: Finding relevant sensors...")
        locations_response = local_client.locations.list(
            coordinates=[latitude, longitude],
            radius=25000,
            limit=100
        )
        locations = getattr(locations_response, 'results', [])
        if not locations:
            local_client.close()
            return {"message": f"No locations found near {city_name}"}, 404
        print(f"Found {len(locations)} locations near {city_name}.")

        # Extract all relevant sensor IDs
        sensor_ids_with_pollutant = {}
        for loc in locations:
            if hasattr(loc, 'sensors'):
                for s in loc.sensors:
                    parameter_name = getattr(getattr(s, 'parameter', None), "name", "").lower()
                    if parameter_name in target_pollutants:
                        sensor_ids_with_pollutant.setdefault(s.id, []).append(parameter_name)
        
        if not sensor_ids_with_pollutant:
            local_client.close()
            return {"message": "No relevant sensors found for target pollutants"}, 404
        print(f"Found {len(sensor_ids_with_pollutant)} unique sensors with target pollutants.")

        # ✅ Limit sensors to avoid excessive API calls
        if test_mode:
            print("Running in test mode. Limiting to the first 5 sensors.")
            test_sensor_ids = list(sensor_ids_with_pollutant.keys())[:5]
        else:
            if len(sensor_ids_with_pollutant) > max_sensors:
                print(f"⚠️ Found {len(sensor_ids_with_pollutant)} sensors. Limiting to {max_sensors} to prevent overload.")
            test_sensor_ids = list(sensor_ids_with_pollutant.keys())[:max_sensors]

        # Step 2: Fetch measurements with corrected data extraction
        print("Step 2: Fetching measurements for each sensor...")
        for sensor_id in test_sensor_ids:
            pollutants = sensor_ids_with_pollutant[sensor_id]
            print(f"  - Fetching for sensor {sensor_id} (pollutants: {', '.join(pollutants)})...")
            page = 1
            limit = 1000
            
            while True:
                try:
                    measurements_response = local_client.measurements.list(
                        sensors_id=sensor_id,
                        datetime_from=start_date.isoformat(),
                        datetime_to=end_date.isoformat(),
                        limit=limit,
                        page=page
                    )

                    if not hasattr(measurements_response, "results") or not measurements_response.results:
                        print(f"    - Page {page}: No more results found for sensor {sensor_id}.")
                        break
                    
                    results_count = len(measurements_response.results)
                    print(f"    - Page {page}: Found {results_count} measurements.")
                    
                    # Extract data with correct attribute names and proper datetime handling
                    extracted_count = 0
                    for m in measurements_response.results:
                        try:
                            # Get timestamp from period object
                            timestamp = None
                            if hasattr(m, 'period'):
                                period = m.period
                                if hasattr(period, 'datetimeFrom'):
                                    datetime_obj = period.datetimeFrom
                                    if hasattr(datetime_obj, 'utc'):
                                        timestamp = datetime_obj.utc
                                    elif hasattr(datetime_obj, 'isoformat'):
                                        timestamp = datetime_obj.isoformat()
                                    else:
                                        timestamp = str(datetime_obj)
                                elif hasattr(period, 'datetime_from'):
                                    datetime_obj = period.datetime_from
                                    if hasattr(datetime_obj, 'utc'):
                                        timestamp = datetime_obj.utc
                                    elif hasattr(datetime_obj, 'isoformat'):
                                        timestamp = datetime_obj.isoformat()
                                    else:
                                        timestamp = str(datetime_obj)
                                elif hasattr(period, 'label'):
                                    timestamp = period.label
                            
                            # If still no timestamp, try coverage object
                            if not timestamp and hasattr(m, 'coverage'):
                                coverage = m.coverage
                                if hasattr(coverage, 'datetimeFrom'):
                                    datetime_obj = coverage.datetimeFrom
                                    if hasattr(datetime_obj, 'utc'):
                                        timestamp = datetime_obj.utc
                                    elif hasattr(datetime_obj, 'isoformat'):
                                        timestamp = datetime_obj.isoformat()
                                    else:
                                        timestamp = str(datetime_obj)
                            
                            # Convert timestamp to string if it's still an object
                            if timestamp is not None:
                                if hasattr(timestamp, 'isoformat'):
                                    timestamp = timestamp.isoformat()
                                elif not isinstance(timestamp, str):
                                    timestamp = str(timestamp)
                            
                            # Get parameter name
                            parameter = None
                            if hasattr(m, 'parameter'):
                                if hasattr(m.parameter, 'name'):
                                    parameter = m.parameter.name.lower()
                                elif isinstance(m.parameter, str):
                                    parameter = m.parameter.lower()
                            
                            # Get value
                            value = getattr(m, 'value', None)
                            
                            if timestamp and parameter and value is not None:
                                all_measurements.append({
                                    "timestamp": timestamp,
                                    "parameter": parameter,
                                    "value": value,
                                    "sensor_id": sensor_id
                                })
                                extracted_count += 1
                                
                        except Exception:
                            continue
                    
                    print(f"    - Extracted {extracted_count} valid measurements from page {page}")
                    
                    page += 1
                    
                except (RateLimitError, ReadTimeout) as e:
                    wait_time = 60
                    print(f"    - Exception caught ({type(e).__name__}) for sensor {sensor_id}. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                except Exception as e:
                    print(f"    - Error fetching data for sensor {sensor_id}, page {page}: {e}")
                    traceback.print_exc()
                    break

        print(f"✓ Total measurements fetched: {len(all_measurements)}")

        if not all_measurements:
            local_client.close()
            return {"message": "No historical data available for the specified period"}, 404

        # Step 3: Process data with proper datetime conversion
        df = pd.DataFrame(all_measurements)
        
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            print(f"Error converting timestamps: {e}")
            df["timestamp"] = df["timestamp"].apply(lambda x: 
                pd.to_datetime(x.utc if hasattr(x, 'utc') else str(x))
            )
        
        df["date"] = df["timestamp"].dt.date
        
        # Aggregate daily data
        daily_data = df.groupby(["date", "parameter"])["value"].mean().unstack(fill_value=None)
        daily_data = daily_data.reset_index()

        # Step 4: Store aggregated daily data
        stored_count = 0
        for _, row in daily_data.iterrows():
            pollutants = {p: row.get(p) for p in target_pollutants}
            if store_historical_data(city_name, (latitude, longitude), pollutants, row["date"]):
                stored_count += 1

        print(f"✅ Stored {stored_count} days of historical data")
        
        local_client.close()
        
        return {
            "status": "success",
            "city": city_name,
            "days_collected": stored_count,
            "date_range": {
                "start": str(daily_data['date'].min()),
                "end": str(daily_data['date'].max())
            }
        }, 200

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        traceback.print_exc()
        local_client.close()
        return {"error": str(e)}, 500


def prepare_training_data(city_name):
    """Prepare data from database for ML training"""
    try:
        conn = sqlite3.connect(DB_NAME)
        query = '''
            SELECT timestamp, pm25, pm10, so2, no2, co, o3
            FROM historical_data
            WHERE city = ?
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn, params=(city_name,))
        conn.close()
        
        # Need at least 50 days for reliable predictions with 40-day lookback
        min_required = LOOKBACK_PERIOD + 20
        if df.empty or len(df) < min_required:
            print(f"  ⚠️ Insufficient data: Found {len(df)} rows, need at least {min_required}")
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Remove outliers (values beyond 3 standard deviations)
        pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        for pollutant in pollutants:
            if pollutant in df.columns:
                mean = df[pollutant].mean()
                std = df[pollutant].std()
                if std > 0:  # Avoid division by zero
                    df[pollutant] = df[pollutant].clip(lower=mean-3*std, upper=mean+3*std)
        
        # Interpolate missing values instead of forward fill
        df[pollutants] = df[pollutants].interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining NaN with median
        df[pollutants] = df[pollutants].fillna(df[pollutants].median())
        
        print(f"  ✓ Prepared {len(df)} days of training data")
        return df
    except Exception as e:
        print(f"Error preparing training data: {e}")
        traceback.print_exc()
        return None

def create_features(df, lookback=LOOKBACK_PERIOD):
    """Create features for time series prediction with enhanced lookback"""
    features = []
    targets = []
    pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
    
    for i in range(lookback, len(df)):
        # Features: past lookback days of data
        feature_row = []
        for pollutant in pollutants:
            if pollutant in df.columns:
                feature_row.extend(df[pollutant].iloc[i-lookback:i].values)
        
        # Target: next day's values
        target_row = [df[p].iloc[i] for p in pollutants if p in df.columns]
        
        if len(feature_row) > 0 and len(target_row) > 0:
            features.append(feature_row)
            targets.append(target_row)
    
    return np.array(features), np.array(targets)

def train_prediction_model(city_name, force_retrain=False):
    """
    Train XGBoost model with advanced hyperparameter tuning for 80%+ accuracy.
    Uses smart model caching - only retrains if model doesn't exist or is too old.
    
    Args:
        city_name: Name of the city
        force_retrain: If True, retrain even if recent model exists
    """
    print(f"\n🤖 Training XGBoost model for {city_name}...")
    
    # Check for existing model
    if not force_retrain:
        exists, model_data, age_days = check_existing_model(city_name, MODEL_RETRAINING_DAYS)
        
        if exists and age_days is not None and age_days <= MODEL_RETRAINING_DAYS:
            print(f"  ✅ Using existing model (only {age_days} days old)")
            return {
                "status": "success",
                "message": "Using existing trained model",
                "model_type": model_data.get('model_type', 'xgboost'),
                "city": city_name,
                "model_age_days": age_days,
                "trained_at": model_data['trained_at'],
                "train_score": model_data.get('train_score'),
                "test_score": model_data.get('test_score'),
                "cv_score": model_data.get('cv_score'),
                "test_mae": model_data.get('test_mae'),
                "model_file": f"model_{city_name.replace(' ', '_')}.pkl",
                "retraining_skipped": True
            }, 200
    
    # Prepare data
    df = prepare_training_data(city_name)
    if df is None:
        return {"error": "Insufficient historical data for training"}, 400
    
    # Create features with extended lookback
    X, y = create_features(df, lookback=LOOKBACK_PERIOD)
    
    if len(X) == 0:
        return {"error": "Not enough data points for training"}, 400
    
    print(f"  Training samples: {len(X)}")
    print(f"  Features shape: {X.shape}, Targets shape: {y.shape}")
    print(f"  Lookback period: {LOOKBACK_PERIOD} days")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )
    
    # Advanced scaling with robust scaler
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost with optimized parameters
    result = _train_xgboost_optimized(X_train_scaled, X_test_scaled, y_train, y_test, scaler, city_name)
    return result['response'], result['status']


def _train_xgboost_optimized(X_train_scaled, X_test_scaled, y_train, y_test, scaler, city_name):
    """Train XGBoost with enhanced hyperparameter tuning for maximum accuracy"""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  ⚠️  XGBoost not installed. Install with: pip install xgboost")
        return {
            'response': {"error": "XGBoost not installed"},
            'status': 400,
            'test_score': 0
        }
    
    print("  🚀 Training XGBoost with optimized hyperparameters...")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # Enhanced parameter grid for better accuracy
    param_grid = {
        'n_estimators': [500, 700, 1000, 1200, 1500],
        'max_depth': [8, 10, 12, 15, 18, 20],
        'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        'subsample': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
        'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3, 4, 5],
        'gamma': [0, 0.01, 0.05, 0.1, 0.15, 0.2],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 1.5],
        'reg_lambda': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    }
    
    xgb = XGBRegressor(
        random_state=42,
        n_jobs=-1,
        tree_method='hist',  # Faster training
        objective='reg:squarederror',
        booster='gbtree',
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    # Increased iterations for better hyperparameter search
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=150,  # Increased from 100
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    print("  ⏳ Hyperparameter search in progress (this may take a while)...")
    random_search.fit(
        X_train_scaled, 
        y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    cv_score = random_search.best_score_
    
    # Additional metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    y_pred_test = model.predict(X_test_scaled)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    print(f"\n  ✅ XGBoost Performance:")
    print(f"     Test R²: {test_score:.4f} ({test_score*100:.1f}%)")
    print(f"     CV R²: {cv_score:.4f} ({cv_score*100:.1f}%)")
    print(f"     Train R²: {train_score:.4f}")
    print(f"     Test MAE: {test_mae:.4f}")
    print(f"     Test RMSE: {test_rmse:.4f}")
    print(f"     Test MAPE: {test_mape:.2f}%")
    print(f"\n  🎯 Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"     {param}: {value}")
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_type': 'xgboost',
        'city': city_name,
        'trained_at': datetime.now().isoformat(),
        'train_score': float(train_score),
        'test_score': float(test_score),
        'cv_score': float(cv_score),
        'oob_score': None,
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_mape': float(test_mape),
        'best_params': best_params,
        'lookback': LOOKBACK_PERIOD,
        'days_back': DAYS_BACK
    }
    
    model_filename = f"model_{city_name.replace(' ', '_')}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"  💾 Model saved to {model_filename}")
    
    return {
        'response': {
            "status": "success",
            "model_type": "xgboost",
            "city": city_name,
            "train_score": float(train_score),
            "test_score": float(test_score),
            "cv_score": float(cv_score),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "test_mape": float(test_mape),
            "best_params": best_params,
            "model_file": model_filename,
            "lookback_days": LOOKBACK_PERIOD,
            "training_data_days": DAYS_BACK
        },
        'status': 200,
        'test_score': test_score
    }


def predict_next_week(city_name):
    """Predict pollutant levels for next 7 days using cached or trained model"""
    print(f"\n🔮 Generating predictions for {city_name}...")
    
    # Load model
    model_filename = f"model_{city_name.replace(' ', '_')}.pkl"
    
    try:
        with open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        model_type = model_data.get('model_type', 'unknown')
        lookback = model_data.get('lookback', LOOKBACK_PERIOD)
        
        print(f"  ✓ Model loaded: {model_type.upper()}")
        print(f"  ✓ Lookback period: {lookback} days")
        print(f"  ✓ Model age: {(datetime.now() - datetime.fromisoformat(model_data['trained_at'])).days} days")
    except FileNotFoundError:
        return {"error": f"Model not trained yet for {city_name}. Please train the model first."}, 400
    except Exception as e:
        return {"error": f"Error loading model: {str(e)}"}, 500
    
    # Get recent data
    df = prepare_training_data(city_name)
    if df is None or len(df) < lookback:
        return {"error": f"Insufficient data for prediction (need {lookback} days)"}, 400
    
    pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
    recent_data = df[pollutants].iloc[-lookback:].values.flatten()
    
    predictions = []
    current_input = recent_data.copy()
    
    for day in range(7):
        # Predict next day
        input_scaled = scaler.transform(current_input.reshape(1, -1))
        prediction = model.predict(input_scaled)[0]
        
        # Store prediction with realistic bounds
        pred_date = datetime.now() + timedelta(days=day+1)
        predictions.append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'day': pred_date.strftime('%A'),
            'pm25': float(np.clip(prediction[0], 0, 500)),
            'pm10': float(np.clip(prediction[1], 0, 600)),
            'so2': float(np.clip(prediction[2], 0, 100)),
            'no2': float(np.clip(prediction[3], 0, 200)),
            'co': float(np.clip(prediction[4], 0, 50)),
            'o3': float(np.clip(prediction[5], 0, 300))
        })
        
        # Update input for next prediction
        current_input = np.concatenate([current_input[6:], prediction])
    
    print(f"  ✅ Generated {len(predictions)} days of predictions using {model_type.upper()}")
    
    return {
        "status": "success",
        "city": city_name,
        "predictions": predictions,
        "model_info": {
            "model_type": model_type,
            "trained_at": model_data['trained_at'],
            "train_score": model_data.get('train_score'),
            "test_score": model_data.get('test_score'),
            "cv_score": model_data.get('cv_score'),
            "test_mae": model_data.get('test_mae'),
            "test_rmse": model_data.get('test_rmse'),
            "best_params": model_data.get('best_params', {}),
            "lookback_days": lookback
        }
    }, 200

def generate_map_tiles(lat, lon, radius_km=100):
    """Generate interactive map tile URL"""
    if not GEE_AVAILABLE:
        return None
    
    try:
        print(f"🗺️ Generating map tiles for ({lat}, {lon})...")
        
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(radius_km * 1000)
        
        end_date = ee.Date(datetime.now())
        start_date = end_date.advance(-7, 'day')
        
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
        
        # Normalize and combine pollutants
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
        
        # Combine layers
        combined_aqi = normalized_layers[0]
        for layer in normalized_layers[1:]:
            combined_aqi = combined_aqi.max(layer)
        
        # Define visualization
        vis_params = {
            'min': 0,
            'max': 1,
            'palette': [
                '#00FF00',  # Green - Safe
                '#90EE90',
                '#FFFF00',  # Yellow - Moderate
                '#FFA500',  # Orange
                '#FF4500',
                '#FF0000',  # Red - Severe
                '#8B0000'
            ]
        }
        
        # Get map ID
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
                'start': start_date.format('YYYY-MM-dd').getInfo(),
                'end': end_date.format('YYYY-MM-dd').getInfo()
            }
        }
        
    except Exception as e:
        print(f"  ✗ Error generating map tiles: {e}")
        traceback.print_exc()
        return None

def fetch_current_air_quality(city_name):
    """Fetch current air quality data"""
    
    coords = get_coordinates(city_name)
    if not coords:
        local_client.close()
        return {"error": f"Could not find coordinates for {city_name}"}, 404

    try:
        locations_response = local_client.locations.list(
            coordinates=coords,
            radius=25000,
            limit=100
        )
        
        locations = locations_response.results if hasattr(locations_response, 'results') else []
        
        if not locations:
            local_client.close()
            return {"message": f"No stations found near {city_name}"}, 404

        target_pollutants = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
        city_pollutants = {}
        
        for loc in locations:
            if hasattr(loc, 'sensors') and loc.sensors:
                added_parameters = set()
                
                for sensor in loc.sensors:
                    try:
                        parameter = sensor.parameter.name if hasattr(sensor.parameter, 'name') else 'unknown'
                        unit = sensor.parameter.units if hasattr(sensor.parameter, 'units') else 'unknown'
                        display_name = sensor.parameter.display_name if hasattr(sensor.parameter, 'display_name') else parameter
                        sensor_id = sensor.id
                        
                        parameter_lower = parameter.lower()
                        
                        if display_name not in added_parameters and any(p in parameter_lower for p in target_pollutants):
                            sensor_measurements = local_client.measurements.list(
                                sensors_id=sensor_id,
                                limit=1
                            )
                            
                            if sensor_measurements and hasattr(sensor_measurements, 'results') and sensor_measurements.results:
                                latest = sensor_measurements.results[0]
                                
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
                                
                                if display_name not in city_pollutants:
                                    city_pollutants[display_name] = measurement_data
                                
                                added_parameters.add(display_name)
                                
                    except Exception as e:
                        continue
        
        # Generate map tiles
        map_tiles = None
        if GEE_AVAILABLE:
            map_tiles = generate_map_tiles(coords[0], coords[1], radius_km=100)
        
        result = {
            "city": city_name,
            "coordinates": {"latitude": coords[0], "longitude": coords[1]},
            "pollutants": city_pollutants,
            "map_tiles": map_tiles,
            "timestamp": datetime.now().isoformat()
        }
        
        local_client.close()
        return result, 200

    except Exception as e:
        print(f"Error fetching current data: {e}")
        traceback.print_exc()
        local_client.close()
        return {"error": str(e)}, 500

# --- Flask API Endpoints ---

@app.route('/api/air-quality', methods=['GET'])
def air_quality_endpoint():
    """Get current air quality data"""
    city_name = request.args.get('city')
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400

    data, status_code = fetch_current_air_quality(city_name)
    return jsonify(data), status_code

@app.route('/api/collect-historical', methods=['POST'])
def collect_historical():
    """Collect historical data (default: 70 days)"""
    data = request.get_json()
    city_name = data.get('city')
    days_back = data.get('days_back', DAYS_BACK)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = fetch_historical_openaq_data(city_name, days_back=days_back)
    return jsonify(result), status_code

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train prediction model (uses caching by default)"""
    data = request.get_json()
    city_name = data.get('city')
    force_retrain = data.get('force_retrain', False)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = train_prediction_model(city_name, force_retrain=force_retrain)
    return jsonify(result), status_code

@app.route('/api/predict', methods=['GET'])
def predict():
    """Get predictions for next 7 days"""
    city_name = request.args.get('city')
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    result, status_code = predict_next_week(city_name)
    return jsonify(result), status_code

@app.route('/api/full-analysis', methods=['POST'])
def full_analysis():
    """Complete workflow: collect data, train model (if needed), and predict"""
    data = request.get_json()
    city_name = data.get('city')
    force_retrain = data.get('force_retrain', False)
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    print(f"\n🚀 Starting full analysis for {city_name}")
    
    # Step 1: Check if we need to collect data
    print("\n📊 Step 1: Checking historical data...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM historical_data WHERE city = ?", (city_name,))
    existing_records = cursor.fetchone()[0]
    conn.close()
    
    if existing_records < LOOKBACK_PERIOD + 20:
        print(f"  ⚠️ Insufficient data ({existing_records} records), collecting {DAYS_BACK} days...")
        hist_result, hist_status = fetch_historical_openaq_data(city_name, days_back=DAYS_BACK)
        if hist_status != 200:
            return jsonify(hist_result), hist_status
    else:
        print(f"  ✅ Sufficient data exists ({existing_records} records)")
        hist_result = {"status": "skipped", "message": "Sufficient data already exists"}
    
    # Step 2: Train model (with smart caching)
    print("\n🤖 Step 2: Training/Loading ML model...")
    train_result, train_status = train_prediction_model(city_name, force_retrain=force_retrain)
    if train_status != 200:
        return jsonify(train_result), train_status
    
    # Step 3: Generate predictions
    print("\n🔮 Step 3: Generating predictions...")
    pred_result, pred_status = predict_next_week(city_name)
    if pred_status != 200:
        return jsonify(pred_result), pred_status
    
    # Step 4: Get current data
    print("\n📊 Step 4: Fetching current data...")
    current_result, current_status = fetch_current_air_quality(city_name)
    
    return jsonify({
        "status": "success",
        "city": city_name,
        "historical_data": hist_result,
        "model_training": train_result,
        "predictions": pred_result,
        "current_data": current_result,
        "optimization_settings": {
            "days_back": DAYS_BACK,
            "lookback_period": LOOKBACK_PERIOD,
            "model_retraining_days": MODEL_RETRAINING_DAYS,
            "max_sensors": MAX_SENSORS
        }
    }), 200

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "gee_available": GEE_AVAILABLE,
        "database": DB_NAME,
        "service": "Air Quality API with ML Predictions (Optimized)",
        "optimization_settings": {
            "days_back": DAYS_BACK,
            "lookback_period": LOOKBACK_PERIOD,
            "model_retraining_days": MODEL_RETRAINING_DAYS,
            "max_sensors": MAX_SENSORS
        }
    }), 200

@app.route('/api/database-stats', methods=['GET'])
def database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Get total records
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        total_records = cursor.fetchone()[0]
        
        # Get cities
        cursor.execute("SELECT DISTINCT city FROM historical_data")
        cities = [row[0] for row in cursor.fetchall()]
        
        # Get date range per city
        city_stats = {}
        for city in cities:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*) 
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
            "cities": city_stats
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check model status for a city"""
    city_name = request.args.get('city')
    
    if not city_name:
        return jsonify({"error": "Missing 'city' parameter"}), 400
    
    exists, model_data, age_days = check_existing_model(city_name, MODEL_RETRAINING_DAYS)
    
    if not exists:
        return jsonify({
            "status": "not_found",
            "city": city_name,
            "message": "No trained model exists for this city"
        }), 404
    
    return jsonify({
        "status": "found",
        "city": city_name,
        "model_type": model_data.get('model_type'),
        "trained_at": model_data.get('trained_at'),
        "age_days": age_days,
        "needs_retraining": age_days > MODEL_RETRAINING_DAYS,
        "performance": {
            "train_score": model_data.get('train_score'),
            "test_score": model_data.get('test_score'),
            "cv_score": model_data.get('cv_score'),
            "test_mae": model_data.get('test_mae'),
            "test_rmse": model_data.get('test_rmse')
        },
        "best_params": model_data.get('best_params', {}),
        "lookback_days": model_data.get('lookback', LOOKBACK_PERIOD)
    }), 200

# --- Run Flask App ---
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌍 Air Quality API with ML Predictions (OPTIMIZED)")
    print("="*70)
    print(f"GEE Status: {'✅ Available' if GEE_AVAILABLE else '❌ Not Available'}")
    print(f"Database: {DB_NAME}")
    print(f"\n📈 OPTIMIZATION SETTINGS:")
    print(f"   • Historical Data Collection: {DAYS_BACK} days")
    print(f"   • Model Lookback Period: {LOOKBACK_PERIOD} days")
    print(f"   • Model Cache Duration: {MODEL_RETRAINING_DAYS} days")
    print(f"   • Max Sensors per Query: {MAX_SENSORS}")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)