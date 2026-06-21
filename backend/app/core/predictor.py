import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, roc_auc_score

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    HAS_LGB = False

try:
    from app.core.data_processor import AstraDataProcessor
except ImportError:
    from data_processor import AstraDataProcessor


class AstraPredictor:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.duration_model = None
        self.priority_model = None
        self.closure_model = None
        self.processor = None
        
    def train_models(self, data_path):
        print(f"Loading data from {data_path}...")
        self.processor = AstraDataProcessor(data_path)
        df = self.processor.load_data()
        
        print("Preprocessing dataset...")
        processed_df = self.processor.preprocess(df)
        X, y_dur, y_prio, y_close = self.processor.get_features_and_targets(processed_df)
        
        # 1. Train Duration Model (Regression)
        print("\n--- Training Event Duration Prediction Model ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y_dur, test_size=0.2, random_state=42)
        if HAS_LGB:
            print("Using LightGBM Regressor")
            self.duration_model = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        else:
            print("Using RandomForest Regressor fallback")
            self.duration_model = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
            
        self.duration_model.fit(X_train, y_train)
        preds = self.duration_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"Duration Model Evaluation -> MAE: {mae:.2f} mins, R2 Score: {r2:.4f}")
        
        # 2. Train Priority Model (Classification)
        print("\n--- Training Priority Classification Model ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y_prio, test_size=0.2, random_state=42)
        if HAS_LGB:
            print("Using LightGBM Classifier")
            self.priority_model = LGBMClassifier(n_estimators=100, random_state=42)
        else:
            print("Using RandomForest Classifier fallback")
            self.priority_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
            
        self.priority_model.fit(X_train, y_train)
        preds = self.priority_model.predict(X_test)
        proba = self.priority_model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds, average='binary')
        auc = roc_auc_score(y_test, proba)
        print(f"Priority Model Evaluation -> F1 Score: {f1:.4f}, ROC-AUC: {auc:.4f}")
        
        # 3. Train Road Closure Model (Classification)
        print("\n--- Training Road Closure Classification Model ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y_close, test_size=0.2, random_state=42)
        if HAS_LGB:
            self.closure_model = LGBMClassifier(n_estimators=100, random_state=42)
        else:
            self.closure_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
            
        self.closure_model.fit(X_train, y_train)
        preds = self.closure_model.predict(X_test)
        proba = self.closure_model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds, average='binary')
        auc = roc_auc_score(y_test, proba)
        print(f"Closure Model Evaluation -> F1 Score: {f1:.4f}, ROC-AUC: {auc:.4f}")
        
        # Save models and processor
        self.save_models()
        
    def save_models(self):
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.duration_model, os.path.join(self.model_dir, 'duration_model.joblib'))
        joblib.dump(self.priority_model, os.path.join(self.model_dir, 'priority_model.joblib'))
        joblib.dump(self.closure_model, os.path.join(self.model_dir, 'closure_model.joblib'))
        joblib.dump(self.processor, os.path.join(self.model_dir, 'processor.joblib'))
        print(f"All models saved successfully inside {self.model_dir}")
        
    def load_models(self):
        self.duration_model = joblib.load(os.path.join(self.model_dir, 'duration_model.joblib'))
        self.priority_model = joblib.load(os.path.join(self.model_dir, 'priority_model.joblib'))
        self.closure_model = joblib.load(os.path.join(self.model_dir, 'closure_model.joblib'))
        self.processor = joblib.load(os.path.join(self.model_dir, 'processor.joblib'))
        print("All models loaded successfully.")
        
    def predict_incident(self, input_data: dict):
        """
        input_data should look like:
        {
            'latitude': 12.971598,
            'longitude': 77.594562,
            'event_type': 'unplanned',
            'event_cause': 'vehicle_breakdown',
            'requires_road_closure': False,
            'start_datetime': '2026-06-21 12:00:00',
            'description': 'bmtc bus off road near junction',
            'veh_type': 'bmtc_bus',
            'corridor': 'Bellary Road 1',
            'priority': 'High',
            'police_station': 'Cubbon Park',
            'zone': 'Central Zone 2',
            'junction': 'QueensStatueCircle'
        }
        """
        # Create a single row DataFrame from input_data
        # We need to run preprocessing on this single row
        # However, to use the same LabelEncoder and preprocessing, we can write a single row converter
        df_row = pd.DataFrame([input_data])
        
        # 1. Parse datetime
        df_row['start_datetime'] = pd.to_datetime(df_row['start_datetime'], errors='coerce')
        
        # 2. Fill baseline values
        df_row['endlatitude'] = df_row.get('endlatitude', pd.Series([np.nan]))
        df_row['endlongitude'] = df_row.get('endlongitude', pd.Series([np.nan]))
        df_row['endlatitude'] = df_row['endlatitude'].fillna(df_row['latitude'])
        df_row['endlongitude'] = df_row['endlongitude'].fillna(df_row['longitude'])
        
        # Fill categorical columns
        df_row['veh_type'] = df_row.get('veh_type', pd.Series(['unknown_vehicle'])).fillna('unknown_vehicle')
        df_row['corridor'] = df_row.get('corridor', pd.Series(['Non-corridor'])).fillna('Non-corridor')
        df_row['priority'] = df_row.get('priority', pd.Series(['Low'])).fillna('Low')
        df_row['event_cause'] = df_row.get('event_cause', pd.Series(['others'])).fillna('others')
        df_row['police_station'] = df_row.get('police_station', pd.Series(['unknown_station'])).fillna('unknown_station')
        df_row['zone'] = df_row.get('zone', pd.Series(['unknown_zone'])).fillna('unknown_zone')
        df_row['junction'] = df_row.get('junction', pd.Series(['unknown_junction'])).fillna('unknown_junction')
        
        # 3. Temporal calculations
        df_row['hour'] = df_row['start_datetime'].dt.hour.fillna(12).astype(int)
        df_row['dayofweek'] = df_row['start_datetime'].dt.dayofweek.fillna(0).astype(int)
        df_row['month'] = df_row['start_datetime'].dt.month.fillna(3).astype(int)
        df_row['is_weekend'] = df_row['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        df_row['hour_sin'] = np.sin(2 * np.pi * df_row['hour'] / 24.0)
        df_row['hour_cos'] = np.cos(2 * np.pi * df_row['hour'] / 24.0)
        df_row['day_sin'] = np.sin(2 * np.pi * df_row['dayofweek'] / 7.0)
        df_row['day_cos'] = np.cos(2 * np.pi * df_row['dayofweek'] / 7.0)
        
        # 4. Domain specific weights
        df_row['is_planned'] = df_row['event_type'].apply(lambda x: 1 if x == 'planned' else 0)
        df_row['requires_road_closure_encoded'] = df_row['requires_road_closure'].apply(lambda x: 1 if x == True or str(x).upper() == 'TRUE' else 0)
        
        heavy_vehicle_map = {
            'bmtc_bus': 3, 'heavy_vehicle': 3, 'truck': 3, 'private_bus': 2, 'ksrtc_bus': 2,
            'lcv': 2, 'private_car': 1, 'taxi': 1, 'auto': 1, 'others': 1, 'unknown_vehicle': 1
        }
        df_row['vehicle_blockage_weight'] = df_row['veh_type'].map(heavy_vehicle_map).fillna(1)
        
        cause_weight_map = {
            'vehicle_breakdown': 2.5, 'accident': 3.0, 'tree_fall': 3.0, 'water_logging': 3.5,
            'construction': 2.0, 'pot_holes': 1.5, 'road_conditions': 1.5, 'congestion': 1.0,
            'public_event': 2.5, 'procession': 2.5, 'vip_movement': 3.0, 'protest': 3.5,
            'Debris': 2.0, 'debris': 2.0, 'others': 1.5
        }
        df_row['cause_severity_weight'] = df_row['event_cause'].map(cause_weight_map).fillna(1.5)
        
        df_row['description_len'] = df_row.get('description', pd.Series([''])).fillna('').apply(len)
        df_row['desc_has_slow'] = df_row.get('description', pd.Series([''])).fillna('').str.lower().apply(lambda x: 1 if 'slow' in x or 'traffic' in x or 'movement' in x else 0)
        df_row['desc_has_blocked'] = df_row.get('description', pd.Series([''])).fillna('').str.lower().apply(lambda x: 1 if 'block' in x or 'closed' in x or 'stop' in x else 0)
        
        df_row['estimated_impact_radius_km'] = df_row.apply(
            lambda row: 2.0 if row['event_type'] == 'planned' 
            else (1.0 if row['event_cause'] in ['water_logging', 'protest', 'accident', 'tree_fall'] else 0.3), axis=1
        )
        
        # Encode Categorical features using loaded LabelEncoders
        cat_cols = ['event_cause', 'corridor', 'priority', 'veh_type', 'police_station', 'zone', 'junction']
        for col in cat_cols:
            le = self.processor.label_encoders[col]
            val = df_row[col].astype(str).values[0]
            if val in le.classes_:
                df_row[col + '_encoded'] = le.transform([val])[0]
            else:
                # Handle unseen category
                df_row[col + '_encoded'] = -1
                
        # 5. Extract Feature Matrix
        feature_cols = [
            'latitude', 'longitude', 'endlatitude', 'endlongitude',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'month',
            'is_planned', 'requires_road_closure_encoded', 'vehicle_blockage_weight', 'cause_severity_weight',
            'description_len', 'desc_has_slow', 'desc_has_blocked', 'estimated_impact_radius_km',
            'event_cause_encoded', 'corridor_encoded', 'priority_encoded', 'veh_type_encoded', 
            'police_station_encoded', 'zone_encoded', 'junction_encoded'
        ]
        
        X_row = df_row[feature_cols]
        
        # 6. Run Predictions
        predicted_duration = self.duration_model.predict(X_row)[0]
        priority_prio_proba = self.priority_model.predict_proba(X_row)[0][1]
        closure_proba = self.closure_model.predict_proba(X_row)[0][1]
        
        return {
            'predicted_duration_mins': float(round(predicted_duration, 2)),
            'priority_probability_high': float(round(priority_prio_proba, 4)),
            'road_closure_probability': float(round(closure_proba, 4))
        }

if __name__ == '__main__':
    # Test training and inference
    file_path = r"C:\Users\tkurm\Downloads\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    predictor = AstraPredictor()
    predictor.train_models(file_path)
    
    # Test a sample prediction
    test_input = {
        'latitude': 12.971598,
        'longitude': 77.594562,
        'event_type': 'unplanned',
        'event_cause': 'vehicle_breakdown',
        'requires_road_closure': False,
        'start_datetime': '2024-03-07 17:00:00',
        'description': 'bmtc bus broke down blocking left lane',
        'veh_type': 'bmtc_bus',
        'corridor': 'Bellary Road 1',
        'priority': 'High',
        'police_station': 'Cubbon Park',
        'zone': 'Central Zone 2',
        'junction': 'QueensStatueCircle'
    }
    
    result = predictor.predict_incident(test_input)
    print("\nSample Prediction Output:")
    print(result)
