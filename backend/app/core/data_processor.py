import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

class AstraDataProcessor:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.label_encoders = {}
        self.imputers = {}
        
    def load_data(self):
        if not self.file_path or not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found at {self.file_path}")
        return pd.read_csv(self.file_path)
        
    def preprocess(self, df):
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # 1. Parse Datetime columns
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df['closed_datetime'] = pd.to_datetime(df['closed_datetime'], errors='coerce')
        df['resolved_datetime'] = pd.to_datetime(df['resolved_datetime'], errors='coerce')
        
        # 2. Target Variable Engineering
        # Incident/Event Duration in Minutes
        df['closed_duration_mins'] = (df['closed_datetime'] - df['start_datetime']).dt.total_seconds() / 60.0
        
        # Handle outliers in target (negative values or values > 7 days/10080 mins)
        # We replace negative values with 0 and cap extremely long events to 10080 minutes to avoid skewing regression models
        df['closed_duration_mins'] = df['closed_duration_mins'].clip(lower=0, upper=10080)
        
        # Fill missing targets with median duration (64.5 minutes based on data analysis)
        df['closed_duration_mins'] = df['closed_duration_mins'].fillna(64.5)
        
        # 3. Handle Missing Values in Features
        # Replace 0 coordinates with NaN for imputation
        df['endlatitude'] = df['endlatitude'].replace(0, np.nan)
        df['endlongitude'] = df['endlongitude'].replace(0, np.nan)
        
        # Fill missing end latitudes with start latitudes
        df['endlatitude'] = df['endlatitude'].fillna(df['latitude'])
        df['endlongitude'] = df['endlongitude'].fillna(df['longitude'])
        
        # Fill categorical missing values
        df['veh_type'] = df['veh_type'].fillna('unknown_vehicle')
        df['corridor'] = df['corridor'].fillna('Non-corridor')
        df['priority'] = df['priority'].fillna('Low')
        df['event_cause'] = df['event_cause'].fillna('others')
        df['police_station'] = df['police_station'].fillna('unknown_station')
        df['zone'] = df['zone'].fillna('unknown_zone')
        df['junction'] = df['junction'].fillna('unknown_junction')
        
        # 4. Temporal Feature Engineering
        # Extract features from start_datetime
        df['hour'] = df['start_datetime'].dt.hour.fillna(12).astype(int)
        df['dayofweek'] = df['start_datetime'].dt.dayofweek.fillna(0).astype(int)
        df['month'] = df['start_datetime'].dt.month.fillna(3).astype(int)
        df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Cyclical encoding of hour and day of week
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
        
        # 5. Domain Specific Engineered Features
        # Is Planned Event
        df['is_planned'] = df['event_type'].apply(lambda x: 1 if x == 'planned' else 0)
        
        # Requires Road Closure
        df['requires_road_closure_encoded'] = df['requires_road_closure'].apply(lambda x: 1 if x == True or str(x).upper() == 'TRUE' else 0)
        
        # Vehicle Type Severity Index (BMTC busses, Heavy vehicles and Trucks block more lanes)
        heavy_vehicle_map = {
            'bmtc_bus': 3,
            'heavy_vehicle': 3,
            'truck': 3,
            'private_bus': 2,
            'ksrtc_bus': 2,
            'lcv': 2,
            'private_car': 1,
            'taxi': 1,
            'auto': 1,
            'others': 1,
            'unknown_vehicle': 1
        }
        df['vehicle_blockage_weight'] = df['veh_type'].map(heavy_vehicle_map).fillna(1)
        
        # Event Cause Severity Weight
        cause_weight_map = {
            'vehicle_breakdown': 2.5,
            'accident': 3.0,
            'tree_fall': 3.0,
            'water_logging': 3.5,
            'construction': 2.0,
            'pot_holes': 1.5,
            'road_conditions': 1.5,
            'congestion': 1.0,
            'public_event': 2.5,
            'procession': 2.5,
            'vip_movement': 3.0,
            'protest': 3.5,
            'Debris': 2.0,
            'debris': 2.0,
            'others': 1.5
        }
        df['cause_severity_weight'] = df['event_cause'].map(cause_weight_map).fillna(1.5)
        
        # Text-based features from description (e.g. length of description, key traffic indicators)
        df['description_len'] = df['description'].fillna('').apply(len)
        df['desc_has_slow'] = df['description'].fillna('').str.lower().apply(lambda x: 1 if 'slow' in x or 'traffic' in x or 'movement' in x else 0)
        df['desc_has_blocked'] = df['description'].fillna('').str.lower().apply(lambda x: 1 if 'block' in x or 'closed' in x or 'stop' in x else 0)
        
        # Estimated Impact Radius (planned public events have larger impact radius than simple breakdowns)
        df['estimated_impact_radius_km'] = df.apply(
            lambda row: 2.0 if row['event_type'] == 'planned' 
            else (1.0 if row['event_cause'] in ['water_logging', 'protest', 'accident', 'tree_fall'] else 0.3), axis=1
        )
        
        # 6. Categorical Variable Encoding
        cat_cols = ['event_cause', 'corridor', 'priority', 'veh_type', 'police_station', 'zone', 'junction']
        for col in cat_cols:
            le = LabelEncoder()
            # Convert to string to avoid mixed type issues
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        return df
        
    def get_features_and_targets(self, df):
        feature_cols = [
            'latitude', 'longitude', 'endlatitude', 'endlongitude',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'month',
            'is_planned', 'requires_road_closure_encoded', 'vehicle_blockage_weight', 'cause_severity_weight',
            'description_len', 'desc_has_slow', 'desc_has_blocked', 'estimated_impact_radius_km',
            'event_cause_encoded', 'corridor_encoded', 'priority_encoded', 'veh_type_encoded', 
            'police_station_encoded', 'zone_encoded', 'junction_encoded'
        ]
        
        X = df[feature_cols]
        # Targets:
        # 1. Event duration (Regression)
        y_duration = df['closed_duration_mins']
        # 2. Priority High vs Low (Classification)
        y_priority = df['priority_encoded'] # Encoded priority
        # 3. Requires Road Closure (Classification)
        y_closure = df['requires_road_closure_encoded']
        
        return X, y_duration, y_priority, y_closure

if __name__ == '__main__':
    # Test DataProcessor
    file_path = r"C:\Users\tkurm\Downloads\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    processor = AstraDataProcessor(file_path)
    df = processor.load_data()
    processed_df = processor.preprocess(df)
    X, y_dur, y_prio, y_close = processor.get_features_and_targets(processed_df)
    print("Preprocessed features shape:", X.shape)
    print("Features sample:")
    print(X.head(2).to_string())
    print("Targets description:")
    print(y_dur.describe())
