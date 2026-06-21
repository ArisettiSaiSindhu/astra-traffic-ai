import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional


from app.core.data_processor import AstraDataProcessor
from app.core.predictor import AstraPredictor
from app.core.optimizer import AstraOptimizer
from app.core.simulator import AstraSimulator
from app.core.copilot import AstraCopilot

app = FastAPI(title="AstraTraffic AI Backend", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
DATA_PATH = r"C:\Users\tkurm\Downloads\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
predictor = AstraPredictor()
optimizer = AstraOptimizer()
simulator = AstraSimulator()
copilot = AstraCopilot()

# Load models on startup
@app.on_event("startup")
def startup_event():
    try:
        predictor.load_models()
    except Exception as e:
        print(f"Could not load models, training fresh... {e}")
        try:
            predictor.train_models(DATA_PATH)
        except Exception as ex:
            print(f"Error training models: {ex}")

# Pydantic Schemas
class IncidentInput(BaseModel):
    latitude: float
    longitude: float
    event_type: str
    event_cause: str
    requires_road_closure: bool
    start_datetime: str
    description: Optional[str] = ""
    veh_type: Optional[str] = "unknown_vehicle"
    corridor: Optional[str] = "Non-corridor"
    priority: Optional[str] = "Low"
    police_station: Optional[str] = "unknown_station"
    zone: Optional[str] = "unknown_zone"
    junction: Optional[str] = "unknown_junction"

class IncidentOptimizeInput(BaseModel):
    id: str
    severity_score: float
    priority: str
    junction: str
    corridor: str
    latitude: Optional[float] = 12.971598
    longitude: Optional[float] = 77.594562
    min_officers: Optional[int] = 1
    max_officers: Optional[int] = 4


class OptimizeRequest(BaseModel):
    incidents: List[IncidentOptimizeInput]
    total_officers: int

class SimulateRequest(BaseModel):
    event_location: dict  # {'latitude': float, 'longitude': float}
    baseline_incidents: List[dict]
    attendance_multiplier: float
    is_raining: bool
    closed_segments: List[str]

class CopilotRequest(BaseModel):
    query: str
    context: dict

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to AstraTraffic AI Backend API"}

@app.post("/api/predict")
def predict_congestion(payload: IncidentInput):
    try:
        results = predictor.predict_incident(payload.dict())
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/optimize")
def optimize_resources(payload: OptimizeRequest):
    try:
        incidents_list = [inc.dict() for inc in payload.incidents]
        allocations = optimizer.optimize_police_allocation(incidents_list, payload.total_officers)
        
        # Calculate signal adjustments and barricades for the highest priority spot
        signal_adjustments = []
        barricades = []
        if incidents_list:
            highest_severity_inc = max(incidents_list, key=lambda x: x['severity_score'])
            # Mock predicted duration and priority probability based on score for recommendation details
            signal_adjustments = optimizer.calculate_signal_adjustments(
                highest_severity_inc['junction'], 
                predicted_duration=120.0, 
                priority_prio_proba=0.85 if highest_severity_inc['priority'] == 'High' else 0.4
            )
            barricades = optimizer.recommend_barricades(highest_severity_inc)
            
        return {
            "allocations": allocations,
            "signal_strategy": signal_adjustments,
            "barricade_strategy": barricades
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
def run_simulation_scenario(payload: SimulateRequest):
    try:
        results = simulator.run_simulation(
            payload.event_location,
            payload.baseline_incidents,
            payload.attendance_multiplier,
            payload.is_raining,
            payload.closed_segments
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/copilot")
def chat_copilot(payload: CopilotRequest):
    try:
        response = copilot.chat_with_copilot(payload.query, payload.context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/incidents")
def get_sample_incidents():
    """
    Returns a sample of real-world active and closed incidents from the dataset
    to populate the frontend dashboard.
    """
    try:
        processor = AstraDataProcessor(DATA_PATH)
        df = processor.load_data()
        
        # Sample 10 incidents, clean missing data and parse columns
        sample_df = df.sample(10, random_state=42).copy()
        
        # Fill standard missing values for serialization
        sample_df['description'] = sample_df['description'].fillna('No description provided')
        sample_df['corridor'] = sample_df['corridor'].fillna('Non-corridor')
        sample_df['priority'] = sample_df['priority'].fillna('Low')
        sample_df['veh_type'] = sample_df['veh_type'].fillna('others')
        sample_df['police_station'] = sample_df['police_station'].fillna('City Station')
        sample_df['junction'] = sample_df['junction'].fillna('City Intersection')
        sample_df['event_cause'] = sample_df['event_cause'].fillna('others')
        
        results = []
        for idx, row in sample_df.iterrows():
            results.append({
                "id": str(row['id']),
                "event_type": str(row['event_type']),
                "event_cause": str(row['event_cause']),
                "latitude": float(row['latitude']),
                "longitude": float(row['longitude']),
                "requires_road_closure": bool(row['requires_road_closure']),
                "start_datetime": str(row['start_datetime']),
                "description": str(row['description']),
                "veh_type": str(row['veh_type']),
                "corridor": str(row['corridor']),
                "priority": str(row['priority']),
                "police_station": str(row['police_station']),
                "junction": str(row['junction']),
                "severity_score": float(9.2 if row['priority'] == 'High' else 4.5) # calculated mock severity for dashboard
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
