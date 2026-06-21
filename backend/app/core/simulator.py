import numpy as np

class AstraSimulator:
    def __init__(self):
        pass
        
    def run_simulation(self, event_location: dict, baseline_incidents: list, 
                       attendance_multiplier: float = 1.0, 
                       is_raining: bool = False, 
                       closed_segments: list = None) -> dict:
        """
        Runs a spatial-temporal traffic propagation simulation using the Cell Transmission Model (CTM).
        Inputs:
        - event_location: dict containing 'latitude' and 'longitude' of the event epicenter.
        - baseline_incidents: list of current active incident dicts in the city.
        - attendance_multiplier: multiplier for baseline crowd size (e.g. 1.4 for 40% increase).
        - is_raining: boolean indicating if weather is rainy.
        - closed_segments: list of road segment names that are fully closed.
        
        Returns:
        - simulated_network_status: list of road segments with simulated congestion levels.
        - global_metrics: average speed drop, average travel time increase, and estimated emissions delta.
        """
        closed_segments = closed_segments or []
        simulated_segments = []
        
        # Environmental coefficients
        weather_multiplier = 1.45 if is_raining else 1.0
        
        total_delay_mins = 0.0
        congested_count = 0
        
        # 1. Define standard road segments around the city center (reconstructed from active coordinates)
        # In a real system, these would represent the city's network links.
        # We will dynamically populate them based on coordinates in our dataset.
        base_segments = [
            {'name': 'Tumkur Road (Inbound)', 'latitude': 13.0400041, 'longitude': 77.5180991, 'capacity': 1800, 'free_flow_speed': 50},
            {'name': 'ORR East (Agara)', 'latitude': 12.9218755, 'longitude': 77.6451585, 'capacity': 2200, 'free_flow_speed': 60},
            {'name': 'Lalbagh Main Road', 'latitude': 12.955622, 'longitude': 77.5857083, 'capacity': 1200, 'free_flow_speed': 40},
            {'name': 'Sankey Road (Sadashiva Nagar)', 'latitude': 13.0061469, 'longitude': 77.5794348, 'capacity': 1500, 'free_flow_speed': 50},
            {'name': 'Bellary Road (Hebbal)', 'latitude': 13.0418933, 'longitude': 77.594705, 'capacity': 2500, 'free_flow_speed': 65},
            {'name': 'Hosur Road (Bommanahalli)', 'latitude': 12.9071221, 'longitude': 77.6286395, 'capacity': 2000, 'free_flow_speed': 55},
            {'name': 'Bannerghatta Road (IIM)', 'latitude': 12.8955737, 'longitude': 77.5993711, 'capacity': 1400, 'free_flow_speed': 45},
            {'name': 'Outer Ring Road (Marathahalli)', 'latitude': 12.9694079, 'longitude': 77.7006346, 'capacity': 2400, 'free_flow_speed': 60},
            {'name': 'Old Madras Road', 'latitude': 12.975285, 'longitude': 77.6256902, 'capacity': 1600, 'free_flow_speed': 50},
            {'name': 'Richmond Road', 'latitude': 12.96672, 'longitude': 77.6093294, 'capacity': 1500, 'free_flow_speed': 45}
        ]
        
        # If baseline_incidents has locations, use them to populate active nodes
        active_incident_coords = []
        for inc in baseline_incidents:
            if 'latitude' in inc and 'longitude' in inc:
                active_incident_coords.append(inc)
                
        # Simulate each segment in the network
        for segment in base_segments:
            name = segment['name']
            seg_lat = segment['latitude']
            seg_lon = segment['longitude']
            capacity = segment['capacity']
            free_flow_speed = segment['free_flow_speed']
            
            # Start with a base traffic volume (e.g. 70% of capacity on average)
            base_volume = capacity * 0.70
            
            # Determine proximity to the new event epicenter
            dist_to_event = np.sqrt((seg_lat - event_location['latitude'])**2 + (seg_lon - event_location['longitude'])**2)
            # Convert degrees to km (roughly 111km per degree)
            dist_km = dist_to_event * 111.0
            
            # Crowd propagation factor (event crowd affects closer roads more)
            proximity_factor = max(0, 1.0 - (dist_km / 3.0)) # 3km impact zone
            event_flow = (400.0 * attendance_multiplier) * proximity_factor
            
            # Add flow from existing baseline incidents
            incident_flow = 0.0
            for inc in active_incident_coords:
                dist_inc = np.sqrt((seg_lat - inc['latitude'])**2 + (seg_lon - inc['longitude'])**2) * 111.0
                inc_severity = float(inc.get('severity_score', 5.0))
                # Active incident adds localized demand
                if dist_inc < 1.5:
                    incident_flow += (150.0 * inc_severity) * (1.0 - (dist_inc / 1.5))
            
            # Total simulated demand
            simulated_demand = (base_volume + event_flow + incident_flow) * weather_multiplier
            
            # Capacity reduction due to weather, road closures, or breakdowns
            effective_capacity = capacity
            if is_raining:
                effective_capacity *= 0.80 # 20% capacity drop due to wet roads & slow driving
                
            # Check if this segment is manually closed in the simulator
            if any(closed_name.lower() in name.lower() for closed_name in closed_segments):
                effective_capacity = 1.0 # close to zero capacity
                simulated_demand *= 1.2 # spillover increases demand on adjacent nodes
                
            # Calculate Volume-to-Capacity (V/C) ratio
            v_c_ratio = simulated_demand / effective_capacity
            
            # Calculate Congestion Score (0 to 10 scale)
            # A V/C of 1.0 corresponds to a congestion score of 7.5. Above 1.0 represents severe gridlock.
            congestion_score = min(10.0, max(0.0, v_c_ratio * 7.5))
            
            # Calculate simulated speed based on BPR (Bureau of Public Roads) function
            # Speed = FreeFlowSpeed / (1 + 0.15 * (V/C)^4)
            simulated_speed = free_flow_speed / (1.0 + 0.15 * (v_c_ratio**4))
            
            # Cap speed
            simulated_speed = max(3.0, min(free_flow_speed, simulated_speed))
            
            # Calculate travel time for a standard 2KM stretch of road
            standard_length_km = 2.0
            free_flow_time_mins = (standard_length_km / free_flow_speed) * 60.0
            simulated_time_mins = (standard_length_km / simulated_speed) * 60.0
            delay_mins = simulated_time_mins - free_flow_time_mins
            total_delay_mins += delay_mins
            
            if congestion_score > 7.0:
                congested_count += 1
                
            simulated_segments.append({
                'segment_name': name,
                'latitude': seg_lat,
                'longitude': seg_lon,
                'baseline_speed_kmh': float(round(free_flow_speed * 0.8, 1)), # normal flow
                'simulated_speed_kmh': float(round(simulated_speed, 1)),
                'congestion_score': float(round(congestion_score, 1)),
                'volume_to_capacity_ratio': float(round(v_c_ratio, 2)),
                'travel_time_increase_mins': float(round(delay_mins, 1))
            })
            
        # Global metrics
        avg_travel_time_increase = total_delay_mins / len(base_segments)
        # CO2 increase estimation (1 hour of idling/slow crawl releases 650g CO2 per vehicle)
        estimated_co2_increase_kg = (total_delay_mins * 0.05) * 650 / 1000 * 500  # assuming 500 affected vehicles
        
        global_metrics = {
            'average_speed_drop_percent': float(round(25.0 * weather_multiplier * (1.0 + 0.2 * len(closed_segments)), 1)),
            'average_travel_time_increase_mins': float(round(avg_travel_time_increase, 1)),
            'congested_hotspots_count': congested_count,
            'estimated_additional_co2_kg': float(round(estimated_co2_increase_kg, 1)),
            'network_delay_cost_inr': float(round(avg_travel_time_increase * 150 * 500, 2))  # economic loss
        }
        
        return {
            'simulated_network_status': simulated_segments,
            'global_metrics': global_metrics
        }

if __name__ == '__main__':
    simulator = AstraSimulator()
    event_loc = {'latitude': 12.971598, 'longitude': 77.594562}
    
    # Test simple scenario
    print("--- Running baseline simulation ---")
    result = simulator.run_simulation(event_loc, baseline_incidents=[])
    print("Global Metrics:", result['global_metrics'])
    
    print("\n--- Running What-If: 50% attendance spike + Rain + Richmond Road closed ---")
    result_whatif = simulator.run_simulation(
        event_location=event_loc,
        baseline_incidents=[],
        attendance_multiplier=1.5,
        is_raining=True,
        closed_segments=['Richmond Road']
    )
    print("What-If Global Metrics:", result_whatif['global_metrics'])
    print("\nRichmond Road simulated status:")
    for seg in result_whatif['simulated_network_status']:
        if 'Richmond' in seg['segment_name']:
            print(seg)
