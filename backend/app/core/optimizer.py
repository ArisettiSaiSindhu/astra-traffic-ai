import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

class AstraOptimizer:
    def __init__(self):
        pass
        
    def optimize_police_allocation(self, incidents: list, total_officers: int) -> list:
        """
        Allocates total_officers across a list of incidents/junctions to maximize mitigation impact.
        MILP formulation:
        Maximize Sum(Incident_Severity_i * Allocated_Officers_i * Intersection_Weight_i)
        Subject to:
        - Sum(Allocated_Officers_i) <= total_officers
        - Allocated_Officers_i >= Min_Required_i
        - Allocated_Officers_i <= Max_Limit_i
        
        incidents is a list of dicts:
        [
            {
                'id': 'FKID000000',
                'severity_score': 7.5, # congestion score or priority weight
                'junction': 'Jalahalli Cross Junction',
                'min_officers': 1,
                'max_officers': 5
            },
            ...
        ]
        """
        if not incidents or total_officers <= 0:
            return []
            
        n = len(incidents)
        
        # Coefficients for objective function (we want to maximize, so we negate for minimization in scipy.optimize.milp)
        # c represents the coefficients of the linear objective function: c^T * x
        c = []
        for inc in incidents:
            weight = 2.0 if inc.get('priority') == 'High' else 1.0
            # Objective coefficient = - (severity * priority_weight)
            c.append(-float(inc.get('severity_score', 5.0) * weight))
        c = np.array(c)
        
        # Bounds on variables: min_officers <= x_i <= max_officers
        lower_bounds = np.array([int(inc.get('min_officers', 1)) for inc in incidents])
        upper_bounds = np.array([int(inc.get('max_officers', 4)) for inc in incidents])
        
        # If the minimum officers required exceeds total available, we scale down the minimums
        if sum(lower_bounds) > total_officers:
            # Fallback: distribute officers proportionally
            allocated = np.zeros(n, dtype=int)
            remaining = total_officers
            severities = np.array([float(inc.get('severity_score', 5.0)) for inc in incidents])
            ranks = np.argsort(-severities)
            
            # Allocate at least 1 to high-severity until out of officers
            for idx in ranks:
                if remaining > 0:
                    allocated[idx] = 1
                    remaining -= 1
            return [{**incidents[i], 'allocated_officers': int(allocated[i])} for i in range(n)]
            
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Constraint: Sum(x_i) <= total_officers
        # A_eq * x <= b_eq
        A = np.ones((1, n))
        constraint = LinearConstraint(A, lb=0, ub=total_officers)
        
        # Variable types: integers (1 for all variables)
        integrality = np.ones(n, dtype=int)
        
        # Run MILP
        res = milp(c=c, bounds=bounds, constraints=constraint, integrality=integrality)
        
        allocated = np.zeros(n, dtype=int)
        if res.success:
            allocated = np.round(res.x).astype(int)
        else:
            # Fallback greedy allocation
            allocated = lower_bounds.copy()
            remaining = total_officers - sum(allocated)
            severities = np.array([float(inc.get('severity_score', 5.0)) for inc in incidents])
            
            # Greedy allocation of remaining officers to highest severity spots
            ranks = np.argsort(-severities)
            while remaining > 0:
                allocated_any = False
                for idx in ranks:
                    if allocated[idx] < upper_bounds[idx] and remaining > 0:
                        allocated[idx] += 1
                        remaining -= 1
                        allocated_any = True
                if not allocated_any:
                    break
                    
        result = []
        for i in range(n):
            result.append({
                'id': incidents[i].get('id'),
                'junction': incidents[i].get('junction'),
                'corridor': incidents[i].get('corridor'),
                'severity_score': float(incidents[i].get('severity_score', 5.0)),
                'min_officers': int(lower_bounds[i]),
                'max_officers': int(upper_bounds[i]),
                'allocated_officers': int(allocated[i])
            })
            
        return result

    def calculate_signal_adjustments(self, junction_name: str, predicted_duration: float, priority_prio_proba: float) -> dict:
        """
        Calculates signal cycle adjustments based on Webster's delay model.
        Returns suggested cycle length and green light offset adjustments.
        """
        base_cycle = 90  # standard 90-second cycle
        
        # Priority and duration determine the level of signal preemption required
        if priority_prio_proba > 0.7:
            # Major event/breakdown -> trigger cycle extension for the critical approach
            adjustment = min(30, int(30 * priority_prio_proba))
            recommended_cycle = base_cycle + adjustment
            phase_allocation = {
                'critical_phase_green_secs': 45 + adjustment,
                'secondary_phase_green_secs': 30 - int(adjustment * 0.3),
                'pedestrian_crossing_secs': 15
            }
            action = "EXTEND_CRITICAL_PHASE"
        elif predicted_duration > 120:
            # Long duration incident -> extend cycle slightly to prevent queue spillback
            recommended_cycle = base_cycle + 15
            phase_allocation = {
                'critical_phase_green_secs': 50,
                'secondary_phase_green_secs': 28,
                'pedestrian_crossing_secs': 17
            }
            action = "INCREASE_CYCLE_LENGTH"
        else:
            recommended_cycle = base_cycle
            phase_allocation = {
                'critical_phase_green_secs': 40,
                'secondary_phase_green_secs': 35,
                'pedestrian_crossing_secs': 15
            }
            action = "MAINTAIN_NORMAL_CYCLE"
            
        return {
            'junction': junction_name,
            'recommended_cycle_seconds': recommended_cycle,
            'phase_allocations': phase_allocation,
            'mitigation_strategy': action
        }

    def recommend_diversion_routes(self, incident_location: dict, entry_points: list, exit_points: list) -> list:
        """
        Generates simulated diversion routing options bypassing the incident location.
        Since full spatial graph computation is done via NetworkX/OSM on the backend,
        this provides a lightweight routing model based on euclidean node offsets with congestion weights.
        """
        diversions = []
        
        inc_lat = incident_location['latitude']
        inc_lon = incident_location['longitude']
        
        for idx, entry in enumerate(entry_points):
            entry_lat = entry['latitude']
            entry_lon = entry['longitude']
            
            # Standard path to exit bypassing the incident center
            # We calculate a checkpoint to route around the incident
            # Simple vector rotation of 90 degrees to bypass the bottleneck
            vector_y = inc_lat - entry_lat
            vector_x = inc_lon - entry_lon
            
            # Offset checkpoint 500 meters to the side
            bypass_lat = inc_lat + (-vector_x * 0.005)
            bypass_lon = inc_lon + (vector_y * 0.005)
            
            for jdx, ex in enumerate(exit_points):
                exit_lat = ex['latitude']
                exit_lon = ex['longitude']
                
                # Check travel time difference
                # Congested path runs through the incident
                dist_direct = np.sqrt((entry_lat - inc_lat)**2 + (entry_lon - inc_lon)**2) + \
                              np.sqrt((inc_lat - exit_lat)**2 + (inc_lon - exit_lon)**2)
                              
                dist_bypass = np.sqrt((entry_lat - bypass_lat)**2 + (entry_lon - bypass_lon)**2) + \
                              np.sqrt((bypass_lat - exit_lat)**2 + (bypass_lon - exit_lon)**2)
                
                # Travel times: direct path is congested (speed = 5 km/h), bypass path is free-flow (speed = 30 km/h)
                time_direct_mins = (dist_direct * 111.0) / (5.0 / 60.0)  # 1 degree approx 111km
                time_bypass_mins = (dist_bypass * 111.0) / (25.0 / 60.0)
                
                delay_saved = max(0, time_direct_mins - time_bypass_mins)
                
                diversions.append({
                    'id': f"DIV_{idx}_{jdx}",
                    'entry_point': entry['name'],
                    'exit_point': ex['name'],
                    'direct_route_est_time_mins': float(round(time_direct_mins, 1)),
                    'diversion_route_est_time_mins': float(round(time_bypass_mins, 1)),
                    'estimated_delay_saved_mins': float(round(delay_saved, 1)),
                    'bypass_nodes': [
                        {'latitude': entry_lat, 'longitude': entry_lon},
                        {'latitude': bypass_lat, 'longitude': bypass_lon},
                        {'latitude': exit_lat, 'longitude': exit_lon}
                    ]
                })
                
        # Sort by delay saved descending
        diversions.sort(key=lambda x: x['estimated_delay_saved_mins'], reverse=True)
        return diversions

    def recommend_barricades(self, incident: dict) -> list:
        """
        Recommends specific barricade layouts based on incident properties.
        """
        cause = incident.get('event_cause')
        lat = incident.get('latitude') or 12.971598
        lon = incident.get('longitude') or 77.594562
        
        barricades = []

        
        # Check if closure is required
        if incident.get('requires_road_closure') or incident.get('priority') == 'High':
            # Needs outer containment rings to divert traffic early
            barricades.append({
                'type': 'CONTAINMENT_BARRICADE',
                'distance_meters': 200,
                'count': 6,
                'location': {'latitude': lat + 0.0018, 'longitude': lon},
                'direction': 'INCOMING',
                'description': 'Stop traffic before junction entry. Display DIVERSION sign.'
            })
            barricades.append({
                'type': 'BUFFER_BARRICADE',
                'distance_meters': 50,
                'count': 4,
                'location': {'latitude': lat - 0.0005, 'longitude': lon + 0.0005},
                'direction': 'CROSS_FLOW',
                'description': 'Protect response vehicles and active cleanup crew.'
            })
        else:
            # Partial lane closure only
            barricades.append({
                'type': 'LANE_MERGE_BARRICADE',
                'distance_meters': 30,
                'count': 3,
                'location': {'latitude': lat - 0.0003, 'longitude': lon - 0.0003},
                'direction': 'ALONG_FLOW',
                'description': 'Taper traffic into right lane. Place warning cone 50m upstream.'
            })
            
        return barricades

if __name__ == '__main__':
    # Test optimization
    optimizer = AstraOptimizer()
    
    test_incidents = [
        {'id': 'FKID001', 'junction': 'Jalahalli Cross', 'priority': 'High', 'severity_score': 8.2, 'min_officers': 1, 'max_officers': 4},
        {'id': 'FKID002', 'junction': 'Bashyam Circle', 'priority': 'Low', 'severity_score': 3.5, 'min_officers': 1, 'max_officers': 2},
        {'id': 'FKID003', 'junction': 'Agara Junction', 'priority': 'High', 'severity_score': 9.0, 'min_officers': 2, 'max_officers': 5},
    ]
    
    allocated = optimizer.optimize_police_allocation(test_incidents, total_officers=6)
    print("MILP Allocated Police Officers:")
    for a in allocated:
        print(f"Junction: {a['junction']}, Severity: {a['severity_score']}, Allocated: {a['allocated_officers']}")
        
    print("\nSignal Adjustments Test:")
    print(optimizer.calculate_signal_adjustments("Agara Junction", 180.0, 0.95))
