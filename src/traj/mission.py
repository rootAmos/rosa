import numpy as np
from traj.hvr.hover_perfo import analyze_hover_performance
from traj.crz.crz_perfo import CruiseSegment

class MissionSegment:
    def __init__(self, MTOM_kg, density_kg_m3, diameter_m, airspeeds_m_s, aero_filenames, num_lift_prplrs=8, num_fwd_prplrs=4):
        """
        Initialize mission segment with velocity profile.
        
        Args:
            MTOM_kg: Maximum takeoff mass (kg)
            density_kg_m3: Air density (kg/m³)
            diameter_m: Propeller diameter (m)
            airspeeds_m_s: Array of airspeeds (m/s)
            aero_filenames: List of aero data files
            num_lift_prplrs: Number of lift propellers
            num_fwd_prplrs: Number of forward propellers
        """
        self.MTOM_kg = MTOM_kg
        self.density_kg_m3 = density_kg_m3
        self.diameter_m = diameter_m
        self.airspeeds_m_s = np.array(airspeeds_m_s)
        self.aero_filenames = aero_filenames
        self.num_lift_prplrs = num_lift_prplrs
        self.num_fwd_prplrs = num_fwd_prplrs

    def analyze(self):
        """Analyze complete mission segment."""
        results = []
        
        for v in self.airspeeds_m_s:
            if v < 1.0:  # Hover condition (near-zero forward speed)
                segment_results = analyze_hover_performance(
                    self.MTOM_kg, 
                    self.density_kg_m3, 
                    self.diameter_m
                )
                segment_results['segment_type'] = 'hover'
                segment_results['airspeed_m_s'] = v
                
            else:  # Forward flight
                crz = CruiseSegment(
                    self.MTOM_kg,
                    self.density_kg_m3,
                    self.diameter_m,
                    v,
                    self.aero_filenames,
                    self.num_lift_prplrs,
                    self.num_fwd_prplrs
                )
                segment_results = crz.analyze()
                segment_results['segment_type'] = 'cruise'
                segment_results['airspeed_m_s'] = v
                
            results.append(segment_results)
            
        return results

if __name__ == "__main__":
    # Example usage
    MTOM_kg = 5500
    density_kg_m3 = 1.225
    diameter_m = 2.7
    
    # Define velocity profile (hover -> transition -> cruise -> hover)
    airspeeds = [0, 15, 30, 60, 60, 30, 15, 0]
    
    
    # Get aero data files
    import glob
    filenames = glob.glob("data/aero/*.txt")
    
    # Create and analyze mission
    mission = MissionSegment(MTOM_kg, density_kg_m3, diameter_m, airspeeds, filenames)
    results = mission.analyze()
    
    # Print results
    print("\nMission Analysis Results:")
    print("-" * 50)
    for i, res in enumerate(results):
        print(f"\nSegment {i+1} ({res['segment_type']}):")
        print(f"Airspeed: {res['airspeed_m_s']:.1f} m/s")
        if res['segment_type'] == 'hover':
            print(f"Thrust per motor: {res['thrust_per_motor_N']:.1f} N")
            print(f"Power per motor: {res['power_per_motor_W']/1000:.1f} kW")
            print(f"Power elec per motor: {res['power_elec_per_motor_W']/1000:.1f} kW")
        else:
            print(f"Total thrust: {res['thrust_N'][0]:.1f} N")
            print(f"Power per motor: {res['power_W'][0]/1000:.1f} kW")
            print(f"Power elec per motor: {res['power_elec_W'][0]/1000:.1f} kW")
        print(f"Operating RPM: {res['operating_rpm'][0]:.1f}")
        print(f"Motor efficiency: {res['eta_motor']:.3f}")