import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from tools.lift_prop_import import EmpiricalPropeller
from tools.emotor_import_rbf import EmpiricalMotor

def analyze_hover_performance(MTOM_kg, density_kgm3, diameter_m):
    """
    Analyze hover performance for 8-motor configuration.
    
    Args:
        MTOM_kg: Maximum takeoff mass (kg)
        density_kgm3: Air density (kg/m³)
        diameter_m: Propeller diameter (m)
    """
    # Load propeller and motor data
    prop = EmpiricalPropeller()

    # Calculate required thrust per motor (weight / 8)
    weight_N = MTOM_kg * 9.806
    thrust_per_motor_N = weight_N / 8

    power_per_motor_W, rpm, coll, error, mach_tip = prop.calculate_power(thrust_per_motor_N, density_kgm3, diameter_m)
    
    torque_Nm = power_per_motor_W / (rpm * 2 * np.pi / 60)
    
    motor = EmpiricalMotor()    
    interp_func = motor.create_efficiency_interpolator(kernel='thin_plate_spline')

    eta_motor = interp_func([[rpm, torque_Nm]])[0]

    p_elec_motor_W = power_per_motor_W / eta_motor
    
    return {
        'thrust_per_motor_N': thrust_per_motor_N,
        'power_per_motor_W': power_per_motor_W,
        'power_elec_per_motor_W': p_elec_motor_W,
        'operating_rpm': rpm,
        'collective_deg': coll,
        'eta_motor': eta_motor,
        'mach_tip': mach_tip
    }

if __name__ == "__main__":
    # Example usage
    MTOM_kg = 5500  # Example MTOM
    density_kgm3 = 1.225
    diameter_m = 2.7
    
    results = analyze_hover_performance(MTOM_kg, density_kgm3, diameter_m)
    
    print("\nHover Performance Results:")
    print(f"Thrust per motor: {results['thrust_per_motor_N']:.1f} N")
    print(f"Power per motor: {results['power_per_motor_W']/1000:.1f} kW")
    print(f"Power elec per motor: {results['power_elec_per_motor_W']/1000:.1f} kW")
    print(f"Operating RPM: {results['operating_rpm']:.1f}")
    print(f"Collective pitch: {results['collective_deg']:.1f}°")
    print(f"Motor efficiency: {results['eta_motor']:.1f}")
    print(f"Mach tip: {results['mach_tip']:.3f}")
