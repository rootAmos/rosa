import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from tools.emotor_import_rbf import EmpiricalMotor
from tools.dfan_import import EmpiricalDfan
from tools.flow5_import import EmpiricalAero

import glob
import pdb

class CruiseSegment:
    def __init__(self, MTOM_kg, density_kg_m3, diameter_m, airspeed_m_s,aero_filenames, num_lift_prplrs,num_fwd_prplrs):
        self.MTOM_kg = MTOM_kg
        self.density_kg_m3 = density_kg_m3
        self.diameter_m = diameter_m
        self.airspeed_m_s = airspeed_m_s
        self.aero_filenames = aero_filenames
        self.num_lift_prplrs = num_lift_prplrs
        self.num_fwd_prplrs = num_fwd_prplrs

    def analyze(self):
        """
        Analyze hover performance for 8-motor configuration.
        
        Args:
            MTOM_kg: Maximum takeoff mass (kg)
            density_kgm3: Air density (kg/m³)
            diameter_m: Propeller diameter (m)
        """
        # Load propeller and motor data
        dfan = EmpiricalDfan()

        # Unpack Segment
        MTOM_kg = self.MTOM_kg
        density_kg_m3 = self.density_kg_m3
        diameter_m = self.diameter_m
        airspeed_m_s = self.airspeed_m_s
        aero_filenames = self.aero_filenames
        num_fwd_prplrs = self.num_fwd_prplrs

        # Calculate CL required
        CL = (MTOM_kg * 9.806) / (0.5 * density_kg_m3 * airspeed_m_s**2 * np.pi * diameter_m**2) 

        # Determine angle of attack
        aero = EmpiricalAero(aero_filenames)

        alpha_from_CL = aero.interp_alpha_from_CL()

        alpha = alpha_from_CL([airspeed_m_s, CL])[0]

        CD = aero.interp_CD([airspeed_m_s, alpha])

        drag_N = 0.5 * density_kg_m3 * airspeed_m_s**2 * np.pi * diameter_m**2 * CD

        thrust_N = drag_N

        unit_thrust_N = thrust_N / num_fwd_prplrs


        power_W, rpm = dfan.calculate_power(unit_thrust_N, airspeed_m_s, density_kg_m3, diameter_m)

        torque_Nm = power_W / (rpm * 2 * np.pi / 60)

        motor = EmpiricalMotor()    
        interp_func = motor.create_efficiency_interpolator(kernel='thin_plate_spline')

        eta_motor = interp_func([[rpm, torque_Nm]])[0]

        p_elec_motor_W = power_W / eta_motor
        
        return {
            'thrust_N': thrust_N,
            'power_W': power_W,
            'power_elec_W': p_elec_motor_W,
            'operating_rpm': rpm,
            'eta_motor': eta_motor,
        }

if __name__ == "__main__":

    # Example usage
    MTOM_kg = 5500  # Example MTOM
    density_kgm3 = 1.225
    diameter_m = 2.7
    airspeed_m_s = 60
    
    # Get all txt files in data folder
    filenames = glob.glob("data/aero/*.txt")
    crz = CruiseSegment(MTOM_kg, density_kgm3, diameter_m, airspeed_m_s, filenames, 4, 4)
    
    results = crz.analyze()
    
    print("\nHover Performance Results:")
    print(f"Thrust per motor: {results['thrust_N'][0]:.1f} N")
    print(f"Power per motor: {results['power_W'][0]/1000:.1f} kW")
    print(f"Power elec per motor: {results['power_elec_W'][0]/1000:.1f} kW")
    print(f"Operating RPM: {results['operating_rpm'][0]:.1f}")
    print(f"Motor efficiency: {results['eta_motor']:.3f}")
