import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import Rbf
from itertools import product

class EmpiricalPropeller:
    def __init__(self, vehicle, data_dir='data/prop'):
        """Load propeller performance data."""
        from scipy.interpolate import Rbf
        
        self.data = pd.read_csv(f'{data_dir}/0_15-Scale_JVX_Three-Bladed_Proprotor.csv')
        self.ref_diam_ft = 68.2/12
        
        # Store original points and values
        rpm = self.data['Vtip'] / ((self.ref_diam_ft/2) * 2* np.pi / 60)
        coll = self.data['Coll']
        
        # Create RBF interpolators
        self.CT_interp = Rbf(rpm, coll, self.data['CT'], function='cubic')
        self.CP_interp = Rbf(rpm, coll, self.data['CP'], function='cubic')
        
        self.rpm_min = vehicle['lift_prplsr_rpm_min']
        self.rpm_max = vehicle['lift_prplsr_rpm_max']
        self.coll_min = vehicle['lift_prplsr_beta_min']
        self.coll_max = vehicle['lift_prplsr_beta_max']
        self.N_sweep = 1000

    def calculate_power(self, phase, vehicle):
        """Calculate power required for given thrust in hover."""
        # Convert to imperial units
        thrust_req_lbf = phase['thrust_unit_N'] * 0.224809
        density_slugft3 = phase['density_kgm3'] * 0.00194032
        diam_ft = vehicle['lift_prplsr_diam_m'] * 3.28084
        radius_ft = diam_ft/2

        # Create sweep points
        rpm_sweep = np.linspace(self.rpm_min, self.rpm_max, self.N_sweep)[:, np.newaxis]
        coll_sweep = np.linspace(-7, 21, self.N_sweep)[:, np.newaxis]

        # Create all combinations using product
        combinations = np.array(list(product(rpm_sweep, coll_sweep)))
        rpm_all = combinations[:, 0]
        coll_all = combinations[:, 1]
        # Pass to interpolator
        CT = self.CT_interp(rpm_all, coll_all)
        
        # Calculate thrust matrix
        vtip_ft_s = rpm_all * radius_ft * np.pi / 60 * 2
        thrust_calc_lbf = CT * density_slugft3 * vtip_ft_s**2 * radius_ft**2 * np.pi
        
        # Find best match
        error = abs(thrust_calc_lbf - thrust_req_lbf)
        idx = np.argmin(error, axis=0)

        min_errors = error[idx, np.arange(phase['N'])]
        min_errors_rel = min_errors / thrust_req_lbf
        if np.any(min_errors_rel > 0.05):
            print("Warning: Large errors in power calculation")
        # end

        rpm = rpm_all[idx].flatten()
        coll = coll_all[idx].flatten()
        
        # Get final CP (also needs column_stack)
        CP = self.CP_interp(rpm, coll)
        
        # Calculate power
        vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
        power_lbfts = CP * density_slugft3 * vtip_ft_s**3 * radius_ft**2 * np.pi
        power_W = power_lbfts * 1.35582

        return power_W, rpm, coll

    def calculate_thrust(self, phase, vehicle):
        """Calculate thrust available for given power in hover."""
        from itertools import product
        
        # Convert to imperial units
        power_lbfts = phase['power_W'] * 0.737562
        density_slugft3 = phase['density_kgm3'] * 0.00194032
        diam_ft = vehicle['lift_prplsr_diam_m'] * 3.28084
        radius_ft = diam_ft/2

        # Create sweep points
        rpm_sweep = np.linspace(self.rpm_min, self.rpm_max, self.N_sweep)[:, np.newaxis]
        coll_sweep = np.linspace(-7, 21, self.N_sweep)[:, np.newaxis]
        
        # Create all combinations using product
        combinations = np.array(list(product(rpm_sweep, coll_sweep)))
        rpm_all = combinations[:, 0]
        coll_all = combinations[:, 1]
        
        # Get CP for all combinations
        CP = self.CP_interp(rpm_all, coll_all)
        
        # Calculate power vector
        vtip_ft_s = rpm_all * radius_ft * np.pi / 60 * 2
        power_calc_lbfts = CP * density_slugft3 * vtip_ft_s**3 * radius_ft**2 * np.pi
        
        # Find best match
        error = abs(power_calc_lbfts - power_lbfts)
        idx = np.argmin(error, axis=0)

        min_errors = error[idx, np.arange(phase['N'])]
        min_errors_rel = min_errors / power_lbfts
        if np.any(min_errors_rel > 0.05):
            print("Warning: Large errors in thrust calculation")
        # end

        rpm = rpm_all[idx].flatten()
        coll = coll_all[idx].flatten()
        
        # Get final CT
        CT = self.CT_interp(rpm, coll)
        
        # Calculate thrust
        vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
        thrust_lbf = CT * density_slugft3 * vtip_ft_s**2 * radius_ft**2 * np.pi
        thrust_N = thrust_lbf * 4.44822
        
        return thrust_N, rpm, coll

if __name__ == "__main__":

    vehicle = {}
    vehicle['lift_prplsr_rpm_min'] = 2200
    vehicle['lift_prplsr_rpm_max'] = 2700
    vehicle['lift_prplsr_beta_min'] = -7
    vehicle['lift_prplsr_beta_max'] = 21
    vehicle['lift_prplsr_diam_m'] = 2.7


    phase = {}
    phase['N_phase'] = 10
    phase['thrust_unit_N'] = 5500*(1.1*9.806)/8  * np.ones(phase['N_phase'])
    phase['density_kgm3'] = 1.225 * np.ones(phase['N_phase'])

    # Forward calculation

    prop = EmpiricalPropeller(vehicle)

    power_W, rpm_lift, coll = prop.calculate_power(phase,vehicle)
    
    np.set_printoptions(precision=3, suppress=True)

    print("\nForward calculation:")
    print(f"Input thrust: {phase['thrust_unit_N']} N")
    print(f"Required power: {power_W/1000} kW")
    print(f"Required RPM: {rpm_lift}")
    print(f"Collective angle: {coll} degrees")

    phase['power_W'] = power_W


    # Reverse calculation
    thrust_unit_N, rpm_rev, coll = prop.calculate_thrust(phase, vehicle)
    
    print("\nReverse calculation:")
    print(f"Input power: {phase['power_W']/1000} kW")
    print(f"Available thrust: {thrust_unit_N} N")
    print(f"Required RPM: {rpm_rev}")
    print(f"Collective angle: {coll} degrees")

