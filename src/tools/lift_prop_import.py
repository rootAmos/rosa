import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import RegularGridInterpolator

class EmpiricalPropeller:
    def __init__(self, data_dir='data/prop'):

        """
        Load propeller performance data.
        """

        self.data = pd.read_csv(f'{data_dir}/0_15-Scale_JVX_Three-Bladed_Proprotor.csv')
        self.ref_diam_ft = 68.2/12
        self.points = np.column_stack(((self.data['Vtip'] / ((self.ref_diam_ft/2) * 2* np.pi / 60)), self.data['Coll']))

    # Create interpolation functions using griddata
    def rpm_coll_to_CT(self, rpm_coll):
        return griddata(self.points, self.data['CT'], rpm_coll, method='linear')
        
    def rpm_coll_to_CP(self, rpm_coll):
        return griddata(self.points, self.data['CP'], rpm_coll, method='linear')

    def calculate_power(self, thrust_N, density_kgm3,diameter_m):
        """Calculate power required for given thrust in hover."""
        # Convert to imperial units
        thrust_req_lbf = thrust_N * 0.224809
        density_slugft3 = density_kgm3 * 0.00194032
        diameter_ft = diameter_m * 3.28084
        radius_ft = diameter_ft/2
        
        def rpm_coll_solver(x):
            rpm, coll = x
            vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
            
            # Get interpolated CT for these conditions
            CT = self.rpm_coll_to_CT(np.array([[rpm, coll]]))
            
            # Calculate actual thrust at these conditions
            thrust_calc_lbf = CT * density_slugft3 * vtip_ft_s**2 * radius_ft**2 * np.pi

            # Debug prints
            #print(f"\nRPM: {rpm:.1f}, Coll: {coll:.1f}")
            #print(f"Vtip: {vtip_ft_s:.1f}, CT: {CT}")
            #print(f"Thrust calc: {thrust_calc_lbf[0]:.1f} lbf")
            #print(f"Thrust req: {thrust_req_lbf:.1f} lbf")
            #print(f"Error: {abs(thrust_calc_lbf[0] - thrust_req_lbf):.1f}")
            
            return abs(thrust_calc_lbf - thrust_req_lbf)
        # end
        
        # Initial guess
        initial_guess = [2300, 10]  # rpm=1000, coll=10deg
        bounds = [(2100, 2700), (-7, 21)]  # rpm and coll bounds
        
        # Find RPM and collective that give required thrust
        from scipy.optimize import minimize
        result = minimize(rpm_coll_solver, initial_guess, bounds=bounds, method='SLSQP')
        rpm, coll = result.x
        error = result.fun
        
        # Get final CP from interpolation
        vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
        mach_tip = vtip_ft_s / 1125

        CP = self.rpm_coll_to_CP(np.array([[rpm, coll]]))
        
        # Calculate power using vtip
        power_lbfts = CP * density_slugft3 * vtip_ft_s**3 * radius_ft**2 * np.pi
        power_W = power_lbfts * 1.35582

        
        return power_W[0], rpm, coll, error, mach_tip

    def calculate_thrust(self, power_W, density_kgm3, diameter_m ):
        """Calculate thrust available for given power in hover."""
        # Convert to imperial units
        power_req_lbfts = power_W * 0.737562
        density_slugft3 = density_kgm3 * 0.00194032
        diameter_ft = diameter_m * 3.28084
        radius_ft = diameter_ft/2
        
        def rpm_coll_solver(x):
            rpm, coll = x
            vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2

            # Get interpolated CP for these conditions
            CP = self.rpm_coll_to_CP(np.array([[rpm, coll]]))
            
            # Calculate actual power at these conditions
            power_calc_lbfts = CP * density_slugft3 * vtip_ft_s**3 * radius_ft**2 * np.pi
            
            # Debug prints
            #print(f"\nRPM: {rpm:.1f}, Coll: {coll:.1f}")
            #print(f"Vtip: {vtip_ft_s:.1f}, CP: {CP}")
            #print(f"Power calc: {power_calc_lbfts[0]:.1f} lbf")
            #print(f"Power req: {power_req_lbfts:.1f} lbf")
            #print(f"Error: {abs(power_calc_lbfts[0] - power_req_lbfts):.1f}")
        
            return abs(power_calc_lbfts - power_req_lbfts)
        
        # Initial guess
        initial_guess = [2300, 10]  # rpm=1000, coll=10deg
        bounds = [(2100, 2700), (-7, 21)]  # rpm and coll bounds
        
        # Find RPM and collective that give required power
        from scipy.optimize import minimize
        result = minimize(rpm_coll_solver, initial_guess, bounds=bounds, method='SLSQP')
        rpm, coll = result.x
        error = result.fun
        
        # Get final CT from interpolation
        vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
        mach_tip = vtip_ft_s / 1125
        CT = self.rpm_coll_to_CT(np.array([[rpm, coll]]))
        
        # Calculate thrust using vtip
        thrust_lbf = CT * density_slugft3 * vtip_ft_s**2 * radius_ft**2 * np.pi
        thrust_N = thrust_lbf * 4.44822
        
        return thrust_N[0], rpm, coll, error, mach_tip

if __name__ == "__main__":

    prop = EmpiricalPropeller()
    
    # Example calculations
    thrust_N = 5500*(1.1*9.806)/8
    #thrust_N  = 41500 / 2.20462 * 9.806 / 4
    density_kgm3 = 1.225
    diameter_m = 2.7
    #diameter_m = 15.625/3.048
    # Forward calculation
    power_W, rpm_fwd, coll, error, mach_tip = prop.calculate_power(thrust_N, density_kgm3, diameter_m)
    
    print("\nForward calculation:")
    print(f"Input thrust: {thrust_N:.1f} N")
    print(f"Required power: {power_W/1000:.1f} kW")
    print(f"Required RPM: {rpm_fwd:.1f}")
    print(f"Collective angle: {coll:.1f} degrees")
    print(f"Error: {error:.1e}")
    print(f"Mach tip: {mach_tip:.3f}")

    # Reverse calculation
    thrust_N, rpm_rev, coll, error, mach_tip = prop.calculate_thrust(power_W, density_kgm3, diameter_m)
    
    print("\nReverse calculation:")
    print(f"Input power: {power_W/1000:.1f} kW")
    print(f"Available thrust: {thrust_N:.1f} N")
    print(f"Required RPM: {rpm_rev:.1f}")
    print(f"Collective angle: {coll:.1f} degrees")
    print(f"Error: {error:.1e}")
    print(f"Mach tip: {mach_tip:.3f}")