import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pdb
from scipy.interpolate import RegularGridInterpolator

def load_prop_data(data_dir='data/prop'):
    """Load propeller performance data."""
    data = pd.read_csv(f'{data_dir}/2FF16A1-4A_HoverPerfo_Diam_15_625.csv')
    
    # Create points array from RPM (derived from Vtip) and Coll
    diam_ft = 15.625
    points = np.column_stack(((data['Vtip'] / ((diam_ft/2) * 2* np.pi / 60)), data['Coll']))

    # Create interpolation functions using griddata
    def rpm_coll_to_CT(rpm_coll):
        return griddata(points, data['CT'], rpm_coll, method='linear')
        
    def rpm_coll_to_CP(rpm_coll):
        return griddata(points, data['CT']/data['CT/CP'], rpm_coll, method='linear')
    
    return rpm_coll_to_CT, rpm_coll_to_CP, data

def calculate_power(thrust_N, density_kgm3, diameter_m, rpm_coll_to_CT, rpm_coll_to_CP):
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
        CT = rpm_coll_to_CT(np.array([[rpm, coll]]))
        
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
    initial_guess = [1000, 10]  # rpm=1000, coll=10deg
    bounds = [(800, 1200), (-2, 20)]  # rpm and coll bounds
    
    # Find RPM and collective that give required thrust
    from scipy.optimize import minimize
    result = minimize(rpm_coll_solver, initial_guess, bounds=bounds, method='SLSQP')
    rpm, coll = result.x
    error = result.fun
    
    # Get final CP from interpolation
    vtip_ft_s = rpm * radius_ft * np.pi / 60
    CP = rpm_coll_to_CP(np.array([[rpm, coll]]))
    
    # Calculate power using vtip
    power_lbfts = CP * density_slugft3 * vtip_ft_s**3 * radius_ft**2 * np.pi
    power_W = power_lbfts * 1.35582

    
    return power_W[0], rpm, coll, error

def calculate_thrust(power_W, density_kgm3, diameter_m, rpm_coll_to_CT, rpm_coll_to_CP):
    """Calculate thrust available for given power in hover."""
    # Convert to imperial units
    power_req_lbfts = power_W * 0.737562
    density_slugft3 = density_kgm3 * 0.00194032
    diameter_ft = diameter_m * 3.28084
    radius_ft = diameter_ft/2
    
    def rpm_coll_solver(x):
        rpm, coll = x
        vtip_ft_s = rpm * radius_ft * np.pi / 60

        # Get interpolated CP for these conditions
        CP = rpm_coll_to_CP(np.array([[rpm, coll]]))
        
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
    initial_guess = [1000, 10]  # rpm=1000, coll=10deg
    bounds = [(800, 1200), (-2, 20)]  # rpm and coll bounds
    
    # Find RPM and collective that give required power
    from scipy.optimize import minimize
    result = minimize(rpm_coll_solver, initial_guess, bounds=bounds, method='SLSQP')
    rpm, coll = result.x
    error = result.fun
    
    # Get final CT from interpolation
    vtip_ft_s = rpm * radius_ft * np.pi / 60 * 2
    CT = rpm_coll_to_CT(np.array([[rpm, coll]]))
    
    # Calculate thrust using vtip
    thrust_lbf = CT * density_slugft3 * vtip_ft_s**2 * radius_ft**2 * np.pi
    thrust_N = thrust_lbf * 4.44822
    
    return thrust_N[0], rpm, coll, error

if __name__ == "__main__":
    # Load data
    rpm_coll_to_CT, rpm_coll_to_CP, data = load_prop_data()
    
    # Example calculations
    #thrust_N = 5700*9.806/8
    thrust_N  = 41500 / 2.20462 * 9.806 / 4
    density_kgm3 = 1.225
    #diameter_m = 3.09
    diameter_m = 15.625/3.048
    
    # Forward calculation
    power_W, rpm_fwd, coll, error = calculate_power(thrust_N, density_kgm3, diameter_m, 
                                     rpm_coll_to_CT, rpm_coll_to_CP)
    
    print("\nForward calculation:")
    print(f"Input thrust: {thrust_N:.1f} N")
    print(f"Required power: {power_W/1000:.1f} kW")
    print(f"Required RPM: {rpm_fwd:.1f}")
    print(f"Collective angle: {coll:.1f} degrees")
    print(f"Error: {error:.1e}")


    # Reverse calculation
    thrust_N, rpm_rev, coll, error = calculate_thrust(power_W, density_kgm3, diameter_m, 
                                         rpm_coll_to_CT, rpm_coll_to_CP)
    
    print("\nReverse calculation:")
    print(f"Input power: {power_W/1000:.1f} kW")
    print(f"Available thrust: {thrust_N:.1f} N")
    print(f"Required RPM: {rpm_rev:.1f}")
    print(f"Collective angle: {coll:.1f} degrees")
    print(f"Error: {error:.1e}")