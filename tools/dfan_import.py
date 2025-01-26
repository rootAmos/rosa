import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def load_dfan_data(data_dir='data/dfan'):
    """Load ducted fan performance data.
    
    Returns:
        CT_CP_interp: Function that takes CT and returns CP
        J_CP_interp: Function that takes J and returns CP
        J_CT_interp: Function that takes J and returns CT
    """
    # Load data
    ct_cp_data = pd.read_csv(f'{data_dir}/ct_cp.csv', header=None)
    j_cp_data = pd.read_csv(f'{data_dir}/j_cp.csv', header=None)
    j_ct_data = pd.read_csv(f'{data_dir}/j_ct.csv', header=None)
    
    # Create interpolation functions
    CT_CP_interp = interp1d(ct_cp_data[0], ct_cp_data[1], bounds_error=False, fill_value='extrapolate')
    J_CP_interp = interp1d(j_cp_data[0], j_cp_data[1], bounds_error=False, fill_value='extrapolate')
    J_CT_interp = interp1d(j_ct_data[0], j_ct_data[1], bounds_error=False, fill_value='extrapolate')
    
    return CT_CP_interp, J_CP_interp, J_CT_interp

def calculate_power(thrust_N, airspeed_ms, density_kgm3, diameter_m, J_CT_interp, CT_CP_interp):
    """
    Calculate power required for given thrust and airspeed.
    
    Args:
        thrust_N: Required thrust (N)
        airspeed_ms: Forward airspeed (m/s)
        density_kgm3: Air density (kg/m³)
        diameter_m: Fan diameter (m)
        J_CT_interp: Function that takes J and returns CT
        CT_CP_interp: Function that takes CT and returns CP
    
    Returns:
        power_W: Power required (W)
        rpm: Required RPM
    """
    # Convert to imperial units for coefficient calculations
    thrust_lbf = thrust_N * 0.224809
    airspeed_fts = airspeed_ms * 3.28084
    density_slugft3 = density_kgm3 * 0.00194032
    diameter_ft = diameter_m * 3.28084
    
    def rpm_solver(rpm):
        n = rpm / 60  # convert to rev/s
        J = airspeed_fts / (n * diameter_ft)
        CT_target = thrust_lbf / (density_slugft3 * diameter_ft**4 * n**2)
        CT_actual = J_CT_interp(J)
        return abs(CT_actual - CT_target)
    
    # Find RPM that matches required thrust
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(rpm_solver, bounds=(1, 10000), method='bounded')
    rpm = result.x
    
    # Calculate final coefficients
    n = rpm / 60
    J = airspeed_fts / (n * diameter_ft)
    CT = J_CT_interp(J)
    CP = CT_CP_interp(CT)
    
    # Calculate power and convert to SI
    power_lbfts = CP * density_slugft3 * diameter_ft**5 * n**3
    power_W = power_lbfts * 1.35582
    
    return power_W, rpm

def calculate_thrust(power_W, airspeed_ms, density_kgm3, diameter_m, J_CP_interp, CT_CP_interp):
    """
    Calculate thrust available for given power and airspeed.
    
    Args:
        power_W: Available power (W)
        airspeed_ms: Forward airspeed (m/s)
        density_kgm3: Air density (kg/m³)
        diameter_m: Fan diameter (m)
        J_CP_interp: Function that takes J and returns CP
        CT_CP_interp: Function that takes CT and returns CP
    
    Returns:
        thrust_N: Available thrust (N)
        rpm: Required RPM
    """
    # Convert to imperial units for coefficient calculations
    power_lbfts = power_W * 0.737562
    airspeed_fts = airspeed_ms * 3.28084
    density_slugft3 = density_kgm3 * 0.00194032
    diameter_ft = diameter_m * 3.28084
    
    def rpm_solver(rpm):
        n = rpm / 60
        J = airspeed_fts / (n * diameter_ft)
        CP_target = power_lbfts / (density_slugft3 * diameter_ft**5 * n**3)
        CP_actual = J_CP_interp(J)
        return abs(CP_actual - CP_target)
    
    # Find RPM that matches required power
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(rpm_solver, bounds=(1, 10000), method='bounded')
    rpm = result.x
    
    # Calculate final coefficients
    n = rpm / 60
    J = airspeed_fts / (n * diameter_ft)
    CP = J_CP_interp(J)
    
    def ct_solver(ct):
        return abs(CT_CP_interp(ct) - CP)
    
    ct_result = minimize_scalar(ct_solver, bounds=(0, 1), method='bounded')
    CT = ct_result.x
    
    # Calculate thrust and convert to SI
    thrust_lbf = CT * density_slugft3 * diameter_ft**4 * n**2
    thrust_N = thrust_lbf * 4.44822
    
    return thrust_N, rpm

def plot_curves(CT_CP_interp, J_CP_interp, J_CT_interp):
    """Plot scaled performance curves."""
    # Plot J vs CT and CP
    J = np.linspace(0, 2, 100)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(J, J_CT_interp(J))
    plt.xlabel('J')
    plt.ylabel('CT ')
    plt.grid(True)
    plt.title('J vs CT')
    
    plt.subplot(1, 2, 2)
    plt.plot(J, J_CP_interp(J))
    plt.xlabel('J')
    plt.ylabel('CP ')
    plt.grid(True)
    plt.title('J vs CP')
    
    plt.tight_layout()
    plt.show()
    
    # Plot CT vs CP
    CT = np.linspace(0, max(J_CT_interp(J)), 100)
    plt.figure(figsize=(8, 6))
    plt.plot(CT, CT_CP_interp(CT))
    plt.xlabel('CT ')
    plt.ylabel('CP ')
    plt.title('CT vs CP')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data
    CT_CP_interp, J_CP_interp, J_CT_interp = load_dfan_data()
    
    # Example calculations with SI inputs
    thrust_N = 3000
    airspeed_ms = 74
    density_kgm3 = 1.225
    diameter_m = 1.58
    
    # Forward calculation
    power_W, rpm_fwd = calculate_power(thrust_N, airspeed_ms, density_kgm3, 
                                     diameter_m, J_CT_interp, CT_CP_interp)
    
    print("Forward calculation:")
    print(f"Input thrust: {thrust_N:.1f} N")
    print(f"Required power: {power_W/1000:.1f} kW")
    print(f"Required RPM: {rpm_fwd:.1f}")
    
    # Reverse calculation
    thrust_rev, rpm_rev = calculate_thrust(power_W, airspeed_ms, density_kgm3,
                                         diameter_m, J_CP_interp, CT_CP_interp)
    
    print("\nReverse calculation:")
    print(f"Input power: {power_W/1000:.1f} kW")
    print(f"Available thrust: {thrust_rev:.1f} N")
    print(f"Required RPM: {rpm_rev:.1f}")
    
    # Verify calculations match
    print("\nVerification:")
    print(f"Thrust error: {abs(thrust_N - thrust_rev):.3f} N")
    print(f"RPM error: {abs(rpm_fwd - rpm_rev):.3f}")
    
    # Plot curves
    plot_curves(CT_CP_interp, J_CP_interp, J_CT_interp) 