import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def load_hover_data():
    """Load hover performance data and create interpolation functions."""
    data = pd.read_csv('data/2FF16A1-4A_HoverPerfo_Diam_15_625 - Sheet1.csv')
    FM_interp = interp1d(data['CT'], data['FM'], bounds_error=False, fill_value='extrapolate')
    return data, FM_interp

def get_power_required(thrust_N, density_SI, RPM, radius_m, FM_interp):
    """Calculate power required for a propeller in hover.
    
    Args:
        thrust_N: Required thrust in Newtons
        density_SI: Air density in kg/m³
        RPM: Propeller rotation speed in RPM
        radius_m: Propeller radius in meters
    
    Returns:
        power_W: Required power in Watts
    """
    # Convert inputs to imperial units
    thrust_lbf = thrust_N * 0.224809  # N to lbf
    density_slug_ft3 = density_SI * 0.00194032  # kg/m³ to slug/ft³
    radius_ft = radius_m * 3.28084  # m to ft
    
    # Calculate tip speed
    omega_rev_s = RPM / 60  # rev/s
    tip_speed_ft_s = omega_rev_s * radius_ft * 2 * np.pi  # ft/s
    
    # Calculate CT for current diameter
    CT = thrust_lbf / (density_slug_ft3 * np.pi * radius_ft**2 * tip_speed_ft_s**2)
    
    # Scale CT to reference diameter (15.625 ft)
    D_ref = 15.625  # ft
    D_scaled = 2 * radius_ft  # ft
    CT_ref = CT / (D_scaled/D_ref)**4

    # Get FM for this CT_ref
    FM = float(FM_interp(CT_ref))
    
    # Calculate CP_ref
    CP_ref = CT_ref**(3/2) / (np.sqrt(2) * FM)
    
    # Scale CP to actual diameter
    CP = CP_ref * (D_scaled/D_ref)**5
    
    # Calculate power in ft-lbf/s
    power_ftlbf = CP * density_slug_ft3 * np.pi * radius_ft**2 * tip_speed_ft_s**3
    
    # Convert to Watts
    power_W = power_ftlbf * 1.35582
    
    return power_W

def plot_CT_CP_sweep(thrust_N, density_SI, RPM_range, radius_m, data, FM_interp):
    """Create a plot of CT vs CP for a range of RPMs with diameter scaling."""
    import matplotlib.pyplot as plt
    
    # Convert inputs to imperial units
    thrust_lbf = thrust_N * 0.224809
    density_slug_ft3 = density_SI * 0.00194032
    radius_ft = radius_m * 3.28084
    
    # Reference diameter
    D_ref_ft = 15.625  # ft
    D_scaled_ft = 2 * radius_ft  # ft
    
    # Calculate tip speeds (vectorized)
    omega_rev_s = RPM_range / 60
    tip_speed_ft_s = omega_rev_s * radius_ft * 2 * np.pi
    
    # Calculate scaled CT (vectorized)
    CTs_scaled = thrust_lbf / (density_slug_ft3 * np.pi * radius_ft**2 * tip_speed_ft_s**2)
    
    # Convert to reference CT (vectorized)
    CTs_ref = CTs_scaled / (D_scaled_ft/D_ref_ft)**4
    
    # Get FM for these CT_refs (vectorized)
    FMs = FM_interp(CTs_ref)
    
    # Calculate reference CP (vectorized)
    CPs_ref = CTs_ref**(3/2) / (np.sqrt(2) * FMs)
    
    # Scale CP to actual diameter (vectorized)
    CPs_scaled = CPs_ref * (D_scaled_ft/D_ref_ft)**5
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(CTs_ref, CPs_ref, '--', color='gray', label=f'D = {2*radius_m:.2f}m')
    plt.plot(data['CT'], data['CT'] / data['CT/CP'], 'rx', markersize=4, label='Test Data (D = 15.625ft)')
    
    plt.xlabel('CT')
    plt.ylabel('CP')
    plt.title('CT vs CP for Different RPMs')
    plt.grid(True)
    plt.legend()
    plt.show()
        

if __name__ == "__main__":
    # Example usage
    thrust_N = 20000 * 9.806 / 4  # N
    density = 1.225  # kg/m³
    RPM = 800
    radius_m = 15.625 / 3.048 / 2
    import pdb;
    data, FM_interp = load_hover_data()
    #pdb.set_trace()
    power = get_power_required(thrust_N, density, RPM, radius_m, FM_interp)
    print(f"Power required: {power/1000:.1f} kW") 
    pdb.set_trace()

    RPM_range = np.linspace(750, 3000, 100)
    plot_CT_CP_sweep(thrust_N, density, RPM_range, radius_m, data, FM_interp)