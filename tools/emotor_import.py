import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import glob

def load_efficiency_data(data_dir='data/emotor'):
    """
    Load efficiency curve data from CSV files.
    
    Args:
        data_dir: Directory containing efficiency curve CSV files
        
    Returns:
        points: List of (speed, torque, efficiency) points
    """
    points = []
    
    # Get all CSV files in directory
    files = glob.glob(f"{data_dir}/0_*.csv")
    
    for file in files:
        # Extract efficiency from filename (0_xx.csv)
        eff = float(file.split('_')[1].split('.')[0])/100
        
        # Load speed and torque points from unlabeled columns
        data = pd.read_csv(file, header=None)
        speeds = data[0]  # First column is speed
        torques = data[1]  # Second column is torque
        
        # Add points for this efficiency curve
        for speed, torque in zip(speeds, torques):
            points.append([speed, torque, eff])
            
    return np.array(points)

def create_efficiency_interpolator(rpm_opt, tq_opt, eff_opt, rpm_max, tq_max, k0):
    """
    Create interpolation function for motor efficiency based on literature loss models.
    
    Args:
        rpm_opt: Optimum operating speed (RPM)
        tq_opt: Optimum operating torque (Nm)
        eff_opt: Maximum efficiency (decimal)
        rpm_max: Maximum speed (RPM)
        tq_max: Maximum torque (Nm)
        k0: Core loss coefficient
        
    Returns:
        interp_func: Function that takes (speed, torque) and returns efficiency
    """
    # Convert RPM to rad/s for optimal speed
    w_opt = rpm_opt * 2 * np.pi / 60
    w_max = rpm_max * 2 * np.pi / 60
    
    # Motor Constants from design parameters
    kw = w_max / w_opt
    kq = tq_max / tq_opt
    max_power = tq_max * w_max  # Max power in Watts
    kp = max_power / (tq_opt * w_opt)
    
    # Loss coefficients derived from literature
    # C_0: Core losses (hysteresis and eddy current)
    C_0 = k0 * w_opt * tq_opt / 6 * (1 - eff_opt) / eff_opt
    
    # C_1: Speed dependent losses
    C_1 = -3 * C_0 / (2 * w_opt) + tq_opt * (1 - eff_opt) / (4 * eff_opt)
    
    # C_2: Windage losses (cubic with speed)
    C_2 = C_0 / (2 * w_opt**3) + tq_opt * (1 - eff_opt) / (4 * eff_opt * w_opt**2)
    
    # C_3: Copper losses (proportional to torque squared)
    C_3 = w_opt * (1 - eff_opt) / (2 * tq_opt * eff_opt)
    
    # Create regular grid for interpolation
    rpm_grid = np.linspace(0, rpm_max, 100)
    tq_grid = np.linspace(0, tq_max, 100)
    
    # Create mesh for efficiency calculation
    RPM, TQ = np.meshgrid(rpm_grid, tq_grid)
    W = RPM * 2 * np.pi / 60  # Convert RPM to rad/s
    
    # Calculate power losses using the full model
    P_loss = C_0 + C_1 * W + C_2 * W**3 + C_3 * TQ**2
    
    # Calculate mechanical output power
    P_out = W * TQ
    
    # Calculate electrical input power
    P_in = P_out + P_loss
    
    # Calculate efficiency map
    eff_map = np.where(P_in > 0, P_out / P_in, 0)
    
    # Create interpolator
    return RegularGridInterpolator((rpm_grid, tq_grid), eff_map.T,
                                 bounds_error=False, fill_value=0)

def plot_efficiency_map(points, interp_func, rpm_max, tq_max, rpm_min=0, tq_min=0):
    """
    Plot experimental data points and interpolated efficiency map.
    
    Args:
        points: Array of [speed, torque, efficiency] points
        interp_func: Interpolation function from create_efficiency_interpolator
        rpm_max: Maximum speed to plot (RPM)
        tq_max: Maximum torque to plot (Nm)
        rpm_min: Minimum speed to plot (RPM), default 0
        tq_min: Minimum torque to plot (Nm), default 0
    """
    # Create grid for contour plot
    rpm = np.linspace(rpm_min, rpm_max, 100)
    tq = np.linspace(tq_min, tq_max, 100)
    RPM, TQ = np.meshgrid(rpm, tq)
    
    # Get efficiencies from interpolator
    points_to_eval = np.array([[r, t] for r, t in zip(RPM.flat, TQ.flat)])
    EFF = interp_func(points_to_eval).reshape(RPM.shape)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot interpolated contours
    levels = np.arange(0.90, 0.96, 0.01)
    contour = plt.contourf(RPM, TQ, EFF, levels=levels, cmap='RdYlBu_r')
    plt.colorbar(contour, label='Efficiency')
    
    # Plot experimental points
    plt.scatter(points[:,0], points[:,1], c=points[:,2], 
               cmap='RdYlBu_r', marker='x', s=20)
    
    plt.xlabel('Motor Speed [rpm]')
    plt.ylabel('Motor Torque [Nm]')
    plt.title('Combined Motor and Inverter Efficiency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load experimental data
    points = load_efficiency_data()
    
    # Create interpolator with motor specifications
    rpm_opt = 2500
    tq_opt = 400
    eff_opt = 0.95
    rpm_max = 2700
    tq_max = 1400
    k0 = 0.725
    
    interp_func = create_efficiency_interpolator(rpm_opt, tq_opt, eff_opt, rpm_max, tq_max, k0)
    
    # Create plot with full range
    plot_efficiency_map(points, interp_func, rpm_max, tq_max)
    
    # Test interpolator at optimal point
    eff = interp_func([[rpm_opt, tq_opt]])[0]
    print(f"Efficiency at optimal point: {eff:.3f}")