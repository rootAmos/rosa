def parse_aero_coeffs(filenames):
    """Parse key aerodynamic coefficients vs alpha for multiple files."""
    all_data = {}
    
    # First pass: collect alpha ranges to find intersection
    alpha_mins = []
    alpha_maxs = []
    for filename in filenames:
        with open(filename, 'r') as f:
            headers = next(f).strip().split()
            alpha_idx = headers.index('Alpha')
            next(f)  # Skip blank line
            alphas = []
            for line in f:
                if line.strip():
                    values = [float(x) for x in line.split()]
                    alphas.append(values[alpha_idx])
            alpha_mins.append(min(alphas))
            alpha_maxs.append(max(alphas))
    
    # Use the most restrictive range
    alpha_min = max(alpha_mins)  # Highest minimum
    alpha_max = min(alpha_maxs)  # Lowest maximum
    
    # Create common alpha array
    import numpy as np
    common_alphas = np.linspace(alpha_min, alpha_max, 50)  # 50 points in the valid range
    
    # Second pass: interpolate data to common alpha values
    from scipy.interpolate import interp1d
    for filename in filenames:
        data = {
            'Alpha': [],
            'CL': [],
            'CD': [],
            'CD_viscous': [],
            'CD_induced': []
        }
        
        with open(filename, 'r') as f:
            headers = next(f).strip().split()
            idx = {
                'Alpha': headers.index('Alpha'),
                'CL': headers.index('CL'),
                'CD': headers.index('CD'),
                'CD_viscous': headers.index('CD_viscous'),
                'CD_induced': headers.index('CD_induced')
            }
            next(f)
            
            # Read raw data
            raw_data = {'Alpha': [], 'CL': [], 'CD': [], 'CD_viscous': [], 'CD_induced': []}
            for line in f:
                if line.strip():
                    values = [float(x) for x in line.split()]
                    for key, index in idx.items():
                        raw_data[key].append(values[index])
            
            # Interpolate to common alpha values
            for key in ['CL', 'CD', 'CD_viscous', 'CD_induced']:
                interp = interp1d(raw_data['Alpha'], raw_data[key])
                data[key] = list(interp(common_alphas))
            data['Alpha'] = list(common_alphas)
        
        # Simplified case name extraction - just get the number
        case_name = filename.split('.txt')[0].split('\\')[-1].split('_')[0]
        all_data[case_name] = data
    
    return all_data

def create_2d_interp(aero_data, coeff):
    """Create 2D interpolation function for given coefficient."""
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np
    
    # Extract unique speeds and alphas and sort them
    speeds = sorted([float(case) for case in aero_data.keys()])
    speeds = np.array(speeds)
    alphas = np.array(aero_data[list(aero_data.keys())[0]]['Alpha'])
    
    # Create 2D grid of values
    values = np.zeros((len(speeds), len(alphas)))
    for i, speed in enumerate(speeds):
        # Simplified case name format
        case = str(int(speed))
        values[i,:] = aero_data[case][coeff]
    
    # Create interpolator
    return RegularGridInterpolator((speeds, alphas), values)


def plot_3d_coeff(aero_data, coeff_name, interp_func):
    """Create 3D surface plot of coefficient vs speed and alpha."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Create dense grid of speed and alpha values
    speeds = np.array([float(case) for case in aero_data.keys()])
    alphas = np.array(aero_data[list(aero_data.keys())[0]]['Alpha'])
    
    speed_grid = np.linspace(min(speeds), max(speeds), 50)
    alpha_grid = np.linspace(min(alphas), max(alphas), 50)
    
    # Create meshgrid for surface plot
    S, A = np.meshgrid(speed_grid, alpha_grid)
    
    # Calculate coefficient values at each point
    Z = np.zeros_like(S)
    for i in range(len(speed_grid)):
        for j in range(len(alpha_grid)):
            Z[j,i] = interp_func([speed_grid[i], alpha_grid[j]])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(S, A, Z, cmap='viridis')
    
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Angle of Attack (deg)')
    ax.set_zlabel(coeff_name)
    ax.set_title(f'{coeff_name} vs Speed and Angle of Attack')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, label=coeff_name)
    
    plt.show()


if __name__ == "__main__":
    import glob
    import numpy as np
    
    # Get all txt files in data folder
    filenames = glob.glob("data/*.txt")
    aero = parse_aero_coeffs(filenames)
    
    # Create interpolation functions
    interp_CL = create_2d_interp(aero, 'CL')
    interp_CD = create_2d_interp(aero, 'CD')
    interp_CD_visc = create_2d_interp(aero, 'CD_viscous')
    interp_CD_ind = create_2d_interp(aero, 'CD_induced')

    # Do a sweep of speeds and alphas
    speeds = [10, 40, 75, 100]  # m/s
    alphas = [-1, 0, 5, 10]     # degrees
    
    print("\nAerodynamic coefficients at different speeds and angles:")
    print("Speed (m/s) | Alpha (deg) |    CL    |    CD")
    print("-" * 55)
    
    for speed in speeds:
        for alpha in alphas:
            cl = float(interp_CL([speed, alpha])[0])  # Extract scalar with [0]
            cd = float(interp_CD([speed, alpha])[0])  # Extract scalar with [0]
            print(f"{speed:10.1f} | {alpha:10.1f} | {cl:8.4f} | {cd:8.4f}")
        print("-" * 55)  # Separator 
    # 3D Plot example
    #plot_3d_coeff(aero, 'CL', interp_CL)    