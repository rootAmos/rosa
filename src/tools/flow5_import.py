class EmpiricalAero:
    def __init__(self, filenames):
        self.load_aero_data(filenames)
        self.parse_aero_coeffs(filenames)

        # Create interpolation functions
        self.interp_CL = self.interp_coeff_from_alpha('CL')
        self.interp_CD = self.interp_coeff_from_alpha('CD')
        self.interp_CD_visc = self.interp_coeff_from_alpha('CD_viscous')
        self.interp_CD_ind = self.interp_coeff_from_alpha('CD_induced')

    def load_aero_data(self, filenames):
        self.aero_data = self.parse_aero_coeffs(filenames)


    def parse_aero_coeffs(self, filenames):
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

    def interp_coeff_from_alpha(self, coeff):
        """Create 2D interpolation function for given coefficient."""
        from scipy.interpolate import RegularGridInterpolator
        import numpy as np

        aero_data = self.aero_data
        
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

    def interp_alpha_from_CL(self):
        """Create interpolation function for alpha(speed, CL)."""
        from scipy.interpolate import RegularGridInterpolator
        import numpy as np

        aero_data = self.aero_data
        
        # Extract unique speeds and CLs and sort them
        speeds = sorted([float(case) for case in aero_data.keys()])
        speeds = np.array(speeds)
        alphas = np.array(aero_data[list(aero_data.keys())[0]]['Alpha'])
        
        # Create 2D grid of CL values
        values = np.zeros((len(speeds), len(alphas)))
        for i, speed in enumerate(speeds):
            case = str(int(speed))
            values[i,:] = aero_data[case]['CL']
        
        # Create meshgrid for interpolation
        speed_grid, alpha_grid = np.meshgrid(speeds, alphas)
        
        # Return interpolator for alpha(speed, CL)
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
    import pdb
    
    # Get all txt files in data folder
    filenames = glob.glob("data/aero/*.txt")

    aero = EmpiricalAero(filenames)

    alpha_from_CL = aero.interp_alpha_from_CL()

    alpha = alpha_from_CL([10, 0.7])
    print(f"Alpha: {alpha[0]:.2f} deg")

    pdb.set_trace()
    

    # Do a sweep of speeds and alphas
    speeds = [10, 40, 75, 100]  # m/s
    alphas = [-1, 0, 5, 10]     # degrees
    
    print("\nAerodynamic coefficients at different speeds and angles:")
    print("Speed (m/s) | Alpha (deg) |    CL    |    CD")
    print("-" * 55)
    
    for speed in speeds:
        for alpha in alphas:
            cl = float(aero.interp_CL([speed, alpha])[0])  # Extract scalar with [0]
            cd = float(aero.interp_CD([speed, alpha])[0])  # Extract scalar with [0]
            print(f"{speed:10.1f} | {alpha:10.1f} | {cl:8.4f} | {cd:8.4f}")
        print("-" * 55)  # Separator 
    # 3D Plot example
    #plot_3d_coeff(aero, 'CL', interp_CL)    