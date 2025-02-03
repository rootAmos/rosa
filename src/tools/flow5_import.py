class EmpiricalAero:
    def __init__(self, filenames):
        self.load_aero_data(filenames)

        self.CL_data, self.CD_data, self.CD_viscous_data, self.CD_induced_data = self.parse_aero_coeffs(filenames)


    def load_aero_data(self, filenames):
        self.aero_data = self.parse_aero_coeffs(filenames)

    def parse_aero_coeffs(self, filenames):
        """Parse aerodynamic coefficients into separate nx3 matrices."""
        import numpy as np
        from scipy.interpolate import interp1d
        
        # Read all data files
        raw_data = [np.loadtxt(f, skiprows=1) for f in filenames]
        speeds = np.array([float(f.split('.txt')[0].split('\\')[-1].split('_')[0]) for f in filenames])
        
        # Create speed-alpha mesh for each file
        data_points = []
        for speed, data in zip(speeds, raw_data):
            speed_col = np.full_like(data[:, 0], speed)
            data_points.append(np.column_stack((data[:, 1], speed_col, data[:, 3], data[:, 4], data[:, 5], data[:, 6])))
        
        # Stack all data points
        all_points = np.vstack(data_points)

        
        # Split into separate matrices [alpha, speed, value]
        CL = all_points[:, [0, 1, 2]]          # alpha, speed, CL
        CD = all_points[:, [0, 1, 3]]          # alpha, speed, CD
        CD_viscous = all_points[:, [0, 1, 4]]  # alpha, speed, CD_viscous
        CD_induced = all_points[:, [0, 1, 5]]  # alpha, speed, CD_induced

        return {
            'CL': CL,
            'CD': CD,
            'CD_viscous': CD_viscous,
            'CD_induced': CD_induced
        }

    def interp_coeff_from_alpha(self, coeff_type):
        """Create vectorized interpolation function using RBF."""
        from scipy.interpolate import RBFInterpolator
        import numpy as np
        
        data = self.aero_data[coeff_type]
        points = data[:, :2]  # alpha and speed columns
        values = data[:, 2]   # coefficient values
        
        rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
        return lambda_wx: rbf(np.column_stack((x[1], x[0])))  # Stack inputs correctly

    def interp_alpha_from_CL(self):
        """Create vectorized interpolation for alpha(speed, CL) using RBF."""
        from scipy.interpolate import RBFInterpolator
        import numpy as np
        
        data = self.aero_data['CL']
        points = data[:, [1, 2]]  # speed and CL columns
        values = data[:, 0]       # alpha values
        
        rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
        return lambda_wx: rbf(np.column_stack((x[0], x[1])))  # Stack inputs correctly

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

    phase = {}
    phase['N'] = 10
    phase['CL'] = 0.7 * np.ones(phase['N'])
    phase['u_m_s'] = 60 * np.ones(phase['N'])

    # Stack arrays vertically for interpolation
    input_data = np.vstack((phase['u_m_s'], phase['CL']))
    alpha = alpha_from_CL(input_data)

    print("-"*50)
    print("\nDetermine alpha from CL")
    print(f"Speed (m/s): {phase['u_m_s']}")
    print(f"CL: {phase['CL']}")
    print(f"Alpha (deg): {alpha}")
    
    coeff_from_alpha = aero.interp_coeff_from_alpha('CL')

    phase = {}
    phase['N'] = 10
    phase['alpha'] = 3 * np.ones(phase['N'])
    phase['u_m_s'] = 60 * np.ones(phase['N'])

    # Stack arrays vertically for interpolation
    input_data = np.vstack((phase['u_m_s'], phase['alpha']))
    CL = coeff_from_alpha(input_data)

    print("-"*50)
    print("\nDetermine CL from alpha")
    print(f"Speed (m/s): {phase['u_m_s']}")
    print(f"Alpha (deg): {phase['alpha']}")
    print(f"CL: {CL}")

    # Separator 
    # 3D Plot example
    #plot_3d_coeff(aero, 'CL', interp_CL)    