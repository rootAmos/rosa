import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RBFInterpolator
import glob

class EmpiricalMotor:
    def __init__(self, data_dir='data/emotor'):
        self.data_dir = data_dir
        self.points = self.load_efficiency_data()
        self.interp_func = self.create_efficiency_interpolator()

    def load_efficiency_data(self):
        """
        Load efficiency curve data from CSV files.
    
        Args:
            data_dir: Directory containing efficiency curve CSV files
            
        Returns:
            points: List of (speed, torque, efficiency) points
        """
        points = []
        
        # Get all CSV files in directory
        files = glob.glob(f"{self.data_dir}/0_*.csv")
        
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

    def create_efficiency_interpolator(self, kernel='thin_plate_spline', epsilon=0.1):
        """
        Create RBF interpolation function for motor efficiency.
        
        Args:
            kernel: RBF kernel type ('thin_plate_spline', 'multiquadric', 'gaussian', etc)
            epsilon: Shape parameter for RBF
            
        Returns:
            interp_func: Function that takes (speed, torque) and returns efficiency
        """
        # Unpack points 
        points = self.points

        # Normalize inputs to improve numerical stability
        speed_scale = np.max(points[:,0])
        torque_scale = np.max(points[:,1])
        
        X = np.column_stack([
            points[:,0] / speed_scale,
            points[:,1] / torque_scale
        ])
        y = points[:,2]
        
        # Create RBF interpolator
        rbf = RBFInterpolator(X, y, kernel=kernel, epsilon=epsilon)
        
        # Create wrapper function that handles normalization
        def interp_func(points):
            points_array = np.array(points)  # Convert input to numpy array
            points_norm = np.column_stack([
                points_array[:,0] / speed_scale,
                points_array[:,1] / torque_scale
            ])
            return rbf(points_norm)
        
        return interp_func

    def plot_efficiency_map(self, interp_func, rpm_max, tq_max, rpm_min=0, tq_min=0):
        """Plot experimental data points and interpolated efficiency map."""
        # Create grid for contour plot

        # Unpack points 
        points = self.points
        rpm = np.linspace(rpm_min, rpm_max, 100)
        tq = np.linspace(tq_min, tq_max, 100)
        RPM, TQ = np.meshgrid(rpm, tq)
        
        # Get efficiencies from interpolator
        points_to_eval = np.array([[r, t] for r, t in zip(RPM.flat, TQ.flat)])
        EFF = interp_func(points_to_eval).reshape(RPM.shape)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot interpolated contours with extended range
        levels = np.arange(0.90, 0.97, 0.01)  # Extended to 0.97 to show >0.95
        contour = plt.contourf(RPM, TQ, EFF, levels=levels, cmap='RdYlBu_r', extend='both')
        plt.colorbar(contour, label='Efficiency')
        
        # Plot experimental points
        plt.scatter(points[:,0], points[:,1], c=points[:,2], 
                cmap='RdYlBu_r', marker='x', s=20, vmin=0.90, vmax=0.97)
        
        plt.xlabel('Motor Speed [rpm]')
        plt.ylabel('Motor Torque [Nm]')
        plt.title('Combined Motor and Inverter Efficiency (RBF Interpolation)')
        plt.grid(True)
        plt.show()

    """
    
    def plot_cross_validation(self, kernel='thin_plate_spline', epsilon=0.1):
        Plot cross-validation results to assess interpolation quality
        from sklearn.model_selection import KFold

        # Unpack points 
        points = self.points
        
        errors = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(points):
            # Create interpolator with training data
            train_points = points[train_idx]
            test_points = points[test_idx]
            
            interp = self.create_efficiency_interpolator(train_points, kernel, epsilon)
            
            # Calculate error on test set
            pred_eff = interp(test_points[:,:2])
            error = np.sqrt(np.mean((pred_eff - test_points[:,2])**2))
            errors.append(error)
        
        print(f"Cross-validation RMSE: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        """

if __name__ == "__main__":
    # Load experimental data
    motor = EmpiricalMotor()
    
    # Motor specifications
    rpm_max = 2700
    tq_max = 1400
    rpm_opt = 2500
    tq_opt = 400
    
    # Create 'thin_plate_spline'
    interp_func = motor.create_efficiency_interpolator(kernel='thin_plate_spline')
    
    # Plot efficiency map
    motor.plot_efficiency_map(interp_func, rpm_max, tq_max)
    
    # Perform cross-validation
    #motor.plot_cross_validation(kernel='thin_plate_spline')
    
    # Test interpolator at optimal point
    eff = interp_func([[rpm_opt, tq_opt]])[0]
    print(f"Efficiency at optimal point: {eff:.3f}")