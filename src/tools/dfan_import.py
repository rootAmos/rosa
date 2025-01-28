import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pdb

class EmpiricalDfan:  

    def __init__(self, vehicle, data_dir='data/dfan'):
        """
        Initialize the EmpiricalDfan class.
        
        Args:
            vehicle: Vehicle data
            data_dir: Directory containing the ducted fan data
        """

        self.data_dir = data_dir
        self.load_dfan_data()
        self.N_sweep = 1000
        self.rpm_min = vehicle['fwd_prplsr_rpm_min']
        self.rpm_max = vehicle['fwd_prplsr_rpm_max']

    def load_dfan_data(self):
        """Load ducted fan performance data.
        
        Returns:
        CT_CP_interp: Function that takes CT and returns CP
        J_CP_interp: Function that takes J and returns CP
        J_CT_interp: Function that takes J and returns CT
        """
        # Load data
        ct_cp_data = pd.read_csv(f'{self.data_dir}/ct_cp.csv', header=None)
        j_cp_data = pd.read_csv(f'{self.data_dir}/j_cp.csv', header=None)
        j_ct_data = pd.read_csv(f'{self.data_dir}/j_ct.csv', header=None)
        
        # Create interpolation functions
        self.CT_CP_interp = interp1d(ct_cp_data[0], ct_cp_data[1], bounds_error=False, fill_value='extrapolate')
        self.J_CP_interp = interp1d(j_cp_data[0], j_cp_data[1], bounds_error=False, fill_value='extrapolate')
        self.J_CT_interp = interp1d(j_ct_data[0], j_ct_data[1], bounds_error=False, fill_value='extrapolate')
        

    def calculate_power(self, vehicle, phase):
        """
        Calculate power required for given thrust and airspeed.
        
        Args:
            vehicle: Vehicle data
            phase: Phase data
            J_CT_interp: Function that takes J and returns CT
            CT_CP_interp: Function that takes CT and returns CP
        
        Returns:
            power_W: Power required (W)
            rpm: Required RPM
        """
        # Convert to imperial units for coefficient calculations
        thrust_lbf = phase['hor_thrust_unit_N'] * 0.224809
        airspeed_fts = phase['u_m_s'] * 3.28084
        density_slugft3 = phase['density_kgm3'] * 0.00194032
        diameter_ft = vehicle['fwd_prplsr_diam_m'] * 3.28084
        
        rpm_sweep = np.linspace(self.rpm_min, self.rpm_max, self.N_sweep)[:,np.newaxis]
        n = rpm_sweep / 60  # convert to rev/s

        J = airspeed_fts / (n * diameter_ft)

        thrust_calc_sweep_lbf = self.J_CT_interp(J) * (density_slugft3 * diameter_ft**4 * n**2)
        error = np.abs(thrust_calc_sweep_lbf - np.broadcast_to(thrust_lbf, thrust_calc_sweep_lbf.shape))

        min_error_idx = np.argmin(error, axis = 0)
        rpm = rpm_sweep[min_error_idx].flatten()

        min_errors = error[min_error_idx, np.arange(phase['N'])]
        min_errors_rel = min_errors / thrust_lbf

        #pdb.set_trace()
        if np.any(min_errors_rel > 0.05):
            pdb.set_trace()
            print("Warning: Large errors in ducted fan power calculation")
        # end

        
        # Calculate final coefficients
        n = rpm / 60
        J = airspeed_fts / (n * diameter_ft)
        CT = self.J_CT_interp(J)
        CP = self.CT_CP_interp(CT)
        
        # Calculate power and convert to SI
        power_lbfts = CP * density_slugft3 * diameter_ft**5 * n**3
        power_W = power_lbfts * 1.35582
                
        return power_W, rpm

    def calculate_thrust(self, vehicle, phase):
        """
        Calculate thrust available for given power and airspeed.
        
        Args:
            vehicle: Vehicle data
            phase: Phase data
            J_CP_interp: Function that takes J and returns CP
            CT_CP_interp: Function that takes CT and returns CP
        
        Returns:
            thrust_N: Available thrust (N)
            rpm: Required RPM
        """
        # Convert to imperial units for coefficient calculations
        power_lbfts = phase['fwd_unit_shaft_power_W'] * 0.737562
        airspeed_fts = phase['u_m_s'] * 3.28084
        density_slugft3 = phase['density_kgm3'] * 0.00194032
        diameter_ft = vehicle['fwd_prplsr_diam_m'] * 3.28084
        
        rpm_sweep = np.linspace(self.rpm_min, self.rpm_max, self.N_sweep)[:,np.newaxis]
        n = rpm_sweep / 60  # convert to rev/s

        J = airspeed_fts / (n * diameter_ft)    

        power_calc_sweep_lbfts = self.J_CP_interp(J) * (density_slugft3 * diameter_ft**5 * n**3)
        error = np.abs(power_calc_sweep_lbfts - np.broadcast_to(power_lbfts, power_calc_sweep_lbfts.shape))

        min_error_idx = np.argmin(error, axis = 0)

        min_errors = error[min_error_idx, np.arange(phase['N'])]
        min_errors_rel = min_errors / power_lbfts
        if np.any(min_errors_rel > 0.05):
            print("Warning: Large errors in ducted fan thrust calculation")
        # end

        rpm = rpm_sweep[min_error_idx].flatten()
        
        # Calculate final coefficients
        n = rpm / 60
        J = airspeed_fts / (n * diameter_ft)
        CT = self.J_CT_interp(J)

        # Calculate thrust and convert to SI
        thrust_lbf = CT * density_slugft3 * diameter_ft**4 * n**2
        thrust_N = thrust_lbf * 4.44822
        
        return thrust_N, rpm

    def plot_curves(self):
        """Plot scaled performance curves."""
        # Plot J vs CT and CP
        J = np.linspace(0, 2, 100)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(J, self.J_CT_interp(J))
        plt.xlabel('J')
        plt.ylabel('CT ')
        plt.grid(True)
        plt.title('J vs CT')
        
        plt.subplot(1, 2, 2)
        plt.plot(J, self.J_CP_interp(J))
        plt.xlabel('J')
        plt.ylabel('CP ')
        plt.grid(True)
        plt.title('J vs CP')
        
        plt.tight_layout()
        plt.show()
        
        # Plot CT vs CP
        CT = np.linspace(0, max(self.J_CT_interp(J)), 100)
        plt.figure(figsize=(8, 6))
        plt.plot(CT, self.CT_CP_interp(CT))
        plt.xlabel('CT ')
        plt.ylabel('CP ')
        plt.title('CT vs CP')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":


    # Example calculations with SI inputs

    N_phase = 10

    phase = {}
    vehicle = {}

    
    L_D = 18
    phase['hor_thrust_unit_N'] = 5700*9.806/L_D / 2 * np.ones(N_phase)
    phase['u_m_s'] = 74 * np.ones(N_phase)
    phase['density_kgm3'] = 1.225 * np.ones(N_phase)
    vehicle['fwd_prplsr_diam_m'] = 1.58
    vehicle['fwd_prplsr_rpm_min'] = 500
    vehicle['fwd_prplsr_rpm_max'] = 3000

        # Load data
    dfan = EmpiricalDfan(vehicle)

    
    # Forward calculation
    power_W, rpm_fwd = dfan.calculate_power(vehicle, phase)
    
    print("\nForward calculation:")
    print(f"Input thrust (N): {phase['hor_thrust_unit_N']}")
    print(f"Required power (kW): {power_W/1000}")
    print(f"Required RPM: {rpm_fwd}")
    
    # Reverse calculation
    phase['fwd_unit_shaft_power_W'] = power_W
    thrust_rev, rpm_rev = dfan.calculate_thrust(vehicle, phase)
    
    print("\nReverse calculation:")
    print(f"Input power (kW): {power_W/1000}")
    print(f"Available thrust (N): {thrust_rev}")
    print(f"Required RPM: {rpm_rev}")
    
    # Verify calculations match
    print("\nVerification:")
    print(f"Thrust error (N): {np.abs(phase['hor_thrust_unit_N'] - thrust_rev)}")
    print(f"RPM error: {np.abs(rpm_fwd - rpm_rev)}")
    
    # Plot curves
    dfan.plot_curves() 