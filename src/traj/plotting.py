import numpy as np
import matplotlib.pyplot as plt

def plot_mission(phases):
    """
    Plot combined mission data across multiple phases.
    
    Args:
        phases: List of phase dictionaries or single phase dictionary
    """
    # Convert single phase to list if necessary
    if not isinstance(phases, list):
        phases = [phases]
    
    # Concatenate time-series data
    time_s = np.concatenate([phase['time_s'] for phase in phases])
    time_min = time_s / 60.0  # Convert to minutes
    
    # Kinematics
    x_m = np.concatenate([phase['x_m'] for phase in phases])
    z_m = np.concatenate([phase['z_m'] for phase in phases])
    xdot_m_s = np.concatenate([phase['xdot_m_s'] for phase in phases])
    zdot_m_s = np.concatenate([phase['zdot_m_s'] for phase in phases])
    udot_m_s2 = np.concatenate([phase['udot_m_s2'] for phase in phases])
    
    # Forces
    lift_N = np.concatenate([phase.get('lift_N', np.zeros_like(phase['time_s'])) for phase in phases])
    drag_N = np.concatenate([phase.get('drag_N', np.zeros_like(phase['time_s'])) for phase in phases])
    
    # Calculate phase transition times in minutes
    transition_times = [phase['t1_s']/60.0 for phase in phases[:-1]]
    
    def add_transition_lines(ax):
        for t in transition_times:
            ax.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    # Create subplots for kinematics
    fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig1.suptitle('Mission Kinematics')
    
    # Position and velocity plots
    ax1.plot(time_min, x_m/1000, label='x')  # Convert to km
    add_transition_lines(ax1)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Distance (km)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(time_min, z_m *3.048, label='z')  # Convert to ft
    add_transition_lines(ax2)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Altitude (ft)')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(time_min, xdot_m_s, label='$\dot{x}$')
    add_transition_lines(ax3)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Earth axis X-Velocity (m/s)')
    ax3.grid(True)
    ax3.legend()
    
    ax4.plot(time_min, zdot_m_s, label='$\dot{z}$')
    add_transition_lines(ax4)
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('Earth axis Z-Velocity (m/s)')
    ax4.grid(True)
    ax4.legend()
    
    # Acceleration plot
    fig2, ax5 = plt.subplots(figsize=(10, 4))
    fig2.suptitle('Body Acceleration')
    ax5.plot(time_min, udot_m_s2, label='$\ddot{u}$')
    add_transition_lines(ax5)
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.grid(True)
    ax5.legend()
    
    # Forces plots
    fig3, (ax6, ax7) = plt.subplots(2, 1, figsize=(10, 8))
    fig3.suptitle('Forces')
    
    # Initialize arrays for thrust components
    time_min = np.array([])
    ver_thrust = np.array([])
    hor_thrust = np.array([])
    
    # Concatenate thrust data across phases
    for phase in phases:
        phase_time = phase['time_s']/60.0
        time_min = np.concatenate([time_min, phase_time])
        
        # Vertical thrust
        thrust = phase.get('ver_thrust_unit_N', np.zeros_like(phase['time_s']))
        if not isinstance(thrust, np.ndarray):
            thrust = thrust * np.ones_like(phase['time_s'])
        ver_thrust = np.concatenate([ver_thrust, thrust])
        
        # Horizontal thrust
        thrust = phase.get('hor_thrust_unit_N', np.zeros_like(phase['time_s']))
        if not isinstance(thrust, np.ndarray):
            thrust = thrust * np.ones_like(phase['time_s'])
        hor_thrust = np.concatenate([hor_thrust, thrust])
    
    # Plot thrust components
    ax6.plot(time_min, ver_thrust, label='Vertical Thrust')
    ax6.plot(time_min, hor_thrust, label='Horizontal Thrust')
    
    add_transition_lines(ax6)
    ax6.set_xlabel('Time (min)')
    ax6.set_ylabel('Thrust per unit (N)')
    ax6.grid(True)
    ax6.legend()
    
    # Lift and Drag
    ax7.plot(time_min, lift_N, label='Lift')
    ax7.plot(time_min, drag_N, label='Drag')
    
    add_transition_lines(ax7)
    ax7.set_xlabel('Time (min)')
    ax7.set_ylabel('Force (N)')
    ax7.grid(True)
    ax7.legend()
    
    # Power and efficiency plots
    fig4, ((ax8, ax9), (ax10, ax11)) = plt.subplots(2, 2, figsize=(12, 8))
    fig4.suptitle('Power and Efficiency')
    
    # Initialize arrays
    lift_motor_power = np.array([])
    fwd_motor_power = np.array([])
    lift_rpm = np.array([])
    fwd_rpm = np.array([])
    lift_torque = np.array([])
    fwd_torque = np.array([])
    lift_eta = np.array([])
    fwd_eta = np.array([])
    
    # Concatenate data across all phases
    for phase in phases:
        # Power
        lift_motor_power = np.concatenate([lift_motor_power, 
            phase.get('power_lift_motor_W', np.zeros_like(phase['time_s']))/1000])
        fwd_motor_power = np.concatenate([fwd_motor_power, 
            phase.get('power_fwd_motor_W', np.zeros_like(phase['time_s']))/1000])
        
        # RPM
        lift_rpm = np.concatenate([lift_rpm, 
            phase.get('rpm_lift_motor', np.zeros_like(phase['time_s']))])
        fwd_rpm = np.concatenate([fwd_rpm, 
            phase.get('rpm_fwd_motor', np.zeros_like(phase['time_s']))])
        
        # Torque
        lift_torque = np.concatenate([lift_torque, 
            phase.get('torque_lift_motor_Nm', np.zeros_like(phase['time_s']))])
        fwd_torque = np.concatenate([fwd_torque, 
            phase.get('torque_fwd_motor_Nm', np.zeros_like(phase['time_s']))])
        
        # Efficiency
        lift_eta = np.concatenate([lift_eta, 
            phase.get('eta_lift_motor', np.zeros_like(phase['time_s']))])
        fwd_eta = np.concatenate([fwd_eta, 
            phase.get('eta_fwd_motor', np.zeros_like(phase['time_s']))])
    
    # Plot power
    ax8.plot(time_min, lift_motor_power, label='Lift Motor Power')
    ax8.plot(time_min, fwd_motor_power, label='Forward Motor Power')
    
    add_transition_lines(ax8)
    ax8.set_xlabel('Time (min)')
    ax8.set_ylabel('Power (kW)')
    ax8.grid(True)
    ax8.legend()
    
    # Plot RPM
    ax9.plot(time_min, lift_rpm, label='Lift Motor RPM')
    ax9.plot(time_min, fwd_rpm, label='Forward Motor RPM')
    
    add_transition_lines(ax9)
    ax9.set_xlabel('Time (min)')
    ax9.set_ylabel('RPM')
    ax9.grid(True)
    ax9.legend()
    ax9.set_ylim(bottom=2000)
    
    # Plot torque
    ax10.plot(time_min, lift_torque, label='Lift Motor Torque')
    ax10.plot(time_min, fwd_torque, label='Forward Motor Torque')
    
    add_transition_lines(ax10)
    ax10.set_xlabel('Time (min)')
    ax10.set_ylabel('Torque (Nm)')
    ax10.grid(True)
    ax10.legend()
    
    # Plot efficiency
    ax11.plot(time_min, lift_eta, label='Lift Motor Efficiency')
    ax11.plot(time_min, fwd_eta, label='Forward Motor Efficiency')
    
    add_transition_lines(ax11)
    ax11.set_xlabel('Time (min)')
    ax11.set_ylabel('Efficiency')
    ax11.grid(True)
    ax11.legend()
    ax11.set_ylim(bottom=0.8)
    
    # Aerodynamic coefficients
    fig5, (ax12, ax13) = plt.subplots(2, 1, figsize=(10, 8))
    fig5.suptitle('Aerodynamic Coefficients')
    
    # Initialize and concatenate aero coefficients
    cl_all = np.array([])
    cd_all = np.array([])
    
    for phase in phases:
        cl_all = np.concatenate([cl_all, phase.get('CL', np.zeros_like(phase['time_s']))])
        cd_all = np.concatenate([cd_all, phase.get('CD', np.zeros_like(phase['time_s']))])
    
    ax12.plot(time_min, cl_all, label='CL')
    ax13.plot(time_min, cd_all, label='CD')
    
    add_transition_lines(ax12)
    add_transition_lines(ax13)
    
    ax12.set_xlabel('Time (min)')
    ax12.set_ylabel('$C_L$')
    ax12.grid(True)
    ax12.legend()
    ax12.set_ylim(0, 2)  # Set CL limits between 0 and 2

    
    ax13.set_xlabel('Time (min)')
    ax13.set_ylabel('$C_D$')
    ax13.grid(True)
    ax13.legend()
    ax13.set_ylim(0, 0.2)  # Set CD limits between 0 and 0.2
    
    plt.tight_layout()
    plt.show() 