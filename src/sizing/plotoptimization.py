import openmdao.api as om
import matplotlib.pyplot as plt
import numpy as np

def plot_optimization_history(case_recorder_filename):
    """
    Plot the evolution of fuel weight, range, and MTOM margin during optimization
    """
    cr = om.CaseReader(case_recorder_filename)
    driver_cases = cr.get_cases('driver')
    
    # Extract data
    iterations = range(len(driver_cases))
    fuel_weights = []
    ranges = []
    mtom_margins = []
    
    for case in driver_cases:
        fuel_weights.append(case.get_design_vars()['w_fuel'][0]/9.81)  # Convert to kg
        ranges.append(case.get_objectives()['range'][0]/-1000/10000)  # Convert to km
        mtom_margins.append(case.get_constraints()['mtom_margin'][0]/9.81)  # Convert to kg
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot fuel weight
    ax1.plot(iterations, fuel_weights, 'o-', linewidth=2)
    ax1.set_ylabel('Fuel Weight [kg]')
    ax1.grid(True)
    ax1.set_title('Fuel Weight')
    
    # Plot range
    ax2.plot(iterations, ranges, 'o-', linewidth=2)
    ax2.set_ylabel('Range [km]')
    ax2.grid(True)
    ax2.set_title('Range')
    
    # Plot MTOM margin
    ax3.plot(iterations, mtom_margins, 'o-', linewidth=2)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('MTOM Margin [kg]')
    ax3.grid(True)
    ax3.set_title('MTOM Set - MTOM Calc')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_optimization_history('cases.sql') 