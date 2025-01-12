from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from computehover import ComputeHoverForce
from computehover import ComputeHoverPowerEnergy
from convergecruiserange import ConvergeCruiseRange
from computeweightfractions import AircraftWeight
from computeptrainweights import ComputePTrainWeights
from computeamos import ComputeAtmos
from computecruise import Cruise
from computeductedfan import ComputeDuctedfan

class ConvergeAircraft(om.Group):
    """
    Global sizing loop that converges aircraft MTOM through power, range, and weight calculations.
    """
    
    def initialize(self):
        self.options.declare('fuel_type', default='JET_A1', values=['JET_A1', 'JET_A', 'JP_8','LNG'],
                           desc='Type of fuel to use')
    
    def setup(self):
        # Add hover power and energy calculation

        
        self.add_subsystem("atmos", ComputeAtmos(), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("hover",
                          ComputeHoverPowerEnergy(),
                          promotes_inputs=["mass_aircraft", "hover_time",
                                         "epsilon_hover", "n_lift_motors",
                                         "n_gens", "eta_motor", "eta_cable",
                                         "eta_electronics", "eta_generator",
                                         "eta_gasturbine", "rho"])
        
        self.add_subsystem("cruise", Cruise(), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("dfan",
                          ComputeDuctedfan(),
                          promotes_inputs=["*"],
                          promotes_outputs=[])

        
        # Add cruise range convergence
        self.add_subsystem("cruiserange",
                          ConvergeCruiseRange(fuel_type=self.options['fuel_type']),
                          promotes_inputs=["bat_energy_density"])
        
        # Add weight computation
        self.add_subsystem("weights",
                          AircraftWeight(),
                          promotes_outputs=["w_struct"])
        
        self.add_subsystem("ptrain_weights", ComputePTrainWeights(), 
                           promotes_inputs=["*"], 
                           promotes_outputs=["*"])
        
        # Connect hover outputs to cruise and weights
        self.connect("hover.bat_energy", "cruise.bat_energy")
        self.connect("hover.bat_power", ["weights.bat_power", "cruise.bat_power"])
        self.connect("hover.electrical_power", "weights.electrical_power")
        self.connect("hover.motor_power", "weights.motor_power")
        self.connect("hover.generator_power", "weights.generator_power")
        self.connect("hover.gasturbine_power", "weights.gasturbine_power")
        
        # Connect cruise outputs to weights
        self.connect("cruise.fuel_weight.w_fuel", "weights.w_fuel")
        
        # Connect weight back to mass_aircraft for next iteration
        self.connect("w_struct", "mass_aircraft", units="N")
        
        # Add nonlinear solver for MTOM convergence
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['iprint'] = 2
        
        # Add linear solver
        self.linear_solver = om.DirectSolver()


if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Add independent variables
    indep = prob.model.add_subsystem("indep", om.IndepVarComp())
    
    # Initial mass guess
    indep.add_output("mass_aircraft", val=2000.0, units="kg")
    
    # Mission parameters
    indep.add_output("hover_time", val=120.0, units="s")
    indep.add_output("epsilon_hover", val=0.4)
    
    # Component counts
    indep.add_output("n_lift_motors", val=8.0)
    indep.add_output("n_gens", val=2.0)
    
    # Efficiencies
    indep.add_output("eta_motor", val=0.95)
    indep.add_output("eta_cable", val=0.98)
    indep.add_output("eta_electronics", val=0.95)
    indep.add_output("eta_generator", val=0.95)
    indep.add_output("eta_gasturbine", val=0.30)
    
    # Environmental conditions
    indep.add_output("rho", val=1.225, units="kg/m**3")
    
    # bat characteristics
    indep.add_output("bat_energy_density", val=200.0, units="W*h/kg")
    
    # Add convergence group
    prob.model.add_subsystem("converge",
                            ConvergeAircraft(fuel_type='JET_A1'),
                            promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nAircraft Sizing Results:")
    print("Final MTOM:", prob.get_val("converge.w_struct")[0]/9.81, "kg")
    print("Hover Power Required:", prob.get_val("converge.hover.electrical_power")[0]/1000, "kW")
    print("bat Energy:", prob.get_val("converge.hover.bat_energy")[0]/3.6e6, "kWh")
    print("Fuel Weight:", prob.get_val("converge.cruise.fuel_weight.w_fuel")[0]/9.81, "kg")
    print("Empty Weight Fraction:", prob.get_val("converge.weights.empty_weight_fraction")[0])
    
    # Check model
    prob.check_partials(compact_print=True) 