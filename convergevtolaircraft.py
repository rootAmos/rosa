from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any
from hoverenergy import HoverEnergyGroup
from convergecruiserange import ConvergeCruiseRange
from computeweights import AircraftWeight


class ConvergeAircraft(om.Group):
    """
    Global sizing loop that converges aircraft MTOM through power, range, and weight calculations.
    """
    
    def initialize(self):
        self.options.declare('fuel_type', default='JET_A1', values=['JET_A1', 'JET_A', 'JP_8'],
                           desc='Type of fuel to use')
    
    def setup(self):
        # Create cycle for MTOM convergence
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        
        # Add hover power and energy calculation
        cycle.add_subsystem("hover",
                          HoverEnergyGroup(),
                          promotes_inputs=["mass_aircraft", "hover_time",
                                         "epsilon_hover", "n_lift_motors",
                                         "n_generators", "eta_motor", "eta_cable",
                                         "eta_electronics", "eta_generator",
                                         "eta_gasturbine", "rho"])
        
        # Add cruise range convergence
        cycle.add_subsystem("cruise",
                          ConvergeCruiseRange(fuel_type=self.options['fuel_type']),
                          promotes_inputs=["battery_energy_density"])
        
        # Add weight computation
        cycle.add_subsystem("weights",
                          AircraftWeight(),
                          promotes_outputs=["w_struct"])
        
        # Connect hover outputs to cruise and weights
        cycle.connect("hover.battery_energy", "cruise.battery_energy")
        cycle.connect("hover.battery_power", ["weights.battery_power", "cruise.battery_power"])
        cycle.connect("hover.electrical_power", "weights.electrical_power")
        cycle.connect("hover.motor_power", "weights.motor_power")
        cycle.connect("hover.generator_power", "weights.generator_power")
        cycle.connect("hover.gasturbine_power", "weights.gasturbine_power")
        
        # Connect cruise outputs to weights
        cycle.connect("cruise.fuel_weight.w_fuel", "weights.w_fuel")
        
        # Connect weight back to mass_aircraft for next iteration
        cycle.connect("w_struct", "mass_aircraft", units="N")
        
        # Add nonlinear solver for MTOM convergence
        cycle.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        cycle.nonlinear_solver.options['maxiter'] = 50
        cycle.nonlinear_solver.options['iprint'] = 2
        
        # Add linear solver
        cycle.linear_solver = om.DirectSolver()


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
    indep.add_output("n_generators", val=2.0)
    
    # Efficiencies
    indep.add_output("eta_motor", val=0.95)
    indep.add_output("eta_cable", val=0.98)
    indep.add_output("eta_electronics", val=0.95)
    indep.add_output("eta_generator", val=0.95)
    indep.add_output("eta_gasturbine", val=0.30)
    
    # Environmental conditions
    indep.add_output("rho", val=1.225, units="kg/m**3")
    
    # Battery characteristics
    indep.add_output("battery_energy_density", val=200.0, units="W*h/kg")
    
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
    print("Battery Energy:", prob.get_val("converge.hover.battery_energy")[0]/3.6e6, "kWh")
    print("Fuel Weight:", prob.get_val("converge.cruise.fuel_weight.w_fuel")[0]/9.81, "kg")
    print("Empty Weight Fraction:", prob.get_val("converge.weights.empty_weight_fraction")[0])
    
    # Check model
    prob.check_partials(compact_print=True) 