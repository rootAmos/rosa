from __future__ import annotations

import openmdao.api as om
from computecruiserange import ComputeCruiseRange
from convergecruisehybridization import ConvergeCruiseHybridization


class CruiseRangeGroup(om.Group):
    """
    Group that combines range calculation and hybridization ratio convergence.
    """
    
    def setup(self):
        # Add the range calculator
        self.add_subsystem("range_calc", 
                          ComputeCruiseRange(),
                          promotes_inputs=["cl", "cd", "eta_i", "eta_m", "eta_p", 
                                         "eta_g", "epsilon", "cb", "cp", "wt0_max",
                                         "w_pay", "f_we"])
        
        # Add the hybridization solver
        self.add_subsystem("hybridization_solver",
                          ConvergeCruiseHybridization(),
                          promotes_inputs=["target_range"])
        
        # Connect range_calc output to hybridization_solver input
        self.connect("range_calc.range_total", "hybridization_solver.actual_range")
        
        # Connect epsilon back to range calculator (creating the cycle)
        self.connect("hybridization_solver.epsilon", "epsilon")
        
        # Add a nonlinear solver
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 100
        self.nonlinear_solver.options['atol'] = 1e-6
        self.nonlinear_solver.options['rtol'] = 1e-6
        
        # Add a linear solver
        self.linear_solver = om.DirectSolver()


if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("cl", val=0.5, units=None)
    ivc.add_output("cd", val=0.03, units=None)
    ivc.add_output("eta_i", val=0.95, units=None)
    ivc.add_output("eta_m", val=0.95, units=None)
    ivc.add_output("eta_p", val=0.85, units=None)
    ivc.add_output("eta_g", val=0.95, units=None)
    ivc.add_output("cb", val=400*3600, units="J/kg")
    ivc.add_output("cp", val=0.3/60/60/1000, units="1/s")
    ivc.add_output("wt0_max", val=5500.0*9.806, units="N")
    ivc.add_output("w_pay", val=200.0*9.806, units="N")
    ivc.add_output("f_we", val=0.6, units=None)
    ivc.add_output("target_range", val=1000000.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('cruise_analysis', CruiseRangeGroup(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nResults for 1000 km range:')
    print('Hybridization ratio (epsilon):', prob.get_val('cruise_analysis.hybridization_solver.epsilon')[0])
    print('Actual range achieved:', prob.get_val('cruise_analysis.range_calc.range_total')[0]/1000, 'km')
    
    # Try another target range (1500 km)
    prob.set_val('target_range', 1500000.0, units='m')
    prob.run_model()
    
    print('\nResults for 1500 km range:')
    print('Hybridization ratio (epsilon):', prob.get_val('cruise_analysis.hybridization_solver.epsilon')[0])
    print('Actual range achieved:', prob.get_val('cruise_analysis.range_calc.range_total')[0]/1000, 'km') 