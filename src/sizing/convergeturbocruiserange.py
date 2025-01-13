from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any

from openmdao.core.explicitcomponent import ExplicitComponent

class PropulsiveEfficiency(ExplicitComponent):
    """Multiplies fan, duct, and propulsive efficiencies to get total propulsive efficiency."""
    
    def setup(self):
        # Inputs
        self.add_input('eta_fan', val=1.0, desc='Fan efficiency')
        self.add_input('eta_duct', val=1.0, desc='Duct efficiency')
        self.add_input('eta_prplsv', val=1.0, desc='isentropic propulsive efficiency')
        
        # Output
        self.add_output('eta_pt', val=1.0, desc='Total propulsive efficiency')
        
        # Declare partials
        self.declare_partials('eta_pt', ['eta_fan', 'eta_duct', 'eta_prplsv'])
        
    def compute(self, inputs, outputs):
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']
        eta_prplsv = inputs['eta_prplsv']
        
        outputs['eta_pt'] = eta_fan * eta_duct * eta_prplsv
        
    def compute_partials(self, inputs, partials):
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']
        eta_prplsv = inputs['eta_prplsv']
        
        partials['eta_pt', 'eta_fan'] = eta_duct * eta_prplsv
        partials['eta_pt', 'eta_duct'] = eta_fan * eta_prplsv
        partials['eta_pt', 'eta_prplsv'] = eta_fan * eta_duct
        
class ConvergeTurboCruiseRange(om.ImplicitComponent):
    """
    Solves for fuel required given a target range using analytical derivatives.
    """

    def setup(self):
        # Add input variables (parameters) - all with val=1 and units
        self.add_input("cl", val=1.0, units=None,
                      desc="Lift coefficient")
        self.add_input("cd", val=1.0, units=None,
                      desc="Drag coefficient")
        self.add_input("eta_pe", val=1.0, units=None,
                      desc="Inverter efficiency")
        self.add_input("eta_motor", val=1.0, units=None,
                      desc="Motor efficiency")
        self.add_input("eta_pt", val=1.0, units=None,
                      desc="total propulsor efficiency incl fan, duct, and isentropic losses")
        self.add_input("eta_gen", val=1.0, units=None,
                      desc="Generator efficiency")
        self.add_input("cp", val=1.0, units="kg/W/s",
                      desc="Shaft power specific fuel consumption")
        self.add_input("w_mto", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")

        # Add state variable (epsilon)
        self.add_output("w_fuel", val=1.0, units="N", 
                       desc="fuel mass required")

        # Declare partials
        self.declare_partials('w_fuel', '*', method='exact')

        # Add nonlinear solver
        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        #self.nonlinear_solver = om.NonlinearBlockGS()
        #self.linear_solver = om.DirectSolver()

        #self.nonlinear_solver.options['maxiter'] = 100
        #self.nonlinear_solver.options['atol'] = 1e-8
        #self.nonlinear_solver.options['rtol'] = 1e-8
        #self.nonlinear_solver.options['iprint'] = 2  # Print iteration info
        
        # Add line search to help with convergence
        #self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        #self.nonlinear_solver.linesearch.options['bound_enforcement'] = 'scalar'
        
        # Add linear solver

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Compute residual between actual and target range."""
        g = 9.806

        w_fuel = outputs["w_fuel"]
        
        # Calculate actual range
        range_actual = self._compute_range(inputs, w_fuel)
        
        # Compute residual
        residuals["w_fuel"] = range_actual - inputs["target_range"]

    def _compute_range(self, inputs, w_fuel):
        """Helper method to compute range for given inputs and epsilon."""
        g = 9.806
        
        # Common terms
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_pe = inputs["eta_pe"]
        eta_m = inputs["eta_motor"]
        eta_pt = inputs["eta_pt"]
        eta_gen = inputs["eta_gen"]
        cp = inputs["cp"]
        w_mto = inputs["w_mto"]
        
        range = -np.log( (w_mto - w_fuel)/ w_mto) * cl/cd * eta_gen * eta_pe * eta_m * eta_pt / g / cp;

        return range
    


    def linearize(self, inputs, outputs, partials):
        """Compute analytical partial derivatives."""
        g = 9.806

        w_fuel = outputs["w_fuel"]
        
        # Common terms
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_pe = inputs["eta_pe"]
        eta_m = inputs["eta_motor"]
        eta_pt = inputs["eta_pt"]
        eta_gen = inputs["eta_gen"]
        cp = inputs["cp"]
        w_mto = inputs["w_mto"]

        partials["w_fuel", "w_fuel"] = -(cl*eta_gen*eta_pe*eta_m*eta_pt)/(cd*cp*g*(w_fuel - w_mto))

        partials["w_fuel", "target_range"] = -1.0
        
        partials["w_fuel", "cl"] = -(eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["w_fuel", "cd"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd**2*cp*g)


        partials["w_fuel", "eta_pe"] = -(cl*eta_gen*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["w_fuel", "eta_motor"] = -(cl*eta_gen*eta_pe*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["w_fuel", "eta_pt"] = -(cl*eta_gen*eta_pe*eta_m*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["w_fuel", "eta_gen"] = -(cl*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["w_fuel", "cp"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp**2*g)


        partials["w_fuel", "w_mto"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*w_mto*(1/w_mto + (w_fuel - w_mto)/w_mto**2))/(cd*cp*g*(w_fuel - w_mto))


if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem(reports=False)
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("cl", val=0.5, units=None)
    ivc.add_output("cd", val=0.03, units=None)
    ivc.add_output("eta_pe", val=0.95, units=None)
    ivc.add_output("eta_motor", val=0.95, units=None)
    ivc.add_output("eta_pt", val=0.85, units=None)
    ivc.add_output("eta_gen", val=0.95, units=None)
    ivc.add_output("cp", val=0.5/60/60/1000, units="kg/W/s")
    ivc.add_output("w_mto", val=5000.0*9.806, units="N")
    ivc.add_output("target_range", val=800*1000.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('cruise_analysis', ConvergeTurboCruiseRange(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.linear_solver = om.DirectSolver()
    model.nonlinear_solver.options['maxiter'] = 100
    model.nonlinear_solver.options['atol'] = 1e-8
    model.nonlinear_solver.options['rtol'] = 1e-8
    model.nonlinear_solver.options['iprint'] = 2  # Print iteration info
    
    # Generate N2 diagram
    #om.n2(prob)

    prob.set_val('cruise_analysis.w_fuel', 500*9.8)
    
 

    import pdb
    #pdb.set_trace()

    # Run the model
    prob.run_model()
    
    # Print results
    g = 9.806
    print('\nResults for 1000 km range:')
    print('Required fuel (kg):', prob.get_val('cruise_analysis.w_fuel')[0]/g)
    
    # Try another target range (1500 km)
    prob.set_val('target_range', 1500000.0, units='m')
    prob.run_model()
    
    print('\nResults for 1500 km range:')
    print('Required fuel (kg):', prob.get_val('cruise_analysis.w_fuel')[0]/g    )

   # Get other parameters

    inputs = {}
    inputs["cl"] = prob.get_val('cruise_analysis.cl')[0]
    inputs["cd"] = prob.get_val('cruise_analysis.cd')[0]
    inputs["eta_pe"] = prob.get_val('cruise_analysis.eta_pe')[0]
    inputs["eta_motor"] = prob.get_val('cruise_analysis.eta_motor')[0]
    inputs["eta_pt"] = prob.get_val('cruise_analysis.eta_pt')[0]
    inputs["eta_gen"] = prob.get_val('cruise_analysis.eta_gen')[0]
    inputs["cp"] = prob.get_val('cruise_analysis.cp')[0]
    inputs["w_mto"] = prob.get_val('cruise_analysis.w_mto')[0]
    inputs["target_range"] = prob.get_val('cruise_analysis.target_range')[0]
    w_fuel = prob.get_val('cruise_analysis.w_fuel')[0]

    # Verify the solution by computing range with the solved epsilon
    achieved_range = ConvergeTurboCruiseRange._compute_range(None, inputs, w_fuel)/1000
    print(f'Back calculated range: {achieved_range:.1f} km')
    prob.check_partials(compact_print=True)