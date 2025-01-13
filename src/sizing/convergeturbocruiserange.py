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
        self.add_input("eta_i", val=1.0, units=None,
                      desc="Inverter efficiency")
        self.add_input("eta_m", val=1.0, units=None,
                      desc="Motor efficiency")
        self.add_input("eta_pt", val=1.0, units=None,
                      desc="total propulsor efficiency incl fan, duct, and isentropic losses")
        self.add_input("eta_g", val=1.0, units=None,
                      desc="Generator efficiency")
        self.add_input("cp", val=1.0, units="kg/W/s",
                      desc="Shaft power specific fuel consumption")
        self.add_input("wto_max", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")

        # Add state variable (epsilon)
        self.add_output("w_fuel", val=1.0, units="N",
                       desc="fuel mass required")

        # Declare partials
        self.declare_partials('w_fuel', '*', method='exact')

        # Add nonlinear solver
        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['maxiter'] = 50
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['iprint'] = 2  # Print iteration info
        
        # Add line search to help with convergence
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        
        # Add linear solver
        self.linear_solver = om.DirectSolver()

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
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_pt = inputs["eta_pt"]
        eta_g = inputs["eta_g"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]
        
        range = -np.log( (wto_max - w_fuel)/ wto_max) * cl/cd * eta_g * eta_i * eta_m * eta_pt / g / cp;

        return range
    


    def linearize(self, inputs, outputs, partials):
        """Compute analytical partial derivatives."""
        g = 9.806

        w_fuel = outputs["w_fuel"]
        
        # Common terms
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_pt = inputs["eta_pt"]
        eta_g = inputs["eta_g"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]

        partials["w_fuel", "w_fuel"] = (cl*eta_g*eta_i*eta_m*eta_pt)/(cd*cp*(wto_max - w_fuel))

        partials["w_fuel", "target_range"] = -1.0
        
        partials["w_fuel", "cl"] = -(eta_g*eta_i*eta_m*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd*cp*g)

        partials["w_fuel", "cd"] = (cl*eta_g*eta_i*eta_m*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd**2*cp*g)

        partials["w_fuel", "eta_i"] = -(cl*eta_g*eta_m*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd*cp*g)

        partials["w_fuel", "eta_m"] = -(cl*eta_g*eta_i*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd*cp*g)

        partials["w_fuel", "eta_pt"] = -(cl*eta_g*eta_i*eta_m*np.log((wto_max - w_fuel)/wto_max))/(cd*cp*g)

        partials["w_fuel", "eta_g"] = -(cl*eta_i*eta_m*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd*cp*g)

        partials["w_fuel", "cp"] = (cl*eta_g*eta_i*eta_m*eta_pt*np.log((wto_max - w_fuel)/wto_max))/(cd*cp**2*g)

        partials["w_fuel", "wto_max"] = (cl*eta_g*eta_i*eta_m*eta_pt*wto_max*(1/wto_max + (w_fuel - wto_max)/wto_max**2))/(cd*cp*g*(w_fuel - wto_max))

if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem(reports=False)
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("cl", val=0.5, units=None)
    ivc.add_output("cd", val=0.03, units=None)
    ivc.add_output("eta_i", val=0.95, units=None)
    ivc.add_output("eta_m", val=0.95, units=None)
    ivc.add_output("eta_pt", val=0.85, units=None)
    ivc.add_output("eta_g", val=0.95, units=None)
    ivc.add_output("cp", val=0.5/60/60/1000, units="kg/W/s")
    ivc.add_output("wto_max", val=5500.0*9.806, units="N")
    ivc.add_output("target_range", val=1030*1000.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('cruise_analysis', ConvergeTurboCruiseRange(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    # om.n2(prob)
    
 

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
    inputs["eta_i"] = prob.get_val('cruise_analysis.eta_i')[0]
    inputs["eta_m"] = prob.get_val('cruise_analysis.eta_m')[0]
    inputs["eta_pt"] = prob.get_val('cruise_analysis.eta_pt')[0]
    inputs["eta_g"] = prob.get_val('cruise_analysis.eta_g')[0]
    inputs["cp"] = prob.get_val('cruise_analysis.cp')[0]
    inputs["wto_max"] = prob.get_val('cruise_analysis.wto_max')[0]
    inputs["target_range"] = prob.get_val('cruise_analysis.target_range')[0]
    w_fuel = prob.get_val('cruise_analysis.w_fuel')[0]

    # Verify the solution by computing range with the solved epsilon
    achieved_range = ConvergeTurboCruiseRange._compute_range(None, inputs, w_fuel)/1000
    print(f'Back calculated range: {achieved_range:.1f} km')
    prob.check_partials(compact_print=True)