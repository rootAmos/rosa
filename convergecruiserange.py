from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class ConvergeCruiseRange(om.ImplicitComponent):
    """
    Solves for epsilon given a target range using analytical derivatives.
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
        self.add_input("eta_p", val=1.0, units=None,
                      desc="Propulsive efficiency")
        self.add_input("eta_g", val=1.0, units=None,
                      desc="Generator efficiency")
        self.add_input("cb", val=1.0, units="J/kg",
                      desc="Battery specific energy")
        self.add_input("cp", val=1.0, units="1/s",
                      desc="Shaft power specific fuel consumption")
        self.add_input("wto_max", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("w_pay", val=1.0, units="N",
                      desc="Payload weight")
        self.add_input("f_we", val=1.0, units=None,
                      desc="Empty weight fraction")
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")

        # Add state variable (epsilon)
        self.add_output("epsilon", val=1.0, units=None,
                       desc="Required hybridization ratio",
                       lower=0.0, upper=1.0)

        # Declare partials
        self.declare_partials('epsilon', '*', method='exact')

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
        epsilon = outputs["epsilon"]
        
        # Calculate actual range
        range_actual = self._compute_range(inputs, epsilon)
        
        # Compute residual
        residuals["epsilon"] = range_actual - inputs["target_range"]

    def _compute_range(self, inputs, epsilon):
        """Helper method to compute range for given inputs and epsilon."""
        g = 9.806
        
        # Common terms
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        cb = inputs["cb"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]
        
        # Terms for readability
        term1 = (cl/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp)
        term2 = epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max)
        
        return (
            term1 * np.log(
                ((epsilon + (1 - epsilon) * cb * cp) * wto_max) / term2
            ) + (epsilon * cb * ((1 - f_we) * wto_max - w_pay)) / term2
        )
    


    def linearize(self, inputs, outputs, partials):
        """Compute analytical partial derivatives."""
        g = 9.806
        epsilon = outputs["epsilon"]
        
        # Common terms
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_i = inputs["eta_i"]
        eta_m = inputs["eta_m"]
        eta_p = inputs["eta_p"]
        eta_g = inputs["eta_g"]
        cb = inputs["cb"]
        cp = inputs["cp"]
        wto_max = inputs["wto_max"]
        w_pay = inputs["w_pay"]
        f_we = inputs["f_we"]
        
        # Intermediate terms for readability
        aero_term = (cl/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp)
        denom = epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max)
        num = (epsilon + (1 - epsilon) * cb * cp) * wto_max
        
        # Partial derivatives
        
        # With respect to epsilon
        d_log = wto_max * (1 - cb * cp) / num - (wto_max - cb * cp * (w_pay + f_we * wto_max)) / denom
        d_second = (cb * ((1-f_we) * wto_max - w_pay) * denom - 
                   epsilon * cb * ((1-f_we) * wto_max - w_pay) * 
                   (wto_max - cb * cp * (w_pay + f_we * wto_max))) / (denom ** 2)
        partials["epsilon", "epsilon"] = aero_term * d_log + d_second
        
        # With respect to target_range
        partials["epsilon", "target_range"] = -1.0
        
        # With respect to cl
        partials["epsilon", "cl"] = (1/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp) * np.log(num/denom)
        
        # With respect to cd
        partials["epsilon", "cd"] = -(cl/(cd**2)) * (eta_i * eta_m * eta_p / g) * (eta_g / cp) * np.log(num/denom)
        
        # With respect to efficiencies
        partials["epsilon", "eta_i"] = (cl/cd) * (eta_m * eta_p / g) * (eta_g / cp) * np.log(num/denom)
        partials["epsilon", "eta_m"] = (cl/cd) * (eta_i * eta_p / g) * (eta_g / cp) * np.log(num/denom)
        partials["epsilon", "eta_p"] = (cl/cd) * (eta_i * eta_m / g) * (eta_g / cp) * np.log(num/denom)
        partials["epsilon", "eta_g"] = (cl/cd) * (eta_i * eta_m * eta_p / g) * (1 / cp) * np.log(num/denom)
        
        # With respect to energy parameters
        d_cb = ((1 - epsilon) * cp * wto_max / num - 
                (1 - epsilon) * cp * (w_pay + f_we * wto_max) / denom)
        partials["epsilon", "cb"] = aero_term * d_cb + epsilon * ((1-f_we) * wto_max - w_pay) / denom
        
        d_cp = ((1 - epsilon) * cb * wto_max / num - 
                (1 - epsilon) * cb * (w_pay + f_we * wto_max) / denom)
        partials["epsilon", "cp"] = aero_term * d_cp - aero_term * np.log(num/denom) / cp
        
        # With respect to weights
        d_wto = (1/wto_max + (epsilon + (1-epsilon) * cb * cp) / num - 
                (epsilon + (1-epsilon) * cb * cp * f_we) / denom)
        partials["epsilon", "wto_max"] = aero_term * d_wto + epsilon * cb * (1-f_we) / denom
        
        d_wpay = -(1-epsilon) * cb * cp / denom
        partials["epsilon", "w_pay"] = aero_term * d_wpay - epsilon * cb / denom
        
        d_fwe = -(1-epsilon) * cb * cp * wto_max / denom
        partials["epsilon", "f_we"] = aero_term * d_fwe - epsilon * cb * wto_max / denom


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
    ivc.add_output("wto_max", val=5500.0*9.806, units="N")
    ivc.add_output("w_pay", val=200.0*9.806, units="N")
    ivc.add_output("f_we", val=0.6, units=None)
    ivc.add_output("target_range", val=1000000.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('cruise_analysis', ConvergeCruiseRange(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    # om.n2(prob)
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nResults for 1000 km range:')
    print('Required hybridization ratio (epsilon):', prob.get_val('cruise_analysis.epsilon')[0])
    
    # Try another target range (1500 km)
    prob.set_val('target_range', 1500000.0, units='m')
    prob.run_model()
    
    print('\nResults for 1500 km range:')
    print('Required hybridization ratio (epsilon):', prob.get_val('cruise_analysis.epsilon')[0]) 

   # Get other parameters
    inputs = {}
    inputs["cl"] = prob.get_val('cruise_analysis.cl')[0]
    inputs["cd"] = prob.get_val('cruise_analysis.cd')[0]
    inputs["eta_i"] = prob.get_val('cruise_analysis.eta_i')[0]
    inputs["eta_m"] = prob.get_val('cruise_analysis.eta_m')[0]
    inputs["eta_p"] = prob.get_val('cruise_analysis.eta_p')[0]
    inputs["eta_g"] = prob.get_val('cruise_analysis.eta_g')[0]
    inputs["cb"] = prob.get_val('cruise_analysis.cb')[0]
    inputs["cp"] = prob.get_val('cruise_analysis.cp')[0]
    inputs["wto_max"] = prob.get_val('cruise_analysis.wto_max')[0]
    inputs["w_pay"] = prob.get_val('cruise_analysis.w_pay')[0]
    inputs["f_we"] = prob.get_val('cruise_analysis.f_we')[0]
    epsilon = prob.get_val('cruise_analysis.epsilon')[0]

    # Verify the solution by computing range with the solved epsilon
    achieved_range = ConvergeCruiseRange._compute_range(None, inputs, epsilon)/1000
    print(f'Back calculated range: {achieved_range:.1f} km')
