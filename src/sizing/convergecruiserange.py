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
                      desc="bat specific energy")
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
        

        partials["epsilon", "epsilon"] = -(cl*eta_i*eta_m*eta_p*((cb*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (cb*epsilon*(wto_max - cb*cp*(w_pay + f_we*wto_max))*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2 + (eta_g*((wto_max*(cb*cp - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) + (wto_max*(wto_max - cb*cp*(w_pay + f_we*wto_max))*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2)*(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)))/(cp*wto_max*(epsilon - cb*cp*(epsilon - 1)))))/(cd*g)

        # With respect to target_range
        partials["epsilon", "target_range"] = -1.0
        
        # With respect to cl
        partials["epsilon", "cl"] = (eta_i*eta_m*eta_p*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp - (cb*epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd*g)

        
        # With respect to cd
        partials["epsilon", "cd"] = -(cl*eta_i*eta_m*eta_p*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp - (cb*epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd**2*g)

        
        # With respect to efficiencies
        partials["epsilon", "eta_i"] = (cl*eta_m*eta_p*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp - (cb*epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd*g)

        partials["epsilon", "eta_m"] = (cl*eta_i*eta_p*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp - (cb*epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd*g)

        partials["epsilon", "eta_p"] = (cl*eta_i*eta_m*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp - (cb*epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd*g)

        partials["epsilon", "eta_g"] = (cl*eta_i*eta_m*eta_p*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/(cd*cp*g)

        partials["epsilon", "cb"] = -(cl*eta_i*eta_m*eta_p*((epsilon*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) + (eta_g*((cp*wto_max*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (cp*wto_max*(w_pay + f_we*wto_max)*(epsilon - cb*cp*(epsilon - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2)*(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)))/(cp*wto_max*(epsilon - cb*cp*(epsilon - 1))) + (cb*cp*epsilon*(w_pay + f_we*wto_max)*(w_pay + wto_max*(f_we - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2))/(cd*g)
        partials["epsilon", "cp"] = -(cl*eta_i*eta_m*eta_p*((eta_g*np.log((wto_max*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))))/cp**2 + (eta_g*((cb*wto_max*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (cb*wto_max*(w_pay + f_we*wto_max)*(epsilon - cb*cp*(epsilon - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2)*(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)))/(cp*wto_max*(epsilon - cb*cp*(epsilon - 1))) + (cb**2*epsilon*(w_pay + f_we*wto_max)*(w_pay + wto_max*(f_we - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2))/(cd*g)


        partials["epsilon", "wto_max"] = (cl*eta_i*eta_m*eta_p*((cb*epsilon*(epsilon - cb*cp*f_we*(epsilon - 1))*(w_pay + wto_max*(f_we - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2 - (cb*epsilon*(f_we - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) + (eta_g*((epsilon - cb*cp*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (wto_max*(epsilon - cb*cp*f_we*(epsilon - 1))*(epsilon - cb*cp*(epsilon - 1)))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2)*(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)))/(cp*wto_max*(epsilon - cb*cp*(epsilon - 1)))))/(cd*g)
        partials["epsilon", "w_pay"] = -(cl*eta_i*eta_m*eta_p*((cb*epsilon)/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (cb*eta_g*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) + (cb**2*cp*epsilon*(w_pay + wto_max*(f_we - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2))/(cd*g)
        partials["epsilon", "f_we"] = -(cl*eta_i*eta_m*eta_p*((cb*epsilon*wto_max)/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) - (cb*eta_g*wto_max*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1)) + (cb**2*cp*epsilon*wto_max*(w_pay + wto_max*(f_we - 1))*(epsilon - 1))/(epsilon*wto_max - cb*cp*(w_pay + f_we*wto_max)*(epsilon - 1))**2))/(cd*g)



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

    prob.check_partials(compact_print=True)