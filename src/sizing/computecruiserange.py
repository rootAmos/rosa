from __future__ import annotations

import numpy as np
import openmdao.api as om
from scipy.optimize import brentq
from typing import Any


class ComputeCruiseRange(om.ExplicitComponent):
    """
    Computes either the range given epsilon, or epsilon given a target range.
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
        self.add_input("wt0_max", val=1.0, units="N",
                      desc="Maximum takeoff weight")
        self.add_input("w_pay", val=1.0, units="N",
                      desc="Payload weight")
        self.add_input("f_we", val=1.0, units=None,
                      desc="Empty weight fraction")
        self.add_input("target_range", val=1.0, units="m",
                      desc="Target range")

        # Add outputs
        self.add_output("epsilon", val=0.3, units=None,
                       desc="Required hybridization ratio")

        # Declare partials
        self.declare_partials('epsilon', ['cl', 'cd', 'eta_i', 'eta_m', 'eta_p', 
                                        'eta_g', 'cb', 'cp', 'wt0_max', 'w_pay', 
                                        'f_we', 'target_range'])

    def compute(self, inputs, outputs):
        """Compute epsilon for target range using root finding."""
        
        def range_residual(epsilon):
            """Compute difference between actual and target range."""
            g = 9.806
            
            # Calculate actual range for given epsilon
            range_actual = (
                (inputs["cl"] / inputs["cd"]) * (inputs["eta_i"] * inputs["eta_m"] * inputs["eta_p"] / g)
                * np.log(
                    ((epsilon + (1 - epsilon) * inputs["cb"] * inputs["cp"]) * inputs["wt0_max"])
                    / (epsilon * inputs["wt0_max"] + (1 - epsilon) * inputs["cb"] * inputs["cp"] 
                       * (inputs["w_pay"] + inputs["f_we"] * inputs["wt0_max"]))
                ) * inputs["eta_g"] / inputs["cp"]
                + (epsilon * inputs["cb"] * ((1 - inputs["f_we"]) * inputs["wt0_max"] - inputs["w_pay"]))
                / (epsilon * inputs["wt0_max"] + (1 - epsilon) * inputs["cb"] * inputs["cp"] 
                   * (inputs["w_pay"] + inputs["f_we"] * inputs["wt0_max"]))
            )
            
            return range_actual - inputs["target_range"]
        
        # Find epsilon that gives target range (between 0 and 1)
        outputs["epsilon"] = brentq(range_residual, 0.001, 0.999)

    def compute_partials(self, inputs, partials):
        """Compute partial derivatives using implicit function theorem."""
        g = 9.806
        epsilon = self._compute_epsilon(inputs)  # Current converged epsilon
        
        # Get partial of residual with respect to epsilon
        dR_deps = self._compute_range_partial_epsilon(inputs, epsilon)
        
        # Compute partials for each input
        for input_name in inputs:
            # Compute partial of residual with respect to input using finite difference
            inp_val = inputs[input_name]
            delta = 1e-6 * (1.0 + abs(inp_val))
            
            # Perturbed inputs
            inputs_plus = inputs.copy()
            inputs_plus[input_name] += delta
            
            # Compute residual difference
            R1 = self._compute_range(inputs, epsilon)
            R2 = self._compute_range(inputs_plus, epsilon)
            
            dR_dx = (R2 - R1) / delta
            
            # Apply implicit function theorem: deps/dx = -(dR/deps)^-1 * dR/dx
            partials['epsilon', input_name] = -dR_dx / dR_deps

    def _compute_epsilon(self, inputs):
        """Helper method to compute epsilon using root finding."""
        def range_residual(eps):
            return self._compute_range(inputs, eps) - inputs["target_range"]
        
        return brentq(range_residual, 0.001, 0.999)

    def _compute_range(self, inputs, epsilon):
        """Helper method to compute range for given inputs and epsilon."""
        g = 9.806
        
        return (
            (inputs["cl"] / inputs["cd"]) * (inputs["eta_i"] * inputs["eta_m"] * inputs["eta_p"] / g)
            * np.log(
                ((epsilon + (1 - epsilon) * inputs["cb"] * inputs["cp"]) * inputs["wt0_max"])
                / (epsilon * inputs["wt0_max"] + (1 - epsilon) * inputs["cb"] * inputs["cp"] 
                   * (inputs["w_pay"] + inputs["f_we"] * inputs["wt0_max"]))
            ) * inputs["eta_g"] / inputs["cp"]
            + (epsilon * inputs["cb"] * ((1 - inputs["f_we"]) * inputs["wt0_max"] - inputs["w_pay"]))
            / (epsilon * inputs["wt0_max"] + (1 - epsilon) * inputs["cb"] * inputs["cp"] 
               * (inputs["w_pay"] + inputs["f_we"] * inputs["wt0_max"]))
        )

    def _compute_range_partial_epsilon(self, inputs, epsilon):
        """Helper method to compute partial of range with respect to epsilon."""
        delta = 1e-6 * (1.0 + abs(epsilon))
        R1 = self._compute_range(inputs, epsilon)
        R2 = self._compute_range(inputs, epsilon + delta)
        return (R2 - R1) / delta


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
    model.add_subsystem('cruise_analysis', ComputeCruiseRange(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Generate N2 diagram
    om.n2(prob)
    
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