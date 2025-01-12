from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class Aerodynamics(om.ExplicitComponent):
    """
    Computes aerodynamic coefficients for the aircraft.
    """
    
    def setup(self):
        # Add inputs
        self.add_input("mach", val=0.1, units=None,
                      desc="Mach number")
        self.add_input("alpha", val=0.0, units="rad",
                      desc="Angle of attack")
        self.add_input("s_ref", val=1.0, units="m**2",
                      desc="Reference wing area")
        self.add_input("ar_w", val=1.0, units=None,
                      desc="Wing aspect ratio")
        self.add_input("e_w", val=1.0, units=None,
                      desc="Wing Oswald efficiency factor")
        self.add_input("cd0", val=0.02, units=None,
                      desc="Parasitic drag coefficient")
        
        # Add outputs
        self.add_output("cl", units=None,
                       desc="Lift coefficient")
        self.add_output("cd", units=None,
                       desc="Drag coefficient")
        
        # Declare partials
        self.declare_partials("cl", ["alpha"], method="exact")
        self.declare_partials("cd", ["cl", "ar_w", "e_w", "cd0"], method="exact")
        
    def compute(self, inputs, outputs):
        
        """Compute lift and drag coefficients."""
        alpha = inputs["alpha"]
        ar_w = inputs["ar_w"]
        e_w = inputs["e_w"]
        cd0 = inputs["cd0"]
        
        # Compute lift coefficient (assuming linear region)
        outputs["cl"] = 2 * np.pi * alpha
        
        # Compute drag coefficient
        outputs["cd"] = cd0 + (outputs["cl"] ** 2) / (np.pi * ar_w * e_w)
        
    def compute_partials(self, inputs, partials):
        """Compute partial derivatives analytically."""
        alpha = inputs["alpha"]
        ar_w = inputs["ar_w"]
        e_w = inputs["e_w"]
        
        # Partial of cl with respect to alpha
        partials["cl", "alpha"] = 2 * np.pi
        
        # Partial of cd with respect to inputs
        cl = 2 * np.pi * alpha
        partials["cd", "cl"] = 2 * cl / (np.pi * ar_w * e_w)
        partials["cd", "ar_w"] = -(cl ** 2) / (np.pi * (ar_w ** 2) * e_w)
        partials["cd", "e_w"] = -(cl ** 2) / (np.pi * ar_w * (e_w ** 2))
        partials["cd", "cd0"] = 1.0


if __name__ == "__main__":
    import openmdao.api as om
    
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("mach", val=0.1)
    ivc.add_output("alpha", val=np.radians(2.0), units="rad")
    ivc.add_output("s_ref", val=25.0, units="m**2")
    ivc.add_output("ar_w", val=10.0)
    ivc.add_output("e_w", val=0.85)
    ivc.add_output("cd0", val=0.02)
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('aero', Aerodynamics(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nAerodynamic Results:')
    print('CL:', prob.get_val('aero.cl')[0])
    print('CD:', prob.get_val('aero.cd')[0])
    
    # Check partials
    prob.check_partials(compact_print=True) 