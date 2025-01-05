from __future__ import annotations

import numpy as np
import openmdao.api as om
from typing import Any


class PropellerPerformance(om.ExplicitComponent):
    """
    Computes propeller performance using blade element momentum theory.
    """
    
    def setup(self):
        # Add inputs
        self.add_input("thrust", val=1.0, units="N",
                      desc="Required thrust")
        self.add_input("velocity", val=1.0, units="m/s",
                      desc="Flight velocity")
        self.add_input("rho", val=1.225, units="kg/m**3",
                      desc="Air density")
        self.add_input("diameter", val=2.0, units="m",
                      desc="Propeller diameter")
        
        # Add outputs
        self.add_output("power", units="W",
                       desc="Power required")
        self.add_output("efficiency", units=None,
                       desc="Propeller efficiency")
        
        # Declare partials
        self.declare_partials("*", "*", method="exact")
        
    def compute(self, inputs, outputs):
        # Extract inputs
        T = inputs["thrust"]
        V = inputs["velocity"]
        rho = inputs["rho"]
        D = inputs["diameter"]
        
        # Disk area
        A = np.pi * (D/2)**2
        
        # Compute induced velocity using momentum theory
        if V == 0:  # Hover condition
            vi = np.sqrt(T / (2 * rho * A))
            P = T * vi
            eta = 0.0  # Efficiency undefined in hover
        else:  # Forward flight
            # Solve for induced velocity using quadratic formula
            a = 1
            b = 2 * V
            c = -(T / (2 * rho * A))
            vi = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            
            # Total velocity at disk
            V_disk = np.sqrt((V + vi)**2 + vi**2)
            
            # Power and efficiency
            P = T * V_disk
            eta = T * V / P
        
        outputs["power"] = P
        outputs["efficiency"] = eta
        
    def compute_partials(self, inputs, partials):
        # Extract inputs
        T = inputs["thrust"]
        V = inputs["velocity"]
        rho = inputs["rho"]
        D = inputs["diameter"]
        A = np.pi * (D/2)**2
        
        if V == 0:  # Hover condition
            vi = np.sqrt(T / (2 * rho * A))
            
            # Partial derivatives for hover
            partials["power", "thrust"] = 1.5 * vi
            partials["power", "velocity"] = 0
            partials["power", "rho"] = -T * vi / (2 * rho)
            partials["power", "diameter"] = -T * vi / D
            
            partials["efficiency", "thrust"] = 0
            partials["efficiency", "velocity"] = 0
            partials["efficiency", "rho"] = 0
            partials["efficiency", "diameter"] = 0
            
        else:  # Forward flight
            # Compute induced velocity
            a = 1
            b = 2 * V
            c = -(T / (2 * rho * A))
            vi = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
            
            # Total velocity at disk
            V_disk = np.sqrt((V + vi)**2 + vi**2)
            
            # Partial of vi with respect to inputs
            dvi_dT = -1 / (4 * rho * A * np.sqrt(V**2 + T/(2*rho*A)))
            dvi_dV = -1 + V / np.sqrt(V**2 + T/(2*rho*A))
            dvi_drho = T / (4 * rho**2 * A * np.sqrt(V**2 + T/(2*rho*A)))
            dvi_dD = T / (4 * rho * A * D * np.sqrt(V**2 + T/(2*rho*A)))
            
            # Partial of V_disk with respect to vi
            dVdisk_dvi = ((V + vi) + vi) / V_disk
            
            # Power partials
            partials["power", "thrust"] = V_disk + T * dVdisk_dvi * dvi_dT
            partials["power", "velocity"] = T * ((V + vi) / V_disk + dVdisk_dvi * dvi_dV)
            partials["power", "rho"] = T * dVdisk_dvi * dvi_drho
            partials["power", "diameter"] = T * dVdisk_dvi * dvi_dD
            
            # Efficiency partials
            P = T * V_disk
            deta_dT = V * (P - T * partials["power", "thrust"]) / P**2
            deta_dV = (P - T * V * partials["power", "velocity"]) / P**2
            deta_drho = -T * V * partials["power", "rho"] / P**2
            deta_dD = -T * V * partials["power", "diameter"] / P**2
            
            partials["efficiency", "thrust"] = deta_dT
            partials["efficiency", "velocity"] = deta_dV
            partials["efficiency", "rho"] = deta_drho
            partials["efficiency", "diameter"] = deta_dD


if __name__ == "__main__":
    # Create problem
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output("thrust", val=1000.0, units="N")
    ivc.add_output("velocity", val=100.0, units="m/s")
    ivc.add_output("rho", val=1.225, units="kg/m**3")
    ivc.add_output("diameter", val=2.0, units="m")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('propeller', PropellerPerformance(), promotes_inputs=["*"])
    
    # Setup problem
    prob.setup()
    
    # Run the model
    prob.run_model()
    
    # Print results
    print('\nPropeller Performance Results:')
    print('Power Required:', prob.get_val('propeller.power')[0]/1000, 'kW')
    print('Efficiency:', prob.get_val('propeller.efficiency')[0])
    
    # Check partials
    prob.check_partials(compact_print=True)
