import openmdao.api as om

from computecruise import Cruise
from computeductedfan import ComputeUnitThrust, ComputeDuctedfan
from convergeturbocruiserange import PropulsiveEfficiency

class CruiseMaxEfficiency(om.Group):
    """
    Group that determines propulsive efficiency at design cruise condition.
    Combines cruise performance, thrust distribution, ducted fan calculations,
    and propulsive efficiency computation.
    """
    
    def setup(self):
        # Add cruise calculation
        self.add_subsystem("cruise",
                          Cruise(),
                          promotes_inputs=["*"],
                          promotes_outputs=["thrust_total"])
        
        # Add unit thrust calculation
        self.add_subsystem("unit_thrust",
                          ComputeUnitThrust(),
                          promotes_inputs=["n_fans"],
                          promotes_outputs=["thrust_unit"])
        
        # Add ducted fan calculation
        self.add_subsystem("ducted_fan",
                          ComputeDuctedfan(),
                          promotes_inputs=["*"],
                          promotes_outputs=["eta_prplsv"])
        
        # Add propulsive efficiency calculation
        self.add_subsystem("prop_efficiency",
                          PropulsiveEfficiency(),
                          promotes_inputs=["*"],
                          promotes_outputs=["eta_pt"])
        
        # Connect total thrust to unit thrust calculator
        self.connect("thrust_total", "unit_thrust.thrust_total")

                # Add nonlinear and linear solvers
        #self.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6)
        #self.linear_solver = om.DirectSolver()
