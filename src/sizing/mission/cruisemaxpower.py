import openmdao.api as om

from computecruise import Cruise
from ComputeDuctedFan import ComputeUnitThrust, ComputeDuctedFan

class CruiseMaxpower(om.Group):
    """
    Group that determines maximum shaft power needed for maximum speed operation.
    Combines cruise performance, thrust distribution, and ducted fan calculations.
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
                          ComputeDuctedFan(),
                          promotes_inputs=["*"],
                          promotes_outputs=["p_shaft_unit"])
        
        # Connect total thrust to unit thrust calculator
        self.connect("thrust_total", "unit_thrust.thrust_total")
        

                # Add nonlinear and linear solvers
        #self.nonlinear_solver = om.NonlinearBlockGS(maxiter=100, atol=1e-6, rtol=1e-6)
        #self.linear_solver = om.DirectSolver()
