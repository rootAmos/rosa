import openmdao.api as om
import numpy as np

class ComputeTurboCruiseRange(om.ExplicitComponent):
    """
    Computes cruise range based on fuel weight using the Breguet range equation
    """
    
    def setup(self):
        # Inputs
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
        self.add_input("w_fuel", val=1.0, units="N",
                      desc="Fuel weight")
        
        # Outputs
        self.add_output('range', val=1.0, units='m', desc='Cruise range')
        
    def setup_partials(self):
        # Declare all partial derivatives
        self.declare_partials('range', '*')
        
    def compute(self, inputs, outputs):
        
        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_pe = inputs["eta_pe"]
        eta_m = inputs["eta_motor"]
        eta_pt = inputs["eta_pt"]
        eta_gen = inputs["eta_gen"]
        cp = inputs["cp"]
        w_mto = inputs["w_mto"]
        w_fuel = inputs["w_fuel"]

        g = 9.806
        
        # Breguet range equation
        range = -np.log( (w_mto - w_fuel)/ w_mto) * cl/cd * eta_gen * eta_pe * eta_m * eta_pt / g / cp;

        
        outputs['range'] = range
        
    def compute_partials(self, inputs, partials):

        cl = inputs["cl"]
        cd = inputs["cd"]
        eta_pe = inputs["eta_pe"]
        eta_m = inputs["eta_motor"]
        eta_pt = inputs["eta_pt"]
        eta_gen = inputs["eta_gen"]
        cp = inputs["cp"]
        w_mto = inputs["w_mto"]
        w_fuel = inputs["w_fuel"]

        g = 9.806
    
        # Partial derivatives
        partials["range", "w_fuel"] = -(cl*eta_gen*eta_pe*eta_m*eta_pt)/(cd*cp*g*(w_fuel - w_mto))

        
        partials["range", "cl"] = -(eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["range", "cd"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd**2*cp*g)


        partials["range", "eta_pe"] = -(cl*eta_gen*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["range", "eta_motor"] = -(cl*eta_gen*eta_pe*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["range", "eta_pt"] = -(cl*eta_gen*eta_pe*eta_m*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["range", "eta_gen"] = -(cl*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp*g)


        partials["range", "cp"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*np.log(-(w_fuel - w_mto)/w_mto))/(cd*cp**2*g)


        partials["range", "w_mto"] = (cl*eta_gen*eta_pe*eta_m*eta_pt*w_mto*(1/w_mto + (w_fuel - w_mto)/w_mto**2))/(cd*cp*g*(w_fuel - w_mto))

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
    ivc.add_output("w_fuel", val=500*9.8, units="N")
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('cruise_analysis', ComputeTurboCruiseRange(), promotes_inputs=["*"])
    
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
    
 

    import pdb
    #pdb.set_trace()

    # Run the model
    prob.run_model()
    
    # Print results
    g = 9.806
    print('\nResults for 500 kg fuel:')
    print('Range (km):', prob.get_val('cruise_analysis.range')[0]/1000)


    prob.check_partials(compact_print=True)