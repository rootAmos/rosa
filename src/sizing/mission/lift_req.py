import numpy as np
import openmdao.api as om
import pdb
class Cruise(om.ExplicitComponent):
    """Computes required thrust for steady flight at any flight path angle."""
    
    def setup(self):

        # Inputs
        self.add_input('w_mto',val=1.0, units='N', desc='Maximum takeoff weight')
        self.add_input('gamma', val=1.0, units='rad', desc='Flight path angle (0 for cruise)')

        # Outputs
        self.add_output('lift_req', val=1.0, units='N', desc='Required lift')



    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        w_mto = inputs['w_mto']
        gamma = inputs['gamma']

        # Constants
        g = 9.806 # m/s**2

        # Required lift coefficient on both wings?
        lift_req = (w_mto * np.cos(gamma))
        
        # Pack outputs
        outputs['lift_req'] = lift_req
        
    def compute_partials(self, inputs, partials):

        # Unpack inputs
        w_mto = inputs['w_mto']
        gamma = inputs['gamma']

        # Parials with respect to thrust_total
        partials['lift_req', 'w_mto'] = np.cos(gamma)
        partials['lift_req', 'gamma'] = -w_mto*np.sin(gamma)


if __name__ == "__main__":


    # Test case
    prob = om.Problem()

        # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output('w_mto', val=100000.0, units='N')
    ivc.add_output('gamma', val=5, units='deg')

    
    # Add the component
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem('cruise', Cruise(), promotes=['*'])
    
    # Connect the inputs and outputs
    
    prob.setup()

    prob.model.nonlinear_solver = om.NewtonSolver()
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()



    prob.run_model()

    print('lift_req: ', prob['lift_req'], 'N')

    prob.check_partials(compact_print=True)
    

