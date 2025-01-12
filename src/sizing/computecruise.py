import numpy as np
import openmdao.api as om

class Cruise(om.ExplicitComponent):
    """Computes required thrust for steady flight at any flight path angle."""
    
    def setup(self):

        # Inputs
        self.add_input('w_mto', units='N', desc='Maximum takeoff weight')
        self.add_input('vel', units='m/s', desc='True airspeed')
        self.add_input('s_ref', units='m**2', desc='Reference wing area')
        self.add_input('cd0', desc='Zero-lift drag coefficient')
        self.add_input('ar', desc='Aspect ratio')
        self.add_input('e', desc='Oswald efficiency factor')
        self.add_input('gamma', units='rad', desc='Flight path angle (0 for cruise)')
        self.add_input('rho', units='kg/m**3', desc='Air density')

        # Outputs
        self.add_output('thrust_total', units='N', desc='Required thrust')
        self.add_output('CL', desc='Lift coefficient')
        self.add_output('CD', desc='Drag coefficient')

    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        w_mto = inputs['w_mto']
        vel = inputs['vel']
        s_ref = inputs['s_ref']
        cd0 = inputs['cd0']
        ar = inputs['ar']
        e = inputs['e']
        gamma = inputs['gamma']
        rho = inputs['rho']

        # Constants
        g = 9.806 # m/s**2

        # Compute mass at takeoff. Assume constant for cruise.
        m_mto = w_mto / g 
        
        # Required lift coefficient
        CL = (w_mto * np.cos(gamma)) / (0.5 * rho * vel**2 * s_ref)
        
        # Drag coefficient (parabolic polar)
        CD = cd0 + CL**2 / (np.pi * ar * e)

        # Required thrust
        drag = 0.5 * rho * vel**2 * s_ref * CD  # Drag
        thrust = drag + w_mto * np.sin(gamma)  # Thrust required
        
        # Pack outputs
        outputs['thrust_total'] = thrust
        outputs['CL'] = CL
        outputs['CD'] = CD
        
    def compute_partials(self, inputs, partials):


        # Unpack inputs
        w_mto = inputs['w_mto']
        vel = inputs['vel']
        s_ref = inputs['s_ref']
        cd0 = inputs['cd0']
        ar = inputs['ar']
        e = inputs['e']
        gamma = inputs['gamma']
        rho = inputs['rho']

        
        # Parials with respect to thrust_total
        partials['thrust_total', 'w_mto'] = np.sin(gamma) + (4*w_mto*np.cos(gamma)**2)/(ar*e*rho*s_ref*vel**2*np.pi)

        partials['thrust_total', 'gamma'] = w_mto*np.cos(gamma) - (4*w_mto**2*np.cos(gamma)*np.sin(gamma))/(ar*e*rho*s_ref*vel**2*np.pi)

        partials['thrust_total', 'rho'] = (s_ref*vel**2*(cd0 + (4*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**2*vel**4*np.pi)))/2 - (4*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref*vel**2*np.pi)

        partials['thrust_total', 'vel'] = rho*s_ref*vel*(cd0 + (4*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**2*vel**4*np.pi)) - (8*w_mto**2*np.cos(gamma)**2)/(ar*e*rho*s_ref*vel**3*np.pi)

        partials['thrust_total', 's_ref'] = (rho*vel**2*(cd0 + (4*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**2*vel**4*np.pi)))/2 - (4*w_mto**2*np.cos(gamma)**2)/(ar*e*rho*s_ref**2*vel**2*np.pi)

        partials['thrust_total', 'ar'] = -(2*w_mto**2*np.cos(gamma)**2)/(ar**2*e*rho*s_ref*vel**2*np.pi)

        partials['thrust_total', 'e'] = -(2*w_mto**2*np.cos(gamma)**2)/(ar*e**2*rho*s_ref*vel**2*np.pi)

        partials['thrust_total', 'cd0'] =   (rho*s_ref*vel**2)/2

        
        # Partials with respect to CL
        partials['CL', 'w_mto'] = (2*np.cos(gamma))/(rho*s_ref*vel**2)
        partials['CL', 'gamma'] = -(2*w_mto*np.sin(gamma))/(rho*s_ref*vel**2)
        partials['CL', 'vel'] = -(4*w_mto*np.cos(gamma))/(rho*s_ref*vel**3)
        partials['CL', 's_ref'] = -(2*w_mto*np.cos(gamma))/(rho*s_ref**2*vel**2)
        partials['CL', 'rho'] = -(2*w_mto*np.cos(gamma))/(rho**2*s_ref*vel**2)


        
        # Partials with respect to CD
        partials['CD', 'w_mto'] = (8*w_mto*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**2*vel**4*np.pi)

        partials['CD', 'gamma'] = -(8*w_mto**2*np.cos(gamma)*np.sin(gamma))/(ar*e*rho**2*s_ref**2*vel**4*np.pi)

        partials['CD', 'rho'] = -(8*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**3*s_ref**2*vel**4*np.pi)

        partials['CD', 'vel'] = -(16*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**2*vel**5*np.pi)

        partials['CD', 's_ref'] = -(8*w_mto**2*np.cos(gamma)**2)/(ar*e*rho**2*s_ref**3*vel**4*np.pi)

        partials['CD', 'ar'] = -(4*w_mto**2*np.cos(gamma)**2)/(ar**2*e*rho**2*s_ref**2*vel**4*np.pi)

        partials['CD', 'e'] = -(4*w_mto**2*np.cos(gamma)**2)/(ar*e**2*rho**2*s_ref**2*vel**4*np.pi)

        partials['CD', 'cd0'] = 1




if __name__ == "__main__":


    # Test case
    prob = om.Problem()

        # Create independent variable component
    ivc = om.IndepVarComp()
    ivc.add_output('w_mto', val=100000.0, units='N')
    ivc.add_output('vel', val=100.0, units='m/s')
    ivc.add_output('s_ref', val=100.0, units='m**2')
    ivc.add_output('cd0', val=0.02)
    ivc.add_output('ar', val=10.0)
    ivc.add_output('e', val=0.8)
    ivc.add_output('gamma', val=0.0, units='rad')
    ivc.add_output('rho', val=1.225, units='kg/m**3')

    
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

    print('thrust_total: ', prob['thrust_total'], 'N')
    print('CL: ', prob['CL'])
    print('CD: ', prob['CD'])

    #prob.check_partials(compact_print=True)
    

