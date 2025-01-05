import numpy as np  
import openmdao.api as om



class ComputeThrustFromAcc(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n', default=1, desc='number of data points')
        self.options.declare('g', default=9.806, desc='gravitational acceleration')


    def setup(self):

        # Inputs    
        self.add_input('drag', val= np.ones(self.options['n']), desc='drag force', units='N')
        self.add_input('mass', val=1, desc='mass', units='kg')
        self.add_input('gamma', val= np.ones(self.options['n']) * np.pi/180, desc='flight path angle', units='rad')
        self.add_input('acc', val= np.zeros(self.options['n']), desc='acceleration in longitudinal direction of body-fixed frame', units='m/s**2')

        # Outputs
        self.add_output('total_thrust_req', val= np.ones(self.options['n']), desc='thrust force', units='N', lower=1e-3)

    def setup_partials(self):
        self.declare_partials('total_thrust_req', 'drag')    
        self.declare_partials('total_thrust_req', 'mass')
        self.declare_partials('total_thrust_req', 'gamma')
        self.declare_partials('total_thrust_req', 'acc')

    def compute(self, inputs, outputs):

        # Unpack inputs
        Drag = inputs['drag']
        mass = inputs['mass']
        gamma = inputs['gamma']
        acc = inputs['acc']

        # Unpack constants
        g = self.options['g']

        # Accelerations in the body-fixed frame
        thrust = mass * acc + Drag + mass * g * np.sin(gamma)

        if np.any(thrust < 1e-3):
            print('Negative thrust detected')

        # Check for negative thrust
        thrust = np.where(thrust < 1e-3, 1e-3, thrust)
    
        # Pack outputs
        outputs['total_thrust_req'] = thrust

    def compute_partials(self, inputs, J):

        # Unpack inputs
        mass = inputs['mass']
        acc = inputs['acc']
        gamma = inputs['gamma']

        # Unpack constants
        g = self.options['g']
        n = self.options['n']

        # Compute partials
        J['total_thrust_req', 'drag'] = np.eye(n) 
        J['total_thrust_req', 'mass'] = acc 
        J['total_thrust_req', 'gamma'] = np.eye(n) * mass * g * np.cos(gamma)
        J['total_thrust_req', 'acc'] = mass * np.eye(n)


if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    n = 10

    ivc = om.IndepVarComp()
    ivc.add_output('acc', 0.1 * np.ones(n), units='m/s**2')
    ivc.add_output('drag', 1000 * np.ones(n), units='N')
    ivc.add_output('mass', 8600, units='kg')
    ivc.add_output('gamma', 0 * np.ones(n), units='rad')


    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('ComputeThrustFromAcc', ComputeThrustFromAcc(n=n), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('thrust = ', p['ComputeThrustFromAcc.total_thrust_req'])
