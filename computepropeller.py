import numpy as np  
import openmdao.api as om



class Propeller(om.ExplicitComponent):
    """
    Compute the power required for a propeller.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    
    """

    def setup(self):

        # Inputs    
        self.add_input('d_blade', val=1, desc='blade diameter', units='m')
        self.add_input('d_hub', val=0.5, desc='hub diameter', units='m')
        self.add_input('rho', val= 1, desc='air density', units='kg/m**3')
        self.add_input('thrust_total', val= 1, desc='total aircraft thrust required', units='N')
        self.add_input('n_motors', val=2, desc='number of engines', units=None)
        self.add_input('vel', val= 1, desc='true airspeed', units='m/s')
        self.add_input('eta_prop', val=1, desc='propeller efficiency', units=None) 
        self.add_input('eta_hover', val=1, desc='hover efficiency', units=None)

        # Outputs
        self.add_output('shaft_power_unit', val= 1, desc='power required per engine', units='W')
        self.add_output('eta_prplsv', val= 1, desc='propeller isentropic efficiency', units=None)

    def setup_partials(self):
        self.declare_partials('shaft_power_unit', 'thrust_total')
        self.declare_partials('shaft_power_unit', 'vel')
        self.declare_partials('shaft_power_unit', 'd_blade')
        self.declare_partials('shaft_power_unit', 'd_hub')
        self.declare_partials('shaft_power_unit', 'rho')
        self.declare_partials('shaft_power_unit', 'n_motors')
        self.declare_partials('shaft_power_unit', 'eta_prop')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        vel = inputs['vel']
        eta_hover = inputs['eta_hover']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        thrust_unit = thrust_total / n_motors 


        # Compute induced airspeed [1] Eq 15-76
        v_ind = 0.5 * ( - vel + ( vel**2 + 2*thrust_unit / (0.5 * rho * diskarea) ) **(0.5) )

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = np.where(vel>1e-9, 2 / (1 + v3/ (1e-9 + vel)), eta_hover)   

        # Compute the power required [1] Eq 15-75
        propulsive_power_unit = thrust_unit * vel  + thrust_unit ** 1.5/ np.sqrt(2 * rho * diskarea)
   

        # Compute the power required [1] Eq 15-78
        shaft_power_unit = propulsive_power_unit / eta_prop / eta_prplsv

        # Pack outputs
        outputs['shaft_power_unit'] = shaft_power_unit
        outputs['eta_prplsv'] = eta_prplsv

    def compute_partials(self, inputs, J):  

        # Unpack inputs
        vel = inputs['vel']
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        eta_prop = inputs['eta_prop']


        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)
        d_blade_d_diskarea = np.pi * d_blade / 2
        d_hub_d_diskarea = -np.pi * d_hub / 2
        thrust_unit = thrust_total / n_motors  

        # Compute induced airspeed [1] Eq 15-76
        v_ind = 0.5 * ( - vel + ( vel**2 + 2*thrust_unit / (0.5 * rho * diskarea) ) **(0.5) )

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/ (1e-9 + vel))   

        propulsive_power_unit = thrust_unit * vel  + thrust_unit ** 1.5/ np.sqrt(2 * rho * diskarea)


        J['shaft_power_unit', 'thrust_total'] =  (vel + 1.5 * thrust_total**0.5 / np.sqrt(2 * rho * diskarea)) / n_motors / eta_prop / eta_prplsv
        J['shaft_power_unit', 'vel'] =  thrust_unit / eta_prop / eta_prplsv
        J['shaft_power_unit', 'd_blade'] = thrust_unit ** 1.5 * (-0.5)*  (2 * rho *diskarea)**(-1.5) * d_blade_d_diskarea/ eta_prop / eta_prplsv
        J['shaft_power_unit', 'd_hub'] = thrust_unit ** 1.5 * (-0.5)*  (2 * rho *diskarea)**(-1.5) * d_hub_d_diskarea/ eta_prop / eta_prplsv
        J['shaft_power_unit', 'rho'] =  thrust_unit ** 1.5 * (-0.5) *  (2 * rho *diskarea)**(-1.5) * ( 2 * diskarea) / eta_prop / eta_prplsv
        J['shaft_power_unit', 'eta_prop'] =  -propulsive_power_unit / eta_prop**2 / eta_prplsv
        J['shaft_power_unit', 'n_motors'] =  -thrust_total*vel/ eta_prop / eta_prplsv / n_motors**2  -  1.5 * thrust_unit**0.5  * thrust_total/ n_motors **2 / np.sqrt(2 * rho * diskarea) / eta_prop / eta_prplsv
        J['shaft_power_unit', 'eta_hover'] =  -propulsive_power_unit / eta_prop / eta_prplsv

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('d_blade', 1.5, units='m')
    ivc.add_output('d_hub', 0.5, units='m')
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('thrust_total', 5000, units='N')
    ivc.add_output('n_motors', 2, units=None)
    ivc.add_output('vel', 100, units='m/s')
    ivc.add_output('eta_prop', 0.9 , units=None)
    ivc.add_output('eta_hover', 0.5, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('Propeller', Propeller(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('shaft_power_unit = ', p['Propeller.shaft_power_unit']/1000, 'kW')
    print('eta_prplsv = ', p['Propeller.eta_prplsv'])
