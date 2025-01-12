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
        self.add_input('d_blades', val=1, desc='fan blade diameter', units='m')
        self.add_input('d_hub', val=1, desc='hub diameter', units='m')
        self.add_input('rho', val= np.ones(self.options['n']), desc='air density', units='kg/m**3')
        self.add_input('thrust_unit', val= np.ones(self.options['n']), desc='unit thrust required', units='N')
        self.add_input('vel', val= np.ones(self.options['n']), desc='true airspeed', units='m/s')
        self.add_input('epsilon_r', val=1, desc='expansion ratio', units=None)
        self.add_input('eta_fan', val=1, desc='fan efficiency', units=None)
        self.add_input('eta_duct', val=1, desc='duct efficiency', units=None)

        # Outputs
        self.add_output('p_shaft_unit', val=0, desc='power required per engine', units='W')

    def setup_partials(self):
        self.declare_partials('p_shaft_unit', '*')

    def compute(self, inputs, outputs):

        # Unpack inputs
        d_blade = inputs['d_blade']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_unit = inputs['thrust_unit']
        vel = inputs['vel']
        epsilon_r = inputs['epsilon_r']
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        # Compute the power required [1] Eq 15-90
        p_prplsv_unit = 3/4 * vel * thrust_unit + np.sqrt((thrust_unit **2 * vel**2 / 4 **2) + ( thrust_unit**3/ ( 4* rho * diskarea  * epsilon_r)))

        # Compute induced airspeed [1] Eq 15-91
        v_ind = ( 0.5 * epsilon_r - 1)* vel + np.sqrt( (vel * epsilon_r /2)**2 + epsilon_r * thrust_unit  / (rho * diskarea) )
        
        # Compute station 3 velocity [1] Eq 15-87
        v3 = vel +  v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = 2 / (1 + v3/vel)    

        # Compute the power required [1] Eq 15-78
        p_shaft_unit = p_prplsv_unit / eta_fan / eta_duct / eta_prplsv

        # Pack outputs
        outputs['p_shaft_unit'] = p_shaft_unit

    def compute_partials(self, inputs, partials):  

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


        partials['p_shaft_unit', 'thrust_total'] =  (vel + 1.5 * thrust_total**0.5 / np.sqrt(2 * rho * diskarea)) / n_motors / eta_prop / eta_prplsv
        partials['p_shaft_unit', 'vel'] =  thrust_unit / eta_prop / eta_prplsv
        partials['p_shaft_unit', 'd_blade'] = thrust_unit ** 1.5 * (-0.5)*  (2 * rho *diskarea)**(-1.5) * d_blade_d_diskarea/ eta_prop / eta_prplsv
        partials['p_shaft_unit', 'd_hub'] = thrust_unit ** 1.5 * (-0.5)*  (2 * rho *diskarea)**(-1.5) * d_hub_d_diskarea/ eta_prop / eta_prplsv
        partials['p_shaft_unit', 'rho'] =  thrust_unit ** 1.5 * (-0.5) *  (2 * rho *diskarea)**(-1.5) * ( 2 * diskarea) / eta_prop / eta_prplsv
        partials['p_shaft_unit', 'eta_prop'] =  -propulsive_power_unit / eta_prop**2 / eta_prplsv
        partials['p_shaft_unit', 'n_motors'] =  -thrust_total*vel/ eta_prop / eta_prplsv / n_motors**2  -  1.5 * thrust_unit**0.5  * thrust_total/ n_motors **2 / np.sqrt(2 * rho * diskarea) / eta_prop / eta_prplsv
        partials['p_shaft_unit', 'eta_hover'] =  -propulsive_power_unit / eta_prop / eta_prplsv

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

    print('p_shaft_unit = ', p['Propeller.p_shaft_unit']/1000, 'kW')
    print('eta_prplsv = ', p['Propeller.eta_prplsv'])
