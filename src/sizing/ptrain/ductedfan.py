import numpy as np  
import openmdao.api as om

class ComputeUnitThrust(om.ExplicitComponent):
    """
    Compute the thrust required per ducted fan unit by dividing total thrust by number of fans.
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']
        # Inputs
        self.add_input('thrust_total', val=1.0 * np.ones(N), desc='total thrust required', units='N')
        self.add_input('num_ducts', val=1.0, desc='number of fans', units=None)


        # Outputs
        self.add_output('thrust_unit', val=1.0 * np.ones(N), desc='thrust required per fan', units='N')

    def setup_partials(self):
        self.declare_partials('thrust_unit', ['thrust_total', 'num_ducts'])

    def compute(self, inputs, outputs):
        outputs['thrust_unit'] = inputs['thrust_total'] / inputs['num_ducts']

    def compute_partials(self, inputs, partials):
        N = self.options['N']
        partials['thrust_unit', 'thrust_total'] = np.eye(N) * 1.0 / inputs['num_ducts']

        partials['thrust_unit', 'num_ducts'] = -inputs['thrust_total'] / inputs['num_ducts']**2



class ComputeDuctedFan(om.ExplicitComponent):
    """
    Compute the power required for a ducted fan.

    References
    [1] GA Guddmundsson. "General Aviation Aircraft Design".
    
    """

    def initialize(self):   
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):

        N = self.options['N']

        # Inputs    

        self.add_input('d_blades', val=1, desc='fan blade diameter', units='m')
        self.add_input('d_hub', val=1, desc='hub diameter', units='m')
        self.add_input('rho', val= 1 * np.ones(N), desc='air density', units='kg/m**3')
        self.add_input('thrust_unit', val= 1 * np.ones(N), desc='total thrust required', units='N')
        self.add_input('u', val= 1 * np.ones(N), desc='true airspeed', units='m/s')

        self.add_input('epsilon_r', val=1, desc='expansion ratio', units=None)
        self.add_input('eta_fan', val=1, desc='fan efficiency', units=None)
        self.add_input('eta_duct', val=1, desc='duct efficiency', units=None)
        self.add_input('num_ducts', val=1, desc='number of fans', units=None)

        # Outputs
        self.add_output('p_shaft_unit', val=1 * np.ones(N), desc='power required per engine', units='W')
        self.add_output('eta_prplsv', val=1 * np.ones(N), desc='propulsive efficiency', units=None)


    def setup_partials(self):
        self.declare_partials('p_shaft_unit', '*')
        self.declare_partials('eta_prplsv', '*')

    def compute(self, inputs, outputs):

        # Unpack inputs
        d_blades = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_unit = inputs['thrust_unit']
        vel = inputs['u']
        epsilon_r = inputs['epsilon_r']
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']

        diskarea = np.pi * ((d_blades/2)**2 - (d_hub/2)**2)

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
        outputs['eta_prplsv'] = eta_prplsv

    def compute_partials(self, inputs, partials):  

        N = self.options['N']

        # Unpack inputs
        vel = inputs['u']
        d_blades = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        epsilon_r = inputs['epsilon_r']
        thrust_unit = inputs['thrust_unit']
        eta_fan = inputs['eta_fan']
        eta_duct = inputs['eta_duct']
 

        partials['p_shaft_unit', 'd_blades'] = - (d_blades*thrust_unit**3*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(16*epsilon_r*eta_fan*eta_duct*rho*np.pi*(d_blades**2/4 - d_hub**2/4)**2*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2)) - (d_blades*epsilon_r*thrust_unit*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(8*eta_fan*eta_duct*rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2)

        partials['p_shaft_unit', 'd_hub'] = (d_hub*thrust_unit**3*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(16*epsilon_r*eta_fan*eta_duct*rho*np.pi*(d_blades**2/4 - d_hub**2/4)**2*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2)) + (d_hub*epsilon_r*thrust_unit*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(8*eta_fan*eta_duct*rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2)

        partials['p_shaft_unit', 'rho'] =np.eye(N) * - (thrust_unit**3*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(8*epsilon_r*eta_fan*eta_duct*rho**2*np.pi*(d_blades**2/4 - d_hub**2/4)*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2)) - (epsilon_r*thrust_unit*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(4*eta_fan*eta_duct*rho**2*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4))


        partials['p_shaft_unit', 'thrust_unit'] = np.eye(N) * (((3*vel)/4 + ((3*thrust_unit**2)/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit*vel**2)/8)/(2*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2)))*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(eta_fan*eta_duct) + (epsilon_r*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(4*eta_fan*eta_duct*rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4))


        partials['p_shaft_unit', 'u'] = np.eye(N) * (((epsilon_r/2 + (epsilon_r**2*vel)/(4*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)))/(2*vel) - (vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel**2))*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(eta_fan*eta_duct) + (((3*thrust_unit)/4 + (thrust_unit**2*vel)/(16*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2)))*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(eta_fan*eta_duct)

        partials['p_shaft_unit', 'epsilon_r'] = ((vel/2 + ((epsilon_r*vel**2)/2 + thrust_unit/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))/(2*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)))*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(2*eta_fan*eta_duct*vel) - (thrust_unit**3*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2))/(8*epsilon_r**2*eta_fan*eta_duct*rho*np.pi*(d_blades**2/4 - d_hub**2/4)*(thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2))

        partials['p_shaft_unit', 'eta_fan'] = -(((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2)*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(eta_fan**2*eta_duct)

        partials['p_shaft_unit', 'eta_duct'] = -(((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/(2*vel) + 1/2)*((thrust_unit**3/(4*epsilon_r*np.pi*rho*(d_blades**2/4 - d_hub**2/4)) + (thrust_unit**2*vel**2)/16)**(1/2) + (3*thrust_unit*vel)/4))/(eta_fan*eta_duct**2)


        
        
        partials['eta_prplsv', 'epsilon_r'] = -(2*(vel/2 + ((epsilon_r*vel**2)/2 + thrust_unit/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))/(2*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2))))/(vel*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2)

        partials['eta_prplsv', 'u'] = np.eye(N) * -(2*((epsilon_r/2 + (epsilon_r**2*vel)/(4*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)))/vel - (vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel**2))/((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2


        partials['eta_prplsv', 'rho'] =np.eye(N) * (epsilon_r*thrust_unit)/(rho**2*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2)


        partials['eta_prplsv', 'thrust_unit'] =np.eye(N) * -epsilon_r/(rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2)

        partials['eta_prplsv', 'd_blades'] = (d_blades*epsilon_r*thrust_unit)/(2*rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2)

        partials['eta_prplsv', 'd_hub'] = -(d_hub*epsilon_r*thrust_unit)/(2*rho*vel*np.pi*((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2*((vel + ((epsilon_r**2*vel**2)/4 + (thrust_unit*epsilon_r)/(np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2) + vel*(epsilon_r/2 - 1))/vel + 1)**2)

        partials['eta_prplsv', 'eta_fan'] = 0
        partials['eta_prplsv', 'eta_duct'] = 0
        
if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('d_blades', 1.5, units='m')
    ivc.add_output('d_hub', 0.5, units='m')
    ivc.add_output('rho', 1.225, units='kg/m**3')
    ivc.add_output('thrust_unit', 5000, units='N')
    ivc.add_output('u', 100, units='m/s')
    ivc.add_output('epsilon_r', 1.5, units=None)
    ivc.add_output('eta_fan', 0.9 , units=None)
    ivc.add_output('eta_duct', 0.5, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    model.add_subsystem('DuctedFan', ComputeDuctedFan(), promotes_inputs=['*'])

    model.nonlinear_solver = om.NewtonSolver()
    model.linear_solver = om.DirectSolver()

    model.nonlinear_solver.options['iprint'] = 2
    model.nonlinear_solver.options['maxiter'] = 200
    model.nonlinear_solver.options['solve_subsystems'] = True
    model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

    p.setup()

    #om.n2(p)
    p.run_model()

    print('p_shaft_unit = ', p['DuctedFan.p_shaft_unit']/1000, 'kW')
    #print('eta_prplsv = ', p['DuctedFan.eta_prplsv'])

    #p.check_partials(compact_print=True)

