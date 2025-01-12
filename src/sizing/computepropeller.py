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
        self.add_input('d_blades', val=1, desc='blade diameter', units='m')
        self.add_input('d_hub', val=0.5, desc='hub diameter', units='m')
        self.add_input('rho', val= 1, desc='air density', units='kg/m**3')
        self.add_input('thrust_total', val= 1, desc='total aircraft thrust required', units='N')
        self.add_input('n_motors', val=2, desc='number of engines', units=None)
        self.add_input('vel', val= 1, desc='true airspeed', units='m/s')
        self.add_input('eta_prop', val=1, desc='propeller efficiency', units=None) 
        self.add_input('eta_hover', val=1, desc='hover efficiency', units=None)

        # Outputs
        self.add_output('p_shaft_unit', val= 1, desc='power required per engine', units='W')
        self.add_output('eta_prplsv', val= 1, desc='propeller isentropic efficiency', units=None)

    def setup_partials(self):

        self.declare_partials('p_shaft_unit', 'thrust_total')
        self.declare_partials('p_shaft_unit', 'vel')
        self.declare_partials('p_shaft_unit', 'd_blades')
        self.declare_partials('p_shaft_unit', 'd_hub')
        self.declare_partials('p_shaft_unit', 'rho')
        self.declare_partials('p_shaft_unit', 'n_motors')
        self.declare_partials('p_shaft_unit', 'eta_prop')
        self.declare_partials('p_shaft_unit', 'eta_hover')

        self.declare_partials('eta_prplsv', 'vel')
        self.declare_partials('eta_prplsv', 'd_blades')
        self.declare_partials('eta_prplsv', 'd_hub')
        self.declare_partials('eta_prplsv', 'thrust_total')
        self.declare_partials('eta_prplsv', 'rho')
        self.declare_partials('eta_prplsv', 'eta_hover')
        self.declare_partials('eta_prplsv', 'n_motors')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        vel = inputs['vel']
        eta_hover = inputs['eta_hover']

        vel = np.where(vel < 1e-9, 1e-9, vel)

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        thrust_unit = thrust_total / n_motors 


        # Compute induced airspeed [1] Eq 15-76
        v_ind =  0.5 * ( - vel + ( vel**2 + 2*thrust_unit / (0.5 * rho * diskarea) ) **(0.5) )

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv = np.where(vel>1e-9, 2 / (1 + v3/ (1e-9 + vel)), eta_hover)   

        # Compute the power required [1] Eq 15-75
        p_prplsv_unit = thrust_unit * vel  + thrust_unit ** 1.5/ np.sqrt(2 * rho * diskarea)
   

        # Compute the power required [1] Eq 15-78
        p_shaft_unit = p_prplsv_unit / eta_prop / eta_prplsv

        # Pack outputs
        outputs['p_shaft_unit'] = p_shaft_unit
        outputs['eta_prplsv'] = eta_prplsv

    def compute_partials(self, inputs, partials):  

        # Unpack inputs
        vel = inputs['vel']
        d_blades = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        eta_prop = inputs['eta_prop']
        eta_hover = inputs['eta_hover']

        # Intermediate Derivative Terms
        sigma_diskarea = (d_blades**2 / 4) - (d_hub**2 / 4)
        sigma_prime = np.sqrt(vel**2 + (4 * thrust_total) / (n_motors * rho * np.pi * sigma_diskarea))


        # Partial derivatives: Shaft power / Thrust
        dp_dthrust = (
            (
                vel / n_motors
                + (3 * np.sqrt(2) * np.sqrt(thrust_total / n_motors)) /
                (4 * n_motors * np.sqrt(np.pi) * np.sqrt(rho * sigma_diskarea))
            )
            * (
                (sigma_prime / (2 * (vel + 1 / 1e9))) + 0.5
            )
            / eta_prop
            + (
                thrust_total * vel / n_motors
                + (np.sqrt(2) * (thrust_total / n_motors)**(3/2)) /
                (2 * np.sqrt(np.pi) * np.sqrt(rho * sigma_diskarea))
            )
            / (eta_prop * n_motors * rho * np.pi * (vel + 1 / 1e9) * sigma_prime * sigma_diskarea)
        )

        partials['p_shaft_unit', 'thrust_total'] = dp_dthrust
        
        # Partial derivatives: Shaft power / Velocity
        dp_dvel = (
            thrust_total / (eta_prop * n_motors)
            * (
                (sigma_prime / (2 * (vel + 1 / 1e9))) + 0.5
            )
            - (
                (thrust_total * vel / n_motors)
                + (np.sqrt(2) * (thrust_total / n_motors)**(3/2)) / (2 * np.sqrt(np.pi) * np.sqrt(rho * sigma_diskarea))
            )
            * (
                (sigma_prime / (2 * (vel + 1 / 1e9)**2)) - (vel / (2 * (vel + 1 / 1e9)* sigma_prime)) 
            ) / eta_prop 
        )

        partials['p_shaft_unit', 'vel'] = dp_dvel
        
        # Partial derivatives: Shaft power / Blade diameter
        partials['p_shaft_unit', 'd_blades'] = - (2**(1/2)*d_blades*rho*(thrust_total/n_motors)**(3/2)*((vel**2 + (4*thrust_total)\
                                                /(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(2*(vel + 1/1e9)) + 1/2))\
                                                    /(8*eta_prop*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(3/2)) - (d_blades*thrust_total*((thrust_total*vel)\
                                                    /n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2))))/(2*eta_prop*n_motors*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2)

        partials['p_shaft_unit', 'd_hub'] = (2**(1/2)*d_hub*rho*(thrust_total/n_motors)**(3/2)*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(2*(vel + 1/1e9)) + 1/2))/(8*eta_prop*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(3/2)) + (d_hub*thrust_total*((thrust_total*vel)/n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2))))/(2*eta_prop*n_motors*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2)

        partials['p_shaft_unit', 'rho'] =  - (2**(1/2)*(thrust_total/n_motors)**(3/2)*(d_blades**2/4 - d_hub**2/4)*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(2*(vel + 1/1e9)) + 1/2))/(4*eta_prop*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(3/2)) - (thrust_total*((thrust_total*vel)/n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2))))/(eta_prop*n_motors*rho**2*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4))

        partials['p_shaft_unit', 'eta_prop'] =  -(((thrust_total*vel)/n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2)))*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(2*(vel + 1/1e9)) + 1/2))/eta_prop**2

        partials['p_shaft_unit', 'n_motors'] = - (((thrust_total*vel)/n_motors**2 + (3*2**(1/2)*thrust_total*(thrust_total/n_motors)**(1/2))/(4*n_motors**2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2)))*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(2*(vel + 1/1e9)) + 1/2))/eta_prop - (thrust_total*((thrust_total*vel)/n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2))))/(eta_prop*n_motors**2*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4))

        partials['p_shaft_unit', 'eta_hover'] =  -((thrust_total*vel)/n_motors + (2**(1/2)*(thrust_total/n_motors)**(3/2))/(2*np.pi**(1/2)*(rho*(d_blades**2/4 - d_hub**2/4))**(1/2)))/(eta_prop*eta_hover**2)


        
        # Partial derivatives: Propulsive efficiency
        partials['eta_prplsv', 'vel'] = (2*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9)**2 - vel/((vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2))))/((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2

        partials['eta_prplsv', 'd_blades'] = (2*d_blades*thrust_total)/(n_motors*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2)

        partials['eta_prplsv', 'd_hub'] = -(2*d_hub*thrust_total)/(n_motors*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)**2*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2)

        partials['eta_prplsv', 'thrust_total'] = -4/(n_motors*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2)

        partials['eta_prplsv', 'rho'] = (4*thrust_total)/(n_motors*rho**2*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2)

        partials['eta_prplsv', 'eta_hover'] = 0.0

        partials['eta_prplsv', 'n_motors'] = (4*thrust_total)/(n_motors**2*rho*np.pi*(vel + 1/1e9)*(vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)*(d_blades**2/4 - d_hub**2/4)*((vel**2 + (4*thrust_total)/(n_motors*np.pi*rho*(d_blades**2/4 - d_hub**2/4)))**(1/2)/(vel + 1/1e9) + 1)**2)

        

if __name__ == "__main__":
    import openmdao.api as om

    p = om.Problem()
    model = p.model

    ivc = om.IndepVarComp()
    ivc.add_output('d_blades', 1.5, units='m')
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

    # p.check_partials(compact_print=True)
