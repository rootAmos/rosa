import numpy as np  
import openmdao.api as om


class HoverPropeller(om.ExplicitComponent):
    """
    Compute the power required for a propeller in hover.

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
        self.add_input('eta_prop', val=1, desc='propeller efficiency', units=None) 
        self.add_input('eta_hover', val=1, desc='hover efficiency', units=None)

        # Outputs
        self.add_output('p_shaft_unit', val= 1, desc='power required per engine', units='W')

    def setup_partials(self):

        self.declare_partials('p_shaft_unit', 'thrust_total')
        self.declare_partials('p_shaft_unit', 'd_blades')
        self.declare_partials('p_shaft_unit', 'd_hub')
        self.declare_partials('p_shaft_unit', 'rho')
        self.declare_partials('p_shaft_unit', 'n_motors')
        self.declare_partials('p_shaft_unit', 'eta_prop')
        self.declare_partials('p_shaft_unit', 'eta_hover')

    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        eta_hover = inputs['eta_hover']


        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        thrust_unit = thrust_total / n_motors 

        # Compute the power required [1] Eq 15-75
        p_prplsv_unit = thrust_unit ** 1.5/ np.sqrt(2 * rho * diskarea)
   

        # Compute the power required [1] Eq 15-78
        p_shaft_unit = p_prplsv_unit / eta_prop / eta_hover

        # Pack outputs
        outputs['p_shaft_unit'] = p_shaft_unit

    def compute_partials(self, inputs, partials):  

        # Unpack inputs
        d_blades = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        eta_prop = inputs['eta_prop']
        eta_hover = inputs['eta_hover']

         
        # Partial derivatives: Shaft power / Blade diameter
        partials['p_shaft_unit', 'thrust_total'] = (0.5984*(thrust_total/n_motors)**(1/2))/(eta_prop*eta_hover*n_motors*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000)

        partials['p_shaft_unit', 'd_blades'] = -(0.0997*d_blades*rho*(thrust_total/n_motors)**1.5000)/(eta_prop*eta_hover*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)

        partials['p_shaft_unit', 'd_hub'] =(0.0997*d_hub*rho*(thrust_total/n_motors)**1.5000)/(eta_prop*eta_hover*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)

        partials['p_shaft_unit', 'n_motors'] = -(0.5984*thrust_total*(thrust_total/n_motors)**(1/2))/(eta_prop*eta_hover*n_motors**2*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000)

        partials['p_shaft_unit', 'eta_prop'] =  -(0.3989*(thrust_total/n_motors)**1.5000)/(eta_prop**2*eta_hover*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000)


        partials['p_shaft_unit', 'rho'] =  -(0.1995*(0.2500*d_blades**2 - 0.2500*d_hub**2)*(thrust_total/n_motors)**1.5000)/(eta_prop*eta_hover*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)


        partials['p_shaft_unit', 'eta_hover'] =  -(0.3989*(thrust_total/n_motors)**1.5000)/(eta_prop*eta_hover**2*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000)




class CruisePropeller(om.ExplicitComponent):
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

        # Outputs
        self.add_output('p_shaft_unit', val= 1, desc='power required per engine', units='W')

    def setup_partials(self):

        self.declare_partials('p_shaft_unit', 'thrust_total')
        self.declare_partials('p_shaft_unit', 'vel')
        self.declare_partials('p_shaft_unit', 'd_blades')
        self.declare_partials('p_shaft_unit', 'd_hub')
        self.declare_partials('p_shaft_unit', 'rho')
        self.declare_partials('p_shaft_unit', 'n_motors')
        self.declare_partials('p_shaft_unit', 'eta_prop')


    def compute(self, inputs, outputs):

        # Unpack inputs    
        eta_prop = inputs['eta_prop']
        d_blade = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        vel = inputs['vel']

        diskarea = np.pi * ((d_blade/2)**2 - (d_hub/2)**2)

        thrust_unit = thrust_total / n_motors 


        # Compute induced airspeed [1] Eq 15-76
        v_ind =  0.5 * ( - vel + ( vel**2 + 2*thrust_unit / (0.5 * rho * diskarea) ) **(0.5) )

        # Compute station 3 velocity [1] Eq 15-73
        v3 = vel + 2 * v_ind

        # Compute propulsive efficiency [1] Eq 15-77
        eta_prplsv =  2 / (1 + v3/ ( vel))

        # Compute the power required [1] Eq 15-75
        p_prplsv_unit = thrust_unit * vel  + thrust_unit ** 1.5/ np.sqrt(2 * rho * diskarea)
   

        # Compute the power required [1] Eq 15-78
        p_shaft_unit = p_prplsv_unit / eta_prop / eta_prplsv

        # Pack outputs
        outputs['p_shaft_unit'] = p_shaft_unit

    def compute_partials(self, inputs, partials):  

        # Unpack inputs
        vel = inputs['vel']
        d_blades = inputs['d_blades']
        d_hub = inputs['d_hub']
        rho = inputs['rho']
        thrust_total = inputs['thrust_total']
        n_motors = inputs['n_motors']
        eta_prop = inputs['eta_prop']


        partials['p_shaft_unit', 'thrust_total'] = (0.3183*((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors))/(eta_prop*n_motors*rho*vel*(0.2500*d_blades**2 - 0.2500*d_hub**2)*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000) + ((vel/n_motors + (0.5984*(thrust_total/n_motors)**(1/2))/(n_motors*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000))*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/eta_prop

        partials['p_shaft_unit', 'vel'] = (thrust_total*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/(eta_prop*n_motors) - (((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors)*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel**2 - 0.5000/(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000))/eta_prop

        partials['p_shaft_unit', 'd_blades'] = - (0.1592*d_blades*thrust_total*((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors))/(eta_prop*n_motors*rho*vel*(0.2500*d_blades**2 - 0.2500*d_hub**2)**2*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000) - (0.0997*d_blades*rho*(thrust_total/n_motors)**1.5000*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/(eta_prop*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)

        partials['p_shaft_unit', 'd_hub'] = (0.1592*d_hub*thrust_total*((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors))/(eta_prop*n_motors*rho*vel*(0.2500*d_blades**2 - 0.2500*d_hub**2)**2*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000) + (0.0997*d_hub*rho*(thrust_total/n_motors)**1.5000*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/(eta_prop*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)

        partials['p_shaft_unit', 'n_motors'] = - (0.3183*thrust_total*((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors))/(eta_prop*n_motors**2*rho*vel*(0.2500*d_blades**2 - 0.2500*d_hub**2)*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000) - (((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000)*((thrust_total*vel)/n_motors**2 + (0.5984*thrust_total*(thrust_total/n_motors)**(1/2))/(n_motors**2*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000)))/eta_prop

        partials['p_shaft_unit', 'eta_prop'] =  -(((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors)*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/eta_prop**2

        partials['p_shaft_unit', 'rho'] =  - (0.3183*thrust_total*((0.3989*(thrust_total/n_motors)**1.5000)/(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**0.5000 + (thrust_total*vel)/n_motors))/(eta_prop*n_motors*rho**2*vel*(0.2500*d_blades**2 - 0.2500*d_hub**2)*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**0.5000) - (0.1995*(0.2500*d_blades**2 - 0.2500*d_hub**2)*(thrust_total/n_motors)**1.5000*((0.5000*(vel**2 + (1.2732*thrust_total)/(n_motors*rho*(0.2500*d_blades**2 - 0.2500*d_hub**2)))**(1/2))/vel + 0.5000))/(eta_prop*(rho*(0.2500*d_blades**2 - 0.2500*d_hub**2))**1.5000)



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
    #ivc.add_output('vel', 100, units='m/s')
    ivc.add_output('eta_prop', 0.9 , units=None)
    ivc.add_output('eta_hover', 0.5, units=None)

    model.add_subsystem('Indeps', ivc, promotes_outputs=['*'])
    #model.add_subsystem('Propeller', CruisePropeller(), promotes_inputs=['*'])
    model.add_subsystem('Propeller', HoverPropeller(), promotes_inputs=['*'])

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

    p.check_partials(compact_print=True)
