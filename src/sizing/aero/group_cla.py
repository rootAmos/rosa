import openmdao.api as om
import numpy as np


class CoupledCLAlphaCanard(om.ExplicitComponent):
    """
    Calculates the effective lift curve slope for the canard, accounting for area ratio and downwash.
    

    Inputs:
        CL_alpha_c : float
            Canard lift curve slope [1/rad]
        deps_c_da : float
            Change in canard downwash angle with respect to angle of attack [-]
        S_c : float
            Canard reference area [m^2]
        S_w : float
            Wing reference area [m^2]
    
    Outputs:
        CL_alpha_c_eff : float
            Effective canard lift curve slope [1/rad]
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    

    def setup(self):
        # Inputs

        self.add_input('CL_alpha_c', val=0.0, units='1/rad',
                      desc='Canard lift curve slope')

        
        if self.options['manta']:
            self.add_input('S_c', val=0.0, units='m**2',
                          desc='Canard reference area')
            self.add_input('S_w', val=0.0, units='m**2',
                          desc='Wing reference area')
            self.add_input('d_eps_c_d_alpha', val=0.0,
                      desc='Canard downwash derivative')
        # end

        
        # Outputs
        self.add_output('CL_alpha_c_eff', val=0.0, units='1/rad',
                       desc='Effective canard lift curve slope')

        
        # Partials
        self.declare_partials('CL_alpha_c_eff', ['*'])
        
    def compute(self, inputs, outputs):
        CL_alpha_c = inputs['CL_alpha_c']
        d_eps_c_d_alpha = inputs['d_eps_c_d_alpha']

        if self.options['manta']:
            S_c = inputs['S_c']
            S_w = inputs['S_w']
            area_ratio = S_c / S_w
            outputs['CL_alpha_c_eff'] = CL_alpha_c  * (1 + d_eps_c_d_alpha) * area_ratio
        else:
            outputs['CL_alpha_c_eff'] = CL_alpha_c 
        # end

        
    def compute_partials(self, inputs, partials):
        d_eps_c_d_alpha = inputs['d_eps_c_d_alpha']
        CL_alpha_c = inputs['CL_alpha_c']

        if self.options['manta'] ==1:
            S_c = inputs['S_c']
            S_w = inputs['S_w']
            area_ratio = S_c / S_w
            partials['CL_alpha_c_eff', 'CL_alpha_c'] = (1 + d_eps_c_d_alpha) * area_ratio
            partials['CL_alpha_c_eff', 'd_eps_c_d_alpha'] = CL_alpha_c  * area_ratio
            partials['CL_alpha_c_eff', 'S_c'] = CL_alpha_c * (1 + d_eps_c_d_alpha) / S_w
            partials['CL_alpha_c_eff', 'S_w'] = -CL_alpha_c * (1 + d_eps_c_d_alpha) * S_c / S_w**2
        else:


            partials['CL_alpha_c_eff', 'CL_alpha_c'] = 1


class CoupledCLAlphaWing(om.ExplicitComponent):
    """
    Calculates the effective lift curve slope for the wing, accounting for downwash.
    

    Inputs:
        CL_alpha_w : float
            Wing lift curve slope [1/rad]
        deps_w_da : float
            Change in wing downwash angle with respect to angle of attack [-]
    
    Outputs:
        CL_alpha_w_eff : float
            Effective wing lift curve slope [1/rad]
    """

    def initialize(self):
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):

        # Inputs
        self.add_input('CL_alpha_w', val=0.0, units='1/rad', 
                      desc='Wing lift curve slope')
        
        if self.options['ray']:
            self.add_input('d_eps_w_d_alpha', val=0.0, 
                      desc='Wing downwash derivative from canard influence')
        # end
        
        # Outputs
        self.add_output('CL_alpha_w_eff', val=0.0, units='1/rad',
                       desc='Effective wing lift curve slope')
        
        # Partials
        self.declare_partials('CL_alpha_w_eff', ['*'])
        

    def compute(self, inputs, outputs):
        CL_alpha_w = inputs['CL_alpha_w']
        d_eps_w_d_alpha = inputs['d_eps_w_d_alpha']

        if self.options['ray'] == 1:
            outputs['CL_alpha_w_eff'] = CL_alpha_w * (1 - d_eps_w_d_alpha)
        else:
            outputs['CL_alpha_w_eff'] = CL_alpha_w

        # end
        

    def compute_partials(self, inputs, partials):
        partials['CL_alpha_w_eff', 'CL_alpha_w'] = 1 - inputs['d_eps_w_d_alpha']
        partials['CL_alpha_w_eff', 'd_eps_w_d_alpha'] = -inputs['CL_alpha_w'] 


class GroupCLAlpha(om.Group):
    """
    Group that computes total lift curve slope by summing contributions
    from wing and canard.
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
    
    def setup(self):

        scaling_factors = [self.options['manta'], self.options['ray']]
        # Wing lift curve slope

        self.add_subsystem('wing_cl_alpha', 
                          CoupledCLAlphaWing(ray=self.options['ray']),
                          promotes_inputs=['CL_alpha_w',
                                         'd_eps_w_d_alpha'],
                          promotes_outputs=['CL_alpha_w_eff'])



        
        # Canard lift curve slope
        self.add_subsystem('canard_cl_alpha',
                          CoupledCLAlphaCanard(manta=self.options['manta']),
                          promotes_inputs=['CL_alpha_c',
                                         'd_eps_c_d_alpha',
                                         'S_c', 'S_w'],
                          promotes_outputs=['CL_alpha_c_eff'])

        

        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('CL_alpha_total',
                          ['CL_alpha_w_eff', 'CL_alpha_c_eff'],
                          desc='Total lift curve slope',
                          scaling_factors=scaling_factors)

        self.add_subsystem('sum_cl_alpha', adder, promotes=['*'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('CL_alpha_w', val=5.5, units='1/rad', desc='Wing lift curve slope')
    ivc.add_output('d_eps_w_d_alpha', val=0.25, desc='Wing downwash derivative')
    
    # Canard parameters
    ivc.add_output('CL_alpha_c', val=4.0, units='1/rad', desc='Canard lift curve slope')
    ivc.add_output('d_eps_c_d_alpha', val=0.1, desc='Canard downwash derivative')
    ivc.add_output('S_c', val=20.0, units='m**2', desc='Canard reference area')
    ivc.add_output('S_w', val=120.0, units='m**2', desc='Wing reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cl_alpha_coupled', GroupCLAlpha(manta=1, ray=1), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()


    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_w")[0]:8.3f} /rad')
    print(f'  Downwash:            {prob.get_val("d_eps_w_d_alpha")[0]:8.3f}')
    print(f'  Effective CL_alpha:   {prob.get_val("CL_alpha_w_eff")[0]:8.3f} /rad')
    
    print('\nCanard:')

    prob.check_partials(compact_print=True)