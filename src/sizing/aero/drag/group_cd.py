import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt
from cdi_component import InducedDrag
from group_cd0 import GroupCD0


class GroupCDi(om.Group):
    """
    Group that computes total induced drag coefficient by summing contributions
    from wing and canard.
    """
    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('scaling_factors', default=[1.0, 1.0], desc='Scaling factors for induced drag summation')

    def setup(self):
        # Wing induced drag

        if self.options['manta']:
            self.add_subsystem('wing_cdi', 
                          InducedDrag(),

                          promotes_inputs=[('CL', 'CL_wing'),
                                         ('aspect_ratio', 'AR_wing'),
                                         ('oswald_no', 'e_wing')],
                          promotes_outputs=[('CDi', 'CDi_wing')])
        # end

        if self.options['ray']:
        
            # Canard induced drag
            self.add_subsystem('canard_cdi',
                            InducedDrag(),
                            promotes_inputs=[('CL', 'CL_canard'),
                                            ('aspect_ratio', 'AR_canard'),
                                            ('oswald_no', 'e_canard')],
                            promotes_outputs=[('CDi', 'CDi_canard')])
        # end
        
        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('CDi_total',
                        ['CDi_wing', 'CDi_canard'],
                        desc='Total induced drag coefficient',
                        scaling_factors=self.options['scaling_factors'])


        self.add_subsystem('sum_cdi', adder, promotes=['*'])
        # end



class GroupCD(om.Group):
    """
    Group that computes total drag coefficient by summing induced drag (CDi)
    and zero-lift drag (CD0) contributions.
    """

    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')
        self.options.declare('scaling_factors', default=[1.0, 1.0], desc='Scaling factors for induced drag summation')

    def setup(self):
        # Add CD0 group

        self.add_subsystem('CD0',
                          GroupCD0(manta=self.options['manta'],
                                   ray=self.options['ray']),
                          promotes_inputs=['S_wet_wing', 'S_wet_canard', 'S_wet_fuselage', 
                                        'S_wet_nacelle', 'S_ref',
                                        'FF_wing', 'FF_canard', 'FF_fuselage', 'FF_nacelle',
                                        'Cf'],
                          promotes_outputs=['CD0_total'])

        
        # Add CDi group
        self.add_subsystem('CDi',
                          GroupCDi(manta=self.options['manta'],
                                   ray=self.options['ray']),
                          promotes_inputs=['CL_wing', 'AR_wing', 'e_wing',
                                         'CL_canard', 'AR_canard', 'e_canard'],
                          promotes_outputs=['CDi_total'])

        
        # Sum the contributions
        adder = om.AddSubtractComp()
        adder.add_equation('CD_total',
                          ['CD0_total', 'CDi_total'],
                          desc='Total drag coefficient')
        
        self.add_subsystem('sum_cd', adder, promotes=['*'])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('CL_wing', val=0.5, desc='Wing lift coefficient')
    ivc.add_output('AR_wing', val=8.0, desc='Wing aspect ratio')
    ivc.add_output('e_wing', val=0.85, desc='Wing Oswald efficiency')
    ivc.add_output('S_wet_wing', val=120.0, units='m**2', desc='Wing wetted area')
    ivc.add_output('FF_wing', val=1.35, desc='Wing form factor')
    
    # Canard parameters
    ivc.add_output('CL_canard', val=0.3, desc='Canard lift coefficient')
    ivc.add_output('AR_canard', val=4.0, desc='Canard aspect ratio')
    ivc.add_output('e_canard', val=0.80, desc='Canard Oswald efficiency')
    ivc.add_output('S_wet_canard', val=20.0, units='m**2', desc='Canard wetted area')
    ivc.add_output('FF_canard', val=1.35, desc='Canard form factor')
    
    # Other parameters
    ivc.add_output('S_wet_fuselage', val=200.0, units='m**2', desc='Fuselage wetted area')
    ivc.add_output('S_wet_nacelle', val=30.0, units='m**2', desc='Nacelle wetted area')
    ivc.add_output('FF_fuselage', val=1.10, desc='Fuselage form factor')
    ivc.add_output('FF_nacelle', val=1.20, desc='Nacelle form factor')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')
    ivc.add_output('Cf', val=0.003, desc='Skin friction coefficient')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CD', GroupCD(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Wing:')
    print(f'  CL:                  {prob.get_val("CL_wing")[0]:8.3f}')
    print(f'  CDi:                 {prob.get_val("CDi_wing")[0]:8.6f}')
    
    print('\nCanard:')
    print(f'  CL:                  {prob.get_val("CL_canard")[0]:8.3f}')
    print(f'  CDi:                 {prob.get_val("CDi_canard")[0]:8.6f}')
    
    print('\nDrag Breakdown:')
    print(f'  CD0 total:           {prob.get_val("CD0_total")[0]:8.6f}')
    print(f'  CDi total:           {prob.get_val("CDi_total")[0]:8.6f}')
    print(f'  CD total:            {prob.get_val("CD_total")[0]:8.6f}')
    
    # Create drag polar
    plt.figure(figsize=(10, 6))
    
    CL_wing_range = np.linspace(0.0, 1.0, 50)
    CD_total = []
    CD0_total = []
    CDi_total = []
    
    for CL in CL_wing_range:
        prob.set_val('CL_wing', CL)
        prob.run_model()
        CD_total.append(prob.get_val('CD_total')[0])
        CD0_total.append(prob.get_val('CD0_total')[0])
        CDi_total.append(prob.get_val('CDi_total')[0])
    
    plt.plot(CD_total, CL_wing_range, label='Total CD')
    plt.plot(CD0_total, CL_wing_range, '--', label='CD0')
    plt.plot(CDi_total, CL_wing_range, ':', label='CDi')
    plt.xlabel('CD')
    plt.ylabel('CL wing')
    plt.grid(True)
    plt.legend()
    plt.title('Drag Polar')
    
    plt.show() 