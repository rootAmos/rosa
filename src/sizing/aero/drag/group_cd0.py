import openmdao.api as om
from cd0_component import ZeroLiftDragComponent
from misc_cd0 import LeakageDrag, ExcrescenceDrag
from group_prplsr_cd0 import NacelleDragGroup, PodDragGroup

class GroupCD0(om.Group):
    """
    Group that combines zero-lift drag from multiple components and adds miscellaneous drag.
    """


    def initialize(self):
        self.options.declare('manta', default=0, desc='Flag for Manta configuration')
        self.options.declare('ray', default=0, desc='Flag for Ray configuration')

    
    def setup(self):

        if self.options['manta'] == 1:

            # Add CD0 components for main surfaces
            self.add_subsystem('cd0_wing', ZeroLiftDragComponent(), 
                            promotes_inputs=[],
                            promotes_outputs=[])
        

            # Add propulsion drag groups
            self.add_subsystem('cd0_nacelles', NacelleDragGroup(),
                          promotes_inputs=[], promotes_outputs=['*'])
            
            self.add_subsystem('cd0_leakage', LeakageDrag(
                CD0_leakage=self.options['CD0_leakage']))
            
            self.add_subsystem('cd0_manta_excrescence', ExcrescenceDrag(
            CD0_excrescence=self.options['CD0_excrescence']))
            
            # Add component to sum all CD0s
            adder_manta = om.AddSubtractComp()
            adder_manta.add_equation('CD0_total',
                            input_names=['CD0_wing', 'CD0_nacelles',
                                        'CD0_leakage', 'CD0_excrescence'],
                            vec_size=1,
                            scaling_factors=[1., 1., 1., 1.],
                            desc='Total zero-lift drag coefficient')
            
            self.add_subsystem('cd0_manta', adder_manta)
        # end

        if self.options['ray'] == 1:
        
            self.add_subsystem('cd0_canard', ZeroLiftDragComponent(),
                            promotes_inputs=[('Cf', 'Cf_canard'),
                                            ('FF', 'FF_canard'),
                                            ('Q', 'Q_canard'),
                                            ('S_wet', 'S_wet_canard'),
                                            'S_ref'])
            
            self.add_subsystem('cd0_pods', PodDragGroup(),
                promotes_inputs=['Cf_pod', 'FF_pod', 
                                         'Q_pod', 'S_wet_pod',
                                         'S_ref', 'num_pods'])
            
            self.add_subsystem('cd0_ray_excrescence', ExcrescenceDrag(
                CD0_excrescence=self.options['CD0_excrescence']))
            
            # Add component to sum all CD0s
            adder_ray = om.AddSubtractComp()
            adder_ray.add_equation('CD0_total',
                            input_names=['CD0_canard', 'CD0_pods',
                                         'CD0_excrescence'],
                            vec_size=1,
                            scaling_factors=[1., 1., 1.],
                            desc='Ray total zero-lift drag coefficient')
            
            self.add_subsystem('cd0_ray', adder_ray)
        # end

        if self.options['ray']  + self.options['manta'] == 2:

            # Add component to sum all CD0s
            adder_mantaray = om.AddSubtractComp()
            adder_mantaray.add_equation('CD0_total',
                            input_names=['CD0_manta', 'CD0_ray'],
                            vec_size=1,
                            scaling_factors=[1., 1.],
                            desc='Manta Ray total zero-lift drag coefficient')
            
            self.add_subsystem('cd0_mantaray', adder_mantaray)
        # end
        
        # Connect CD0 outputs to adder inputs
        self.connect('cd0_wing.CD0', 'cd0_manta.CD0_wing')
        self.connect('cd0_canard.CD0', 'cd0_ray.CD0_canard')
        self.connect('cd0_nacelles.nacelle_total.CD0_total_nacelles', 'cd0_manta.CD0_nacelles')
        self.connect('cd0_pods.pod_total.CD0_total_pods', 'cd0_ray.CD0_pods')
        self.connect('cd0_leakage.CD0_leakage', 'cd0_manta.CD0_leakage')
        self.connect('cd0_manta_excrescence.CD0_excrescence', 'cd0_manta.CD0_excrescence')

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('Cf_wing', val=0.003, desc='Wing skin friction coefficient')
    ivc.add_output('FF_wing', val=1.2, desc='Wing form factor')
    ivc.add_output('Q_wing', val=1.1, desc='Wing interference factor')
    ivc.add_output('S_wet_wing', val=100.0, units='m**2', desc='Wing wetted area')
    
    ivc.add_output('Cf_canard', val=0.003, desc='Canard skin friction coefficient')
    ivc.add_output('FF_canard', val=1.15, desc='Canard form factor')
    ivc.add_output('Q_canard', val=1.1, desc='Canard interference factor')
    ivc.add_output('S_wet_canard', val=20.0, units='m**2', desc='Canard wetted area')
    
    ivc.add_output('S_ref_manta', val=80.0, units='m**2', desc='Manta reference area')
    ivc.add_output('S_ref_ray', val=80.0, units='m**2', desc='Ray reference area')
    
    
    # Add IVC and group to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('cd0_group', GroupCD0(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    om.n2(prob)
    prob.run_model()
    

    # Print results

    print('\nComponent CD0s:')
    print(f'  Wing CD0 = {prob.get_val("cd0_group.cd0_wing.CD0"):.6f}')
    print(f'  Canard CD0 = {prob.get_val("cd0_group.cd0_canard.CD0"):.6f}')
    print(f'  Total CD0 = {prob.get_val("cd0_group.cd0_total.CD0_total"):.6f}')
    
    # Check partials
    prob.check_partials(compact_print=True)