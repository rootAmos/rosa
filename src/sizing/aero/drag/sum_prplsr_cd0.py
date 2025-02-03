import openmdao.api as om
import numpy as np
class TotalDuctDrag(om.ExplicitComponent):
    """
    Computes total zero-lift drag contribution from Manta's ducted fan ducts.
    
    Inputs:
        CD0_duct : float
            Zero-lift drag coefficient per duct [-]
        num_ducts : float
            Number of ducted fan ducts [-]
            
    Outputs:
        CD0_total_ducts : float
            Total zero-lift drag coefficient from all ducts [-]
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')


    def setup(self):
        N = self.options['N']

        self.add_input('CD0_duct', val=1.0 * np.ones(N),
                      desc='Zero-lift drag coefficient per duct')


        self.add_input('num_ducts', val=1.0,
                      desc='Number of ducts')

        self.add_output('CD0_total_ducts',val=1.0 * np.ones(N),
                       desc='Total duct drag coefficient')
        

        self.declare_partials('CD0_total_ducts', ['CD0_duct', 'num_ducts'])
        
    def compute(self, inputs, outputs):
        outputs['CD0_total_ducts'] = inputs['CD0_duct'] * inputs['num_ducts']
        
    def compute_partials(self, inputs, partials):

        N = self.options['N']
        partials['CD0_total_ducts', 'CD0_duct'] = np.eye(N) * inputs['num_ducts']
        partials['CD0_total_ducts', 'num_ducts'] = inputs['CD0_duct']



class TotalPodDrag(om.ExplicitComponent):
    """
    Computes total zero-lift drag contribution from Ray's propeller pods.
    
    Inputs:
        CD0_pod : float
            Zero-lift drag coefficient per pod [-]
        num_pods : float
            Number of propeller pods [-]
            
    Outputs:
        CD0_total_pods : float
            Total zero-lift drag coefficient from all pods [-]
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')


    def setup(self):
        N = self.options['N']

        self.add_input('CD0_pod', val=1.0 * np.ones(N),
                      desc='Zero-lift drag coefficient per pod')

        self.add_input('num_pods', val=1.0,
                      desc='Number of pods')
        
        self.add_output('CD0_total_pods',val=1.0 * np.ones(N),
                       desc='Total pod drag coefficient')
        
        self.declare_partials('CD0_total_pods', ['CD0_pod', 'num_pods'])
        
    def compute(self, inputs, outputs):
        outputs['CD0_total_pods'] = inputs['CD0_pod'] * inputs['num_pods']
        
    def compute_partials(self, inputs, partials):
        partials['CD0_total_pods', 'CD0_pod'] = inputs['num_pods']
        partials['CD0_total_pods', 'num_pods'] = inputs['CD0_pod'] 