import openmdao.api as om

class TotalNacelleDrag(om.ExplicitComponent):
    """
    Computes total zero-lift drag contribution from Manta's ducted fan nacelles.
    
    Inputs:
        CD0_nacelle : float
            Zero-lift drag coefficient per nacelle [-]
        num_nacelles : float
            Number of ducted fan nacelles [-]
            
    Outputs:
        CD0_total_nacelles : float
            Total zero-lift drag coefficient from all nacelles [-]
    """
    
    def setup(self):
        self.add_input('CD0_nacelle', val=0.0,
                      desc='Zero-lift drag coefficient per nacelle')
        self.add_input('num_nacelles', val=1.0,
                      desc='Number of nacelles')
        
        self.add_output('CD0_total_nacelles',
                       desc='Total nacelle drag coefficient')
        
        self.declare_partials('CD0_total_nacelles', ['CD0_nacelle', 'num_nacelles'])
        
    def compute(self, inputs, outputs):
        outputs['CD0_total_nacelles'] = inputs['CD0_nacelle'] * inputs['num_nacelles']
        
    def compute_partials(self, inputs, partials):
        partials['CD0_total_nacelles', 'CD0_nacelle'] = inputs['num_nacelles']
        partials['CD0_total_nacelles', 'num_nacelles'] = inputs['CD0_nacelle']


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
    
    def setup(self):
        self.add_input('CD0_pod', val=0.0,
                      desc='Zero-lift drag coefficient per pod')
        self.add_input('num_pods', val=1.0,
                      desc='Number of pods')
        
        self.add_output('CD0_total_pods',
                       desc='Total pod drag coefficient')
        
        self.declare_partials('CD0_total_pods', ['CD0_pod', 'num_pods'])
        
    def compute(self, inputs, outputs):
        outputs['CD0_total_pods'] = inputs['CD0_pod'] * inputs['num_pods']
        
    def compute_partials(self, inputs, partials):
        partials['CD0_total_pods', 'CD0_pod'] = inputs['num_pods']
        partials['CD0_total_pods', 'num_pods'] = inputs['CD0_pod'] 