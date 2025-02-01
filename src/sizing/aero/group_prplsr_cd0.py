import openmdao.api as om
from .cd0_component import ZeroLiftDragComponent
from .sum_prplsr_cd0 import TotalNacelleDrag, TotalPodDrag

class NacelleDragGroup(om.Group):
    """
    Group that computes total nacelle drag for Manta's ducted fans.
    """
    def setup(self):
        # Add component to compute single nacelle CD0
        self.add_subsystem('nacelle_unit', ZeroLiftDragComponent(),
                          promotes_inputs=[('Cf', 'Cf_nacelle'),
                                         ('FF', 'FF_nacelle'),
                                         ('Q', 'Q_nacelle'),
                                         ('S_wet', 'S_wet_nacelle'),
                                         'S_ref'])
        
        # Add component to multiply by number of nacelles
        self.add_subsystem('nacelle_total', TotalNacelleDrag(),
                          promotes_inputs=['num_nacelles'])
        
        # Connect unit CD0 to total calculator
        self.connect('nacelle_unit.CD0', 'nacelle_total.CD0_nacelle')


class PodDragGroup(om.Group):
    """
    Group that computes total pod drag for Ray's propellers.
    """
    def setup(self):
        # Add component to compute single pod CD0
        self.add_subsystem('pod_unit', ZeroLiftDragComponent(),
                          promotes_inputs=[('Cf', 'Cf_pod'),
                                         ('FF', 'FF_pod'),
                                         ('Q', 'Q_pod'),
                                         ('S_wet', 'S_wet_pod'),
                                         'S_ref'])
        
        # Add component to multiply by number of pods
        self.add_subsystem('pod_total', TotalPodDrag(),
                          promotes_inputs=['num_pods'])
        
        # Connect unit CD0 to total calculator
        self.connect('pod_unit.CD0', 'pod_total.CD0_pod') 