import openmdao.api as om
from cd0_component import ZeroLiftDragComponent
from sum_prplsr_cd0 import TotalDuctDrag, TotalPodDrag
from skin_friction import SkinFriction
from reynolds import ReynoldsNumber



import os
import sys
import pdb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from geometry.duct_form_factor import DuctFormFactor
from geometry.duct_wetted import DuctWettedArea
from geometry.pod_form_factor import PodFormFactor
from geometry.pod_wetted import PodWettedArea


from mission.mach_number import MachNumber
from mission.computeamos import ComputeAtmos



class DuctDragGroup(om.Group):
    """

    Group that computes total nacelle drag for Manta's ducted fans.
    """
    def setup(self):

        self.add_subsystem('atmos', ComputeAtmos(),
                          promotes_inputs=['alt'],
                          promotes_outputs=['*'])
        
        self.add_subsystem('mach', MachNumber(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        
        self.add_subsystem('duct_reynolds', ReynoldsNumber(),
                          promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('duct_cf', SkinFriction(),
                          promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('duct_ff', DuctFormFactor(),
                          promotes_inputs=['*'],promotes_outputs=['*'])


        self.add_subsystem('duct_wet', DuctWettedArea(),
                          promotes_inputs=['*'],promotes_outputs=['*'])

        # Add component to compute single n CD0
        self.add_subsystem('duct_unit', ZeroLiftDragComponent(),
                          promotes_inputs=['*'],promotes_outputs=[])
        
        self.add_subsystem('duct_total', TotalDuctDrag(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        
        self.connect('duct_unit.CD0', 'CD0_duct')
        self.connect('ff_duct', 'ff')
        self.connect('S_wet_duct', 'S_wet')


class PodDragGroup(om.Group):


    """
    Group that computes total pod drag for Ray's propellers.
    """
    def setup(self):
        # Add component to compute single pod CD0

        self.add_subsystem('atmos', ComputeAtmos(),
                          promotes_inputs=['alt'],promotes_outputs=['*'])
        
        self.add_subsystem('mach', MachNumber(),
                          promotes_inputs=['*'],promotes_outputs=['*'])

        self.add_subsystem('pod_reynolds', ReynoldsNumber(),
                          promotes_inputs=['*'],promotes_outputs=['*'])



        self.add_subsystem('pod_cf', SkinFriction(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        
        self.add_subsystem('pod_ff', PodFormFactor(),
                          promotes_inputs=['*'],promotes_outputs=['*'])

        self.add_subsystem('pod_wet', PodWettedArea(),
                          promotes_inputs=['*'],promotes_outputs=['*'])

        self.add_subsystem('pod_unit', ZeroLiftDragComponent(),
                          promotes_inputs=['*'],promotes_outputs=[])

        

        # Add component to multiply by number of pods
        self.add_subsystem('pod_total', TotalPodDrag(),
                          promotes_inputs=['*'],promotes_outputs=['*'])
        

        # Connect unit CD0 to total calculator
        self.connect('pod_unit.CD0', 'CD0_pod') 
        self.connect('ff_pod', 'ff')
        self.connect('S_wet_pod', 'S_wet')



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Test DuctDragGroup
    print("\nTesting DuctDragGroup:")
    print("-----------------------")
    
    # Create problem instance
    prob_duct = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Duct parameters
    ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('duct_l_char', val=1.0, units='m',desc='Duct Characteristic length')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')

    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Duct outer diameter')
    ivc.add_output('id_duct', val=1.0, units='m', desc='Duct inner diameter')
    ivc.add_output('Q_duct', val=1.0, desc='Duct interference factor')
    ivc.add_output('k_lam_duct', val=0.1, desc='Duct laminar flow fraction')
    ivc.add_output('num_ducts', val=2, desc='Number of ducts')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')
    
    # Add subsystems to model
    prob_duct.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob_duct.model.add_subsystem('duct_drag', DuctDragGroup(), promotes=['*'])

    # Make Connections
    prob_duct.model.connect('duct_l_char', 'l_char')
    prob_duct.model.connect('k_lam_duct', 'k_lam')
    prob_duct.model.connect('Q_duct', 'Q')


    



    prob_duct.setup()
    #om.n2(prob_duct)
    prob_duct.run_model()
    

    print('Duct Drag Results:')
    print(f'  Total Duct CD0:   {prob_duct.get_val("CD0_total_ducts")[0]:8.6f}')

    # Test PodDragGroup
    print("\nTesting PodDragGroup:")
    print("-------------------")

    prob_duct.check_partials(compact_print=True)
    
    #pdb.set_trace()
    # Create new problem instance
    prob_pod = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Duct parameters
    ivc.add_output('alt', val=30000, units='ft', desc='Altitude')
    ivc.add_output('u', val=50.0, units='m/s', desc='Flow speed')
    ivc.add_output('pod_l_char', val=1.0, units='m',desc='Pod Characteristic length')
    ivc.add_output('mu', val=1.789e-5, units='Pa*s', desc='Dynamic viscosity')



    ivc.add_output('l_pod', val=3.0, units='m', desc='Pod length')
    ivc.add_output('d_pod', val=2.0, units='m', desc='Pod diameter')

    ivc.add_output('Q_pod', val=1.0, desc='Pod interference factor')
    ivc.add_output('k_lam_pod', val=0.1, desc='Pod laminar flow fraction')
    ivc.add_output('num_pods', val=2, desc='Number of pods')
    ivc.add_output('S_ref', val=100.0, units='m**2', desc='Reference area')
    


    # Add subsystems to model
    prob_pod.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob_pod.model.add_subsystem('pod_drag', PodDragGroup(), promotes=['*'])
    

    # Make Connections
    prob_pod.model.connect('pod_l_char', 'l_char')
    prob_pod.model.connect('k_lam_pod', 'k_lam')
    prob_pod.model.connect('Q_pod', 'Q')


    prob_pod.setup()

    



    om.n2(prob_pod)
    prob_pod.run_model()
    

    print('Pod Drag Results:')
    print(f'  Total Pod CD0:       {prob_pod.get_val("CD0_total_pods")[0]:8.6f}')


    # Create parametric study plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Duct wetted area study
    S_wet_range = np.linspace(10.0, 50.0, 50)
    CD0_duct_values = []
    
    for S_wet in S_wet_range:
        prob_duct.set_val('S_wet_duct', S_wet)
        prob_duct.run_model()
        CD0_duct_values.append(prob_duct.get_val("CD0_total_ducts")[0])
    

    ax1.plot(S_wet_range, CD0_duct_values)
    ax1.set_xlabel('Duct Wetted Area (m²)')
    ax1.set_ylabel('Total Duct CD0')
    ax1.grid(True)
    ax1.set_title('Total Duct CD0 vs Wetted Area')
    
    # Pod wetted area study
    CD0_pod_values = []
    
    for S_wet in S_wet_range:
        prob_pod.set_val('S_wet_pod', S_wet)
        prob_pod.run_model()
        CD0_pod_values.append(prob_pod.get_val("CD0_total_pods")[0])
    

    ax2.plot(S_wet_range, CD0_pod_values)
    ax2.set_xlabel('Pod Wetted Area (m²)')
    ax2.set_ylabel('Total Pod CD0')
    ax2.grid(True)
    ax2.set_title('Total Pod CD0 vs Wetted Area')
    
    plt.tight_layout()
    plt.show()

