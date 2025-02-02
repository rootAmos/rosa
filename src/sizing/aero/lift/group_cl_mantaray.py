import openmdao.api as om
import numpy as np

from cl import LiftCoefficient

class GroupCLMantaRay(om.Group):
    """

    Group that computes total lift coefficient by combining wing and canard.
    """
    
    def initialize(self):
        pass
    def setup(self):


        self.add_subsystem('manta_cl',
                        LiftCoefficient(),
                        promotes_inputs=[],
                        promotes_outputs=[])
        

        self.add_subsystem('ray_cl',
                        LiftCoefficient(),
                        promotes_inputs=[],
                        promotes_outputs=[]) 

        self.connect('manta_cl.CL', 'CL_manta')
        self.connect('ray_cl.CL', 'CL_ray')

        adder = om.AddSubtractComp()
        adder.add_equation('CL_total',
                        ['CL_manta', 'CL_ray'],
                        desc='Total lift coefficient')

        self.add_subsystem('sum_cl', adder, promotes=['*'])



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Wing parameters
    ivc.add_output('CL0_manta', val=0.2, desc='Manta zero-angle lift coefficient')
    ivc.add_output('CL_alpha_manta_eff', val=5.5, units='1/rad',
                   desc='Manta effective lift curve slope')
    

    # Ray parameters
    ivc.add_output('CL0_ray', val=0.1, desc='Ray zero-angle lift coefficient')
    ivc.add_output('CL_alpha_ray_eff', val=4.0, units='1/rad',
                   desc='Ray effective lift curve slope')

    
    # Common parameters
    ivc.add_output('alpha', val=2.0, units='deg', desc='Angle of attack')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('CL', GroupCLMantaRay(), promotes=['*'])
    


    prob.model.connect('CL0_manta', 'manta_cl.CL0')
    prob.model.connect('CL_alpha_manta_eff', 'manta_cl.CL_alpha_eff')


    prob.model.connect('CL0_ray', 'ray_cl.CL0')
    prob.model.connect('CL_alpha_ray_eff', 'ray_cl.CL_alpha_eff')

    prob.model.connect('alpha', 'manta_cl.alpha')


    # Setup problem
    prob.setup()



    om.n2(prob.model)
    
    # Run baseline case
    prob.run_model()
    

    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Alpha:               {np.degrees(prob.get_val("alpha")[0]):8.3f} deg')
    print('\nWing:')
    print(f'  CL0:                 {prob.get_val("CL0_w")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_w_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL_w")[0]:8.3f}')
    
    print('\nCanard:')
    print(f'  CL0:                 {prob.get_val("CL0_c")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha_c_eff")[0]:8.3f} /rad')
    print(f'  CL:                  {prob.get_val("CL_c")[0]:8.3f}')
    
    print('\nTotal:')
    print(f'  CL:                  {prob.get_val("CL_total")[0]:8.3f}')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Alpha sweep
    alpha_range = np.linspace(-5.0, 10.0, 50)  # degrees
    CL_w = []
    CL_c = []
    CL_total = []
    for alpha in alpha_range:
        prob.set_val('alpha', np.radians(alpha))
        prob.run_model()
        CL_w.append(prob.get_val('CL_w')[0])
        CL_c.append(prob.get_val('CL_c')[0])
        CL_total.append(prob.get_val('CL_total')[0])
    
    plt.subplot(221)
    plt.plot(alpha_range, CL_w, label='Wing')
    plt.plot(alpha_range, CL_c, label='Canard')
    plt.plot(alpha_range, CL_total, '--', label='Total')
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    plt.title('Lift Curves')
    
    # Wing CL0 sweep
    prob.set_val('alpha', np.radians(2.0))  # Reset to baseline
    CL0_range = np.linspace(-0.2, 0.4, 50)
    CL_w = []
    CL_total = []
    for CL0 in CL0_range:
        prob.set_val('CL0_w', CL0)
        prob.run_model()
        CL_w.append(prob.get_val('CL_w')[0])
        CL_total.append(prob.get_val('CL_total')[0])
    
    plt.subplot(222)
    plt.plot(CL0_range, CL_w, label='Wing')
    plt.plot(CL0_range, CL_total, '--', label='Total')
    plt.xlabel('Wing CL0')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing CL0')
    
    # Wing CL_alpha sweep
    prob.set_val('CL0_w', 0.2)  # Reset to baseline
    CL_alpha_range = np.linspace(4.0, 7.0, 50)
    CL_w = []
    CL_total = []
    for CL_alpha in CL_alpha_range:
        prob.set_val('CL_alpha_w_eff', CL_alpha)
        prob.run_model()
        CL_w.append(prob.get_val('CL_w')[0])
        CL_total.append(prob.get_val('CL_total')[0])
    
    plt.subplot(223)
    plt.plot(CL_alpha_range, CL_w, label='Wing')
    plt.plot(CL_alpha_range, CL_total, '--', label='Total')
    plt.xlabel('Wing CL_alpha [1/rad]')
    plt.ylabel('CL')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Wing CL_alpha')
    
    # Combined alpha and CL_alpha effect
    alpha_mesh = np.linspace(-5.0, 10.0, 20)  # degrees
    CL_alpha_mesh = np.linspace(4.0, 7.0, 20)
    CL_total = np.zeros((len(alpha_mesh), len(CL_alpha_mesh)))
    
    for i, alpha in enumerate(alpha_mesh):
        for j, CL_alpha in enumerate(CL_alpha_mesh):
            prob.set_val('alpha', np.radians(alpha))
            prob.set_val('CL_alpha_w_eff', CL_alpha)
            prob.run_model()
            CL_total[i,j] = prob.get_val('CL_total')[0]
    
    X, Y = np.meshgrid(CL_alpha_mesh, alpha_mesh)
    plt.subplot(224)
    plt.contour(X, Y, CL_total, levels=20)
    plt.colorbar(label='Total CL')
    plt.xlabel('Wing CL_alpha [1/rad]')
    plt.ylabel('Angle of Attack [deg]')
    plt.title('Combined Effects')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show() 