import openmdao.api as om
import numpy as np

class FormFactor(om.ExplicitComponent):
    """
    Calculates the form factor (ff_wing) for a wing using equation 13.22.
    
    Inputs:
        t_c : float
            Thickness to chord ratio [-]
        x_t : float
            Position of maximum thickness [-]
        mach : float
            machach number [-]
        sweep_max_t : float
            Wing sweep angle at maximum thickness [rad]
    
    Outputs:
        ff_wing : float
            Form factor [-]
    """

    def initialize(self):
        self.options.declare('x_max_t', default=0.3, desc='Position of maximum thickness')
    

    def setup(self):
        # Inputs

        self.add_input('t_c', val=0.0, desc='Thickness to chord ratio')
        self.add_input('mach', val=0.0, desc='Mach number')
        self.add_input('sweep_max_t', val=0.0, units='rad', desc='Sweep angle at max thickness')
        
        # Outputs
        self.add_output('ff_wing', val=1.0, desc='Form factor')
        
        # Declare partials
        self.declare_partials('ff_wing', ['t_c', 'mach', 'sweep_max_t'])
        

    def compute(self, inputs, outputs):

        # Unpack options
        x_max_t = self.options['x_max_t']


        t_c = inputs['t_c']
        mach = inputs['mach']
        sweep_max_t = inputs['sweep_max_t']
        

        # Form factor equation 13.22
        term1 = 1 + (0.6/x_max_t)*(t_c) + 100*(t_c)**4
        term2 = 1.34 * mach**0.18 * (np.cos(sweep_max_t))**0.28
        

        outputs['ff_wing'] = term1 * term2
        
    def compute_partials(self, inputs, partials):

        # Unpack options
        x_max_t = self.options['x_max_t']


        t_c = inputs['t_c']
        mach = inputs['mach']
        sweep_max_t = inputs['sweep_max_t']
        
        partials['ff_wing', 't_c'] = 1.3400*mach**0.1800*np.cos(sweep_max_t)**0.2800*(400*t_c**3 + 0.6000/x_max_t)



        partials['ff_wing', 'mach'] = (0.2412*np.cos(sweep_max_t)**0.2800*(100*t_c**4 + (0.6000*t_c)/x_max_t + 1))/mach**0.8200



        partials['ff_wing', 'sweep_max_t'] = -(0.3752*mach**0.1800*np.sin(sweep_max_t)*(100*t_c**4 + (0.6000*t_c)/x_max_t + 1))/np.cos(sweep_max_t)**0.7200
 

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('t_c', val=0.12, desc='Thickness to chord ratio')
    ivc.add_output('mach', val=0.78, desc='Mach number')
    ivc.add_output('sweep_max_t', val=20.0, units='deg', desc='Sweep angle at max thickness')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('wing_ff', FormFactor(x_max_t=0.3), promotes=['*'])
    

    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  t/c ratio:           {prob.get_val("t_c")[0]:8.3f}')
    print(f'  Mach number:         {prob.get_val("mach")[0]:8.3f}')
    print(f'  Sweep angle:         {prob.get_val("sweep_max_t")[0]:8.3f} deg')
    print(f'  Form Factor:         {prob.get_val("ff_wing")[0]:8.3f}')
    
    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # t/c sweep
    tc_range = np.linspace(0.08, 0.16, 50)
    FF_tc = []
    for tc in tc_range:
        prob.set_val('t_c', tc)
        prob.run_model()
        FF_tc.append(prob.get_val('ff_wing')[0])
    
    plt.subplot(221)
    plt.plot(tc_range, FF_tc)
    plt.xlabel('Thickness/Chord Ratio')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of t/c Ratio')
    
    # Mach sweep
    prob.set_val('t_c', 0.12)  # Reset to baseline
    M_range = np.linspace(0.3, 0.85, 50)
    FF_M = []
    for M in M_range:
        prob.set_val('mach', M)
        prob.run_model()
        FF_M.append(prob.get_val('ff_wing')[0])
    
    plt.subplot(222)
    plt.plot(M_range, FF_M)
    plt.xlabel('Mach Number')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Mach Number')
    
    # Sweep angle sweep
    prob.set_val('mach', 0.78)  # Reset to baseline
    sweep_range = np.linspace(0, 35, 50)
    FF_sweep = []
    for sweep in sweep_range:
        prob.set_val('sweep_max_t', sweep)
        prob.run_model()
        FF_sweep.append(prob.get_val('ff_wing')[0])
    
    plt.subplot(223)
    plt.plot(sweep_range, FF_sweep)
    plt.xlabel('Sweep Angle (deg)')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Effect of Sweep Angle')
    
    # Combined effect of Mach and t/c
    plt.subplot(224)
    tc_values = [0.08, 0.12, 0.16]
    for tc in tc_values:
        prob.set_val('t_c', tc)
        FF_M = []
        for M in M_range:
            prob.set_val('mach', M)
            prob.run_model()
            FF_M.append(prob.get_val('ff_wing')[0])
        plt.plot(M_range, FF_M, label=f't/c = {tc}')
    
    plt.xlabel('Mach Number')
    plt.ylabel('Form Factor')
    plt.grid(True)
    plt.title('Combined Effect of Mach and t/c')
    plt.legend()
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()