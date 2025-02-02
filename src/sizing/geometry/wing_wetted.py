import openmdao.api as om
import numpy as np

class WettedAreaWing(om.ExplicitComponent):
    """
    Calculates the wetted area of a wing.
    
    Inputs:
        S_exp : float
            Exposed planform area [m**2]
        t_c : float
            Thickness to chord ratio [-]
        tau : float
            Airfoil thickness location parameter [-]
        lambda_w : float
            Wing taper ratio [-]
    
    Outputs:
        S_wet : float
            Wetted area [m**2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('S_exp', val=0.0, units='m**2', 
                      desc='Exposed planform area')
        self.add_input('t_c', val=0.0, 
                      desc='Thickness to chord ratio')
        self.add_input('tau', val=0.0, 
                      desc='wing tip thickness to chord ratio / wing root thickness to chord ratio')
        self.add_input('lambda_w', val=0.0, 
                      desc='Wing taper ratio')
        
        # Outputs
        self.add_output('S_wet', val=0.0, units='m**2', 
                       desc='Wetted area')
        
        # Declare partials
        self.declare_partials('S_wet', ['S_exp', 't_c', 'tau', 'lambda_w'])
        
    def compute(self, inputs, outputs):
        S_exp = inputs['S_exp']
        t_c = inputs['t_c']
        tau = inputs['tau']
        lambda_w = inputs['lambda_w']
        
        # Equation from image
        outputs['S_wet'] = 2 * S_exp * (1 + 0.25*(t_c) * ((1 + tau*lambda_w)/(1 + lambda_w)))
        
    def compute_partials(self, inputs, partials):
        S_exp = inputs['S_exp']
        t_c = inputs['t_c']
        tau = inputs['tau']
        lambda_w = inputs['lambda_w']
        
        # Partial derivatives
        partials['S_wet', 'S_exp'] = (0.5000*t_c*(lambda_w*tau + 1))/(lambda_w + 1) + 2

        
        partials['S_wet', 't_c'] = (0.5000*S_exp*(lambda_w*tau + 1))/(lambda_w + 1)

        
        partials['S_wet', 'tau'] = (0.5000*S_exp*lambda_w*t_c)/(lambda_w + 1)

        
        partials['S_wet', 'lambda_w'] = 2*S_exp*((0.2500*t_c*tau)/(lambda_w + 1) - (0.2500*t_c*(lambda_w*tau + 1))/(lambda_w + 1)**2)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('S_exp', val=120.0, units='m**2', desc='Exposed planform area')
    ivc.add_output('t_c', val=0.12, desc='Thickness to chord ratio')
    ivc.add_output('tau', val=0.85, desc='Thickness ratio (tip/root)')
    ivc.add_output('lambda_w', val=0.3, desc='Wing taper ratio')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('wing', ComputeWingWettedAreaLiftingSurface(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Exposed Area:         {prob.get_val("S_exp")} m²')
    print(f'  t/c ratio:           {prob.get_val("t_c")}')
    print(f'  Thickness ratio τ:    {prob.get_val("tau")}')
    print(f'  Taper ratio λ:        {prob.get_val("lambda_w")}')
    print(f'  Wetted Area:         {prob.get_val("S_wet")} m²')
    

    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Exposed area sweep
    S_exp_range = np.linspace(80, 160, 50)
    S_wet_S = []
    for S in S_exp_range:
        prob.set_val('S_exp', S)
        prob.run_model()
        S_wet_S.append(prob.get_val('S_wet')[0])
    
    plt.subplot(221)
    plt.plot(S_exp_range, S_wet_S)
    plt.xlabel('Exposed Area (m²)')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Exposed Area')
    
    # t/c sweep
    prob.set_val('S_exp', 120.0)  # Reset to baseline
    tc_range = np.linspace(0.08, 0.16, 50)
    S_wet_tc = []
    for tc in tc_range:
        prob.set_val('t_c', tc)
        prob.run_model()
        S_wet_tc.append(prob.get_val('S_wet')[0])
    
    plt.subplot(222)
    plt.plot(tc_range, S_wet_tc)
    plt.xlabel('Thickness/Chord Ratio')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of t/c Ratio')
    
    # Thickness ratio sweep
    prob.set_val('t_c', 0.12)  # Reset to baseline
    tau_range = np.linspace(0.6, 1.0, 50)
    S_wet_tau = []
    for tau in tau_range:
        prob.set_val('tau', tau)
        prob.run_model()
        S_wet_tau.append(prob.get_val('S_wet')[0])
    
    plt.subplot(223)
    plt.plot(tau_range, S_wet_tau)
    plt.xlabel('Thickness Ratio τ')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Thickness Ratio')
    
    # Taper ratio sweep
    prob.set_val('tau', 0.85)  # Reset to baseline
    lambda_range = np.linspace(0.1, 0.5, 50)
    S_wet_lambda = []
    for lambda_w in lambda_range:
        prob.set_val('lambda_w', lambda_w)
        prob.run_model()
        S_wet_lambda.append(prob.get_val('S_wet')[0])
    
    plt.subplot(224)
    plt.plot(lambda_range, S_wet_lambda)
    plt.xlabel('Taper Ratio λ')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Taper Ratio')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()
