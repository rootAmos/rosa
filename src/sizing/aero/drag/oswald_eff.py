import openmdao.api as om
import numpy as np

class OswaldEfficiency(om.ExplicitComponent):
    """
    Calculates the Oswald efficiency factor.
    
    Inputs
    ------
    aspect_ratio : float
        aspect_ratiospect ratio [-]
    taper : float
        Taper ratio [-]
    sweep_25 : float
        Quarter-chord sweep angle [rad]
    h : float
        Height of wing above ground [m]
    span : float
        Wing span [m]
    k_WL : float
        Wing-body interference factor [-]
    
    Outputs
    -------
    e : float
        Oswald efficiency factor [-]

    References
    --------
    eq 45
table 4
eq36, 37, 38

    [1] Scholz, Estimating the oswald factor from basic geometrical parameters, 2012.
    """
    
    def setup(self):
        # Inputs
        self.add_input('aspect_ratio', val=0.0, desc='aspect_ratiospect ratio')
        self.add_input('taper', val=1.0, desc='Taper ratio')
        self.add_input('sweep_25', val=0.0, units='deg', desc='Quarter-chord sweep angle')
        self.add_input('h_winglet', val=0.0, units='m', desc='Height above ground')
        self.add_input('span', val=0.0, units='m', desc='Wing span')
        self.add_input('k_WL', val=0.0, desc='Winglet effectiveness factor. [1] Table 4')
        
        # Output
        self.add_output('e', val=0.0, desc='Oswald efficiency factor')
        
        # Declare partials
        self.declare_partials('e', ['aspect_ratio', 'taper', 'sweep_25', 'h_winglet', 'span', 'k_WL'])
        
    def compute(self, inputs, outputs):
        aspect_ratio = inputs['aspect_ratio']
        taper = inputs['taper']
        sweep_25 = inputs['sweep_25']
        h_winglet = inputs['h_winglet']
        span = inputs['span']
        k_WL = inputs['k_WL']
        
        # Compute delta_lambda [1] eq 37
        delta_lambda = -0.357 + 0.45 * np.exp(0.0375 * sweep_25)
        
        # Compute f(λ - Δλ) using 4th order polynomial
        lambda_eff = taper - delta_lambda

        # Compute f(λ - Δλ) [1] eq 36
        f_lambda = (0.0524 * lambda_eff**4 - 
                   0.15 * lambda_eff**3 + 
                   0.1659 * lambda_eff**2 - 
                   0.0706 * lambda_eff + 
                   0.0119)
        
        # Compute theoretical Oswald efficiency [1] eq 38
        e_theo = 1.0 / (1 + f_lambda * aspect_ratio)
        
        # Winglet effect correction term [1] eq 45
        k_e_WL = (1 + 2 * h_winglet/(k_WL * span))**2
        
        # Final Oswald efficiency [1] eq 45
        outputs['e'] = e_theo * k_e_WL
        
    def compute_partials(self, inputs, partials):
        aspect_ratio = inputs['aspect_ratio']
        taper = inputs['taper']
        sweep_25 = inputs['sweep_25']
        h_winglet = inputs['h_winglet']
        span = inputs['span']
        k_WL = inputs['k_WL']
        
        # Final derivatives
        partials['e', 'aspect_ratio'] = (((2*h_winglet)/(span*k_WL) + 1)**2*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133))/(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1)**2

        partials['e', 'taper'] = -(aspect_ratio*((2*h_winglet)/(span*k_WL) + 1)**2*(0.3318*taper - 0.1493*np.exp(0.0375*sweep_25) - 0.4500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.2096*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 + 0.0479))/(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1)**2

        partials['e', 'sweep_25'] = -(aspect_ratio*((2*h_winglet)/(span*k_WL) + 1)**2*(0.0012*np.exp(0.0375*sweep_25) - 0.0056*np.exp(0.0375*sweep_25)*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570) + 0.0076*np.exp(0.0375*sweep_25)*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 - 0.0035*np.exp(0.0375*sweep_25)*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3))/(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1)**2

        partials['e', 'h_winglet'] = -(4*((2*h_winglet)/(span*k_WL) + 1))/(span*k_WL*(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1))

        partials['e', 'span'] = (4*h_winglet*((2*h_winglet)/(span*k_WL) + 1))/(span**2*k_WL*(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1))

        partials['e', 'k_WL'] = (4*h_winglet*((2*h_winglet)/(span*k_WL) + 1))/(span*k_WL**2*(aspect_ratio*(0.0706*taper - 0.0318*np.exp(0.0375*sweep_25) - 0.1659*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**2 + 0.1500*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**3 - 0.0524*(taper - 0.4500*np.exp(0.0375*sweep_25) + 0.3570)**4 + 0.0133) - 1))



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('aspect_ratio', val=5.0, desc='aspect_ratiospect ratio')
    ivc.add_output('taper', val=0.45, desc='Taper ratio')
    ivc.add_output('sweep_25', val=12.0, units='deg', desc='Quarter-chord sweep angle')
    ivc.add_output('h_winglet', val=0.9, units='m', desc='Height above ground')
    ivc.add_output('span', val=30.0, units='m', desc='Wing span')
    ivc.add_output('k_WL', val=2.83, desc='Wing-body interference factor')
    
    # aspect_ratiodd IVC and Oswald efficiency component to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('oswald', OswaldEfficiency(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('\nInputs:')
    print(f'  aspect_ratio:         {prob.get_val("aspect_ratio")}')
    print(f'  Taper Ratio:          {prob.get_val("taper")}')
    print(f'  Sweep angle:          {prob.get_val("sweep_25")} deg')
    print(f'  Height:               {prob.get_val("h_winglet")} m')
    print(f'  Span:                 {prob.get_val("span")} m')
    print(f'  k_WL:                 {prob.get_val("k_WL")}')
    
    print('\nOutput:')
    print(f'  Oswald Efficiency:    {prob.get_val("e")}')
    
    # Check partials
    prob.check_partials(compact_print=True) 