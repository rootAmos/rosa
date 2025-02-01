import openmdao.api as om
import numpy as np

class AlphaRequired(om.ExplicitComponent):
    """
    Computes required angle of attack for a given lift force.
    
    alpha = (L/(0.5 * rho * V**2 * S) - CL0) / CL_alpha
    
    Inputs:
        lift : float
            Required lift force [N]
        rho : float
            Air density [kg/m**3]
        V : float
            Airspeed [m/s]
        S : float
            Reference area [m**2]
        CL0 : float
            Zero-angle lift coefficient [-]
        CL_alpha : float
            Lift curve slope [1/rad]
    
    Outputs:
        alpha : float
            Required angle of attack [rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('lift', val=0.0, units='N',
                      desc='Required lift force')
        self.add_input('rho', val=1.225, units='kg/m**3',
                      desc='Air density')
        self.add_input('u', val=100.0, units='m/s',
                      desc='Airspeed')
        self.add_input('S', val=120.0, units='m**2',
                      desc='Reference area')
        self.add_input('CL0', val=0.0,
                      desc='Zero-angle lift coefficient')
        self.add_input('CL_alpha', val=5.5, units='1/rad',
                      desc='Lift curve slope')
        
        # Outputs
        self.add_output('alpha', val=0.0, units='rad',
                       desc='Required angle of attack')
        
        # Declare partials
        self.declare_partials('alpha', ['lift', 'rho', 'u', 'S', 'CL0', 'CL_alpha'])
        
    def compute(self, inputs, outputs):
        lift = inputs['lift']
        rho = inputs['rho']
        u = inputs['u']
        S = inputs['S']
        CL0 = inputs['CL0']
        CL_alpha = inputs['CL_alpha']

        # Dynamic pressure
        q = 0.5 * rho * u**2
        
        # Required CL
        CL_req = lift / (q * S)
        
        # Required alpha
        outputs['alpha'] = (CL_req - CL0) / CL_alpha
        
    def compute_partials(self, inputs, partials):

        lift = inputs['lift']
        rho = inputs['rho']
        u = inputs['u']
        S = inputs['S']
        CL0 = inputs['CL0']
        CL_alpha = inputs['CL_alpha']
        
        partials['alpha', 'rho'] = -(2*lift)/(CL_alpha*S*rho**2*u**2)

        partials['alpha', 'S'] = -(2*lift)/(CL_alpha*S**2*rho*u**2)

        partials['alpha', 'u'] = -(4*lift)/(CL_alpha*S*rho*u**3)

        partials['alpha', 'CL0'] = -1/CL_alpha

        partials['alpha', 'CL_alpha'] = (CL0 - (2*lift)/(S*rho*u**2))/CL_alpha**2

        partials['alpha', 'lift'] = 2/(CL_alpha*S*rho*u**2)



if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('lift', val=100000.0, units='N', desc='Required lift force')
    ivc.add_output('rho', val=1.225, units='kg/m**3', desc='Air density')
    ivc.add_output('V', val=100.0, units='m/s', desc='Airspeed')
    ivc.add_output('S', val=60.0, units='m**2', desc='Reference area')
    ivc.add_output('CL0', val=0.2, desc='Zero-angle lift coefficient')
    ivc.add_output('CL_alpha', val=5.5, units='1/rad', desc='Lift curve slope')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('alpha', AlphaRequired(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Required Lift:       {prob.get_val("lift")[0]/1000:8.1f} kN')
    print(f'  Density:             {prob.get_val("rho")[0]:8.3f} kg/m³')
    print(f'  Velocity:            {prob.get_val("V")[0]:8.1f} m/s')
    print(f'  Reference Area:      {prob.get_val("S")[0]:8.1f} m²')
    print(f'  CL0:                 {prob.get_val("CL0")[0]:8.3f}')
    print(f'  CL_alpha:            {prob.get_val("CL_alpha")[0]:8.3f} /rad')
    print(f'  Required Alpha:      {np.degrees(prob.get_val("alpha")[0]):8.3f} deg')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Lift sweep
    lift_range = np.linspace(50000, 150000, 50)
    alpha_vals = []
    for lift in lift_range:
        prob.set_val('lift', lift)
        prob.run_model()
        alpha_vals.append(np.degrees(prob.get_val('alpha')[0]))
    
    plt.subplot(221)
    plt.plot(lift_range/1000, alpha_vals)
    plt.xlabel('Required Lift (kN)')
    plt.ylabel('Alpha (deg)')
    plt.grid(True)
    plt.title('Effect of Required Lift')
    
    # Velocity sweep
    prob.set_val('lift', 100000)  # Reset to baseline
    V_range = np.linspace(50, 150, 50)
    alpha_vals = []
    for V in V_range:
        prob.set_val('V', V)
        prob.run_model()
        alpha_vals.append(np.degrees(prob.get_val('alpha')[0]))
    
    plt.subplot(222)
    plt.plot(V_range, alpha_vals)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Alpha (deg)')
    plt.grid(True)
    plt.title('Effect of Velocity')
    
    # CL0 sweep
    prob.set_val('V', 100)  # Reset to baseline
    CL0_range = np.linspace(0.0, 0.4, 50)
    alpha_vals = []
    for CL0 in CL0_range:
        prob.set_val('CL0', CL0)
        prob.run_model()
        alpha_vals.append(np.degrees(prob.get_val('alpha')[0]))
    
    plt.subplot(223)
    plt.plot(CL0_range, alpha_vals)
    plt.xlabel('CL0')
    plt.ylabel('Alpha (deg)')
    plt.grid(True)
    plt.title('Effect of CL0')
    
    # CL_alpha sweep
    prob.set_val('CL0', 0.2)  # Reset to baseline
    CL_alpha_range = np.linspace(4.0, 7.0, 50)
    alpha_vals = []
    for CL_alpha in CL_alpha_range:
        prob.set_val('CL_alpha', CL_alpha)
        prob.run_model()
        alpha_vals.append(np.degrees(prob.get_val('alpha')[0]))
    
    plt.subplot(224)
    plt.plot(CL_alpha_range, alpha_vals)
    plt.xlabel('CL_alpha (1/rad)')
    plt.ylabel('Alpha (deg)')
    plt.grid(True)
    plt.title('Effect of CL_alpha')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show() 