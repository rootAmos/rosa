import openmdao.api as om
import numpy as np

class LiftRequired(om.ExplicitComponent):
    """
    Calculates required lift force based on weight and flight path angle.
    
    lift = MTOW * g * cos(gamma)
    
    Inputs:
        mto : float
            Maximum takeoff weight [kg]
        gamma : float
            Flight path angle [rad]
    
    Outputs:
        lift : float
            Required lift force [N]
    """
    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    

    def setup(self):

        N = self.options['N']
        # Inputs
        self.add_input('mto', val=1.0, units='kg',
                      desc='Maximum takeoff weight')
        self.add_input('gamma', val=1.0 * np.ones(N), units='rad',
                      desc='Flight path angle')

        

        # Outputs
        self.add_output('lift', val=1.0 * np.ones(N), units='N',
                       desc='Required lift force')
        

        # Declare partials
        self.declare_partials('lift', ['mto', 'gamma'])
        
    def compute(self, inputs, outputs):


        mto = inputs['mto']
        gamma = inputs['gamma']
        g = 9.80665  # gravitational acceleration [m/s^2]
        
        outputs['lift'] = mto * g * np.cos(gamma)
        
    def compute_partials(self, inputs, partials):
        
        mto = inputs['mto']
        gamma = inputs['gamma']
        g = 9.80665

        N = self.options['N']
        
        partials['lift', 'mto'] = g * np.cos(gamma)
        partials['lift', 'gamma'] = np.eye(N) * -mto * g * np.sin(gamma)


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('mto', val=10000.0, units='kg', desc='Maximum takeoff weight')
    ivc.add_output('gamma', val=0.0, units='deg', desc='Flight path angle')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('lift', LiftRequired(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  MTOW:                {prob.get_val("mto")[0]:8.1f} kg')
    print(f'  Flight Path Angle:   {prob.get_val("gamma")[0]:8.3f} deg')
    print(f'  Required Lift:       {prob.get_val("lift")[0]/1000:8.1f} kN')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # MTOW sweep
    mtow_range = np.linspace(5000, 15000, 50)
    lift_vals = []
    for mtow in mtow_range:
        prob.set_val('mto', mtow)
        prob.run_model()
        lift_vals.append(prob.get_val('lift')[0])
    
    plt.subplot(121)
    plt.plot(mtow_range, np.array(lift_vals)/1000)
    plt.xlabel('MTOW (kg)')
    plt.ylabel('Required Lift (kN)')
    plt.grid(True)
    plt.title('Effect of Maximum Takeoff Weight')
    
    # Flight path angle sweep
    prob.set_val('mto', 10000)  # Reset to baseline
    gamma_range = np.linspace(-10, 10, 50)  # degrees
    lift_vals = []
    for gamma in gamma_range:
        prob.set_val('gamma', gamma)
        prob.run_model()
        lift_vals.append(prob.get_val('lift')[0])
    
    plt.subplot(122)
    plt.plot(gamma_range, np.array(lift_vals)/1000)
    plt.xlabel('Flight Path Angle (deg)')
    plt.ylabel('Required Lift (kN)')
    plt.grid(True)
    plt.title('Effect of Flight Path Angle')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show() 