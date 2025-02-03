import numpy as np
import openmdao.api as om
import pdb
import matplotlib.pyplot as plt

class ThrustReq(om.ExplicitComponent):
    """Computes required thrust for steady flight at any flight path angle."""

    def initialize(self):
        self.options.declare('g', default=9.806, desc='Acceleration due to gravity')
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):

        N = self.options['N']

        # Inputs

        self.add_input('w_mto',val=1.0 , units='N', desc='Maximum takeoff weight')
        self.add_input('drag', val=1.0 * np.ones(N), units='N', desc='Total drag')
        self.add_input('gamma', val=1.0 * np.ones(N), units='rad', desc='Flight path angle (0 for cruise)')

        self.add_input('udot', val=1.0 * np.ones(N), units='m/s**2', desc='Acceleration in the x-direction of the body axis')


        # Outputs
        self.add_output('thrust_total', val=1.0 * np.ones(N), units='N', desc='Required thrust')


    def setup_partials(self):
        self.declare_partials('*', '*', method='exact')
        
    def compute(self, inputs, outputs):

        # Unpack inputs
        w_mto = inputs['w_mto']
        drag = inputs['drag']
        gamma = inputs['gamma']
        udot = inputs['udot']

        # Constants
        g = self.options['g'] # m/s**2  

        m_mto = w_mto / g 

        thrust = drag + w_mto * np.sin(gamma) + m_mto * udot # Thrust required 
        
        # Pack outputs
        outputs['thrust_total'] = thrust
                
    def compute_partials(self, inputs, partials):

        # Unpack inputs
        w_mto = inputs['w_mto']
        drag = inputs['drag']
        gamma = inputs['gamma']
        udot = inputs['udot']

        g = self.options['g'] # m/s**2
        N = self.options['N']

        # Parials with respect to thrust_total
        partials['thrust_total', 'w_mto'] = np.sin(gamma) + udot/g

        partials['thrust_total', 'gamma'] = np.eye(N) * w_mto*np.cos(gamma)

        partials['thrust_total', 'udot'] = np.eye(N) * w_mto/g

        partials['thrust_total', 'drag'] = np.eye(N) * 1.0





if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('w_mto', val=100000.0, units='N', desc='Maximum takeoff weight')
    ivc.add_output('drag', val=20000.0, units='N', desc='Total drag')
    ivc.add_output('gamma', val=0.0, units='rad', desc='Flight path angle')
    ivc.add_output('udot', val=0.0, units='m/s**2', desc='Acceleration')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('thrust', ThrustReq(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Takeoff Weight:       {prob.get_val("w_mto")} N')
    print(f'  Drag:                {prob.get_val("drag")} N')
    print(f'  Flight Path Angle:    {np.degrees(prob.get_val("gamma"))} deg')
    print(f'  Acceleration:         {prob.get_val("udot")} m/s²')
    print(f'  Required Thrust:      {prob.get_val("thrust_total")} N')
    

    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Weight sweep
    w_range = np.linspace(50000, 150000, 50)
    T_w = []
    for w in w_range:
        prob.set_val('w_mto', w)
        prob.run_model()
        T_w.append(prob.get_val('thrust_total')[0])
    
    plt.subplot(221)
    plt.plot(w_range/1000, np.array(T_w)/1000)
    plt.xlabel('Takeoff Weight (kN)')
    plt.ylabel('Required Thrust (kN)')
    plt.grid(True)
    plt.title('Effect of Weight')
    
    # Drag sweep
    prob.set_val('w_mto', 100000)  # Reset to baseline
    drag_range = np.linspace(10000, 30000, 50)
    T_d = []
    for d in drag_range:
        prob.set_val('drag', d)
        prob.run_model()
        T_d.append(prob.get_val('thrust_total')[0])
    
    plt.subplot(222)
    plt.plot(drag_range/1000, np.array(T_d)/1000)
    plt.xlabel('Drag (kN)')
    plt.ylabel('Required Thrust (kN)')
    plt.grid(True)
    plt.title('Effect of Drag')
    
    # Flight path angle sweep
    prob.set_val('drag', 20000)  # Reset to baseline
    gamma_range = np.linspace(-5, 15, 50)  # degrees
    T_gamma = []
    for gamma in gamma_range:
        prob.set_val('gamma', np.radians(gamma))
        prob.run_model()
        T_gamma.append(prob.get_val('thrust_total')[0])
    
    plt.subplot(223)
    plt.plot(gamma_range, np.array(T_gamma)/1000)
    plt.xlabel('Flight Path Angle (deg)')
    plt.ylabel('Required Thrust (kN)')
    plt.grid(True)
    plt.title('Effect of Flight Path Angle')
    
    # Acceleration sweep
    prob.set_val('gamma', 0.0)  # Reset to baseline
    udot_range = np.linspace(-2, 4, 50)
    T_udot = []
    for udot in udot_range:
        prob.set_val('udot', udot)
        prob.run_model()
        T_udot.append(prob.get_val('thrust_total')[0])
    
    plt.subplot(224)
    plt.plot(udot_range, np.array(T_udot)/1000)
    plt.xlabel('Acceleration (m/s²)')
    plt.ylabel('Required Thrust (kN)')
    plt.grid(True)
    plt.title('Effect of Acceleration')
    
    plt.tight_layout()
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.show()
    

