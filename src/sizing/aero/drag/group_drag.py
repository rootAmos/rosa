import openmdao.api as om
import numpy as np
from group_cd import GroupCD
class DragForce(om.ExplicitComponent):
    """
    Calculates drag force for a lifting surface (wing or canard).
    

    Inputs:
        rho : float
            Air density [kg/m^3]
        V : float
            Airspeed [m/s]
        CD : float
            Drag coefficient [-]
        S : float
            Reference area [m^2]
    
    Outputs:
        D : float
            Drag force [N]
    """
    
    def setup(self):
        # Flow condition inputs
        self.add_input('rho', val=1.225, units='kg/m**3', desc='Air density')
        self.add_input('u', val=0.0, units='m/s', desc='Airspeed')
        
        # Surface inputs
        self.add_input('CD', val=0.0, desc='Drag coefficient')
        self.add_input('S_ref', val=0.0, units='m**2', desc='Reference area')
        
        # Output
        self.add_output('drag', val=0.0, units='N', desc='Drag force')
        
        # Declare partials
        self.declare_partials('drag', ['rho', 'u', 'CD', 'S_ref'])
        
    def compute(self, inputs, outputs):
        
        # Dynamic pressure
        q = 0.5 * inputs['rho'] * inputs['u']**2
        
        # Drag force
        outputs['drag'] = q * inputs['CD'] * inputs['S_ref']
        
    def compute_partials(self, inputs, partials):

        # Common terms
        dq_drho = 0.5 * inputs['u']**2
        dq_dV = inputs['rho'] * inputs['u']
        
        partials['drag', 'rho'] = dq_drho * inputs['CD'] * inputs['S_ref']
        partials['drag', 'u'] = dq_dV * inputs['CD'] * inputs['S_ref']
        partials['drag', 'CD'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['S_ref']
        partials['drag', 'S_ref'] = 0.5 * inputs['rho'] * inputs['u']**2 * inputs['CD']


class GroupDrag(om.Group):
    """
    Group that computes total drag force by summing contributions
    from wing and canard.
    """
    
    def setup(self):
        # Wing drag force
        self.add_subsystem('cd', 
                          GroupCD(manta=self.options['manta'],
                                  ray=self.options['ray']),
                          promotes_inputs=[],
                          promotes_outputs=[])

        
        # Canard drag force
        self.add_subsystem('drag',
                          DragForce(),
                          promotes_inputs=['rho','u'],
                          promotes_outputs=[('drag', 'D_canard')])


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    
    # Flow conditions
    ivc.add_output('rho', val=1.225, units='kg/m**3', desc='Air density')
    ivc.add_output('u', val=100.0, units='m/s', desc='Airspeed')
    
    # Wing parameters
    ivc.add_output('CD_wing', val=0.015, desc='Wing drag coefficient')
    ivc.add_output('S_wing', val=120.0, units='m**2', desc='Wing reference area')
    
    # Canard parameters
    ivc.add_output('CD_canard', val=0.020, desc='Canard drag coefficient')
    ivc.add_output('S_canard', val=20.0, units='m**2', desc='Canard reference area')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('drag', GroupDrag(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print('Flow Conditions:')
    print(f'  Density:             {prob.get_val("rho")[0]:8.3f} kg/m³')
    print(f'  Velocity:            {prob.get_val("V")[0]:8.3f} m/s')
    
    print('\nWing:')
    print(f'  CD:                  {prob.get_val("CD_wing")[0]:8.4f}')
    print(f'  Area:                {prob.get_val("S_wing")[0]:8.3f} m²')
    print(f'  Drag Force:          {prob.get_val("D_wing")[0]/1000:8.3f} kN')
    
    print('\nCanard:')
    print(f'  CD:                  {prob.get_val("CD_canard")[0]:8.4f}')
    print(f'  Area:                {prob.get_val("S_canard")[0]:8.3f} m²')
    print(f'  Drag Force:          {prob.get_val("D_canard")[0]/1000:8.3f} kN')
    
    print('\nTotal:')
    print(f'  Drag Force:          {prob.get_val("D_total")[0]/1000:8.3f} kN')
    
    # Parameter sweeps
    import matplotlib.pyplot as plt
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(12, 10))
    
    # Velocity sweep
    V_range = np.linspace(50, 250, 50)
    D_wing = []
    D_canard = []
    D_total = []
    for V in V_range:
        prob.set_val('u', V)
        prob.run_model()
        D_wing.append(prob.get_val('D_wing')[0])
        D_canard.append(prob.get_val('D_canard')[0])
        D_total.append(prob.get_val('D_total')[0])
    
    plt.subplot(221)
    plt.plot(V_range, np.array(D_wing)/1000, label='Wing')
    plt.plot(V_range, np.array(D_canard)/1000, label='Canard')
    plt.plot(V_range, np.array(D_total)/1000, '--', label='Total')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Drag Force (kN)')
    plt.grid(True)
    plt.legend()
    plt.title('Effect of Velocity')
    
    # Reset velocity
    prob.set_val('u', 100.0)
    
    # CD sweep
    CD_range = np.linspace(0.01, 0.03, 50)
    D_total = []
    for CD in CD_range:
        prob.set_val('CD_wing', CD)
        prob.run_model()
        D_total.append(prob.get_val('D_total')[0])
    
    plt.subplot(222)
    plt.plot(CD_range, np.array(D_total)/1000)
    plt.xlabel('Wing CD')
    plt.ylabel('Total Drag Force (kN)')
    plt.grid(True)
    plt.title('Effect of Wing CD')
    
    # Check partials
    print("\nChecking partials...")
    prob.check_partials(compact_print=True)
    
    plt.tight_layout()
    plt.show() 