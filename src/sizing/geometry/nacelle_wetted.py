import openmdao.api as om
import numpy as np

class NacelleWettedArea(om.ExplicitComponent):
    """
    Calculates the wetted area of a ducted duct nacelle using equation 13.12.
    
    Inputs:
        c_duct : float
            Nacelle length [m]
        od_duct : float
            Nacelle diameter [m]
        mid_c_duct : float
            Inlet length [m]
        id_duct : float
            Highlight diameter [m]
        ed_duct : float
            Exit diameter [m]
    
    Outputs:
        S_wet_duct : float
            Nacelle wetted area [m**2]
    """
    
    def setup(self):
        # Inputs
        self.add_input('c_duct', val=0.0, units='m', desc='Nacelle length')
        self.add_input('od_duct', val=0.0, units='m', desc='Nacelle outer diameter')
        self.add_input('id_duct', val=0.0, units='m', desc='Nacelle inner diameter')
        

        # Output
        self.add_output('S_wet_duct', val=0.0, units='m**2', 
                       desc='Nacelle wetted area')
        
        # Declare partials
        self.declare_partials('S_wet_duct', 
                            ['c_duct', 'od_duct', 'id_duct'])
        

    def compute(self, inputs, outputs):

        # Inputs
        c_duct = inputs['c_duct']
        od_duct = inputs['od_duct']
        id_duct = inputs['id_duct']

        # Duct effective diameter
        ed_duct = (od_duct + id_duct) / 2

        # Duct mid length
        mid_c_duct = c_duct / 2
        
        # Compute terms in brackets
        term1 = 2.0
        term2 = 0.35 * mid_c_duct/c_duct

        term3 = 0.8 * mid_c_duct * id_duct/(c_duct * od_duct)
        term4 = 1.15 * (1 - mid_c_duct/c_duct) * ed_duct/od_duct
        
        # Compute wetted area using equation 13.12
        outputs['S_wet_duct'] = c_duct * od_duct * (term1 + term2 + term3 + term4)
        
    def compute_partials(self, inputs, partials):

        c_duct = inputs['c_duct']
        od_duct = inputs['od_duct']
        id_duct = inputs['id_duct']
        
        # Partial derivatives

        partials['S_wet_duct', 'c_duct'] = od_duct*((0.4000*id_duct)/od_duct + (0.2875*id_duct + 0.2875*od_duct)/od_duct + 2.1750)

        # With respect to od_duct
        partials['S_wet_duct', 'od_duct'] = c_duct*((0.4000*id_duct)/od_duct + (0.2875*id_duct + 0.2875*od_duct)/od_duct + 2.1750) - c_duct*od_duct*((0.4000*id_duct)/od_duct**2 + (0.2875*id_duct + 0.2875*od_duct)/od_duct**2 - 0.2875/od_duct)

        # With respect to id_duct
        partials['S_wet_duct', 'id_duct'] = 0.6875*c_duct

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp
    ivc = om.IndepVarComp()
    ivc.add_output('c_duct', val=3.0, units='m', desc='Nacelle length')
    ivc.add_output('od_duct', val=2.0, units='m', desc='Nacelle outer diameter')
    ivc.add_output('id_duct', val=1.5, units='m', desc='Nacelle inner diameter')
    
    # Add subsystems to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('nacelle', NacelleWettedArea(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run baseline case
    prob.run_model()
    
    print('\nBaseline Configuration:')
    print('----------------------')
    print(f'  Nacelle Length:       {prob.get_val("c_duct")} m')
    print(f'  Outer Diameter:       {prob.get_val("od_duct")} m')
    print(f'  Inner Diameter:       {prob.get_val("id_duct")} m')

    print(f'  Wetted Area:         {prob.get_val("S_wet_duct")} m²')
    
    # Parameter sweeps
    print('\nParameter Sweeps:')
    print('----------------')
    
    # Length sweep
    c_range = np.linspace(2.0, 5.0, 50)
    S_wet_c = []
    for c in c_range:
        prob.set_val('c_duct', c)
        prob.run_model()
        S_wet_c.append(prob.get_val('S_wet_duct')[0])
    
    # Outer diameter sweep
    od_range = np.linspace(1.5, 3.0, 50)
    S_wet_od = []
    prob.set_val('c_duct', 3.0)  # Reset length
    for od in od_range:
        prob.set_val('od_duct', od)
        prob.run_model()
        S_wet_od.append(prob.get_val('S_wet_duct')[0])
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(c_range, S_wet_c)
    plt.xlabel('Nacelle Length (m)')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Nacelle Length')
    
    plt.subplot(122)
    plt.plot(od_range, S_wet_od)
    plt.xlabel('Outer Diameter (m)')
    plt.ylabel('Wetted Area (m²)')
    plt.grid(True)
    plt.title('Effect of Outer Diameter')
    
    plt.tight_layout()
    plt.show()

    prob.check_partials(compact_print=True)