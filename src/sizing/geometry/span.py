import openmdao.api as om
import numpy as np

class Span(om.ExplicitComponent):
    """
    Computes wing span from aspect ratio and wing area using:
    span = sqrt(wing_area * aspect_ratio)
    """
    

    def setup(self):
        # Inputs
        self.add_input('S', val=1.0, units='m**2',
                      desc='Wing reference area')
        self.add_input('aspect_ratio', val=1.0,
                      desc='Wing aspect ratio')
        
        # Outputs
        self.add_output('span', val=1.0, units='m',
                      desc='Wing span')
        
        # Declare partials
        self.declare_partials('span', ['S', 'aspect_ratio'])
        
    def compute(self, inputs, outputs):
        S = inputs['S']
        aspect_ratio = inputs['aspect_ratio']
        
        outputs['span'] = np.sqrt(S * aspect_ratio)
        
    def compute_partials(self, inputs, partials):
        S = inputs['S']
        aspect_ratio = inputs['aspect_ratio']
        
        partials['span', 'S'] = 0.5 * np.sqrt(aspect_ratio/S)
        partials['span', 'aspect_ratio'] = 0.5 * np.sqrt(S/aspect_ratio)

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create independent variable component
    ivc = om.IndepVarComp()
    
    # Example values
    S = 100.0  # m^2
    AR = 10.0
    
    ivc.add_output('S', val=S, units='m**2')
    ivc.add_output('AR', val=AR)
    
    # Build the model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('span', Span(), promotes=['*'])
    
    # Setup problem
    prob.setup()
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nSpan Calculation Results:")
    print("--------------------------")
    print(f"Wing Area: {prob.get_val('S')[0]:.1f} m²")
    print(f"Aspect Ratio: {prob.get_val('AR')[0]:.1f}")
    print(f"Span: {prob.get_val('b')[0]:.1f} m")
    
    # Check partials
    prob.check_partials(compact_print=True)