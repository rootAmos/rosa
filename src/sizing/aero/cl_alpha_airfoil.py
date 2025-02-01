import openmdao.api as om
import numpy as np

class LiftCurveSlopeAirfoil(om.ExplicitComponent):
    """
    Calculates the lift curve slope for a lifting surface using compressibility corrections.
    
    Inputs:
        aspect_ratio : float
            Aspect ratio [-]
        M : float
            Mach number [-]
        phi_50 : float
            50% chord sweep angle [rad]
        cl_alpha_airfoil : float
            Airfoil section lift curve slope [1/rad]
    
    Outputs:
        CL_alpha : float
            Wing lift curve slope [1/rad]
    """
    
    def setup(self):
        # Inputs
        self.add_input('aspect_ratio', val=0.0, desc='Aspect ratio')
        self.add_input('mach', val=0.0, desc='machach number')
        self.add_input('phi_50', val=0.0, units='rad', desc='50% chord sweep angle')
        self.add_input('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Airfoil lift curve slope')
        
        # Outputs
        self.add_output('CL_alpha', val=0.0, units='1/rad', desc='Wing lift curve slope')
        
        # Declare partials
        self.declare_partials('CL_alpha', ['aspect_ratio', 'mach', 'phi_50', 'cl_alpha_airfoil'])
        
    def compute(self, inputs, outputs):
        
        aspect_ratio = inputs['aspect_ratio']
        mach = inputs['mach']
        phi_50 = inputs['phi_50']
        cl_alpha_airfoil = inputs['cl_alpha_airfoil']
        
        # Compute beta (Prandtl-Glauert correction)
        beta = np.sqrt(1 - mach**2)
        
        # Compute kappa
        kappa = cl_alpha_airfoil / (2*np.pi/beta)
        
        # Compute CL_alpha
        term1 = 2*np.pi*aspect_ratio
        term2 = 2 + np.sqrt((aspect_ratio**2 * beta**2 / kappa**2) * (1 + np.tan(phi_50)**2/beta**2)) + 4
        
        outputs['CL_alpha'] = term1/term2
        
    def compute_partials(self, inputs, partials):
        aspect_ratio = inputs['aspect_ratio']
        mach = inputs['mach']
        phi_50 = inputs['phi_50']
        cl_alpha_airfoil = inputs['cl_alpha_airfoil']


        partials['CL_alpha', 'aspect_ratio'] = 6.2832/(6.2832*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**(1/2) + 6) + (39.4784*aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/(cl_alpha_airfoil**2*(6.2832*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**(1/2) + 6)**2*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**0.5000)

        partials['CL_alpha', 'mach'] = -(39.4784*aspect_ratio**3*mach*np.tan(phi_50)**2)/(cl_alpha_airfoil**2*(mach**2 - 1)**2*(6.2832*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**(1/2) + 6)**2*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**0.5000)    

        partials['CL_alpha', 'phi_50'] = (39.4784*aspect_ratio**3*np.tan(phi_50)*(np.tan(phi_50)**2 + 1))/(cl_alpha_airfoil**2*(mach**2 - 1)*(6.2832*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**(1/2) + 6)**2*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**0.5000)

        partials['CL_alpha', 'cl_alpha_airfoil'] = -(39.4784*aspect_ratio**3*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/(cl_alpha_airfoil**3*(6.2832*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**(1/2) + 6)**2*(-(aspect_ratio**2*(np.tan(phi_50)**2/(mach**2 - 1) - 1))/cl_alpha_airfoil**2)**0.5000)

if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Create IndepVarComp for inputs
    ivc = om.IndepVarComp()
    ivc.add_output('aspect_ratio', val=8.0, desc='Aspect ratio')
    ivc.add_output('mach', val=0.3, desc='Mach number')
    ivc.add_output('phi_50', val=np.radians(5.0), units='rad', desc='50% chord sweep angle')
    ivc.add_output('cl_alpha_airfoil', val=2*np.pi, units='1/rad', desc='Airfoil lift curve slope')
    
    # Add IVC and LiftCurveSlope component to model
    prob.model.add_subsystem('inputs', ivc, promotes=['*'])
    prob.model.add_subsystem('wing', LiftCurveSlopeAirfoil(), promotes=['*'])
    
    # Setup and run problem
    prob.setup()
    prob.run_model()
    
    # Print results
    print('\nInputs:')
    print(f'  Aspect Ratio:          {prob.get_val("aspect_ratio")}')
    print(f'  Mach:                  {prob.get_val("mach")}')
    print(f'  Sweep (phi_50):        {prob.get_val("phi_50")}')
    print(f'  Airfoil cl_alpha:      {prob.get_val("cl_alpha_airfoil")}')
    print('\nOutput:')
    print(f'  Wing CL_alpha:         {prob.get_val("CL_alpha")}')
    
    # Check partials
    prob.check_partials(compact_print=True)