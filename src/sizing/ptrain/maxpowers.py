import numpy as np
import openmdao.api as om

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExtractMaxPower(om.ExplicitComponent):
    """

    Extracts maximum power values for powertrain components based on shaft power demands
    and component efficiencies:
    - Maximum unit motor power
    - Maximum unit generator power

    - Maximum unit turbine power
    - Maximum battery power output
    """

    def initialize(self):
        self.options.declare('N_climb', default=1, desc='Number of analysis points')
        self.options.declare('N_cruise', default=1, desc='Number of analysis points')
    

    def setup(self):

        N_climb = self.options['N_climb']
        N_cruise = self.options['N_cruise']
        # Shaft power demands
        self.add_input('p_shaft_manta_climb', val=np.ones(N_climb), units='W',
                      desc='Unit shaft power required by Manta ducted fans')

        self.add_input('p_shaft_ray_climb', val=np.ones(N_climb), units='W',
                      desc='Unit shaft power required by Ray propellers')

        self.add_input('p_shaft_manta_cruise', val=np.ones(N_cruise), units='W',
                      desc='Unit shaft power required by Manta ducted fans')



        self.add_input('ray_elec_power_climb', val=np.ones(N_climb), units='W',
                      desc='Maximum electrical power required by Ray propellers')
        self.add_input('manta_elec_power_climb', val=np.ones(N_climb), units='W',
                      desc='Maximum electrical power required by Manta ducted fans')




        
        # Component counts
        self.add_input('n_motors_manta', val=4,
                      desc='Number of Manta motors')

        self.add_input('n_motors_ray', val=6,
                      desc='Number of Ray motors')
        self.add_input('n_gens', val=2,
                      desc='Number of generators')
        self.add_input('n_turbines', val=2,
                      desc='Number of turbines')
        
        # Efficiencies
        self.add_input('eta_motor', val=0.95,
                      desc='Motor efficiency')
        self.add_input('eta_gen', val=0.95,
                      desc='Generator efficiency')
        self.add_input('eta_cable_manta', val=0.98,
                      desc='Cable efficiency')
        self.add_input('eta_pe', val=0.98,
                      desc='Power electronics efficiency')
        self.add_input('eta_inverter_manta', val=0.98,
                      desc='Inverter efficiency')
        
        # Outputs
        self.add_output('p_motor_manta_unit_max', units='W',
                       desc='Maximum unit motor power')
        self.add_output('p_motor_ray_unit_max', units='W',
                       desc='Maximum unit motor power')
        self.add_output('p_gen_unit_max', units='W',
                       desc='Maximum unit generator power')

        self.add_output('p_turbine_unit_max', units='W',
                       desc='Maximum unit turbine power')
        self.add_output('p_batt_max', units='W',
                       desc='Maximum battery power output')
        self.add_output('p_total_elec_power_climb_max', units='W',
                       desc='Maximum total electrical power required by Manta and Ray')
        self.add_output('p_manta_elec_power_cruise_max', units='W',
                       desc='Maximum electrical power required by Manta')
        
        
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):

        # Extract inputs
        p_shaft_manta_climb = inputs['p_shaft_manta_climb']
        p_shaft_ray_climb = inputs['p_shaft_ray_climb']
        p_shaft_manta_cruise = inputs['p_shaft_manta_cruise']

        n_motors_manta = inputs['n_motors_manta']
        n_motors_ray = inputs['n_motors_ray']
        n_gens = inputs['n_gens']
        n_turbines = inputs['n_turbines']

        # Get efficiencies
        eta_motor = inputs['eta_motor']
        eta_gen = inputs['eta_gen']
        eta_cable_manta = inputs['eta_cable_manta']
        eta_pe = inputs['eta_pe']
        eta_inverter_manta = inputs['eta_inverter_manta']

        # Unpack electrical power
        ray_elec_power_climb = inputs['ray_elec_power_climb']
        manta_elec_power_climb = inputs['manta_elec_power_climb']


        manta_elec_power_cruise = p_shaft_manta_cruise * n_motors_manta / eta_motor / eta_cable_manta / eta_pe / eta_inverter_manta
        total_elec_power_climb = ray_elec_power_climb + manta_elec_power_climb

        
        # Maximum unit motor power (electrical)
        outputs['p_motor_manta_unit_max'] = max(np.concatenate([p_shaft_manta_climb, p_shaft_manta_cruise]))
        outputs['p_motor_ray_unit_max'] = max(p_shaft_ray_climb)

        outputs['p_gen_unit_max'] = max(manta_elec_power_cruise/n_gens)
        outputs['p_turbine_unit_max'] = max(manta_elec_power_cruise/n_turbines/eta_gen)
    
        outputs['p_batt_max'] = max(total_elec_power_climb)
        outputs['p_total_elec_power_climb_max'] = max(total_elec_power_climb)
        outputs['p_manta_elec_power_cruise_max'] = max(manta_elec_power_cruise)
        

    def compute_partials(self, inputs, partials):
        N_climb = self.options['N_climb']
        N_cruise = self.options['N_cruise']
        

        # Extract inputs
        p_shaft_manta_climb = inputs['p_shaft_manta_climb']
        p_shaft_ray_climb = inputs['p_shaft_ray_climb']
        p_shaft_manta_cruise = inputs['p_shaft_manta_cruise']
        n_motors_manta = inputs['n_motors_manta']
        n_motors_ray = inputs['n_motors_ray']
        n_gens = inputs['n_gens']
        n_turbines = inputs['n_turbines']
        
        # Get efficiencies
        eta_motor = inputs['eta_motor']
        eta_gen = inputs['eta_gen']
        eta_cable_manta = inputs['eta_cable_manta']
        eta_pe = inputs['eta_pe']
        eta_inverter_manta = inputs['eta_inverter_manta']
        
        # Unpack electrical power
        ray_elec_power_climb = inputs['ray_elec_power_climb']
        manta_elec_power_climb = inputs['manta_elec_power_climb']
        
        # Calculate electrical power for cruise
        manta_elec_power_cruise = p_shaft_manta_cruise * n_motors_manta / eta_motor / eta_cable_manta / eta_pe / eta_inverter_manta
        
        # Find indices of maximum values
        idx_manta_climb = np.argmax(p_shaft_manta_climb)
        idx_manta_cruise = np.argmax(p_shaft_manta_cruise)
        idx_ray_climb = np.argmax(p_shaft_ray_climb)
        idx_manta_elec_cruise = np.argmax(manta_elec_power_cruise)
        idx_total_elec_climb = np.argmax(ray_elec_power_climb + manta_elec_power_climb)
        
        # Initialize derivative arrays
        d_manta_climb = np.zeros(N_climb)
        d_manta_cruise = np.zeros(N_cruise)
        d_ray_climb = np.zeros(N_climb)
        

        # Set derivatives at maximum points for Manta
        if p_shaft_manta_climb[idx_manta_climb] >= p_shaft_manta_cruise[idx_manta_cruise]:
            d_manta_climb[idx_manta_climb] = 1.0
        else:
            d_manta_cruise[idx_manta_cruise] = 1.0
        
        # Set derivatives for Ray
        d_ray_climb[idx_ray_climb] = 1.0
        
        # Assign partial derivatives for motors
        partials['p_motor_manta_unit_max', 'p_shaft_manta_climb'] = d_manta_climb
        partials['p_motor_manta_unit_max', 'p_shaft_manta_cruise'] = d_manta_cruise
        # Zero out other partials for Manta motor
        partials['p_motor_manta_unit_max', 'p_shaft_ray_climb'] = np.zeros(N_climb)
        
        # Ray motor partials
        partials['p_motor_ray_unit_max', 'p_shaft_ray_climb'] = d_ray_climb
        # Zero out other partials for Ray motor
        partials['p_motor_ray_unit_max', 'p_shaft_manta_climb'] = np.zeros(N_climb)
        partials['p_motor_ray_unit_max', 'p_shaft_manta_cruise'] = np.zeros(N_cruise)
        

        # Generator power derivatives
        d_gen = np.zeros(N_cruise)
        d_gen[idx_manta_elec_cruise] = n_motors_manta / (n_gens * eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        partials['p_gen_unit_max', 'p_shaft_manta_cruise'] = d_gen

        partials['p_gen_unit_max', 'n_motors_manta'] = p_shaft_manta_cruise[idx_manta_elec_cruise] / (n_gens * eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        partials['p_gen_unit_max', 'n_gens'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_gens**2)
        partials['p_gen_unit_max', 'eta_motor'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_gens * eta_motor)
        partials['p_gen_unit_max', 'eta_cable_manta'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_gens * eta_cable_manta)
        partials['p_gen_unit_max', 'eta_pe'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_gens * eta_pe)
        partials['p_gen_unit_max', 'eta_inverter_manta'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_gens * eta_inverter_manta)
        
        # Turbine power derivatives
        d_turbine = np.zeros(N_cruise)
        d_turbine[idx_manta_elec_cruise] = n_motors_manta / (n_turbines * eta_gen * eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        partials['p_turbine_unit_max', 'p_shaft_manta_cruise'] = d_turbine
        partials['p_turbine_unit_max', 'n_motors_manta'] = p_shaft_manta_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen * eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        partials['p_turbine_unit_max', 'n_turbines'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines**2 * eta_gen)
        partials['p_turbine_unit_max', 'eta_gen'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen**2)
        partials['p_turbine_unit_max', 'eta_motor'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen * eta_motor)
        partials['p_turbine_unit_max', 'eta_cable_manta'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen * eta_cable_manta)
        partials['p_turbine_unit_max', 'eta_pe'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen * eta_pe)
        partials['p_turbine_unit_max', 'eta_inverter_manta'] = -manta_elec_power_cruise[idx_manta_elec_cruise] / (n_turbines * eta_gen * eta_inverter_manta)
        
        # Battery power derivatives
        d_batt_ray = np.zeros(N_climb)
        d_batt_manta = np.zeros(N_climb)
        d_batt_ray[idx_total_elec_climb] = 1.0
        d_batt_manta[idx_total_elec_climb] = 1.0

        partials['p_batt_max', 'ray_elec_power_climb'] = d_batt_ray
        partials['p_batt_max', 'manta_elec_power_climb'] = d_batt_manta

        # Total electrical power climb max derivatives
        d_total_elec_climb = np.zeros(N_climb)
        idx_total_elec_climb = np.argmax(ray_elec_power_climb + manta_elec_power_climb)
        d_total_elec_climb[idx_total_elec_climb] = 1.0
        
        partials['p_total_elec_power_climb_max', 'ray_elec_power_climb'] = d_total_elec_climb
        partials['p_total_elec_power_climb_max', 'manta_elec_power_climb'] = d_total_elec_climb
        
        # Manta electrical power cruise max derivatives
        d_manta_cruise = np.zeros(N_cruise)
        idx_manta_cruise = np.argmax(manta_elec_power_cruise)
        d_manta_cruise[idx_manta_cruise] = n_motors_manta / (eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        
        partials['p_manta_elec_power_cruise_max', 'p_shaft_manta_cruise'] = d_manta_cruise
        partials['p_manta_elec_power_cruise_max', 'n_motors_manta'] = (
            p_shaft_manta_cruise[idx_manta_cruise] / (eta_motor * eta_cable_manta * eta_pe * eta_inverter_manta)
        )
        partials['p_manta_elec_power_cruise_max', 'eta_motor'] = (
            -manta_elec_power_cruise[idx_manta_cruise] / eta_motor
        )
        partials['p_manta_elec_power_cruise_max', 'eta_cable_manta'] = (
            -manta_elec_power_cruise[idx_manta_cruise] / eta_cable_manta
        )
        partials['p_manta_elec_power_cruise_max', 'eta_pe'] = (
            -manta_elec_power_cruise[idx_manta_cruise] / eta_pe
        )
        partials['p_manta_elec_power_cruise_max', 'eta_inverter_manta'] = (
            -manta_elec_power_cruise[idx_manta_cruise] / eta_inverter_manta
        )

if __name__ == "__main__":
    import numpy as np
    from openmdao.api import Problem, Group, IndepVarComp
    
    # Create OpenMDAO problem
    prob = Problem()
    model = Group()
    
    # Add independent variables
    N_climb = 10  # Number of analysis points
    N_cruise = 10  # Number of analysis points
    ivc = IndepVarComp()
    

    # Add vector inputs
    ivc.add_output('p_shaft_manta_climb', val=np.linspace(1000, 2000, N_climb), units='W')
    ivc.add_output('p_shaft_ray_climb', val=np.linspace(800, 1500, N_climb), units='W')

    ivc.add_output('p_shaft_manta_cruise', val=np.linspace(500, 1500, N_cruise), units='W')
    ivc.add_output('ray_elec_power_climb', val=np.linspace(1000, 2000, N_climb), units='W')

    ivc.add_output('manta_elec_power_climb', val=np.linspace(1500, 2500, N_climb), units='W')
    

    # Add scalar inputs
    ivc.add_output('n_motors_manta', val=4)
    ivc.add_output('n_motors_ray', val=6)
    ivc.add_output('n_gens', val=2)
    ivc.add_output('n_turbines', val=2)
    ivc.add_output('eta_motor', val=0.95)
    ivc.add_output('eta_gen', val=0.95)
    ivc.add_output('eta_cable_manta', val=0.98)
    ivc.add_output('eta_pe', val=0.98)
    ivc.add_output('eta_inverter_manta', val=0.98)
    
    # Add components to model
    model.add_subsystem('ivc', ivc, promotes_outputs=['*'])
    model.add_subsystem('power_extract', ExtractMaxPower(N_climb=N_climb, N_cruise=N_cruise), promotes_inputs=['*'])
    
    # Set up problem
    prob.model = model
    prob.setup()
    #om.n2(prob)
    prob.run_model()
    
    # Print results
    print('Maximum unit motor power (Manta):', prob['power_extract.p_motor_manta_unit_max'], 'W')
    print('Maximum unit motor power (Ray):', prob['power_extract.p_motor_ray_unit_max'], 'W')
    print('Maximum unit generator power:', prob['power_extract.p_gen_unit_max'], 'W')
    print('Maximum unit turbine power:', prob['power_extract.p_turbine_unit_max'], 'W')
    print('Maximum battery power:', prob['power_extract.p_batt_max'], 'W')
    
    # Check partials
    prob.check_partials(compact_print=True)