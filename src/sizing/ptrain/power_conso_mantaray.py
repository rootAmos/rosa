import openmdao.api as om
import numpy as np
import pdb
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid

class ElecPowerMantaRay(om.ExplicitComponent):
    """

    Computes total electrical power demand accounting for efficiencies.
    
    Inputs:
        r_shaft_power_unit : float
            Shaft power per Ray motor [W]
        m_shaft_power_unit : float
            Shaft power per Manta motor [W]
        num_pods : float
            Number of Ray motors [-]
        num_ducts : float
            Number of Manta motors [-]
        eta_motor_ray : float
            Ray motor efficiency [-]
        eta_motor_manta : float
            Manta motor efficiency [-]
        eta_inverter_ray : float
            Ray inverter efficiency [-]
        eta_inverter_manta : float
            Manta inverter efficiency [-]
        eta_cable_ray : float
            Ray cable efficiency [-]
        eta_cable_manta : float
            Manta cable efficiency [-]
            
    Outputs:
        ray_elec_power : float
            Total Ray electrical power demand [W]
        manta_elec_power : float
            Total Manta electrical power demand [W]
        total_elec_power : float
            Total system electrical power demand [W]
    """

    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):
        N = self.options['N']
        # Inputs - Powers

        self.add_input('r_shaft_power_unit', val=1.0 * np.ones(N), units='W')
        self.add_input('m_shaft_power_unit', val=1.0 * np.ones(N), units='W')
        self.add_input('num_pods', val=1.0)
        self.add_input('num_ducts', val=1.0)
        

        # Inputs - Efficiencies
        self.add_input('eta_motor_ray', val=0.95)
        self.add_input('eta_motor_manta', val=0.95)
        self.add_input('eta_inverter_ray', val=0.95)
        self.add_input('eta_inverter_manta', val=0.95)
        self.add_input('eta_cable_ray', val=0.98)
        self.add_input('eta_cable_manta', val=0.98)
        
        # Outputs
        self.add_output('ray_elec_power', val=1.0 * np.ones(N), units='W')
        self.add_output('manta_elec_power', val=1.0 * np.ones(N), units='W')
        self.add_output('total_elec_power', val=1.0 * np.ones(N), units='W')
        

        self.declare_partials('*','*')
        

    def compute(self, inputs, outputs):
        # Compute total electrical power for Ray
        ray_efficiency = (inputs['eta_motor_ray'] * 
                         inputs['eta_inverter_ray'] * 
                         inputs['eta_cable_ray'])
        
        outputs['ray_elec_power'] = (inputs['r_shaft_power_unit'] * 
                                    inputs['num_pods'] / ray_efficiency)
        
        # Compute total electrical power for Manta
        manta_efficiency = (inputs['eta_motor_manta'] * 
                          inputs['eta_inverter_manta'] * 
                          inputs['eta_cable_manta'])
        
        outputs['manta_elec_power'] = (inputs['m_shaft_power_unit'] * 
                                      inputs['num_ducts'] / manta_efficiency)
        
        # Total electrical power
        outputs['total_elec_power'] = (outputs['ray_elec_power'] + 
                                     outputs['manta_elec_power'])
        

    def compute_partials(self, inputs, partials):

        N = self.options['N']
        # Ray power partials
        ray_efficiency = (inputs['eta_motor_ray'] * 
                         inputs['eta_inverter_ray'] * 
                         inputs['eta_cable_ray'])
        
        partials['ray_elec_power', 'r_shaft_power_unit'] = np.eye(N) * inputs['num_pods'] / ray_efficiency
        partials['ray_elec_power', 'num_pods'] = inputs['r_shaft_power_unit'] / ray_efficiency
        partials['ray_elec_power', 'eta_motor_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_inverter_ray'] * inputs['eta_cable_ray']
        partials['ray_elec_power', 'eta_inverter_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_motor_ray'] * inputs['eta_cable_ray']
        partials['ray_elec_power', 'eta_cable_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_motor_ray'] * inputs['eta_inverter_ray']
        
        # Manta power partials
        manta_efficiency = (inputs['eta_motor_manta'] * 
                          inputs['eta_inverter_manta'] * 
                          inputs['eta_cable_manta'])
        
        partials['manta_elec_power', 'm_shaft_power_unit'] =np.eye(N) * inputs['num_ducts'] / manta_efficiency
        partials['manta_elec_power', 'num_ducts'] = inputs['m_shaft_power_unit'] / manta_efficiency
        partials['manta_elec_power', 'eta_motor_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_ducts'] / (manta_efficiency**2) * inputs['eta_inverter_manta'] * inputs['eta_cable_manta']
        partials['manta_elec_power', 'eta_inverter_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_ducts'] / (manta_efficiency**2) * inputs['eta_motor_manta'] * inputs['eta_cable_manta']
        partials['manta_elec_power', 'eta_cable_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_ducts'] / (manta_efficiency**2) * inputs['eta_motor_manta'] * inputs['eta_inverter_manta']
        
        # Total power partials
        #partials['total_elec_power', 'ray_elec_power'] = np.eye(N)
        #partials['total_elec_power', 'manta_elec_power'] = np.eye(N)


class EnergyConso(om.ExplicitComponent):
    """
    Computes energy consumption and fuel mass based on power loads.
    
    Inputs:
        total_elec_power : float
            Total electrical power demand [W]
        ray_energy_ratio : float
            Ratio of Ray battery energy to total [-]
        duration : float
            Flight phase duration [s]
        eta_battery : float
            Battery system efficiency [-]
        eta_generator : float
            Turbogenerator efficiency [-]
        eta_gen_inverter : float
            Generator inverter efficiency [-]
        psfc : float
            Power specific fuel consumption [kg/J]
            
    Outputs:
        ray_battery_power : float
            Power drawn from Ray battery [W]
        turbogen_power : float
            Power required from turbogenerator [W]
        ray_battery_energy : float
            Energy consumed from Ray battery [J]
        manta_fuel_mass : float
            Fuel mass consumed by turbogenerator [kg]
    """

    def initialize(self):   
        self.options.declare('N', default=1, desc='Number of nodes')
    
    def setup(self):
        N = self.options['N']
        # Inputs

        self.add_input('total_elec_power', val=1.0 * np.ones(N), units='W')
        self.add_input('ray_energy_ratio', val=1.0)
        self.add_input('duration', val=1.0, units='s')
        self.add_input('eta_battery', val=1.0)
        self.add_input('eta_generator', val=1.0)
        self.add_input('eta_gen_inverter', val=1.0)
        self.add_input('psfc', val=1.0, units='kg/J')
        

        # Outputs
        self.add_output('ray_battery_power', val=1.0 * np.ones(N), units='W')
        self.add_output('turbogen_power', val=1.0 * np.ones(N), units='W')
        self.add_output('ray_battery_energy', val=1.0 , units='J')
        self.add_output('manta_fuel_mass', val=1.0 , units='kg')
        


        self.declare_partials('*','*')
        


    def compute(self, inputs, outputs):

        N = self.options['N']



        ray_battery_power = inputs['total_elec_power'] * inputs['ray_energy_ratio']
        turbogen_power = inputs['total_elec_power'] * (1.0 - inputs['ray_energy_ratio']) / (inputs['eta_generator'] * inputs['eta_gen_inverter'])


        time = np.linspace(0, inputs['duration'], N).flatten()
        ray_battery_energy = trapezoid(ray_battery_power / inputs['eta_battery'], time, axis=0)

        fuel_conso = trapezoid(turbogen_power * inputs['psfc'], time, axis=0)
  
        # Fuel mass using PSFC (kg/J * J = kg)
    
        outputs['manta_fuel_mass'] = fuel_conso 
        outputs['ray_battery_power'] = ray_battery_power
        outputs['turbogen_power'] = turbogen_power
        outputs['ray_battery_energy'] = ray_battery_energy



    def compute_partials(self, inputs, partials):
        N = self.options['N']
        
        # Battery power partials
        partials['ray_battery_power', 'total_elec_power'] = np.eye(N) * inputs['ray_energy_ratio']
        partials['ray_battery_power', 'ray_energy_ratio'] =  inputs['total_elec_power']
        
        # Turbogen power partials
        partials['turbogen_power', 'total_elec_power'] = np.eye(N) * (1.0 - inputs['ray_energy_ratio'])
        partials['turbogen_power', 'ray_energy_ratio'] =  -inputs['total_elec_power']
        
        # Battery energy partials (now scalar)
        time = np.linspace(0, inputs['duration'], N).flatten()
        ray_battery_power = inputs['total_elec_power'] * inputs['ray_energy_ratio']

        #partials['ray_battery_energy', 'ray_battery_power'] = trapezoid(np.ones(N) / inputs['eta_battery'], time, axis=0)
        partials['ray_battery_energy', 'eta_battery'] = -trapezoid(ray_battery_power / (inputs['eta_battery']**2), time, axis=0)
        partials['ray_battery_energy', 'duration'] = np.sum(ray_battery_power / inputs['eta_battery']) / N

        turbogen_power = inputs['total_elec_power'] * (1.0 - inputs['ray_energy_ratio']) / (inputs['eta_generator'] * inputs['eta_gen_inverter'])

        # Fuel mass partials (scalar)
        gen_efficiency = inputs['eta_generator'] * inputs['eta_gen_inverter']
        fuel_conso = trapezoid(turbogen_power * inputs['psfc'], time, axis=0)
        
        #partials['manta_fuel_mass', 'turbogen_power'] = np.sum(inputs['psfc'] / gen_efficiency) / N
        partials['manta_fuel_mass', 'duration'] = np.sum(turbogen_power * inputs['psfc'] / gen_efficiency) / N
        partials['manta_fuel_mass', 'eta_generator'] = -fuel_conso / inputs['eta_generator']


        partials['manta_fuel_mass', 'eta_gen_inverter'] = -fuel_conso / inputs['eta_gen_inverter']
        partials['manta_fuel_mass', 'psfc'] = np.sum(turbogen_power / gen_efficiency) / N


class PowerConsoGroup(om.Group):
    """
    Group that combines electrical power demand and energy consumption calculations.
    """
    
    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')
        
    def setup(self):
        N = self.options['N']
        
        # Add the components
        self.add_subsystem('elec_power', 
                       ElecPowerMantaRay(N = N),
                       promotes_inputs=['*'],
                       promotes_outputs=['*'])
    


        self.add_subsystem('energy',
                       EnergyConso(N = N),
                       promotes_inputs=['*'],
                       promotes_outputs=['*'])
        


if __name__ == "__main__":
    # Create problem instance
    prob = om.Problem()
    
    # Independent variable component
    ivc = om.IndepVarComp()

    N = 3
    
    # Add inputs for ElectricalPowerDemand
    ivc.add_output('r_shaft_power_unit', val=50000.0 * np.ones(N), units='W')
    ivc.add_output('m_shaft_power_unit', val=75000.0 * np.ones(N), units='W')
    ivc.add_output('num_pods', val=6)
    ivc.add_output('num_ducts', val=4)

    ivc.add_output('eta_motor_ray', val=0.95)
    ivc.add_output('eta_motor_manta', val=0.95)
    ivc.add_output('eta_inverter_ray', val=0.95)
    ivc.add_output('eta_inverter_manta', val=0.95)
    ivc.add_output('eta_cable_ray', val=0.98)
    ivc.add_output('eta_cable_manta', val=0.98)
    
    # Add inputs for EnergyConsumption
    ivc.add_output('ray_energy_ratio', val=0.3)
    ivc.add_output('duration', val=1200.0, units='s')
    ivc.add_output('eta_battery', val=0.95)
    ivc.add_output('eta_generator', val=0.90)
    ivc.add_output('eta_gen_inverter', val=0.95)
    ivc.add_output('psfc', val=2.0e-7, units='kg/J')
    
    # Build the model
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes=['*'])
    model.add_subsystem('power_conso', PowerConsoGroup(N = N), promotes=['*'])




    prob.setup()
    #om.n2(prob)
    
    # Run model
    prob.run_model()
    
    # Print results
    print("\nPower Demand Results:")
    print(f"Ray Electrical Power: {prob.get_val('ray_elec_power')[0]/1000:.1f} kW")
    print(f"Manta Electrical Power: {prob.get_val('manta_elec_power')[0]/1000:.1f} kW")
    print(f"Total Electrical Power: {prob.get_val('total_elec_power')[0]/1000:.1f} kW")
    
    print("\nEnergy Consumption Results:")
    print(f"Ray Battery Power: {prob.get_val('ray_battery_power')[0]/1000:.1f} kW")
    print(f"Turbogen Power: {prob.get_val('turbogen_power')[0]/1000:.1f} kW")
    print(f"Ray Battery Energy: {prob.get_val('ray_battery_energy')[0]/3.6e6:.1f} kWh")
    print(f"Manta Fuel Mass: {prob.get_val('manta_fuel_mass')[0]:.1f} kg")
    
    # Plot power distribution
    import matplotlib.pyplot as plt
    
    # Create data for pie charts
    power_labels = ['Ray Power', 'Manta Power']
    power_values = [prob.get_val('ray_elec_power')[0], 
                   prob.get_val('manta_elec_power')[0]]
    
    source_labels = ['Battery Power', 'Turbogen Power']
    source_values = [prob.get_val('ray_battery_power')[0],
                    prob.get_val('turbogen_power')[0]]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot power distribution
    ax1.pie(power_values, labels=power_labels, autopct='%1.1f%%')
    ax1.set_title('Power Distribution by Motor Type')
    
    # Plot power source distribution
    ax2.pie(source_values, labels=source_labels, autopct='%1.1f%%')
    ax2.set_title('Power Distribution by Source')
    
    plt.tight_layout()
    plt.show()
    
    # Check partials
    #prob.check_partials(compact_print=True) 