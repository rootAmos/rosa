import openmdao.api as om

class ElectricalPowerMantaRay(om.ExplicitComponent):
    """
    Computes total electrical power demand accounting for efficiencies.
    
    Inputs:
        r_shaft_power_unit : float
            Shaft power per Ray motor [W]
        m_shaft_power_unit : float
            Shaft power per Manta motor [W]
        num_pods : float
            Number of Ray motors [-]
        num_fans : float
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
    
    def setup(self):
        # Inputs - Powers
        self.add_input('r_shaft_power_unit', val=1.0, units='W')
        self.add_input('m_shaft_power_unit', val=1.0, units='W')
        self.add_input('num_pods', val=1.0)
        self.add_input('num_fans', val=1.0)
        
        # Inputs - Efficiencies
        self.add_input('eta_motor_ray', val=0.95)
        self.add_input('eta_motor_manta', val=0.95)
        self.add_input('eta_inverter_ray', val=0.95)
        self.add_input('eta_inverter_manta', val=0.95)
        self.add_input('eta_cable_ray', val=0.98)
        self.add_input('eta_cable_manta', val=0.98)
        
        # Outputs
        self.add_output('ray_elec_power', units='W')
        self.add_output('manta_elec_power', units='W')
        self.add_output('total_elec_power', units='W')
        
        self.declare_partials('ray_elec_power', 
                            ['r_shaft_power_unit', 'num_pods', 
                             'eta_motor_ray', 'eta_inverter_ray', 'eta_cable_ray'])
        
        self.declare_partials('manta_elec_power', 
                            ['m_shaft_power_unit', 'num_fans',
                             'eta_motor_manta', 'eta_inverter_manta', 'eta_cable_manta'])
        
        self.declare_partials('total_elec_power', 
                            ['ray_elec_power', 'manta_elec_power'])
        
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
                                      inputs['num_fans'] / manta_efficiency)
        
        # Total electrical power
        outputs['total_elec_power'] = (outputs['ray_elec_power'] + 
                                     outputs['manta_elec_power'])

    def compute_partials(self, inputs, partials):
        # Ray power partials
        ray_efficiency = (inputs['eta_motor_ray'] * 
                         inputs['eta_inverter_ray'] * 
                         inputs['eta_cable_ray'])
        
        partials['ray_elec_power', 'r_shaft_power_unit'] = inputs['num_pods'] / ray_efficiency
        partials['ray_elec_power', 'num_pods'] = inputs['r_shaft_power_unit'] / ray_efficiency
        partials['ray_elec_power', 'eta_motor_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_inverter_ray'] * inputs['eta_cable_ray']
        partials['ray_elec_power', 'eta_inverter_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_motor_ray'] * inputs['eta_cable_ray']
        partials['ray_elec_power', 'eta_cable_ray'] = -inputs['r_shaft_power_unit'] * inputs['num_pods'] / (ray_efficiency**2) * inputs['eta_motor_ray'] * inputs['eta_inverter_ray']
        
        # Manta power partials
        manta_efficiency = (inputs['eta_motor_manta'] * 
                          inputs['eta_inverter_manta'] * 
                          inputs['eta_cable_manta'])
        
        partials['manta_elec_power', 'm_shaft_power_unit'] = inputs['num_fans'] / manta_efficiency
        partials['manta_elec_power', 'num_fans'] = inputs['m_shaft_power_unit'] / manta_efficiency
        partials['manta_elec_power', 'eta_motor_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_fans'] / (manta_efficiency**2) * inputs['eta_inverter_manta'] * inputs['eta_cable_manta']
        partials['manta_elec_power', 'eta_inverter_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_fans'] / (manta_efficiency**2) * inputs['eta_motor_manta'] * inputs['eta_cable_manta']
        partials['manta_elec_power', 'eta_cable_manta'] = -inputs['m_shaft_power_unit'] * inputs['num_fans'] / (manta_efficiency**2) * inputs['eta_motor_manta'] * inputs['eta_inverter_manta']
        
        # Total power partials
        partials['total_elec_power', 'ray_elec_power'] = 1.0
        partials['total_elec_power', 'manta_elec_power'] = 1.0

class EnergyConsumption(om.ExplicitComponent):
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
    
    def setup(self):
        # Inputs
        self.add_input('total_elec_power', val=1.0, units='W')
        self.add_input('ray_energy_ratio', val=1.0)
        self.add_input('duration', val=1.0, units='s')
        self.add_input('eta_battery', val=1.0)
        self.add_input('eta_generator', val=1.0)
        self.add_input('eta_gen_inverter', val=1.0)
        self.add_input('psfc', val=1.0, units='kg/J')
        

        # Outputs
        self.add_output('ray_battery_power', units='W')
        self.add_output('turbogen_power', units='W')
        self.add_output('ray_battery_energy', units='J')
        self.add_output('manta_fuel_mass', units='kg')
        
        self.declare_partials('ray_battery_power', 
                            ['total_elec_power', 'ray_energy_ratio'])
        
        self.declare_partials('turbogen_power', 
                            ['total_elec_power', 'ray_energy_ratio'])
        
        self.declare_partials('ray_battery_energy', 
                            ['ray_battery_power', 'eta_battery', 'duration'])
        
        self.declare_partials('manta_fuel_mass', 
                            ['turbogen_power', 'duration', 'eta_generator', 
                             'eta_gen_inverter', 'psfc'])
        
    def compute(self, inputs, outputs):
        # Power distribution
        outputs['ray_battery_power'] = (inputs['total_elec_power'] * 
                                      inputs['ray_energy_ratio'])
        outputs['turbogen_power'] = (inputs['total_elec_power'] * 
                                   (1.0 - inputs['ray_energy_ratio']))
        
        # Energy and fuel calculations
        outputs['ray_battery_energy'] = (outputs['ray_battery_power'] / 
                                       inputs['eta_battery'] * 
                                       inputs['duration'])
        
        gen_efficiency = inputs['eta_generator'] * inputs['eta_gen_inverter']
        
        # Energy consumed by turbogen (J)
        turbogen_energy = outputs['turbogen_power'] * inputs['duration'] / gen_efficiency
        
        # Fuel mass using PSFC (kg/J * J = kg)
        outputs['manta_fuel_mass'] = turbogen_energy * inputs['psfc'] 

    def compute_partials(self, inputs, partials):
        # Battery power partials
        partials['ray_battery_power', 'total_elec_power'] = inputs['ray_energy_ratio']
        partials['ray_battery_power', 'ray_energy_ratio'] = inputs['total_elec_power']
        
        # Turbogen power partials
        partials['turbogen_power', 'total_elec_power'] = 1.0 - inputs['ray_energy_ratio']
        partials['turbogen_power', 'ray_energy_ratio'] = -inputs['total_elec_power']
        
        # Battery energy partials
        partials['ray_battery_energy', 'ray_battery_power'] = inputs['duration'] / inputs['eta_battery']
        partials['ray_battery_energy', 'eta_battery'] = -inputs['ray_battery_power'] * inputs['duration'] / (inputs['eta_battery']**2)
        partials['ray_battery_energy', 'duration'] = inputs['ray_battery_power'] / inputs['eta_battery']
        
        # Fuel mass partials
        gen_efficiency = inputs['eta_generator'] * inputs['eta_gen_inverter']
        turbogen_energy = inputs['turbogen_power'] * inputs['duration'] / gen_efficiency
        
        partials['manta_fuel_mass', 'turbogen_power'] = inputs['duration'] * inputs['psfc'] / gen_efficiency
        partials['manta_fuel_mass', 'duration'] = inputs['turbogen_power'] * inputs['psfc'] / gen_efficiency
        partials['manta_fuel_mass', 'eta_generator'] = -turbogen_energy * inputs['psfc'] / inputs['eta_generator']
        partials['manta_fuel_mass', 'eta_gen_inverter'] = -turbogen_energy * inputs['psfc'] / inputs['eta_gen_inverter']
        partials['manta_fuel_mass', 'psfc'] = turbogen_energy 