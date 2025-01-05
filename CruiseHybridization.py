from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

class CruiseHybridization(Discipline):
    def __init__(self) -> None:
        super().__init__()
        
        # Define normalization bounds
        self.cd_low = 0.01
        self.cd_high = 0.05
        
        self.cb_low = 150 * 3600  # Wh/kg to J/kg
        self.cb_high = 800 * 3600
        
        self.cp_low = 0.1 / 60/60/100  # kg/(W*s)
        self.cp_high = 0.5 / 60/60/1000
        
        g = 9.806
        self.wto_max_low = 3000 * g  # N
        self.wto_max_high = 5700 * g
        
        self.w_pay_low = 1 * 100 * g  # N (1 passenger)
        self.w_pay_high = 8 * 100 * g  # (8 passengers)
        
        self.target_range_low = 400 * 1000  # m
        self.target_range_high = 1000 * 1000 # m
        
        self.input_grammar.update_from_names([
            "cl",          # Lift coefficient (dimensionless)
            "cd",          # Drag coefficient (dimensionless)
            "eta_i",       # Inverter efficiency (dimensionless)
            "eta_m",       # Motor efficiency (dimensionless)
            "eta_p",       # Propulsive efficiency (dimensionless)
            "eta_g",       # Generator efficiency (dimensionless)
            "epsilon",     # Fraction of energy provided by batteries (dimensionless)
            "cb",          # Battery specific energy (J/kg)
            "cp",          # Shaft power specific fuel consumption (1/s)
            "wto_max",     # Maximum takeoff weight (N)
            "w_pay",       # Payload weight (N)
            "f_we",        # Empty weight fraction (dimensionless)
            "target_range" # Target range to achieve (m)
        ])
        
        self.output_grammar.update_from_names([
            "range_total",
            "range_residual"  # Difference between computed and target range
        ])

    def _denormalize(self, x_norm, x_min, x_max):
        """Denormalize value from [0,1] interval"""
        return x_min + x_norm * (x_max - x_min)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # Get normalized inputs
        cl = input_data["cl"]
        cd_norm = input_data["cd"]
        eta_i = input_data["eta_i"]
        eta_m = input_data["eta_m"]
        eta_p = input_data["eta_p"]
        eta_g = input_data["eta_g"]
        epsilon = input_data["epsilon"]
        cb_norm = input_data["cb"]
        cp_norm = input_data["cp"]
        wto_max_norm = input_data["wto_max"]
        w_pay_norm = input_data["w_pay"]
        f_we = input_data["f_we"]
        target_range_norm = input_data["target_range"]

        # Denormalize values
        cd = self._denormalize(cd_norm, self.cd_low, self.cd_high)
        cb = self._denormalize(cb_norm, self.cb_low, self.cb_high)
        cp = self._denormalize(cp_norm, self.cp_low, self.cp_high)
        wto_max = self._denormalize(wto_max_norm, self.wto_max_low, self.wto_max_high)
        w_pay = self._denormalize(w_pay_norm, self.w_pay_low, self.w_pay_high)
        target_range = self._denormalize(target_range_norm, self.target_range_low, self.target_range_high)

        g = 9.806
        range_total = (
            (cl / cd) * (eta_i * eta_m * eta_p / g)
            * np.log(
                ((epsilon + (1 - epsilon) * cb * cp) * wto_max)
                / (epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max))
            ) * eta_g / cp
            + (epsilon * cb * ((1 - f_we) * wto_max - w_pay))
            / (epsilon * wto_max + (1 - epsilon) * cb * cp * (w_pay + f_we * wto_max))
        )

        # Calculate residual for implicit solving
        range_residual = range_total - target_range

        return {
            "range_total": range_total,
            "range_residual": range_residual
        }

    def compute_range_derivative_epsilon(self, input_data: StrKeyMapping) -> float:
        """Compute the partial derivative of range_total with respect to epsilon.
        
        Args:
            input_data: Dictionary containing all input parameters
            
        Returns:
            float: Partial derivative dR/dε
        """
        # Get and denormalize inputs - extract float values from arrays
        cl = float(input_data["cl"])
        cd = self._denormalize(float(input_data["cd"]), self.cd_low, self.cd_high)
        eta_i = float(input_data["eta_i"])
        eta_m = float(input_data["eta_m"])
        eta_p = float(input_data["eta_p"])
        eta_g = float(input_data["eta_g"])
        epsilon = float(input_data["epsilon"])
        cb = self._denormalize(float(input_data["cb"]), self.cb_low, self.cb_high)
        cp = self._denormalize(float(input_data["cp"]), self.cp_low, self.cp_high)
        wto_max = self._denormalize(float(input_data["wto_max"]), self.wto_max_low, self.wto_max_high)
        w_pay = self._denormalize(float(input_data["w_pay"]), self.w_pay_low, self.w_pay_high)
        f_we = float(input_data["f_we"])
        
        g = 9.806
        
        # Terms for readability
        term1 = (cl/cd) * (eta_i * eta_m * eta_p / g) * (eta_g / cp)
        term2 = wto_max * (1 - cb * cp)
        term3 = w_pay + f_we * wto_max
        term4 = epsilon * wto_max + (1 - epsilon) * cb * cp * term3
        
        # Derivative of the logarithmic term
        d_log = (wto_max * cb * cp * term3) / (term4 * (epsilon * wto_max + (1 - epsilon) * cb * cp * term3))
        
        # Derivative of the second term
        d_second = (cb * ((1-f_we) * wto_max - w_pay) * wto_max * (1 - cb * cp)) / (term4 ** 2)
        
        return term1 * d_log + d_second

    def _compute_jacobian(
        self,
        input_data: StrKeyMapping,
        output_data: StrKeyMapping
    ) -> StrKeyMapping:
        """Compute the jacobian matrices of the discipline.
        
        Args:
            input_data: Input data from which to compute the jacobian.
            output_data: Output data from which to compute the jacobian.
            
        Returns:
            The jacobian matrices.
        """
        # Convert input_data to dictionary if it's a tuple
        if isinstance(input_data, tuple):
            input_dict = dict(zip(self.input_grammar.names, input_data))
        else:
            input_dict = input_data
            
        derivative = self.compute_range_derivative_epsilon(input_dict)
        
        # Initialize jacobian dictionary
        jacobian = {}
        
        # Set derivatives for range_total
        jacobian["range_total"] = {
            "epsilon": np.array([[derivative]]),
            # Set zero derivatives for other inputs
            "cl": np.zeros((1, 1)),
            "cd": np.zeros((1, 1)),
            "eta_i": np.zeros((1, 1)),
            "eta_m": np.zeros((1, 1)),
            "eta_p": np.zeros((1, 1)),
            "eta_g": np.zeros((1, 1)),
            "cb": np.zeros((1, 1)),
            "cp": np.zeros((1, 1)),
            "wto_max": np.zeros((1, 1)),
            "w_pay": np.zeros((1, 1)),
            "f_we": np.zeros((1, 1)),
            "target_range": np.zeros((1, 1))
        }
        
        # Set derivatives for range_residual
        jacobian["range_residual"] = {
            "epsilon": np.array([[derivative]]),
            "cl": np.zeros((1, 1)),
            "cd": np.zeros((1, 1)),
            "eta_i": np.zeros((1, 1)),
            "eta_m": np.zeros((1, 1)),
            "eta_p": np.zeros((1, 1)),
            "eta_g": np.zeros((1, 1)),
            "cb": np.zeros((1, 1)),
            "cp": np.zeros((1, 1)),
            "wto_max": np.zeros((1, 1)),
            "w_pay": np.zeros((1, 1)),
            "f_we": np.zeros((1, 1)),
            "target_range": np.array([[-1.0]])  # Derivative of (range_total - target_range)
        }
        
        return jacobian

if __name__ == "__main__":
    from gemseo import create_scenario, configure_logger
    from gemseo.algos.design_space import DesignSpace
    
    # Configure logging
    configure_logger()

    # Create discipline instance
    hybridization_discipline = CruiseHybridization()

    # Normalization function
    def normalize(x, x_min, x_max):
        """Normalize value to [0,1] interval with small tolerance"""
        eps = 1e-10  # Small tolerance
        return min(1.0 - eps, max(eps, (x - x_min) / (x_max - x_min)))
    
    def denormalize(x_norm, x_min, x_max):
        """Denormalize value from [0,1] interval"""
        return x_min + x_norm * (x_max - x_min)

    # Define all input parameters
    g = 9.806  # Gravity constant [m/s^2]
    
    cl = 0.7          # Lift coefficient

    cd = 0.03         # Drag coefficient
    cd_low = 0.01
    cd_high = 0.05
    cd_value = normalize(cd, cd_low, cd_high)

    eta_i = 0.95      # Inverter efficiency
    eta_m = 0.95      # Motor efficiency
    eta_p = 0.85      # Propulsive efficiency
    eta_g = 0.92      # Generator efficiency

    cb = 400 * 3600       # Battery specific energy (J/kg). Wh/kg Conversion
    cb_low = 150 * 3600
    cb_high = 800 * 3600
    cb_value = normalize(cb, cb_low, cb_high)

    cp = 0.5/ 60/60/1000        # Shaft power specific fuel consumption (kg/(W*s))
    cp_low = 0.1/60/60/1000
    cp_high = 0.6/60/60/1000
    cp_value = normalize(cp, cp_low, cp_high)

    wto_max = 5500.0 * g  # Maximum takeoff weight (N)
    wto_max_high = 5700 * g
    wto_max_low = 3000 * g
    wto_max_value = normalize(wto_max, wto_max_low, wto_max_high)
    
    w_pay = 200.0 * g    # Payload weight (N)
    w_pay_low = 1 * 100 * g # 1 Passenger
    w_pay_high = 8 * 100 * g # 8 Passengers
    w_pay_value = normalize(w_pay, w_pay_low, w_pay_high)

    f_we = 0.5        # Empty weight fraction

    target_range = 2730 * 1000  # Target range (m)
    target_range_low = 100 * 1000
    target_range_high = 5000 * 1000
    target_range_value = normalize(target_range, target_range_low, target_range_high)

    # Create design space for the MDA
    design_space = DesignSpace()
    
    # Variable to optimize
    design_space.add_variable(
        "epsilon",
        1,
        lower_bound=1e-6,
        upper_bound=1.0 - 1e-6,
        value=0.3
    )
    
    # Fixed variables (same lower and upper bounds)
    design_space.add_variable("cl", 1, lower_bound=cl, upper_bound=cl, value=cl)
    design_space.add_variable("cd", 1, lower_bound=cd_value, upper_bound=cd_value, value=cd_value)
    design_space.add_variable("eta_i", 1, lower_bound=eta_i, upper_bound=eta_i, value=eta_i)
    design_space.add_variable("eta_m", 1, lower_bound=eta_m, upper_bound=eta_m, value=eta_m)
    design_space.add_variable("eta_p", 1, lower_bound=eta_p, upper_bound=eta_p, value=eta_p)
    design_space.add_variable("eta_g", 1, lower_bound=eta_g, upper_bound=eta_g, value=eta_g)
    design_space.add_variable("cb", 1, lower_bound=cb_value, upper_bound=cb_value, value=cb_value)
    design_space.add_variable("cp", 1, lower_bound=cp_value, upper_bound=cp_value, value=cp_value)
    design_space.add_variable("wto_max", 1, lower_bound=wto_max_value, upper_bound=wto_max_value, value=wto_max_value)
    design_space.add_variable("w_pay", 1, lower_bound=w_pay_value, upper_bound=w_pay_value, value=w_pay_value)
    design_space.add_variable("f_we", 1, lower_bound=f_we, upper_bound=f_we, value=f_we)
    design_space.add_variable("target_range", 1, lower_bound=target_range_value, upper_bound=target_range_value, value=target_range_value)

    # Create test input data using the same values
    test_inputs = {
        "cl": cl,
        "cd": cd,
        "eta_i": eta_i,
        "eta_m": eta_m,
        "eta_p": eta_p,
        "eta_g": eta_g,
        "cb": cb,
        "cp": cp,
        "wto_max": wto_max,  # Convert to N
        "w_pay": w_pay,      # Convert to N
        "f_we": f_we,
        "target_range": target_range
    }

    # Set default inputs
    #hybridization_discipline.default_inputs = test_inputs

    # Create and execute MDA scenario
    scenario = create_scenario(
        disciplines=[hybridization_discipline],
        formulation_name="DisciplinaryOpt",
        objective_name="range_residual",  # Objective to minimize
        design_space=design_space,
    )

    # Use exact derivatives instead of finite differences
    scenario.set_differentiation_method("user")

    # Execute the scenario with SLSQP
    scenario.execute(
        algo_name="SLSQP",
        max_iter=1000,
        ftol_rel=1e-6,
        eq_tolerance=1e-6
    )

    # Get optimized results
    print("\nOptimization Results:")
    print(f"Epsilon value for target range: {scenario.design_space.get_current_value(['epsilon'])[0]:.4f}")
    
    # Verify the result
    opt_inputs = {
        "cl": cl,
        "cd": cd,
        "eta_i": eta_i,
        "eta_m": eta_m,
        "eta_p": eta_p,
        "eta_g": eta_g,
        "cb": cb,
        "cp": cp,
        "wto_max": wto_max,
        "w_pay": w_pay,
        "f_we": f_we,
        "target_range": target_range,
        "epsilon": scenario.design_space.get_current_value(['epsilon'])[0]
    }
    result = hybridization_discipline._run(opt_inputs)
    print(f"Achieved range: {result['range_total']/1000:.2f} km")
    print(f"Target range:  {target_range/1000:.2f} km")
    print(f"Range residual: {abs(result['range_residual'])/1000:.2f} km") 