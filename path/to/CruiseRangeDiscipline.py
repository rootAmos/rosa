class CruiseRangeDiscipline:
    def __init__(self):
        # Initialize parameters
        self.cl = None
        self.cd = None
        self.eta_i = None
        self.eta_m = None
        self.eta_p = None
        self.eta_g = None
        self.epsilon = None
        self.cb = None
        self.cp = None
        self.wt0_max = None
        self.w_pay = None
        self.f_we = None

    def calculate_range(self):
        # Import the function from CruiseRange.py
        from CruiseRange import calculate_modified_breguet_range
        
        # Call the function with the parameters
        return calculate_modified_breguet_range(
            self.cl, self.cd, self.eta_i, self.eta_m,
            self.eta_p, self.eta_g, self.epsilon,
            self.cb, self.cp, self.wt0_max, self.w_pay, self.f_we
        ) 