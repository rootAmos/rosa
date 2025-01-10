import openmdao.api as om
import numpy as np

# Define a Mission Analysis component

# Define an Allocation Problem Component
class AllocationProblem(om.ExplicitComponent):
    """
    This component is used to allocate passengers to aircraft.

    References:
    - [1] "Simultaneous aircraft allocation and mission optimization using a modular adjoint approach, 
          by John T. Hwang, Satadru Roy, Jason Y. Kao, Joaquim R. R. A. Martins, and William A. Crossley 
          https://doi.org/10.2514/6.2015-0900
    """

    def initialize(self):
        self.options.declare('n_rt', default=1, desc='Number of routes')
        self.options.declare('n_ac', default=1, desc='Number of aircraft')

    def setup(self):

        n_rt = self.options['n_rt']
        n_ac = self.options['n_ac']


        # Inputs
        self.add_input('price_per_pax', val=np.ones([n_rt, n_ac]), desc='Ticket price per passenger for each route')
        self.add_input('pax_per_flt', val=np.ones([n_rt, n_ac]), desc='Passengers per flight for each route')
        self.add_input('cost_per_flt', val=np.ones([n_rt, n_ac]), desc='Total operating cost (excluding fuel) per flight for each route')
        self.add_input('flt_per_day', val=np.ones([n_rt, n_ac]), desc='Number of flights per day for each route')
        self.add_input('cost_fuel', val=np.ones([n_rt, n_ac]), desc='Fuel cost per unit of fuel', units='1/kg')    
        self.add_input('fuel_flt', val=np.ones([n_rt, n_ac]), desc='Fuel burn per flight for each aircraft', units='kg')
        self.add_input('time_per_flt', val=np.ones([n_rt, n_ac]), desc='Block time per flight for each route', units='h')
        self.add_input('t_maint', val=np.ones([n_ac]), desc='Maintenance time per aircraft. Scaled by number of flights', units='h')
        self.add_input('t_turn', val=1.0, desc='Turnaround time between flights for each route', units='h')
        self.add_input('demand', val=np.ones(n_rt), desc='Demand for each route')
        self.add_input('n_ac', val=np.ones(n_ac), desc='Number of aircraft')

        # Outputs
        self.add_output('profit', val=1.0, desc='Total profit from allocation')
        self.add_output('total_pax_margin', val=1.0 * np.ones(n_rt), desc='Total number of passengers allocated minus demand')
        self.add_output('total_usage_margin', val=1.0 * np.ones(n_ac), desc='Total usage of each aircraft minus capacity')

        # Partial derivatives
        self.declare_partials('*', '*', method='exact')

    def compute(self, inputs, outputs):

        # Extract inputs
        price_per_pax = inputs['price_per_pax']
        pax_per_flt = inputs['pax_per_flt']
        cost_per_flt = inputs['cost_per_flt']
        flt_per_day = inputs['flt_per_day']
        cost_fuel = inputs['cost_fuel']
        fuel_flt = inputs['fuel_flt']
        time_per_flt = inputs['time_per_flt']
        t_maint = inputs['t_maint']
        t_turn = inputs['t_turn']
        demand = inputs['demand']
        n_ac = inputs['n_ac']


        # Simplified profit calculation [1] Eq 14 & 15
        profit = np.sum(flt_per_day * price_per_pax * pax_per_flt)  - np.sum(flt_per_day * (cost_per_flt + cost_fuel * fuel_flt))  # Assume $100 revenue per passenger
        
        # [1] Eq 16
        total_pax = np.sum(pax_per_flt * flt_per_day, axis=1)
        total_pax_margin = total_pax - demand

        # [1] Eq 17
        total_usage = np.sum(flt_per_day * (time_per_flt * (1 + t_maint) + t_turn), axis=0)
        total_usage_margin = total_usage - n_ac * 12 # 12 hours per day

        outputs['total_pax_margin'] = total_pax_margin
        outputs['total_usage_margin'] = total_usage_margin
        outputs['profit'] = profit  


    def compute_partials(self, inputs, partials):

        # Extract inputs
        price_per_pax = inputs['price_per_pax']
        pax_per_flt = inputs['pax_per_flt']
        cost_per_flt = inputs['cost_per_flt']
        flt_per_day = inputs['flt_per_day']
        cost_fuel = inputs['cost_fuel']
        fuel_flt = inputs['fuel_flt']
        time_per_flt = inputs['time_per_flt']
        t_maint = inputs['t_maint']
        t_turn = inputs['t_turn']

        # Partials for profit 
        partials['profit', 'price_per_pax'] = np.sum(flt_per_day * pax_per_flt)
        partials['profit', 'pax_per_flt'] = np.sum(flt_per_day * price_per_pax)
        partials['profit', 'cost_per_flt'] = -np.sum(flt_per_day)
        partials['profit', 'flt_per_day'] = np.sum(price_per_pax * pax_per_flt) - np.sum(cost_per_flt + cost_fuel * fuel_flt)
        partials['profit', 'cost_fuel'] = -np.sum(flt_per_day * fuel_flt)
        partials['profit', 'fuel_flt'] = -np.sum(flt_per_day * cost_fuel)

        # Partials for total_pax
        partials['total_pax_margin', 'pax_per_flt'] = np.sum(flt_per_day, axis=1)  
        partials['total_pax_margin', 'flt_per_day'] = np.sum(pax_per_flt, axis=1)
        partials['total_pax_margin', 'demand'] = -np.ones(n_rt)

        # Partials for total_usage
        partials['total_usage_margin', 'time_per_flt'] = np.sum(flt_per_day * (1 + t_maint), axis=0)
        partials['total_usage_margin', 't_maint'] = np.sum(flt_per_day * time_per_flt, axis=0)
        partials['total_usage_margin', 't_turn'] = np.sum(flt_per_day, axis=0) 
        partials['total_usage_margin', 'n_ac'] = -12 * np.ones(n_ac)
        partials['total_usage_margin', 'flt_per_day'] = np.sum(time_per_flt * (1 + t_maint) + t_turn, axis=0)


if __name__ == "__main__":
    # Main model setup
    prob = om.Problem()


    n_rt = 1 # Number of routes
    n_ac = 1 # Number of aircraft

    ivc = om.IndepVarComp()
    ivc.add_output('price_per_pax', val= 50 * np.ones([n_rt, n_ac]), desc='Ticket price per passenger for each route')
    ivc.add_output('pax_per_flt', val= 60 * np.ones([n_rt, n_ac]), desc='Passengers per flight for each route')
    ivc.add_output('cost_per_flt', val= 1000 * np.ones([n_rt, n_ac]), desc='Total operating cost (excluding fuel) per flight for each route')
    ivc.add_output('flt_per_day', val= 10 * np.ones([n_rt, n_ac]), desc='Number of flights per day for each route')
    ivc.add_output('cost_fuel', val= 10 * np.ones([n_rt, n_ac]), units='1/kg', desc='Fuel cost per flight for each route')    
    ivc.add_output('fuel_flt', val= 100 * np.ones([n_rt, n_ac]), units='kg', desc='Fuel burn per flight for each aircraft')
    ivc.add_output('time_per_flt', val= 2 * np.ones([n_rt, n_ac]), units='h', desc='Block time per flight for each route')
    ivc.add_output('t_maint', val= 1 * np.ones([n_ac]), units = 'h', desc='Maintenance time per aircraft. Scaled by number of flights')
    ivc.add_output('t_turn', val=1.0, units='h', desc='Turnaround time between flights for each route')
    ivc.add_output('demand', val= 2000 * np.ones([n_rt]), desc='Demand for each route')
    ivc.add_output('n_ac', val= 300 * np.ones([n_ac]), desc='Number of aircraft')

    # Add subsystems
    model = prob.model
    model.add_subsystem('inputs', ivc, promotes_outputs=["*"])
    model.add_subsystem('allocation_problem', AllocationProblem(n_rt=n_rt, n_ac=n_ac), promotes_inputs=['*'])


    # Analysis
    prob.model.nonlinear_solver = om.NewtonSolver()
    prob.model.linear_solver = om.DirectSolver()
    prob.model.nonlinear_solver.options['iprint'] = 2
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()


    prob.setup()

    # Setup N2 Chart
    # om.n2(prob)
    # Run model
    prob.run_model()


    # Optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 100

    # Add design variables
    prob.model.add_design_var('flt_per_day', lower=0.0, upper=100.0)
    prob.model.add_design_var('pax_per_flt', lower=0.0, upper=300.0)

    # Add design variables, objective, and constraints
    prob.model.add_constraint('allocation_problem.total_usage_margin', upper=0, ref = 1e-3)
    prob.model.add_constraint('allocation_problem.total_pax_margin', upper=0, ref = 1e-3)
    prob.model.add_objective('allocation_problem.profit', scaler=-1e4)  # Maximizing profit

    # Add recording of iterations
    """
    recorder = om.SqliteRecorder('cases.sql')
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options['record_desvars'] = True
    prob.driver.recording_options['record_objectives'] = True
    prob.driver.recording_options['record_constraints'] = True
    prob.driver.options['debug_print'] = ['desvars', 'objs']
    """

    # Setup and run
    prob.setup()

    #prob.check_partials(compact_print=True)
    prob.run_driver()
    #prob.check_totals(of=['allocation_problem.profit'], wrt=['flt_per_day', 'pax_per_flt'], step=1e-6)

    # Outputs
    print("Profit:", prob['allocation_problem.profit'])
    print("Total pax margin:", prob['allocation_problem.total_pax_margin'])
    print("Total usage margin:", prob['allocation_problem.total_usage_margin'])

    # Print Design Variables
    print("Design Variables:")
    print("flt_per_day:", prob.get_val('flt_per_day'))
    print("pax_per_flt:", prob.get_val('pax_per_flt'))
    prob.cleanup()
