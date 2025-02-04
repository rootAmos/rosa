
class Averager(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('N', default=1, desc='Number of nodes')

    def setup(self):
        N = self.options['N']

        self.add_input('vector', val=1.0 * np.ones(N), desc='Vector to average')
        
        self.add_output('avg', desc='Average of vector')

        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        outputs['avg'] = np.mean(inputs['vector'])


    def compute_partials(self, inputs, partials):
        partials['avg', 'vector'] = 1.0/self.options['N']

