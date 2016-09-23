import numpy as np

INITIAL_SPECIES_SIZE = 500
WEIGHTS_PER_MOBILE = 8

TOO_MANY_LAYERS_FACTOR = 3.0
DUPLICATE_LAYERS_FACTOR = 3.0

# Int32 values are in fixed point form with gram and millimeter precision
# respectively.
weight_dtype = np.dtype([('layer', np.uint8), ('weight', np.int32), ('pos', np.int32)])

def fitness(indiv):
    fit = 0.0
    # If the layer is greater than 7 we need to penalize it somehow. We need
    # one weight per layer for 7 layers total.
    for weight in indiv:
        fit += max(0, weight['layer'] - 7) * TOO_MANY_LAYERS_FACTOR

    # We can't have duplicate layers
    layers = [weight['layer'] for weight in indiv]
    for i in range(WEIGHTS_PER_MOBILE):
        fit += max(0, layers.count(i) - 1) * DUPLICATE_LAYERS_FACTOR

    return fit

class ProgressPrinter:
    def __init__(self, start, end, steps, step_chars='#.', edge_chars='[]'):
        self.start = start
        self.end = end
        self.steps = steps

        self.step_chars = step_chars
        self.edge_chars = edge_chars

        self.current = self.start

        sys.stdout.write(self.edge_chars[0] +
                         self.step_chars[1] * self.steps +
                         self.edge_chars[1])
        sys.stdout.flush()

    def set_progress(self, pt):
        self.current = val

    def add_progress(self, val):
        self.current += val

    def update_printout(self):
        sys.stdout.write('\b' * (self.steps + 2))

        vals_per_step = (self.end - self.start) // self.steps

        steps_completed = self.current // vals_per_step
        steps_not_done = self.steps - steps_completed

        assert steps_completed + steps_not_done == self.steps

        sys.stdout.write(self.edge_chars[0] +
                         self.step_chars[0] * steps_completed +
                         self.step_chars[1] * steps_not_done +
                         self.edge_chars[1])
        sys.stdout.flush()

if __name__ == '__main__':
    # Create the initial species
    rand_source = np.random.bytes(weight_dtype.itemsize *
                                  INITIAL_SPECIES_SIZE *
                                  WEIGHTS_PER_MOBILE)
    species = np.fromstring(rand_source, weight_dtype)
    species.shape = (INITIAL_SPECIES_SIZE, WEIGHTS_PER_MOBILE)

    species_fitness = []
    for indiv in species:
        species_fitness.append((fitness(indiv), indiv))

    # Sort by fitness value
    species_fitness = sorted(species_fitness, key=lambda val: val[0])

    print(species_fitness)
