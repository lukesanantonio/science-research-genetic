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
