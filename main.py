import numpy as np

INITIAL_SPECIES_SIZE = 500
WEIGHTS_PER_MOBILE = 8

# Int32 values are in fixed point form with gram and millimeter precision
# respectively.
weight_dtype = np.dtype([('layer', np.uint8), ('weight', np.int32), ('pos', np.int32)])

if __name__ == '__main__':
    # Create the initial species
    rand_source = np.random.bytes(weight_dtype.itemsize *
                                  INITIAL_SPECIES_SIZE *
                                  WEIGHTS_PER_MOBILE)
    species = np.fromstring(rand_source, weight_dtype)
    species.shape = (INITIAL_SPECIES_SIZE, WEIGHTS_PER_MOBILE)
    print(species[0])
