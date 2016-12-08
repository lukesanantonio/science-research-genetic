import random
import sys
import numpy as np

INITIAL_SPECIES_SIZE = 500
WEIGHTS_PER_MOBILE = 8

HALF_SIZE = INITIAL_SPECIES_SIZE // 2
QUARTER_SIZE = HALF_SIZE // 2

# Must be perfectly divisible
assert QUARTER_SIZE * 4 == INITIAL_SPECIES_SIZE
assert HALF_SIZE * 2 == INITIAL_SPECIES_SIZE

SPECIES_ITERATIONS = 100

TOO_MANY_LAYERS_FACTOR = 3.0
DUPLICATE_LAYERS_FACTOR = 3.0

# This is obvious but we use it so that the calculation is self-explanatory.
BITS_PER_BYTE = 8

# Int32 values are in fixed point form with gram and millimeter precision
# respectively.
weight_dtype = np.dtype([('layer', np.uint8), ('weight', np.int32), ('pos', np.int32)])

def mutater(mutation_rate):
    return lambda x: 1 if x < mutation_rate else 0

def mutate(mutation_rate):
    bit_vals = np.random.sample(weight_dtype.itemsize * BITS_PER_BYTE)

    mutate_fn_vec = np.vectorize(mutater(mutation_rate))

    indiv = np.packbits(mutate_fn_vec(bit_vals)).view(dtype=np.uint8)

    # Xor these bits with the individual, return the individual
    return indiv

def combine(parent1, parent2, children=1, mutation_rate=0.07):
    parent1_bytes = parent1.view(dtype=np.uint8)
    parent2_bytes = parent2.view(dtype=np.uint8)

    ret = []
    for child_i in range(children):
        child = np.zeros(WEIGHTS_PER_MOBILE, dtype=weight_dtype)
        total_bytes = weight_dtype.itemsize * WEIGHTS_PER_MOBILE
        total_bits = total_bytes * BITS_PER_BYTE

        # Find a spot to split between the two parents, in bits
        bit_split = random.randrange(0, total_bits)

        # Reinterpret the data as a list of bytes
        child_bytes = child.view(dtype=np.uint8)

        # Figure out what byte has the splitting point
        byte_split = bit_split // 8

        # Will be a value from one to seven. A value of 0 means no
        bit_in_byte = bit_split % 8

        # Copy sides from the two parents
        child_bytes[:byte_split] = parent1_bytes[:byte_split]
        child_bytes[byte_split + 1:] = parent2_bytes[byte_split + 1:]

        parent1_mask = ~(0xff >> (bit_in_byte))
        parent2_mask = 0xff >> (bit_in_byte)

        assert (parent1_mask | parent2_mask) & 0xff == 0xff

        child_bytes[byte_split] = (parent1_bytes[byte_split] & parent1_mask) | \
                                  (parent2_bytes[byte_split] & parent2_mask)

        ret.append(child)
    return ret

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

# Like random.sample but with replacement
def choose(arr, amount):
    return [random.choice(arr) for _ in range(amount)]

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

    progress = ProgressPrinter(0, SPECIES_ITERATIONS, 10)

    for species_i in range(SPECIES_ITERATIONS):
        progress.update_printout()

        # Figure out the fitness of the current generation
        species_fitness = []
        for indiv in species:
            species_fitness.append((fitness(indiv), indiv))

        # Sort by fitness value
        species_fitness = sorted(species_fitness, key=lambda val: val[0])

        # Make a new list of pairs of individuals (who will be parents).
        # To have a new generation of INITIAL_SPECIES_SIZE we need
        # INITIAL_SPECIES_SIZE * 2 parents.
        # members = []

        # First add all members from the first half, twice. After this members
        # should have a length of INITIAL_SPECIES_SIZE. We need double that
        # members.extend(species_fitness[:HALF_SIZE] * 2)

        # Third quartile
        # members.extend(choose(species_fitness[HALF_SIZE + QUARTER_SIZE:],


        # And the rest
        # members.extend(choose(species_fitness[HALF_SIZE:],
                                     # INITIAL_SPECIES_SIZE * 4))

        members = species_fitness * 2
        random.shuffle(members)

        # Don't bother including the fitness value, we don't need it anymore
        members = [fit_pair[1] for fit_pair in members]

        # We should need twice as many parents as we need children next
        # generation. TODO: Change the method which we use to bread the mobiles.
        assert len(members) == 2 * INITIAL_SPECIES_SIZE

        new_species = []
        # Create pairs
        members = list(zip(*[iter(members)]*2))
        for parents in members:
            new_species.extend(combine(*parents, 1))

        species = new_species

        progress.add_progress(1)

    print(species_fitness)
