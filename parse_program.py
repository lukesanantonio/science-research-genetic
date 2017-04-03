import pyparsing as pp
import pprint
from functools import wraps
import random
import numpy as np
import functools
import math
from inspect import signature

import unittest


class AlreadyExistsError(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "{} already exists".format(self.name)


class FunctionExistsError(AlreadyExistsError):
    def __init__(self, name):
        super(self).__init__(name + ' function')


class TerminalExistsError(AlreadyExistsError):
    def __init__(self, name):
        super(self).__init__(name + ' terminal')


class InvalidPrimitive(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "{} is not valid".format(self.name)


class InvalidFunction(Exception):
    def __init__(self, name):
        super(self).__init__(name + ' function')


class InvalidTerminal(Exception):
    def __init__(self, name):
        super(self).__init__(name + ' terminal')


class PrimitiveDelegate:
    def __init__(self):
        self.func_map = {}
        self.term_map = {}

    def def_func(self, name):
        """Returns a decorator that links the given function with the name."""

        def fn(func):
            """Adds the decorated function to the function set."""

            @wraps(func)
            def log_exec(*args, **kwargs):
                res = func(*args, **kwargs)
                return res

            if name in self.func_map:
                raise FunctionExistsError(name)

            self.func_map[name] = log_exec
            return func

        return fn

    def def_terminal(self, name):
        """Returns a decorator that links a terminal to a getter function."""

        def fn(func):
            """Adds a terminal to the terminal set."""

            @wraps(func)
            def log_exec(*args, **kwargs):
                res = func(*args, **kwargs)
                return res

            if name in self.term_map:
                raise TerminalExistsError(name)

            self.term_map[name] = log_exec
            return func

        return fn

    # Both function and terminals
    def is_primitive(self, prim):
        return self.is_terminal(prim) or self.is_function(prim)

    def pick_random_primitive(self):
        keys = list(self.func_map.keys())
        keys.extend(self.term_map.keys())
        return random.choice(keys)

    # Functions
    def is_function(self, func):
        return True if func in self.func_map else False

    def pick_random_function(self):
        return random.choice(list(self.func_map.keys()))

    def get_function_signature(self, name):
        if not self.is_function(name):
            raise InvalidFunction(name)
        return signature(self.func_map[name])

    # Terminals
    def is_terminal(self, term):
        return True if term in self.term_map else False

    def pick_random_terminal(self):
        return random.choice(list(self.term_map.keys()))

    def eval_list(self, code, state):
        """Returns the evaluated value and list of errors as a tuple."""
        if code is None:
            return None, []

        if isinstance(code, list):
            func_name = code[0]
            if func_name not in self.func_map:
                # Rip, bad function
                raise InvalidFunction(func_name)

            func = self.func_map[func_name]

            def eval_cur_lisp(dsl):
                return self.eval_list(dsl, state)

            # Collect errors
            errs = []

            # Evaluate parameters
            results = list(map(eval_cur_lisp, code[1:]))
            params = []
            for res in results:
                params.append(res[0])
                errs.extend(res[1])

            # Evaluate function
            try:
                return func(state, *params), errs
            except TypeError as e:
                errs.append(e)
                # On error, we don't have another value
                return None, errs

        elif isinstance(code, str):
            # Try parsing true or false
            if code == 'true':
                return True, []
            elif code == 'false':
                return False, []

            # Try parsing an integer
            try:
                val = int(code)
            except ValueError:
                val = None

            if val is not None:
                return val, []

            # Try resolving a terminal reference.
            # Make sure the string is a valid terminal
            if code not in self.term_map:
                raise InvalidTerminal(code)

            # Evaluate the terminal at this time.
            return self.term_map[code](state), []
        else:
            return code, []


class ProceduralState:
    def __init__(self, sizex, sizey, sizez):
        self.abspos_x = 0
        self.abspos_y = 0
        self.abspos_z = 0
        self.gridsize_x = sizex
        self.gridsize_y = sizey
        self.gridsize_z = sizez

        # The actual grid
        self.data = np.zeros((sizex, sizey, sizez), dtype=bool)


delegate = PrimitiveDelegate()


@delegate.def_terminal("curstate")
def curstate_term(state):
    return state.data[state.abspos_x][state.abspos_y][state.abspos_z]


@delegate.def_terminal("abspos_x")
def abspos_y_term(state):
    return state.abspos_x


@delegate.def_terminal("abspos_y")
def abspos_y_term(state):
    return state.abspos_y


@delegate.def_terminal("abspos_z")
def abspos_z_term(state):
    return state.abspos_z


@delegate.def_terminal("gridsize_x")
def gridsize_x_term(state):
    return state.gridsize_x


@delegate.def_terminal("gridsize_y")
def gridsize_y_term(state):
    return state.gridsize_y


@delegate.def_terminal("gridsize_z")
def gridsize_z_term(state):
    return state.gridsize_z


@delegate.def_terminal('pi')
def pi_term(state):
    return math.pi


@delegate.def_func('uniform')
def rand(state, end=None, start=None):
    if end is not None and start is None:
        return random.uniform(0.0, end)
    elif end is not None and start is not None:
        return random.uniform(start, end)
    else:
        return random.uniform(0.0, 1.0)


@delegate.def_func('sin')
def sin(state, val):
    return math.sin(val)


@delegate.def_func('if')
def if_func(state, cond, first, second):
    return first if cond else second


@delegate.def_func('<')
def less_than(state, left, right):
    if left < right:
        return True
    else:
        return False


@delegate.def_func('*')
def multiply(state, left, right):
    return left * right


@delegate.def_func('/')
def divide(state, left, right):
    return left / right


@delegate.def_func('+')
def add(state, left, right):
    return left + right


@delegate.def_func('-')
def minus(state, left, right=None):
    if right is None:
        return 1 - left
    else:
        return left - right


@delegate.def_func('do')
def do(state, *vals):
    return vals[-1]


@delegate.def_func('set')
def set(state, val):
    if (state.abspos_x < 0 or state.gridsize_x < state.abspos_x or
                state.abspos_y < 0 or state.gridsize_y < state.abspos_y or
                state.abspos_z < 0 or state.gridsize_z < state.abspos_z):
        # The position is out of bounds
        return val

    state.data[state.abspos_x][state.abspos_y][state.abspos_z] = val
    return val


@delegate.def_func('move')
def move(state, dx, dy, dz) -> None:
    state.abspos_x += dx
    state.abspos_y += dy
    state.abspos_z += dz


class HyperParameters:
    selection_dist = [0.3, 0.3, 0.4]


def pick_random_node(graph):
    """Returns a random node via list and index.

    This allows the node to modified in-place, however the root node will never
    be chosen.
    """

    # Pick a node, but not the first one
    node_i = random.randrange(1, len(graph))

    if isinstance(graph[node_i], list):
        # If this node is a list, decide randomly whether to traverse it or not
        use_this_node = random.choice([True, False])
        return (graph, node_i) if use_this_node else pick_random_node(
            graph[node_i])
    else:
        # We can't traverse it so return it
        return graph, node_i


def swap_nodes(pick1, pick2):
    """Swap the values of two chosen sub-graphs.

    Params are expected to be a list-index pair."""
    tmp = pick1[0][(pick1[1])]
    pick1[0][(pick1[1])] = pick2[0][(pick2[1])]
    pick2[0][(pick2[1])] = tmp


def weighted_choice(arr):
    # Each item should be a tuple with probabilities as the second element
    total_prob = functools.reduce(lambda total, tup: total + tup[1], arr, 0)

    choice = random.uniform(0.0, total_prob)

    cur_sum = 0.0
    for item in arr:
        cur_sum += item[1]
        if choice < cur_sum:
            return item[0]

    # This will only occur when total_prob is 0
    return None


# Instead of worrying about none functions (and figuring out if the none will
# propagate up), just let it happen and penalize the program when None is
# passed to a function that it shouldn't.

def pick_node(hparams, dgt, function_allowed=True):
    # We can always pick either a terminal or constant, to varying degrees of
    # probability.
    choices = [(dgt.pick_random_terminal(), hparams.selection_dist[1]),
               (random.getrandbits(8), hparams.selection_dist[2] / 2),
               (random.uniform(0.0, 1.0), hparams.selection_dist[2] / 2)]

    # We can only select a function if we have room depth-wise
    if function_allowed:
        choices.append((dgt.pick_random_function(), hparams.selection_dist[0]))

    return weighted_choice(choices)


def generate_random_program(hparams, dgt: PrimitiveDelegate, max_depth=4,
                            cur_depth=0):
    # Only allow None functions to run at the top level
    prim = pick_node(hparams, dgt, cur_depth <= max_depth)

    if not dgt.is_function(prim):
        # We aren't dealing with anything that needs children parameters.
        return prim

    # This is our program
    ret = [prim]

    # Figure out parameters and generate a sub-program for each
    sig = dgt.get_function_signature(prim)
    for param_i in range(len(sig.parameters) - 1):
        # Use type information in annotations of parameter, maybe?
        ret.append(generate_random_program(hparams, dgt, max_depth,
                                           cur_depth + 1))

    return ret


def crossover(g1, g2):
    """Crossover two graphs at some random point at their respective graphs."""
    n1_pick = pick_random_node(g1)
    print(n1_pick)

    n2_pick = pick_random_node(g2)
    print(n2_pick)

    swap_nodes(n1_pick, n2_pick)


class TestEval(unittest.TestCase):
    def setUp(self):
        self.state = ProceduralState(10, 10, 10)

    def test_eval(self):
        res, errors = delegate.eval_list(['+', 5, None], self.state)
        self.assertIsNone(res)
        self.assertEqual(1, len(errors))

        res, errors = delegate.eval_list(['+', 5, 10], self.state)
        self.assertEqual(15, res)
        self.assertEqual(0, len(errors))

RUNS = 1000

if __name__ == '__main__':

    dsl = pp.nestedExpr()
    comment = (';' + pp.restOfLine).suppress()
    dsl.ignore(comment)

    #prog = dsl.parseString(open('initial_seed.lisp', 'r').read()).asList()[0]

    # Generate a new program
    prog = generate_random_program(HyperParameters(), delegate)

    # Print it
    pprint.pprint(prog, indent=4, width=40)

    # Create a new voxel world to run it in
    state = ProceduralState(10, 10, 10)

    for i in range(RUNS):
        # Run the program a few times in the voxel world.
        delegate.eval_list(prog, state)
