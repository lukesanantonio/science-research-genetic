import pyparsing as pp
import pprint
from functools import wraps
import random
import numpy as np
import math

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


class FunctionDelegator:
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
                print('Running', name, '(result =', str(res), ')')
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
                print('Querying', name, '(value =', str(res), ')')
                return res

            if name in self.term_map:
                raise TerminalExistsError(name)

            self.term_map[name] = log_exec
            return func

        return fn

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

delegate = FunctionDelegator()

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
def move(state, dx, dy, dz):
    state.abspos_x += dx
    state.abspos_y += dy
    state.abspos_z += dz
    return None





def eval_lisp(code, state, delegate):
    if isinstance(code, list):
        func_name = code[0]
        if func_name not in delegate.func_map:
            # Rip, bad function
            raise UnknownFunctionError(func_name)

        func = delegate.func_map[func_name]

        def eval_cur_lisp(code):
            return eval_lisp(code, state, delegate)

        # Evaluate parameters
        params = list(map(eval_cur_lisp, code[1:]))

        # Evaluate function
        return func(state, *params)

    elif isinstance(code, str):
        # Try parsing true or false
        if code == 'true':
            return True
        elif code == 'false':
            return False

        # Try parsing an integer
        try:
            val = int(code)
        except ValueError:
            val = None

        if val != None:
            return val

        # Try resolving a terminal reference.
        # Make sure the string is a valid terminal
        if code not in delegate.term_map:
            raise UnknownTerminalError(code)

        # Evaluate the terminal at this time.
        return delegate.term_map[code](state)


RUNS=1000

if __name__ == '__main__':

    dsl = pp.nestedExpr()
    comment = (';' + pp.restOfLine).suppress()
    dsl.ignore(comment)

    prog = dsl.parseString(open('initial_seed.lisp', 'r').read()).asList()[0]
    pprint.pprint(prog, indent=4, width=40)

    state = ProceduralState(10, 10, 10)

    for i in range(RUNS):
        eval_lisp(prog, state, delegate)
