import pyparsing as pp
import pprint
from functools import wraps
import random

class AlreadyExistsError(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "{} already exists".format(self.name)

class FunctionExistsError(AlreadyExistsError):
    def __init__(self, name):
        super().__init__(name + ' function')

class TerminalExistsError(AlreadyExistsError):
    def __init__(self, name):
        super().__init__(name + ' terminal')

class FunctionDelegator:
    def __init__(self):
        self._func_map = {}
        self._term_map = {}

    def def_func(self, name):
        """Returns a decorator that links the given function with the name."""
        def fn(func):
            """Adds the decorated function to the function set."""
            @wraps(func)
            def log_exec(*args, **kwargs):
                res = func(*args, **kwargs)
                print('Running', name, '(result =', str(res), ')')
                return res

            if name in self._func_map:
                raise FunctionExistsError(name)

            self._func_map[name] = log_exec
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

            if name in self._term_map:
                raise TerminalExistsError(name)

            self._term_map[name] = log_exec
            return func

        return fn

delegate = FunctionDelegator()

@delegate.def_func("rand")
def rand(start=None, end=None):
    if start == None and end == None:
        return random.uniform(0.0, 1.0)

    # RETURNING ZERO ALL THE TIME IS NOT RANDOM
    return 0.0

if __name__ == '__main__':

    dsl = pp.nestedExpr()
    comment = (';' + pp.restOfLine).suppress()
    dsl.ignore(comment)

    prog = dsl.parseString(open('initial_seed.lisp', 'r').read()).asList()
    pprint.pprint(prog, indent=4, width=40)

    print(delegate._func_map['rand']())
    print(rand())
