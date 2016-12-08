import pyparsing as pp
import pprint
from functools import wraps
import random

class ExistingFunction(Exception):
    def __init__(self):
        pass

class FuncDelegator:
    def __init__(self):
        self._func_map = {}

    def def_func(self, name):
        """Returns a decorator that links the given function with the name."""
        def fn(func):
            """Adds the decorated function to the dictionary."""
            @wraps(func)
            def log_exec(*args, **kwargs):
                res = func(*args, **kwargs)
                print('Running', name, '(result =', str(res), ')')
                return res

            if name in self._func_map:
                raise KeyError('{} already exists'.format(name))

            self._func_map[name] = log_exec
            return func

        return fn

    def evaluate(self):
        pass

delegate = FuncDelegator()

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