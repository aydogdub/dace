import functools  
import sympy      

from dace import Config, symbolic  

from dace.codegen import cppunparse
from dace.codegen.prettycode import CodeIOStream    


def symbolic_to_cpp(arr):
    """ Converts an array of symbolic variables (or one) to C++ strings. """
    if not isinstance(arr, list):
        return cppunparse.pyexpr2cpp(symbolic.symstr(arr, cpp_mode=True))
    return [cppunparse.pyexpr2cpp(symbolic.symstr(d, cpp_mode=True)) for d in arr]

def get_cuda_dim(idx):
    """ Converts 0 to x, 1 to y, 2 to z, or raises an exception. """
    if idx < 0 or idx > 2:
        raise ValueError(f'idx must be between 0 and 2, got {idx}')
    return ('x', 'y', 'z')[idx]

def product(iterable):
    """
    Computes the symbolic product of elements in the iterable using sympy.Mul.

    This is equivalent to: ```functools.reduce(sympy.Mul, iterable, 1)```.

    Purpose: This function is used to improve readability of the codeGen.
    """
    return functools.reduce(sympy.Mul, iterable, 1)

def emit_sync_debug_checks(backend: str, codestream: CodeIOStream):
    """
    Emit backend sync and error-check calls if synchronous debugging is enabled.
    
    Args:
        backend (str): Backend API prefix (e.g., 'cuda').
        codestream (CodeIOStream): Stream to write code to.
    """
    if Config.get_bool('compiler', 'cuda', 'syncdebug'):
        codestream.write(
            f"DACE_GPU_CHECK({backend}GetLastError());\n"
            f"DACE_GPU_CHECK({backend}DeviceSynchronize());\n"
        )