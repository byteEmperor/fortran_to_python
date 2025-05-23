# structure_formation/numerics/roots/wrappers.py

"""

The purpose of this wrappers.py file is to serve as an abstration layer between the raw numerical implementation
and the rest of the project.

"""


from .zbrent import zbrent

def find_root(func, interval, tol=1e-10):
    """
        Finds a root of the function `func` within the interval [a, b].

        Parameters:
        - func: callable, the function whose root we seek
        - interval: tuple (a, b), where func(a) and func(b) have opposite signs
        - tol: desired precision (default 1e-10)

        Returns:
        - root: float, the approximated root
        """
    a, b = interval
    return zbrent(func, a, b, tol)