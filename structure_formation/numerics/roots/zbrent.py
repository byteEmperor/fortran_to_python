# structure_formation/numerics/roots/zbrent.py

def zbrent(func, x1, x2, tol):
    """
    Brent's method root finding.

    Parameters:
    - func: callable, function for which we find root (func(x) = 0)
    - x1, x2: floats, interval endpoints where root is bracketed (func(x1)*func(x2) < 0)
    - tol: float, tolerance for convergence
    - itmax: int, max iterations (default 100)
    - eps: float, machine precision estimate (default 3e-8)

    Returns:
    - root approximation within tolerance

    Raises:
    - ValueError if root is not bracketed or max iterations exceeded
    """

    itmax = 100
    eps = 1e-8

    a, b = x1, x2
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("Root must be bracketed in zbrent: func(x1) and func(x2) must have opposite signs.")

    c = b
    fc = fb
    d = e = b - a

    for iter in range(itmax):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol1 or fb == 0.0:
            return b  # root found

        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                # Linear interpolation (secant method)
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # Inverse quadratic interpolation
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            p = abs(p)

            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d

        a = b
        fa = fb

        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm >= 0 else -tol1

        fb = func(b)

    raise RuntimeError("zbrent: exceeded maximum iterations")
