'''

scipy.special.elliprj — it's a direct implementation of Carlson’s symmetric elliptic
integral of the third kind, also known as R_J(x, y, z, p).

'''

from scipy.special import elliprj

def mmlind_scipy(x: float, y: float, z: float) -> tuple[float, int]:
    # Error checks
    if min(x, y, z) < 0:
        return float('inf'), 129
    if x + y < 1e-20 or z < 1e-20:
        return float('inf'), 130
    if max(x, y, z) > 1e38:
        return float('inf'), 131

    try:
        value = elliprj(x, y, z, z)
        return value, 0
    except Exception:
        return float('inf'), 999  # catch-all error