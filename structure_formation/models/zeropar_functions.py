import math

def openU(theta, sigma, ztau):
    return sigma * (math.sinh(theta) - theta) - ztau

def closedU(theta, sigma, ztau):
    return sigma * (math.sin(theta) - theta) - ztau
