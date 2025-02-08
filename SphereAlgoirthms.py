import scipy.linalg
from random import random
import numpy as np

def fibonacci_sphere(radius=20., samples=1, randomize=True):
    """ Yields 3D cartesian coordinates of pseudo-equidistributed points on the surface of a sphere of given radius,
    aligned on the origin, using Fibonacci Sphere algorithm.
    Gist from Snord (http://stackoverflow.com/a/26127012/624547)
    @yield 3D point
    """
    rnd = 1.
    if randomize:
        rnd = random() * samples

    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        yield [radius * x, radius * y, radius * z]