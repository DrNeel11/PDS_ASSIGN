"""Synthetic CT data: 2D/3D Shepp-Logan phantoms with a known ground truth."""
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# Modified 3D Shepp-Logan: (intensity, a, b, c, x0, y0, z0, phi_deg)
_ELLIPSOIDS = [
    (1.0, .69, .92, .81, 0, 0, 0, 0),
    (-.8, .6624, .874, .78, 0, -.0184, 0, 0),
    (-.2, .11, .31, .22, .22, 0, 0, -18),
    (-.2, .16, .41, .28, -.22, 0, 0, 18),
    (.1, .21, .25, .41, 0, .35, -.15, 0),
    (.1, .046, .046, .05, 0, .1, .25, 0),
    (.1, .046, .046, .05, 0, -.1, .25, 0),
    (.1, .046, .023, .05, -.08, -.605, 0, 0),
    (.1, .023, .023, .02, 0, -.606, 0, 0),
    (.1, .023, .046, .02, .06, -.605, 0, 0),
]


def phantom2d(n):
    return resize(shepp_logan_phantom(), (n, n), anti_aliasing=True).astype(np.float32)


def phantom3d(n, depth=None):
    depth = depth or n
    z, y, x = np.meshgrid(np.linspace(-1, 1, depth), np.linspace(-1, 1, n),
                          np.linspace(-1, 1, n), indexing="ij")
    vol = np.zeros((depth, n, n), np.float32)
    for A, a, b, c, x0, y0, z0, phi in _ELLIPSOIDS:
        p = np.deg2rad(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        vol[(xr / a) ** 2 + (yr / b) ** 2 + ((z - z0) / c) ** 2 <= 1] += A
    # flip so the "head" is upright like the 2D phantom
    return np.ascontiguousarray(vol[:, ::-1, :])
