import numpy as np
from types import SimpleNamespace

X, Y                 = 0, 1       # coordinates
DIR, NEU, PER        = 0, 1, 2    # boundary condition types
NONE, MINMOD, MONCEN = 0, 1, 2    # slope limiters
HLLC, HLL            = 0, 1       # riemann solvers

NPHOT, XFLUX, YFLUX  = 0, 1, 2    # radiation variables
RHO, VX, VY, P       = 0, 1, 2, 3 # primative variables
RHO, PX, PY, E       = 0, 1, 2, 3 # conserved variables
ION                  = 4          # passive scalars
PAS                  = np.s_[4:]  # passive scalars

sl = SimpleNamespace(
    x    = (slice(0,1),),
    y    = (slice(0,1),),
    ij   = np.s_[:,1:-1,1:-1],
    ijp1 = np.s_[:,1:-1,2:],
    ijm1 = np.s_[:,1:-1,:-2],
    ip1j = np.s_[:,2:,1:-1],
    im1j = np.s_[:,:-2,1:-1]
)