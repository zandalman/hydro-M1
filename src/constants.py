import astropy.units as u
import numpy as np

# All units are in CGS
pc  = 3.086e18                                         # parsec
hr  = 3600                                             # hour
day = 24*hr                                            # day
yr  = 365*day                                          # year
kyr = 1000*yr                                          # kiloyear
c          = 2.99792458e10                             # speed of light
c_r        = 2.99792458e10*6e-4                        # reduced speed of light
ε_γ        = (13.6 * u.eV).cgs.value                   # energy of a single ionizing photon
ε_HI       = (13.6 * u.eV).cgs.value                   # ionization energy of neutral hydrogen
kB         = 1.380658*1e-16                            # boltzmann constant
γ          = gam = 5/3                                 # equation of state parameter
mH         = 1.6733*1e-24                              # Hydrogen mass
X          = 1.0                                       # hydrogen mass fraction
Msol       = 1.99e33                                   # solar mass

def σ(ε):
    """
    Cross section for photoionization of hydrogen

    Args:
    ε: Energy in [eV]

    Returns:
    σ: Cross section in [cm^2]
    """

    # Fitting Parameters for the cross section
    σ0         = 5.475e-14  #cm^2
    ε0         = 0.4298     #eV
    yw, y0, y1 = 0, 0, 0
    ya         = 32.88
    p          = 2.963

    x = ε / ε0 - y0
    y = np.sqrt(x**2 + y1**2)

    if ε >= ε_HI:
        σ = σ0 * ((x - 1)**2 + yw**2) * (y**(0.5 * p - 5.5) / (1 + np.sqrt(y/ya))**p) #cm^2
    else:
        σ = 0

    return σ

# cross sections
σ_N_HI     = σ(13.6)                                   # cross section of neutral hydrogen
σ_E_HI     = σ(13.6)                                   # cross section of neutral hydrogen weighted by energy

