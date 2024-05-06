import constants
import numpy as np

# global variables in CGS
c_r        = constants.c_r                             # reduced speed of light
ε_γ        = constants.ε_γ                             # energy of a single ionizing photon
ε_HI       = constants.ε_HI                            # ionization energy of neutral hydrogen
σ_N_HI     = constants.σ_N_HI                          # cross section of neutral hydrogen
σ_E_HI     = constants.σ_E_HI                          # cross section of neutral hydrogen weighted by energy
kB         = constants.kB                              # boltzmann constant
γ          = constants.γ                               # equation of state parameter
mH         = constants.mH                              # Hydrogen mass
X          = constants.X                               # Hydrogen mass fraction


def βHI(T, prime=False):
    """
    Collisional ionization rate coefficients

    Args:
    T: Temperature in [K]
    prime: If True, return the primed rate coefficient

    Returns:
    β: Collisional ionization rate coefficients [cm^3/s]
    """

    T5 = T/1e5
    f = 1 + np.sqrt(T5) ; hf=0.5/f
    β = 5.85e-11 * np.sqrt(T) / f * np.exp(-157809.1/T)

    if prime:
        dβ_dlogT = (hf+157809.1/T) * np.log(10) * β
        dβ_dT = dβ_dlogT / (T * np.log(10))
        return dβ_dT
    else:
        return β
    

def λHI(T):
    """ 
    Recombination rate coefficient for HI

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    λHI: Recombination rate coefficient unitless
    """
    return 315614 / T

def αB_HII(T, prime):
    """
    The case B recombination coefficient for HII in units of [cm^3/s]

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    αB_HII: Recombination coefficient in [cm^3/s]
    """
    λ = λHI(T)
    f = (1 + (λ/2.74)**0.407)
    α_B = 2.753e-14 * ( λ**1.5 / f**2.242 )
    if prime:
        dα_dlogT = ( 0.912494 * ((f - 1.)/f) - 1.5 ) * np.log(10) * α_B
        dα_dT = dα_dlogT / ( T * np.log(10) )
        return dα_dT
    else:
        return α_B

def ζHI(T, prime):
    """ 
    The Cooling ratefunction for HI in units of [erg cm^3/s]

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    ζHI: Cooling rate in [erg cm^3/s]
    """

    T5 = T / 1e5
    f = ( 1 + np.sqrt(T5) )
    hf = 0.5 / f
    ζ = 1.27e-21 * (np.sqrt(T) /f) * np.exp(-157809.1/T)
    if prime:
        dζ_dlogT = (hf + 157809.1/T) * np.log(10) * ζ
        dζ_dT = dζ_dlogT / (T * np.log(10))
        return dζ_dT
    else:
        return ζ

def ηHII(T, prime):
    """ 
    The recombination rate coefficient for HI in units of [erg cm^3/s] (Case B)

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    ηHI: Recombination rate coefficient in [erg cm^3/s]
    """

    λ = λHI(T)
    f = 1 + (λ/2.25)**0.376
    η = 3.435e-30 * T * ( λ**1.97 / f**3.72 )

    if prime:
        dη_dlogT = (-0.97 + 1.39827*(f - 1.)/f ) * np.log(10) * η
        dη_dT = dη_dlogT / (T * np.log(10))
        return dη_dT
    else:
        return η

def ψHI(T, prime):
    """ 
    The collisional excitation cooling rate function for HI in units of [erg cm^3/s]

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    ψHI: collisional excitation cooling rate in [erg cm^3/s]
    """
    T5 = T / 1e5
    f = ( 1 + np.sqrt(T5) )
    hf = 0.5 / f
    
    ψ = (7.5e-19/f) * np.exp(-118348/T)
    if prime:
        dψ_dlogT = (-118348/T - 0.5 * np.sqrt(T5) / f) * np.log(10) * ψ
        dψ_dT = dψ_dlogT / (T * np.log(10))
        return dψ_dT
    else:
        return ψ

def θHII(T, prime):
    """ 
    The bremsstrahlung cooling rate function for HII in units of [erg cm^3/s]

    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    θHII: bremsstrahlung cooling rate in [erg cm^3/s]
    """
    θ = 1.42e-27 * np.sqrt(T)
    if prime:
        dθ_dlogT = 0.5 * θ * np.log(10)
        dθ_dT = dθ_dlogT / (T * np.log(10))
        return dθ_dT
    else:
        return θ

def pomega(T, prime):
    """ 
    The compton cooling rate function in units of [erg cm^3/s]
    
    Args:
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    pomega: compton cooling rate in [erg cm^3/s]
    """
    a = 1 #cosmological scale factor, assuming z=0
    Ta = 2.727/a
    pomega = 1.017e-37 * Ta**4 * (T - Ta)
    if prime:
        dpomega_dlogT = (T / (T - Ta)) * np.log(10) * pomega
        dpomega_dT = dpomega_dlogT / (T * np.log(10))
        return dpomega_dT
    else:
        return pomega

def cooling_function(U, T, prime):
    """ 
    The total cooling function in units of [erg cm^3/s]

    Args:
    U: State of the system
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    Λ: total cooling function in [erg cm^3/s]
    """

    #unpack the state and convert to useful variables
    rho, momx, momy, E, rho_xHII, N, Fx, Fy = U

    # number densities
    n_H = (rho / mH) * X
    n_HII = n_e = (rho_xHII * X) / mH
    n_HI = n_H - n_HII

    if prime:
        L_prime = ( (ζHI(T, prime=True) + ψHI(T, prime=True)) * n_e * n_HI 
            + ηHII(T, prime=True) * n_e * n_HII 
            + θHII(T, prime=True)* n_e * n_HII 
            + pomega(T, prime=True) * n_e 
            )
        return L_prime
    else:
        L = ( (ζHI(T, prime=False) + ψHI(T, prime=False)) * n_e * n_HI 
                + ηHII(T, prime=False) * n_e * n_HII 
                + θHII(T, prime=False)* n_e * n_HII 
                + pomega(T, prime=False) * n_e 
                )

        return L

def heating_function(U, T):
    """ 
    The heating function in units of [erg cm^3/s]

    Args:
    U: State of the system
    T: Temperature in [K]

    Returns:
    H: Heating term in [erg cm^3/s]
    """
    #unpack the state and convert to useful variables
    rho, momx, momy, E, rho_xHII, N, Fx, Fy = U

    # number densities
    n_H = (rho / mH) * X
    n_HII = n_e = (rho_xHII * X) / mH
    n_HI = n_H - n_HII
    
    H = n_HI * c_r * N * (ε_γ * σ_E_HI - ε_HI * σ_N_HI)
    return H

def Λ_function(U, T, prime):
    """ 
    Energy update function in units of [erg cm^3/s] combining the heating and cooling functions.

    Args:
    U: State of the system
    T: Temperature in [K]
    prime: Boolean to return the derivative

    Returns:
    Λ: Energy update function in [erg cm^3/s] (derivative if prime=True)
    """
    if prime is True:
        return cooling_function(U, T, prime) 
    else:
        return cooling_function(U, T, prime) + heating_function(U, T)
    
def photon_update(U, dt):
    """
    Update the photon number density and flux

    Args:
    U: The state of the system
    dt: The time step

    Returns:
    U_new: The updated state of the system
    """

    #unpack the state and convert to useful variables
    rho, momx, momy, E, rho_xHII, N, Fx, Fy = U
    n_H = (rho / mH) * X
    n_HII = n_e = (rho_xHII * X) / mH
    n_HI = n_H - n_HII
    
    #photon destruction-absorption term
    D = (c_r * n_HI * σ_N_HI)
    

    # Update the photon number density and flux in x & y directions
    # for now we ignore smoothed-RT terms.
    N_new  = N /(1 + dt * D)
    Fx_new = Fx/(1 + dt * D)
    Fy_new = Fy/(1 + dt * D)
    
    if np.any(np.abs(N_new - N) / N > 0.1):
        return False
    else:
        U_new = U
        U_new[5] = N_new
        U_new[6] = Fx_new
        U_new[7] = Fy_new
        
        return U_new

def thermal_update(U, dt):
    """
    Update the thermochemical state of the system

    Args:
    U: The state of the system
    dt: The time step

    Returns:
    U_new: The updated state of the system
    """

    rho, momx, momy, E, rho_xHII, N, Fx, Fy = U
    ε = E - 0.5 * (momx**2 + momy**2) / rho
    x_HII = rho_xHII / rho

    K = ((γ - 1) * mH / (rho * kB))
    μ = 1/(X * (1 + x_HII))
    T = ε * K * μ
    T_μ = T / μ 

    n_H = (rho / mH) * X
    n_HII = n_e = (rho_xHII * X) / mH
    n_HI = n_H - n_HII

    Λ       = Λ_function(U, T, prime=False)
    Λ_prime = Λ_function(U, T, prime=True) * μ
    
    T_μ_new = T_μ + ((Λ * K * dt)/(1 - Λ_prime * K * dt))
    ε_new = T_μ_new / K
        
    #check 10% rule and first-order stability
    if np.any((np.abs(T_μ_new - T_μ) / T_μ) > 0.1) or np.any((np.abs(K * Λ * dt)/T_μ) > 0.1):
        return False
    else:
        U_new = U
        U_new[3] = ε_new + 0.5 * (momx**2 + momy**2) / rho
        return U_new


def hydrogen_ionized_fraction_update(U, dt):
    """
    Update the ionized fraction of the hydrogen

    Args:
    U: The state of the system
    T_μ: The reduced temperature in units of [K]
    dt: The time step

    Returns:
    U_new: The updated state of the system
    """

    #unpack the state and convert to useful variables
    rho, momx, momy, E, rho_xHII, N, Fx, Fy = U
    ε = E - 0.5 * (momx**2 + momy**2) / rho
    x_HII = rho_xHII / rho

    K = ((γ - 1) * mH / (rho * kB))
    μ = 1/(X * (1 + x_HII))
    T = ε * K * μ
    T_μ = T / μ 

    # temperature
    μ = 1/(X * (1 + x_HII))
    T = T_μ * μ

    # number densities
    n_H = (rho / mH) * X
    n_HII = n_e = (rho_xHII * X) / mH
    n_HI = n_H - n_HII

    # Update the ionized fraction
    D = (αB_HII(T, prime=False) * n_e)     #creation rate of HII
    C = (βHI(T) * n_e + σ_N_HI * c_r * N)  #destruction rate of HII

    dC_dx = n_H * βHI(T) - n_e * T_μ * μ**2 * X * βHI(T, prime=True)
    dD_dx = n_H * αB_HII(T, prime=False) - n_e * T_μ * μ**2 * X * αB_HII(T, prime=True)
    
    J = dC_dx - (C + D) - x_HII * (dC_dx + dD_dx)
    x_HII_new = x_HII + dt * ((C - x_HII * (C + D)) / (1 - J * dt)) 

    #check 10% rule and first-order stability
    if (np.all(np.abs(x_HII_new - x_HII) / (x_HII) > 0.1) or np.all( ((np.abs(C - x_HII*(C+D))*dt)/(x_HII)) > 0.1)):
        return False
    else:
        U_new = U
        U_new[4] = x_HII_new * rho
        return U_new


def thermochemical_step(U, dt):
    """
    Given a state U, update the state by dt. This 
    consists of 3 substeps

    I)   Photon Density & Flux Update
    II)  Thermal Update
    III) Hydrogen Ionized Fraction Update

    Args:
    U: The state of the system
    dt: The time step

    Returns:
    U_new: The updated state of the system
    """

    dt_RT = dt
    U_original = U
    t = 0
    N_subdivisions = 0

    while t < dt_RT:
        #perform the substeps
        if N_subdivisions > 20:
            raise AssertionError("Failed to converge in 20 subdivisions.")

        #update the photon number density and flux 
        #if 10% rule failed, reduce the time step and start over
        U = photon_update(U, dt)
        if U is False:
            dt = dt/2
            U = U_original
            t = 0
            N_subdivisions += 1
            print('Photon Update Failed: Halving time step and starting over')
            continue
        #update the thermal update 
        #if 10% rule failed, reduce the time step and start over
        U = thermal_update(U, dt)
        if U is False:
            dt = dt/2
            U = U_original
            t = 0
            N_subdivisions += 1
            print('Thermal Update Failed: Halving time step and starting over')
            continue
        #update the hydrogen ionized fraction
        #if 10% rule failed, reduce the time step and start over
        U = hydrogen_ionized_fraction_update(U, dt)
        if U is False:
            dt = dt/2
            U = U_original
            t = 0
            N_subdivisions += 1
            print('Ion Fraction Failed: Halving time step and starting over')
            continue
        t += dt

    return U
