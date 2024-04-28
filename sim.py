import const
from helper import *

import numpy as np
from types import SimpleNamespace

ndim    = 2 # number of dimensions
nvar    = 5 # number of hydro variables
nvarrad = 3 # number of radiation variables

def con_to_prim(u):
    ''' Convert from primative to conserved variables. '''
    w = np.zeros_like(u)
    w[RHO] = u[RHO]
    w[VX]  = u[PX]/u[RHO]
    w[VY]  = u[PY]/u[RHO]
    w[P]   = (const.gam-1)*(u[E]-u[RHO]/2*(w[VX]**2+w[VY]**2))
    w[PAS] = u[PAS]
    return w

def prim_to_con(w):
    ''' Convert from conservative to primative variables. '''
    u = np.zeros_like(w)
    u[RHO] = w[RHO]
    u[PX]  = w[RHO]*w[VX]
    u[PY]  = w[RHO]*w[VY]
    u[E]   = w[P]/(const.gam-1) + u[RHO]/2*(w[VX]**2+w[VY]**2)
    u[PAS] = w[PAS]
    return u

def calc_flux(u, w, dir):
    ''' Compute flux. '''
    flux = np.zeros_like(u)
    PI = PX if dir==X else PY
    VI = VX if dir==X else VY
    flux[RHO]    = u[PI]
    flux[PX]     = u[PI]*w[VX]+w[P]*(dir==X)
    flux[PY]     = u[PI]*w[VY]+w[P]*(dir==Y)
    flux[E]      = w[VI]*(u[E]+w[P])
    flux[PAS]    = w[VI][None,:,:]*u[PAS]
    return flux

def calc_flux_rad(u):
    ''' Compute flux of radiation variables. '''
    flux   = np.zeros_like([u, u])
    u_red  = np.zeros_like(u)
    chi    = np.full_like(u[NPHOT], 1/3)
    
    u_abs       = np.sqrt(u[XFLUX]**2+u[YFLUX]**2)
    cond_isphot = u[NPHOT]>0
    cond_isflux = u_abs>0
    
    u_red[NPHOT,cond_isphot]            = u_abs[cond_isphot]/(const.c*u[NPHOT,cond_isphot])
    u_red[XFLUX:YFLUX+1][:,cond_isflux] = u[XFLUX:YFLUX+1][:,cond_isflux]/u_abs[cond_isflux]
    chi[cond_isphot]                    = ((3+4*u_red[NPHOT,cond_isphot]**2)/(5+2*np.sqrt(4-3*u_red[NPHOT,cond_isphot]**2)))

    flux[X,NPHOT] = u[XFLUX]
    flux[X,XFLUX] = const.c**2*u[NPHOT]*((1-chi)/2+(3*chi-1)/2*u_red[XFLUX]**2)
    flux[X,YFLUX] = const.c**2*u[NPHOT]*(3*chi-1)/2*u_red[XFLUX]*u_red[YFLUX]

    flux[Y,NPHOT] = u[YFLUX]
    flux[Y,XFLUX] = const.c**2*u[NPHOT]*(3*chi-1)/2*u_red[XFLUX]*u_red[YFLUX]
    flux[Y,YFLUX] = const.c**2*u[NPHOT]*((1-chi)/2+(3*chi-1)/2*u_red[YFLUX]**2)

    return flux

def sed(dw_K, dw_C):
    ''' Smooth extrema detection. '''
    alpha = np.ones_like(dw_K)
    alpha[dw_C<0] = np.minimum(1, np.minimum(0,2*dw_K[dw_C<0])/dw_C[dw_C<0])
    alpha[dw_C>0] = np.minimum(1, np.maximum(0,2*dw_K[dw_C>0])/dw_C[dw_C>0])
    return alpha

def hll(wface_L, wface_R, dir):
    ''' HLL Riemann solver. '''
    VI = [VX, VY][dir]
    fluxface = np.zeros_like(wface_L)
    
    uface_L    = prim_to_con(wface_L)
    uface_R    = prim_to_con(wface_R)
    
    fluxface_L = calc_flux(uface_L, wface_L, dir)
    fluxface_R = calc_flux(uface_R, wface_R, dir)
    
    csface_L   = np.sqrt(const.gam*wface_L[P]/wface_L[RHO])
    csface_R   = np.sqrt(const.gam*wface_R[P]/wface_R[RHO])
    
    wave_L     = np.minimum(wface_L[VI], wface_R[VI]) - np.maximum(csface_L, csface_R)
    wave_R     = np.maximum(wface_L[VI], wface_R[VI]) + np.maximum(csface_L, csface_R)
            
    cond_spR   = np.logical_and(wave_L>0, wave_R>0) # supersonic to the right
    cond_spL   = np.logical_and(wave_L<0, wave_R<0) # supersonic to the left
    cond_sb    = np.logical_and(wave_L<0, wave_R>0) # subsonic
    
    fluxface[:,cond_spR] = fluxface_L[:,cond_spR]
    fluxface[:,cond_spL] = fluxface_R[:,cond_spL]
    fluxface[:,cond_sb]  = ((fluxface_L*wave_R-fluxface_R*wave_L)/(wave_R-wave_L) + wave_R*wave_L/(wave_R-wave_L) * (uface_R-uface_L))[:,cond_sb]

    return fluxface

def hllc(wface_L, wface_R, dir):
    ''' HLLC Riemann solver. '''
    VI      = [VX, VY][dir]
    VJ      = [VY, VX][dir]
    wface   = np.zeros_like(wface_L)
    wstar_L = np.zeros_like(wface_L)
    wstar_R = np.zeros_like(wface_L)
    
    csface_L          = np.sqrt(const.gam*wface_L[P]/wface_L[RHO])
    csface_R          = np.sqrt(const.gam*wface_R[P]/wface_R[RHO])
    
    wave_L            = np.minimum(wface_L[VI], wface_R[VI]) - np.maximum(csface_L, csface_R)
    wave_R            = np.maximum(wface_L[VI], wface_R[VI]) + np.maximum(csface_L, csface_R)

    wavemom_L         = wface_L[RHO]*(wave_L-wface_L[VI])
    wavemom_R         = wface_R[RHO]*(wave_R-wface_R[VI])
    wave_C            = (wface_R[P]-wface_L[P]+wface_L[VI]*wavemom_L-wface_R[VI]*wavemom_R) / (wavemom_L-wavemom_R)
    
    wstar_L[RHO]      = wface_L[RHO]*(wface_L[VI]-wave_L)/(wave_C-wave_L)
    wstar_L[VI]       = wace_C
    wstar_L[VJ]       = wface_L[VJ]
    wstar_L[P]        = wface_L[P]+wavemom_L*(wave_C-wface_L[VI])
    wstar_L[PAS]      = wface_L[PAS]*((wface_L[VI]-wave_L)/(wave_C-wave_L))[None,:,:]
    
    wstar_R[RHO]      = wface_R[RHO]*(wface_R[VI]-wave_R)/(wave_C-wave_R)
    wstar_R[VI]       = wace_C
    wstar_R[VJ]       = wface_R[VJ]
    wstar_R[P]        = wface_R[P]+wavemom_R*(wave_C-wface_R[VI])
    wstar_R[PAS]      = wface_R[PAS]*((wface_L[VI]-wave_L)/(wave_C-wave_L))[None,:,:]
    
    cond_spR          = np.logical_and(wave_L>0, wave_R>0) # supersonic to the right
    cond_spL          = np.logical_and(wave_L<0, wave_R<0) # supersonic to the left
    cond_sbL          = np.logical_and(wave_L<0, wave_C>0) # subsonic to the left
    cond_sbR          = np.logical_and(wave_C<0, wave_R>0) # subsonic to the right
    
    wface[:,cond_spR] = wface_L[:,cond_spR]
    wface[:,cond_spL] = wface_R[:,cond_spL]
    wface[:,cond_sbR] = wstar_L[:,cond_sbR]
    wface[:,cond_sbL] = wstar_R[:,cond_sbL]
    
    uface             = prim_to_con(wface)
    fluxface          = calc_flux(uface, wface, dir)

    return fluxface

def get_shapes(N):
    ''' Get array shapes. '''
    shape = SimpleNamespace(
        scalar      = (      nvar,    N+2, N+2),
        scalar_x    = (      nvar,    N+1, N),
        scalar_y    = (      nvar,    N,   N+1),
        vector      = (ndim, nvar,    N+2, N+2),
        scalarrad   = (      nvarrad, N+2, N+2),
        scalarrad_x = (      nvarrad, N+1, N),
        scalarrad_y = (      nvarrad, N, N+1),
        vectorrad   = (ndim, nvarrad, N+2, N+2)
    )
    return shape

class Init(object):
    ''' Context maanger for initial conditions. '''
    def __init__(self, grid):
        self.grid = grid
    def __enter__(self):
        return self.grid
    def __exit__(self, type, value, traceback):
        self.grid.u = prim_to_con(self.grid.w)

class Grid(object):
    '''
    2D grid object.

    Conserved variables
    0 RHO:   density
    1 PX:    x-momentum
    2 PY:    y-momentum
    3 E:     energy

    Primative variables
    0 RHO:   density
    1 VX:    x-velocity
    2 VY:    y-velocity
    3 P:     pressure
    
    Passive scalars
    4 ION:   HII density

    Radiation variables
    0 NPHOT: photon number
    1 XFLUX: x-flux of photons
    2 YFLUX: y-flux of photons

    Args
    bc_typ:  boundary condition type
    bc_val:  boundary condition value
    C:       Cournat number
    sloper:  slope limiter
    do_sed:  do smooth extrema detection
    rsolve:  riemann solver
    '''
    def __init__(self, N, bc_typ=NEU, bc_val=0, C=0.4, sloper=MINMOD, do_sed=True, rsolve=HLL, do_hydro=True, do_rad=True):

        self.N        = N
        self.bc_typ   = bc_typ
        self.bc_val   = bc_val
        self.C        = C
        self.sloper   = sloper
        self.do_sed   = do_sed
        self.rsolve   = [hllc, hll][rsolve]
        self.do_hydro = do_hydro
        self.do_rad   = do_rad

        self.dx = 1/N
        self.dt = 0
        self.t  = 0

        x1d = np.linspace(-self.dx, 1, N+2)+self.dx/2
        y1d = np.linspace(-self.dx, 1, N+2)+self.dx/2
        self.x, self.y = np.meshgrid(x1d, y1d, indexing='ij')
        
        self.shape = get_shapes(self.N)
        
        self.u               = np.zeros(self.shape.scalar)      # primative variables
        self.w               = np.zeros(self.shape.scalar)      # conserved variables
        self.flux            = np.zeros(self.shape.vector)      # fluxes
        self.urad            = np.zeros(self.shape.scalarrad)   # radiation variables
        self.fluxrad         = np.zeros(self.shape.vectorrad)   # radiation fluxes

        self.dw              = np.zeros(self.shape.vector)      # slope 
        self.whalf           = np.zeros(self.shape.scalar)      # primative variables at half timestep
        
        self.wface_L_x       = np.zeros(self.shape.scalar_x)    # primative variables at left x-dir face
        self.wface_L_y       = np.zeros(self.shape.scalar_y)    # primative variables at left y-dir face
        self.wface_R_x       = np.zeros(self.shape.scalar_x)    # primative variables at right x-dir face
        self.wface_R_y       = np.zeros(self.shape.scalar_y)    # primative variables at right y-dir face
        self.fluxface_x      = np.zeros(self.shape.scalar_x)    # conservative variable fluxes at the x-dir face
        self.fluxface_y      = np.zeros(self.shape.scalar_y)    # conservative fluxes at the y-dir face
        self.fluxfacerad_x   = np.zeros(self.shape.scalarrad_x) # radiation variable fluxes at the x-dir face
        self.fluxfacerad_y   = np.zeros(self.shape.scalarrad_y) # radiation variable fluxes at the y-dir face

    def calc_dt(self):
        ''' Compute timestep according to CFL condition. '''
        if self.do_hydro:
            cs      = np.sqrt(const.gam*self.w[P]/self.w[RHO])
            wavemax = np.max([np.abs(self.w[VX]), np.abs(self.w[VY]), cs])
            if self.do_rad: wavemax = np.max([wavemax, np.full_like(cs, const.c)])
        elif self.do_rad:
            wavemax = const.c
        self.dt = self.C*self.dx/wavemax # CFL condition
    
    def calc_slope(self):
        ''' Compute primative variable slopes. '''
        dw_L, dw_C, dw_R = np.zeros(self.shape.vector), np.zeros(self.shape.vector), np.zeros(self.shape.vector)

        dw_L[sl.x+sl.ij] = (self.w[sl.ij]   - self.w[sl.im1j]) / self.dx
        dw_L[sl.y+sl.ij] = (self.w[sl.ij]   - self.w[sl.ijm1]) / self.dx
        dw_C[sl.x+sl.ij] = (self.w[sl.ip1j] - self.w[sl.im1j]) / (2*self.dx)
        dw_C[sl.y+sl.ij] = (self.w[sl.ijp1] - self.w[sl.ijm1]) / (2*self.dx)
        dw_R[sl.x+sl.ij] = (self.w[sl.ip1j] - self.w[sl.ij])   / self.dx
        dw_R[sl.y+sl.ij] = (self.w[sl.ijp1] - self.w[sl.ij])   / self.dx
        
        # smooth extrema detection
        do_limit = np.ones(self.shape.vector, dtype=bool)
        if self.do_sed:
            alpha_L = sed(dw_L, dw_C)
            alpha_R = sed(dw_R, dw_C)
            alpha = np.minimum(alpha_L, alpha_R)
            do_limit[sl.x+sl.ij] = np.minimum.reduce([alpha[sl.x+sl.im1j], alpha[sl.x+sl.ij], alpha[sl.x+sl.ip1j]])
            do_limit[sl.y+sl.ij] = np.minimum.reduce([alpha[sl.x+sl.ijm1], alpha[sl.x+sl.ij], alpha[sl.x+sl.ijp1]])
        
        # apply slope limiter
        self.dw = dw_C
        if self.sloper == MINMOD:
            self.dw[do_limit] = np.minimum(dw_L, dw_R)[do_limit]
        elif self.sloper == MONCEN:
            self.dw[do_limit] = np.minimum.reduce([2*dw_L, dw_C, 2*dw_R])[do_limit]

    def halfstep(self):
        ''' Evolve primative variables a half timestep using the slope. '''
        self.whalf[RHO] = self.w[RHO] - self.dt/2 * (self.w[VX]*self.dw[X,RHO]            + self.w[VY]*self.dw[Y,RHO]            + self.w[RHO]*self.dw[X,VX]           + self.w[RHO]*self.dw[Y,VY])
        self.whalf[VX]  = self.w[VX]  - self.dt/2 * (self.w[VX]*self.dw[X,VX]             + self.w[VY]*self.dw[Y,VX]             + (1/self.w[RHO])*self.dw[X,P])
        self.whalf[VY]  = self.w[VY]  - self.dt/2 * (self.w[VX]*self.dw[X,VY]             + self.w[VY]*self.dw[Y,VY]             + (1/self.w[RHO])*self.dw[Y,P])
        self.whalf[P]   = self.w[P]   - self.dt/2 * (self.w[VX]*self.dw[X,P]              + self.w[VY]*self.dw[Y,P]              + const.gam*self.w[P]*self.dw[X,VX]   + const.gam*self.w[P]*self.dw[Y,VY]) 
        self.whalf[PAS] = self.w[PAS] - self.dt/2 * (self.w[VX][None,:,:]*self.dw[X][PAS] + self.w[VY][None,:,:]*self.dw[Y][PAS] + self.w[PAS]*self.dw[X,VX][None,:,:] + self.w[PAS]*self.dw[Y,VY][None,:,:])
    
    def calc_face(self):
        ''' Compute primative variables on the interface. '''
        self.wface_L_x = (self.whalf + self.dw[X]*self.dx/2)[:,:-1,1:-1]
        self.wface_L_y = (self.whalf + self.dw[Y]*self.dx/2)[:,1:-1,:-1]
        self.wface_R_x = (self.whalf - self.dw[X]*self.dx/2)[:,1:,1:-1]
        self.wface_R_y = (self.whalf - self.dw[Y]*self.dx/2)[:,1:-1,1:]

    def riemann(self):
        ''' Solve the Riemann problem for hydro variables. '''
        self.fluxface_x = self.rsolve(self.wface_L_x, self.wface_R_x, dir=X)
        self.fluxface_y = self.rsolve(self.wface_L_y, self.wface_R_y, dir=Y)
        
    def addflux(self):
        ''' Evolve the conserved variables a full timestep using the fluxes. '''
        self.u[sl.ij] = self.u[sl.ij] + self.dt/self.dx * (self.fluxface_x[:,:-1,:]-self.fluxface_x[:,1:,:]+self.fluxface_y[:,:,:-1]-self.fluxface_y[:,:,1:])
    
    def step_hydro(self):
        ''' Full timestep for the hydro variables. '''
        self.calc_slope()
        self.halfstep()
        self.calc_face()
        self.riemann()
        self.addflux()
    
    def riemann_rad(self):
        ''' Solve the Riemann problem for radiation variables. '''
        fluxrad            = calc_flux_rad(self.urad)
        self.fluxfacerad_x = (fluxrad[X,:,1:,1:-1]+fluxrad[X,:,:-1,1:-1])/2 - const.c*(self.urad[:,1:,1:-1]-self.urad[:,:-1,1:-1])/2
        self.fluxfacerad_y = (fluxrad[Y,:,1:-1,1:]+fluxrad[Y,:,1:-1,:-1])/2 - const.c*(self.urad[:,1:-1,1:]-self.urad[:,1:-1,:-1])/2
    
    def addflux_rad(self):
        ''' Evolve the radiation variables a full timestep using the fluxes. '''
        self.urad[sl.ij] = self.urad[sl.ij] + self.dt/self.dx * (self.fluxfacerad_x[:,:-1,:]-self.fluxfacerad_x[:,1:,:]+self.fluxfacerad_y[:,:,:-1]-self.fluxfacerad_y[:,:,1:])
    
    def step_rad(self):
        ''' Full timestep for radiation variables. '''
        self.riemann_rad()
        self.addflux_rad()
    
    def bc(self):
        ''' Apply boundary conditions. '''
        pad_mode = ['constant', 'edge', 'wrap'][self.bc_typ]
        if self.do_hydro: self.u    = np.pad(self.u[sl.ij],    [(0,0),(1,1),(1,1)], mode=pad_mode)
        if self.do_rad:   self.urad = np.pad(self.urad[sl.ij], [(0,0),(1,1),(1,1)], mode=pad_mode)

    def step(self):
        ''' Full timestep. '''
        self.bc()
        if self.do_hydro: self.w = con_to_prim(self.u)
        self.calc_dt()
        if self.do_hydro: self.step_hydro()
        if self.do_rad:   self.step_rad()
        self.t += self.dt

    def inject(self, coord, rate, xflux=0, yflux=0):
        ''' Inject photons into the grid. '''
        idx_x = np.searchsorted(self.x[:, 0], coord[X])
        idx_y = np.searchsorted(self.y[0, :], coord[Y])
        self.urad[NPHOT,idx_x,idx_y] += rate*self.dt
        self.urad[XFLUX,idx_x,idx_y] += xflux*rate*self.dt
        self.urad[YFLUX,idx_x,idx_y] += yflux*rate*self.dt
