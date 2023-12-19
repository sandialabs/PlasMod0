#!/usr/bin/env python
import numpy as np
import math
from scipy.constants import e as elementary_charge
from scipy.constants import pi as pi
from scipy.constants import k as k_B
import csv

class Setup():
    def __init__(self):
        import init_states_return_frac_v01 as init_dist
        
        self.xsec_folder = './cross_sections/'
        
        # INPUTS
        ###############################################
        self.sim_name = 'Al_base_v02' 
        self.rxn_file='./Al_6exc_v15.csv'
        self.t0 = 0 # time of phase transition in s
        self.init_radius = 3.16e-7 # m, ablation for 8 mJ and 
        self.E_dc = 0 # V/m
        self.use_boltz = False # use boltzmann solver?
        self.T0 = 2000 # K
        ###############################################
        
        # calculate initial number densities
        rho = 2.7 * 1e6 # convert g/cm^3 to g/m^3
        N_total = 4./3.*math.pi*(self.init_radius)**3*rho/27*6.02e23 # V * rho / atomic_mass * N_A = atoms
        # extract initial volume from expansion function to calculate initial number density
        #[V0, x, x] = self.expansion_rate(self.t0) 
        V0 =4./3.*math.pi*(self.init_radius)**3
        self.n_total = N_total/V0 # atoms/m^3
        
        #load boltzmann distribution of excited states
        boltz = init_dist.initialize(["./InputAl_states.txt"], self.T0)[0]
        for i in boltz:
            boltz[i] = boltz[i]*self.n_total
        # .item converts from numpy.float64 to float
        n_Al = boltz['Al(0)'] + boltz['Al(0.01)']
        n_42S = boltz['Al(3.14)']
        n_34P = boltz['Al(3.598)'] + boltz['Al(3.604)'] + boltz['Al(3.613)']
        n_32D = boltz['Al(4.02)']
        n_42P = boltz['Al(4.085)'] + boltz['Al(4.087)']
        n_52S = boltz['Al(4.673)']
        n_y2D = boltz['Al(4.827)']
        n_e = max(1e17, boltz['Al+'],1./V0)
        
        # Species arrays
        self.species=['e-','Al','Al42S','Al34P','Al32D','Al42P','Al52S','Aly2D','Al+','Te','Tg','emom']
        self.n0 = [n_e, n_Al, n_42S, n_34P, n_32D, n_42P, n_52S, n_y2D, n_e, self.T0, self.T0,0.]
        self.m = [9.11e-31, 4.48e-26,4.48e-26,4.48e-26,4.48e-26,4.48e-26,4.48e-26,4.48e-26,4.48e-26,0.,0.,0.]
        q_int = [-1.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]
        self.q = [x*elementary_charge for x in q_int]
        
        # Timesteps
        dt0 = 1e-17
        tend = 1e-5
        self.t = np.logspace(math.log(dt0,10),math.log(tend,10),num=50000)
        #self.t = np.logspace(math.log(dt0,10),math.log(1e-16,10),num=10)
        self.t[0]=0
        for i in range(len(self.t)):
            self.t[i] += self.t0
            
        # tolerances
        self.mom_tol = 1e-8 # default 
        self.solver_mxstep = 0 # default
        
        # boltzmann solver
        Td = [1e-2,3e-2,1e-1,3e-1,1,3,1.e1,3e1,1e2,3e2,1e3,3e3,1e4] # Td
        self.EN = [x*1e-21 for x in Td] # convert from Td to V*m^2
        self.tboltz_update = np.logspace(math.log(dt0,10),math.log(tend,10),num=10)
        self.tboltz_update = self.tboltz_update[1:-1]
        for i in range(len( self.tboltz_update)):
             self.tboltz_update[i] += self.t0
        print(self.tboltz_update)
        self.tboltz_update=[1e30] # don't update boltz
        
        self.find_specials()
    
    def find_specials(self):
        # Identify position of each 'special' species in the species arrays
        for i,name in enumerate(self.species):
            if name == 'e-':
                self.ie = i
            if name == 'Te':
                self.ite = i
            if name == 'Tg':
                self.itg = i
            if name == 'emom':
                self.iemom = i    

    def expansion_rate(self, t):
        # this function returns the rate of change of the plasma volume in m^3/s
        # it can be a function of time, t

        # This is just a very rough approximation that fits the limits

        t0 = self.t0
        tau = 1e-8
        R0 = self.init_radius

        # calculate velocity we approach asymptotically
        v_inf = (2.*3./2.*k_B*self.T0/(4.48e-26))**0.5 # m/s

        dRdt = v_inf*(1-math.exp(-(t-t0)/tau))
        R = v_inf*(t + tau*math.exp(-(t-t0)/tau) - t0 - tau) + R0
        dVdt = 4.*pi*R**2*dRdt
        V = 4./3.*pi*R**3

        return [V, dVdt, R]
        
    def photon_flux(self, t):
        # this function returns the photon flux in 1/m^2/s
        # it can be a function of time, t
        
        # constant flux
        avg_flux = 2.497e30 # photons/m^2*s,8 mJ
        FWHM = 8e-9
        
        # Gaussian
        sigma = FWHM/2.355
        #mu = 7.9027e-9 # 1% of normal 
        mu = 0
        if t <= 5e-8:
            flux = avg_flux*FWHM /(sigma*(2.*3.14159)**0.5) * math.exp(-0.5*((t-mu)/sigma)**2)
        else:
            flux = 0.0
        return flux 

    def special_rate(self, rxn,n,ite,itg,ie):
        # this function returns the rate of a reaction with a special dependence (not Arrhenius or cross section)
        
        # change this if we find a rate that matches
        found = False
        
        # pressure broadened photoexcitation to Al*** at 4.83 eV by a 4.999 eV photon
        if rxn.rxn_rate_str == '*broadened_photoexc':
            found = True
            nAl = n[1]
            N =     [0,    1e15,    1e26,     2e26,     4e26,     6e26,     8e26,     9e26,     1e27,   1.5e27,     2e27,     3e27,     4e27,     5e27,     6e27,     8e27,     1e28,     2e28,     3e28,     4e28,     6e28,     8e28,     1e29]
            sigma = [0,4.15e-34,4.07e-23,7.823e-23,1.373e-22,1.727e-22,1.893e-22,1.926e-22,1.936e-22,1.813e-22,1.615e-22,1.291e-22,1.078e-22,9.355e-23,8.343e-23,7.018e-23,6.194e-23,4.492e-23,3.911e-23,3.619e-23,3.325e-23,3.177e-23,3.089e-23]

            # interpolate
            for i,n in enumerate(N):
                if n > nAl:
                    rate = sigma[i-1] + (sigma[i]-sigma[i-1])/(N[i]-N[i-1]) * (nAl-N[i-1])
                    break
                    
        # three body recombination from Y. Hahn, "Plasma density effects on the three-body recombination rate coefficients", Physical Letters A 231, 82 (1997)
        if rxn.rxn_rate_str == '*tbr':
            found = True
            lambda_d = (1.222e-34*n[ite]/(max(n[ie],1e-10)*elementary_charge**2.))**0.5
            plasma_param = 4*pi*n[ie]*lambda_d**3
            if n[ite] < 3.82e-5*n[ie]**(1./3.) and n[ite] > 1.19e-7*n[ie]**(1./3.):
                # derived from Eq. 6,8,11 (low Te limit)
                # nD is the minimum only when xn>>1 (low temp)
                rate = 8.22e3*n[ie]**(-7./4.)*n[ite]**(3./4.)
                #print('limiting tbr')
                #print(rate,5.43e-20*n[ite]**-4.5)
            else:
                #if plasma_param < 1:
                if n[ite] < 2.65e-5*n[ie]**(1./3.):
                    #Eq. 15 from Hahn, only possible when xn>>1 (low temp)
                    rate = 5.68e4/n[ite]*n[ie]**(-7./6.)
                else:
                    #Eq. 9 from Hahn
                    # applies to both high and low temp
                    rate = 5.43e-20*n[ite]**-4.5
        
        if rxn.rxn_rate_str == '*rr':
            found = True
            # N. R. Badnell, "Radiative recombination data for modeling dynamic \
            # finite-density plasmas" The Astrophysical Journal Supplement Series, 167, 334 (2006)
            # Eq. 1 and 2
            # constants from http://amdpp.phys.strath.ac.uk/tamoc/RR/
            T = n[ite]
            T0 = 4.623e-02
            T1 = 1.564e9
            T2 = 1.533e5
            
            A = 1.035e-9
            B = 0.6535
            C = 0.1261
            
            B = B + C*math.exp(-1.*T2/T)
            # Eq. 1, converted to m^3/s
            rate = 1e-6*A/((T/T0)**0.5*(1.+(T/T0)**0.5)**(1.-B)*(1.+(T/T1)**(1./2.))**(1.+B)) 
        
        if rxn.rxn_rate_str == '*dr':
            found = True    
            # Z. Altun, A. Yumak, I. Yavuz, N. R. Badnell, S. D. Lock, and M. S. Pindzola, Astronomy and Astrophysics 474, 1041 (2007)
            c = [4.50E-07, 2.61E-06, 3.82E-03, 5.80E-04]
            E = [4.94E+03, 1.40E+04, 8.26E+04, 1.10E+05]
            
            T = n[ite]
            rate = 0.0
            for i in range(len(c)):
                # Eq. 5
                rate += 1e-6*(T**-1.5)*c[i]*math.exp(-E[i]/T)
         
        # if the special rate is not found
        if found == False: 
            print('ERROR: SPECIAL RATE NOT FOUND')
        
        return rate     

