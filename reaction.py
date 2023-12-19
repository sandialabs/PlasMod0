#!/usr/bin/env python
import sys
import math
import numpy as np
from scipy.integrate import odeint
from scipy.constants import m_e as m_e
from scipy.constants import e as elementary_charge
from scipy.constants import k as k_B
from scipy.constants import pi as pi
from scipy.constants import epsilon_0 as eps0

class Reaction():
    def __init__(self, line, species, xsec_folder, use_boltz):
        columns = line.split(',')
        self.rxn_name = columns[0]
        self.rxn_rate_str = columns[1]

        self.identify_reactants(species)        
        self.convert_rate_to_num(xsec_folder, use_boltz)
        self.convert_deltae(columns[2],columns[3])
        
        if not use_boltz: 
            self.momentum_xfer(species,xsec_folder)
            
    def identify_reactants(self, species):
        [lhs_str, rhs_str] = self.rxn_name.split(' -> ')
        self.lhs_num = []
        self.rhs_num = []
        
        reactants = lhs_str.split(' + ')
        products = rhs_str.split(' + ')
     
        for r in reactants: 
            self.lhs_num.append(-1)
            if r == 'hv':
                self.photo = True
                self.lhs_num[-1] = None # 'None' species index indicates photon flux
            for s, spec in enumerate(species): 
                if r == spec:
                    self.lhs_num[-1]=s
            if self.lhs_num[-1] == -1: 
                print('ERROR: species ', r, ' not found')    

        for p in products: 
            self.rhs_num.append(-1)
            for s, spec in enumerate(species): 
                if p == spec:
                    self.rhs_num[-1]=s
            if self.rhs_num[-1] == -1: 
                print('ERROR: species ', p,' not found') 
                
    def convert_rate_to_num(self,xsec_folder,use_boltz):
              
        # see if we are reading a cross section file
        self.use_xs = False
        self.special = False
        if '.dat' in self.rxn_rate_str:
            self.use_xs = True
            #self.te_list = np.concatenate((np.logspace(-6,0,num=10),np.logspace(0.1,6,num=150)))
            self.te_list = np.concatenate((np.logspace(-6,0,num=10),np.logspace(0.1,2.5,num=20),np.logspace(2.6,5,num=350),np.logspace(6,8,num=3)))
            self.rate_list = self.integrate_cross_section(xsec_folder + self.rxn_rate_str, self.te_list, False, use_boltz)

            # for verificaiton calculate rate for ideal molecule
            # analytical_rate = []
            # for te in self.te_list:
                # analytical_rate.append(7.45e-17*math.sqrt(te)*
                # math.exp(-1.62e5/te)*(1.62e5/te + 1))
            
        # if there is not cross section file
        else:
            if self.rxn_rate_str[0] == '*':
                self.special = True
            else:
                # this is an Arrhenius rate
                self.use_te = False
                self.use_tg = False
            
                self.n = 0
                self.Ea = 0
                try: 
                    float(self.rxn_rate_str)
                    self.A = float(self.rxn_rate_str)
                except ValueError:
                    terms = self.rxn_rate_str.split('*')
                    self.A = float(terms[0])
                    for i, term in enumerate(terms): 
                        if term == '': 
                            self.n = float(terms[i+1])
                            break
                    for term in terms: 
                        if 'exp(' in term and '/kTe)' in term:
                            self.use_te = True 
                            term = term.replace('exp(','')
                            term = term.replace('/kTe)','')                  
                            self.Ea = float(term)*elementary_charge/k_B # convert eV to K     
                        if 'exp(' in term and '/kTg)' in term:
                            self.use_tg = True 
                            term = term.replace('exp(','')
                            term = term.replace('/kTg)','')                    
                            self.Ea = float(term)*elementary_charge/k_B # convert J to K                        
                if self.use_te and self.use_tg:
                    print('ERROR: reaction rate was read incorrectly')
            
    def convert_deltae(self,deltae_e_str,deltae_tg_str): 
        self.elastic = False
        try: 
            float(deltae_e_str)
            self.deltae_e = float(deltae_e_str)*elementary_charge # in Joules
        except ValueError:
            if deltae_e_str.lower() == 'elastic':
                self.elastic = True
            else:
                print('ERROR: Cannot read column 3')
        
        # If elastic, check that Tg is also elastic
        if self.elastic:
            try: 
                float(deltae_tg_str)
                print('ERROR: If reaction is elastic, it must be stated', 
                ' in both energy columns')    
            except ValueError:
                if deltae_tg_str.lower() != 'elastic':
                    print('ERROR: Cannot read column 4')
        # If not elastic, just read value of deltae_tg
        else:
            self.deltae_tg = float(deltae_tg_str)*elementary_charge # in Joules                
    
    def calc_elastic(self, m, te, tg):      
        self.deltae_e = -1.5 * (2*m_e/m) * k_B * (te-tg)    
        self.deltae_tg = 1.5 * (2*m_e/m) * k_B * (te-tg)         
    def rate(self, te, tg): 
        rate = None
        error = False
        if self.use_xs:
            if te < self.te_list[0]:
                print('ERROR: Te too low, '+str(te))
                error = True
            elif te > self.te_list[-1]:
                print('ERROR: Te too high, '+str(te))
                error = True
            for i in range(len(self.te_list[:-1])):
                if te > self.te_list[i] and te < self.te_list[i+1]:
                    te_low = self.te_list[i]
                    te_high = self.te_list[i+1]
                    rate_low = self.rate_list[i]
                    rate_high = self.rate_list[i+1]
                    
                    rate = (rate_low + (rate_high - rate_low)
                    /(te_high - te_low)*(te-te_low))
        
        elif self.use_te: 
            rate = self.A * (te)**self.n * math.exp(self.Ea/te)           
        else: 
            rate = self.A * (tg)**self.n * math.exp(self.Ea/tg)

        if rate == None: 
            print('Error: rate not found for '+self.rxn_name)
            print('Te list=',self.te_list)
            print('Rate list=',self.rate_list)
            error = True
        
        if error:
            print('Te=',te)
            print('Te list=',self.te_list)
            exit()

        return rate  
          
    def integrate_cross_section(self, file_name, te_list, momxfer, use_boltz):
        import scipy
        
        of = open(file_name,"r")
        
        self.energy = []
        self.xsec = []
        #velocity = [0.0]
        print(file_name)
        for line in of.readlines():
            words = line.split()
            self.energy.append(float(words[0]))
            if momxfer:
                # if we need to convert to a momentum transfer cross section
                # M. Surendra, D. B. Graves, and G. M. Jellum, "Self-consistent model of a 
                # direct-current glow discharge: Treatment of fast electrons" Phys. Rev. A, 
                # 41, 2 (1990). Eq. 20
                epsilon = self.energy[-1]/elementary_charge
                if math.log(1+epsilon) == 0.:
                    self.xsec.append(float(words[1])) # limit of eqn. below as e --> 0
                else:
                    self.xsec.append(2./math.log(1+epsilon)*(1-math.log(1+epsilon)/epsilon)*float(words[1]))
            else:
                self.xsec.append(float(words[1]))
            #velocity.append(math.sqrt(2*self.energy[-1]/m_e))
        
        # interpolate cross section (linear in energy space)
        # this should provide a more accurate integral because vdf is non-linear
        interp_pts = 100    # points to add between each energy in .dat  
                          #0 for no interpolation
                          # a safe value is >=100
        energy_dense = [] #array with interp_points x points of input
        xsec_dense = []
        for i in range(len(self.energy)-1):
            de = (self.energy[i+1] - self.energy[i])/(interp_pts + 1)
            dxs = (self.xsec[i+1] - self.xsec[i])/(interp_pts + 1)
            for j in range(interp_pts+1):
                energy_dense.append(self.energy[i]+de*j)
                xsec_dense.append(self.xsec[i]+dxs*j)
            #energy_dense.append(lin(self.energy[i],self.energy[i+1],
            #num=interp_pts+1,endpoint=False))
            
        energy_dense.append(self.energy[-1])
        xsec_dense.append(self.xsec[-1])
        
        velocity = []
        for e in energy_dense: 
            velocity.append(math.sqrt(2*e/m_e))
        
        rate = []
        for i,te in enumerate(te_list):
            integrand = [0.0]*len(velocity)
            if not use_boltz:
                for j,v in enumerate(velocity):
                    integrand[j]=self.vmaxwellian(v,te)*xsec_dense[j]
                rate.append(scipy.integrate.trapz(integrand,velocity))
            else: 
                rate.append(0)
        
        return rate
            
    def vmaxwellian(self, v, te):
        
        # returns v*f(v) where v is the velocity and f is a maxwellian velocity distribution funciton

        vf = v * (m_e/(2*pi*k_B*te))**(1.5) * 4*pi*v**2 * math.exp(-1*m_e*v**2/(2*k_B*te))
        
        return vf 
        
    def momentum_xfer(self,species,xsec_folder):
        # if this is an elastic cross section, compute momentum transfer collision rate at 
        # each temperature in te_list
        # this is not used if we're using the boltzmann solver
        if self.elastic:
            self.mom_rate_list = self.integrate_cross_section(xsec_folder + self.rxn_rate_str, self.te_list, True, False)
            
            for r in self.lhs_num:
                if species[r] != 'e-':
                    self.mom_partner = r
            print(self.mom_partner)
            
    def mom_rate(self, te): 
        # return electron-heavy momentum transfer rate by elastic collisions 
        # interpolate values calculated in momentum_xfer
        if te < self.te_list[0]:
            print('ERROR: Te too low, '+str(te))
        for i in range(len(self.te_list[:-2])):
            if te > self.te_list[i] and te < self.te_list[i+1]:
                slope = (self.mom_rate_list[i+1] - self.mom_rate_list[i])/(self.te_list[i+1] - self.te_list[i])
                
                mom_rate = (self.mom_rate_list[i] + slope*(te-self.te_list[i]))  
        return mom_rate                
