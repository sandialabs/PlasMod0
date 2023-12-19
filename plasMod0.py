#!/usr/bin/env python
import sys
import math
import os
#import matlab.engine # uncomment for multibolt coupling
import numpy as np
import copy
from scipy.integrate import odeint
from scipy.constants import m_e as m_e
from scipy.constants import e as elementary_charge
from scipy.constants import k as k_B
from scipy.constants import pi as pi
from scipy.constants import epsilon_0 as eps0

import reaction

def initialize_boltz(conditions,rxns):
    # make the data structure of reaction data to be passed to MultiBolt
    # This requires a modified version of multibolt (making it callable), which is not distributed.
    conditions.boltz_rxns = []
    boltz_spec = []
    conditions.tboltz_update_next=conditions.tboltz_update[0]
    for s in range(len(conditions.species)):
        if conditions.m[s] > 1e-30:       # ignore e, Te, Tg, emom
            boltz_spec.append(s)
            for r,rxn in enumerate(rxns):
                if rxn.use_xs:          #exclude anything that uses rates
                    if s in rxn.lhs_num: #
                        #boltz_rxns_dict.append({'species':s,'rxn name':rxn.rxn_name,'energy':rxn.energy,'xsec':rxn.xsec,'rxn_type':0,'rxn_number':0})
                        # determine rxn type (1:attachment, 2:elastic, 3:excitation, 4:ionization)
                        if rxn.elastic:
                            # elastic
                            rxn_type = 2
                            thresh = 0.
                        elif rxn.rhs_num.count(conditions.ie) < rxn.lhs_num.count(conditions.ie):
                            # attachment
                            rxn_type = 1
                            thresh = rxn.deltae_e
                        elif rxn.rhs_num.count(conditions.ie) > rxn.lhs_num.count(conditions.ie):
                            # ionization
                            rxn_type = 4
                            thresh = rxn.deltae_e
                        else:
                            # excitation
                            rxn_type = 3
                            thresh = rxn.deltae_e
                        conditions.boltz_rxns.append([s,rxn.rxn_name,rxn.energy,rxn.xsec,thresh,rxn_type,r])
    
    # sort reactions by species then rxn_type
    conditions.boltz_rxns.sort(key=lambda x: (x[0], x[5]))
    
    # Multibolt
    print('Initializing MultiBolt')
    conditions.matlab_eng = matlab.engine.start_matlab()
    #conditions.matlab_eng = matlab.engine.start_matlab("-desktop") # for debugging in matlab
    conditions.matlab_eng.cd('spams-0d/MultiBolt_callable')
    #conditions.matlab_eng.eval(dbstop in Initialize_multibolt)
    success = conditions.matlab_eng.Initialize_multibolt(conditions.species,conditions.m,conditions.boltz_rxns,conditions.EN,4,2000)
    if success == False: 
        print('MultiBolt Initialization Failed')
    
    update_boltz(conditions,rxns,conditions.n0)    
    
def update_boltz(conditions,rxns,n):
    
    # update the boltzmann solver and the rates, mean_energy, and transport coefficient
    # n must be a list (not a numpy array
    print('Calling MultiBolt')
    error = False
    
    # must convert numpy array to list to pass to Matlab
    if isinstance(n, np.ndarray):
        n_list = n.tolist()
    else:
        n_list = n
    print('n_list=',n_list)     

    # Calculate rates for gas composition
    [rates,mean_energy,transport,converged] = conditions.matlab_eng.MultiBolt_v201_callable(n_list,nargout=4)

    for i,c in enumerate(converged[0]):
        if not c:
            print('Error: multibolt not converged, E/N =',conditions.EN[i],'V*m^2')
            #error = True

    for rb,boltz_rxn in enumerate(conditions.boltz_rxns):
        r = boltz_rxn[6] # retreive the original rxn number
        rxns[r].te_list = [x*elementary_charge/(3./2.*k_B) for x in mean_energy[0]]
        rxns[r].rate_list = rates[rb] 
        # momentum transfer collision frequency / N
        rxns[r].nu_m_over_N = [elementary_charge/(m_e*x[0]) for x in transport] # transport[0] is a list of mobility*N
        
        for e in range(len(rxns[r].te_list)-1):
            if rxns[r].te_list[e] >= rxns[r].te_list[e+1]:
                print('ERROR: MEAN ENERGIES NON-MONOTONIC')
                print(rxns[r].te_list)
                error=True
        
        if min(rxns[r].rate_list) < 0.0:
            print('ERROR: NEGATIVE REACTION RATE')
            print(rxns[r].rxn_name)
            error=True   
    
    if error:
        exit()
        
def ohmic_heating(t,n,R,conditions,N_tot):
    # this function returns the ohmic heating rate in W/m^2
    e = elementary_charge
    q = conditions.q
    E_dc = conditions.E_dc
    ie = conditions.ie
    ite = conditions.ite
    iemom = conditions.iemom
    
    ne = n[ie]

    # calculate nu_m
    nu_m = coll_freq(n,ite,N_tot)
    
    #nu_m = 1e-13*N_tot# momentum transfer collision frequency - very rough estimate
    laser_freq = 7.595e15 # radians/s
    intensity = 5*elementary_charge * conditions.photon_flux(t) # W/m^2
    efield = (753.5*intensity)**0.5 # E = sqrt(2*I/c*eps0) V/m
    conductivity = (elementary_charge)**2/m_e*n[ie]/max(nu_m,1e-30)
    ohmic_heating = 0.5*efield**2*conductivity*nu_m**2/(laser_freq**2+nu_m**2)
    
    # calculate ion charge density
    ni = 0.
    for s in range(len(n)):
        if s != ie:
            if q[s] != 0: 
                ni += q[s]*n[s]/e # q is in C
    
    # applied electric field
    ohmic_heating += max(n[iemom],0.0)*e*max(E_dc - 2.*(ni-ne)*e*R/(3.*eps0),0.)
    
    return ohmic_heating

def e_diff_flux(n,q,ie,ite,R):
    # calculate the flux of electrons out of the plasma due to diffusion

    ne = n[ie]
    Te = n[ite]
    
    # calculate ion charge density
    ni = 0.
    for s in range(len(n)):
        if s != ie:
            if q[s] != 0: 
                ni += q[s]*n[s]/elementary_charge # q is in C 
                
    e_diff_flux = ne/4.*(8.*k_B*Te/(pi*m_e))**(0.5)*math.exp(
    -1.*elementary_charge**2*(ni-ne)*R**2/(3*eps0*k_B*Te))
    
    return e_diff_flux

def coll_freq(n,ite,N_tot):

    # if we're using the boltzman solver, use mobility from there
    if conditions.use_boltz:
        # te_list and nu_m_over_N 
        te = n[ite]
        if te < rxns[0].te_list[0]:
            print('ERROR: Te too low, '+str(te))
            exit()
        elif te > rxns[0].te_list[-1]:
            print('ERROR: Te too high, '+str(te))
            exit()
        # linearly interpolate
        for i in range(len(rxns[0].te_list[:-1])):
            if te >= rxns[0].te_list[i] and te < rxns[0].te_list[i+1]:
                te_low = rxns[0].te_list[i]
                te_high = rxns[0].te_list[i+1]
                nu_low = rxns[0].nu_m_over_N[i]
                nu_high = rxns[0].nu_m_over_N[i+1]
                
                interp = (nu_low + (nu_high - nu_low)
                /(te_high - te_low)*(te-te_low))
        nu_m = interp*N_tot
    else:
        nu_m = 0.
        for rxn in rxns:
            if rxn.elastic:
                nu_m += n[rxn.mom_partner]*rxn.mom_rate(n[ite])
    return nu_m        
    
def demomdt(n,q,E_dc,ie,iemom,ite,R,N_tot):
    # rate of change of (n_e*u_e)
    e = elementary_charge # C
    ne = n[ie]
    mom = n[iemom]
    
    # calculate ion charge density
    ni = 0.
    for s in range(len(n)):
        if s != ie:
            if q[s] != 0: 
                ni += q[s]*n[s]/e # q is in C
                
    # calculate nu_m
    nu_m = coll_freq(n,ite,N_tot)
    
    # E field term 
    # this term should be allowed to slow down the electrons, but not drive them the other way
    # this approach led to a discontinuous derivative, confusing the solver
    #if mom <= 0:
    #    demomdt = ne*e/m_e*max(E_dc - 2.*(ni-ne)*e*R/(3.*eps0),0.0) 
    #else:
    #    demomdt = ne*e/m_e*(E_dc - 2.*(ni-ne)*e*R/(3.*eps0))
    
    # E field term 
    # this term should be allowed to slow down the electrons, but not drive them the other way
    # we've added an artificial linear function at low drift velocity to keep the derivative continuous (but not smooth!)
    vdrift = mom/ne
    vdrift_lin = 1e-4 # linear transition to 0
    
    E_term = ne*e/m_e*(E_dc - 2.*(ni-ne)*e*R/(3.*eps0))
    if E_term >= 0:
        demomdt = E_term
    else:
        if vdrift >= vdrift_lin:
            demomdt = E_term
        elif (vdrift < vdrift_lin) and (vdrift > 0):
            #print('applying factor=',vdrift/vdrift_lin,'Eterm=',E_term)
            demomdt = E_term*(vdrift/vdrift_lin)
        else:
            demomdt = 0.
    
    # collisions
    demomdt -= mom*nu_m
    
    # convection
    demomdt -= 3./4.*mom**2/(ne*R)
    
    # diffusion
    demomdt -= 3.*mom/ne*e_diff_flux(n,q,ie,ite,R)/R
    
    return demomdt 
    
def derivative(n,t,rxns,conditions):
    # n is list of densities (+temperatures)
    # t is time in seconds
    dndt = [0] * len(n)
    
    # calculate sum of densities of all heavy species
    N_tot = 0.
    for i in range(len(n)):
        if conditions.m[i] > 1e-30:
            N_tot = N_tot + n[i]
        
    photon_absorbed = 0.
    
    # access indices
    ie = conditions.ie
    ite = conditions.ite
    itg = conditions.itg
    iemom = conditions.iemom
    
    # momentum must be positive
    #n[iemom] = max(n[iemom],0.0)
    
    # expansion 
    [V, dVdt, R] = conditions.expansion_rate(t)

    # if we don't have any electrons, the temperature doesn't matter
    if n[ie]*V < 1e-2:
        n[ite] = 300
        if not conditions.limited_te:
            conditions.limited_te = True 
            print('limiting Te, ne=',n[ie],' time=',t)

    # update Boltzmann solver if it's time to do so
    if conditions.use_boltz and t >= conditions.tboltz_update_next:
        update_boltz(conditions,rxns,n)
        # increment the time that will next update the Boltzmann solver
        new_idx = np.where(conditions.tboltz_update==conditions.tboltz_update_next)[0][0]+1
        conditions.tboltz_update_next = conditions.tboltz_update[new_idx]

    # reactions
    # calculate rate R_j (usually k*n1*n2)
    for rxn in rxns:
        if rxn.special: 
            rxn.rate_mult = conditions.special_rate(rxn,n,ite,itg,ie)
        else:
            rxn.rate_mult = rxn.rate(n[ite],n[itg])
        for r in rxn.lhs_num: 
            if r == None:
                rxn.rate_mult = rxn.rate_mult * conditions.photon_flux(t)
            else:
                rxn.rate_mult = rxn.rate_mult * n[r]
        # keep track of how many photons we lose
        for r in rxn.lhs_num:
            if r == None:        
                photon_absorbed += rxn.rate_mult     
    # if we absorbed too many photons, scale back the rates
    if (photon_absorbed*V > conditions.photon_flux(t)*pi*R**2):
        multiplier = conditions.photon_flux(t)*pi*R**2/(photon_absorbed*V)
        for rxn in rxns:
            if None in rxn.lhs_num:
                rxn.rate_mult = rxn.rate_mult*multiplier                     
    for rxn in rxns:
        # add it to reactants and products
        for r in rxn.lhs_num:
            if r != None:        
                dndt[r] -= rxn.rate_mult   
        for p in rxn.rhs_num: 
            dndt[p] += rxn.rate_mult
 
        # for elastic, update deltae_e and deltae_tg        
        if rxn.elastic:
            for r in rxn.lhs_num: 
                if r != ie:
                    rxn.calc_elastic(conditions.m[r],n[ite],n[itg])
            
        # add electron heating (energy density)
        dndt[ite] += rxn.deltae_e*rxn.rate_mult
        
        # add gas heating (energy density)
        dndt[itg] += rxn.deltae_tg*rxn.rate_mult   
    
    # e- losses because of applied field
    dndt[ie] -= max(n[iemom],0.0)*3./(4.*R)
    
    # e- losses due to diffusion
    dndt[ie] -= 3.*e_diff_flux(n,conditions.q,ie,ite,R)/R
    
    # electron energy losses due to diffusion
    dndt[ite] -= (3./2.*k_B*n[ite])*3.*e_diff_flux(n,conditions.q,ie,ite,R)/R

    # Ohmic heating for electrons
    dndt[ite] += ohmic_heating(t,n,R,conditions,N_tot)
    # convective energy losses
    dndt[ite] -= 3./2.*k_B*n[ie]*n[ite]*max(n[iemom],0.0)/n[ie]*4.*R/3. 
    
    # Electron Momentum for separtion, account for expansion
    dndt[iemom] = demomdt(n, conditions.q, conditions.E_dc, ie, iemom, ite, R, N_tot) 
    
    # expansion - applies to densities and momuntum  
    for i in range(len(dndt)):
        if i != ite and i != itg:
            dndt[i] -= n[i]/V*dVdt      

    # expansion - applies to energy densities 
    gamma = 5./3. # adiabatic constant for monatomic gas
    dndt[ite] -= gamma*3./2.*k_B*n[ie]*n[ite]/V*dVdt
    dndt[itg] -= gamma*3./2.*k_B*N_tot*n[itg]/V*dVdt
    
    # calculate dN_tot/dt
    dN_totdt = 0.
    for i in range(len(dndt)):
        if conditions.m[i] > 1e-30:
            dN_totdt = dN_totdt + dndt[i]
    
    # Convert from energy densities to temperatures 
    dndt[ite] = dndt[ite]/(3./2.*k_B*max(n[ie],1)) - dndt[ie]/max(n[ie],1)*n[ite]
    dndt[itg] = dndt[itg]/(3./2.*k_B*N_tot) - n[itg]*dN_totdt/N_tot 

    return dndt
    
def save_output(rxns, solution, conditions, working_dir):
    sim_name = conditions.sim_name
    print(working_dir)
    output_files = [working_dir+sim_name+'_densities.out',working_dir+sim_name+'_rates.out',working_dir+sim_name+'_heating.out']

    # access indices
    ie = conditions.ie
    ite = conditions.ite
    itg = conditions.itg
    iemom = conditions.iemom
    
    # output densities
    of = open(output_files[0],"w")
    outline = 'time,'
    for s in range(len(conditions.species)):
        outline = outline + conditions.species[s] + ','
    outline = outline + 'photon_flux,' + 'volume,'
    of.write(outline + '\n')
    for t in range(len(conditions.t)):
        outline = str(conditions.t[t])+','
        for s in range(len(conditions.species)):
            outline = outline + str(solution[t,s])+','
        flux = conditions.photon_flux(conditions.t[t])
        [V, dVdt, R] = conditions.expansion_rate(conditions.t[t])
        outline = outline + str(flux) + ',' + str(V) + ','
        of.write(outline + '\n')
    of.close()     

    # output reaction rates
    of = open(output_files[1],"w")
    outline = 'time,'
    for rxn in rxns:
        outline = outline + rxn.rxn_name + ','
    of.write(outline + '\n')
    for t in range(len(conditions.t)):
        outline = str(conditions.t[t]) + ','
        n = solution[t,:]
        dndt = derivative(n,conditions.t[t],rxns,conditions)
        for rxn in rxns:
            outline = outline + str(rxn.rate_mult) + ','       
        of.write(outline + '\n')
    of.close()        
    
    # output heating rates
    of = open(output_files[2],"w")
    outline = 'time,'
    temperatures = ['_Te','_Tg']
    for temp in temperatures:
        for rxn in rxns:
            outline = outline + rxn.rxn_name + temp + ','
        if temp == '_Te':
            outline = outline + 'ohmic' + temp + ','
    of.write(outline + '\n')
    for t in range(len(conditions.t)):
        outline = str(conditions.t[t]) + ','
        n = solution[t,:]
        dndt = derivative(n,conditions.t[t],rxns,conditions)
        N_tot = 0
        for i in range(len(n)): 
            if i != ie and i != ite and i != itg and i != iemom: 
                N_tot += n[i]
        for rxn in rxns:
            outline = outline + str(rxn.rate_mult*rxn.deltae_e) + ','
        outline = outline + str(ohmic_heating(conditions.t[t],n,R,conditions,N_tot)) + ','
        for rxn in rxns:
            outline = outline + str(rxn.rate_mult*rxn.deltae_tg) + ','
        of.write(outline + '\n')
    of.close()  

if __name__ == '__main__':

    ################
    # Parse and Initialize
    ################
    
    print(os.path.abspath(sys.argv[1]))
    input_file = os.path.abspath(sys.argv[1])
    
    divide = input_file.rfind('/')+1
    working_dir = input_file[:divide]
    sys.path.insert(1, working_dir)
    input = __import__(input_file[divide:-3])

    conditions = input.Setup()

    #################
    # Read reaction file
    #################
    of = open(os.path.abspath(conditions.rxn_file),"r")
    rxns=[]
    for line in of.readlines()[1:]:
        rxns.append(reaction.Reaction(line,conditions.species,conditions.xsec_folder,conditions.use_boltz))
    
    #################
    # Initialize Boltzmann Solver
    #################
    if conditions.use_boltz:    
        #print('Initializing Boltzmann Solver')
        #initialize_boltz(conditions,rxns)
        print('ERROR: No Boltzmann solver present. Change use_boltz to False.')
        exit()
    
    #################
    # Solve
    ################# 
    
    # flag to show whether or not we're limiting te because ne is too low
    conditions.limited_te = False
    #print(derivative(conditions.n0,0,rxns,conditions))
    
    # increase absolute error bounds for electron momentum
    abs_tol = [1e-8 for x in conditions.n0] # this is the default
    abs_tol[conditions.iemom] = conditions.mom_tol 
    
    # solve
    solution = odeint(derivative,conditions.n0,conditions.t,args=(rxns,conditions),atol=abs_tol,mxstep=conditions.solver_mxstep)
    #print(solution[0:3])
    
    #################
    # Save and Close
    #################
    save_output(rxns,solution,conditions,working_dir)
       
    if conditions.use_boltz:
        conditions.matlab_eng.quit
