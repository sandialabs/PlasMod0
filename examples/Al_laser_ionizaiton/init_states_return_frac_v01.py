#!/usr/bin/env python

# This script calculates the density of each electronic excited state  and vibrational state
# in LTE assuming a Boltzmann distribution. 
# it has not been tested for molecules

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

class Species():
    def __init__(self, input_file, T):
        of = open(input_file,"r")
        self.elec_states = []  

        for line in of.readlines()[1:]:
            self.elec_states.append(ElecState(line,T))
        
        # check if molecule or atom is consistent for all electronic states
        for elec_state in self.elec_states:
            if elec_state.molecule != self.elec_states[0].molecule:
                print('ERROR: some levels are atoms, some are molecules')
                exit()                
        
    def calculate_distribution(self):
        # calculate mole fraction of each state
        self.frac = {} # dictionary
        temp_sum = 0.
        
        # total
        for e in self.elec_states:
            if e.molecule:
                for v in range(max(e.vib_states,1)):
                    temp_sum += e.boltz_fac[v]
            else:
                temp_sum += e.boltz_fac
        
        # normalize        
        for e in self.elec_states:        
            if e.molecule:     
                self.frac[e.name] = []            
                for v in range(max(e.vib_states,1)):
                    self.frac[e.name].append(e.boltz_fac[v]/temp_sum)
            else:     
                self.frac[e.name] = e.boltz_fac/temp_sum
                
        return self.frac
            
class ElecState():
    def __init__(self, line, T):
        words = line.split()
        self.name = words[0]  # Name of each electronic excited state
        self.energy = float(words[1]) # Energy of electronic state (eV)
        self.elec_g = int(words[2]) # Degeneracy of electronic state
            # Number of Vibrational levels in each electronic state (including v=0)
        self.vib_states = int(words[3])  # use 0s for atoms
        self.vib_energy = float(words[4])

        # for atoms
        if self.vib_states == 0:
            self.molecule = False
            self.boltz_fac = self.elec_g*math.exp(-1*self.energy*const.e/(const.k*T))
        # for moleucles
        else:
            self.molecule = True
            self.boltz_fac = []
            for v in range(self.vib_states):
                self.boltz_fac.append(self.elec_g*
                math.exp(-1* (self.energy+self.vib_energy*(2*v+1)) * const.e/(const.k*T)))  

def initialize(input_files, T):

    species_list = []
    output = []
    for input_file in input_files: 
        print("Reading data from " + input_file)
        species_list.append(Species(input_file, T))
        output.append(species_list[-1].calculate_distribution())
        
    return output
