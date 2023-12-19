#!/usr/bin/env python
#
import sys  
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

class Dataset():
    def __init__(self, input_files):
        
        [self.species_names, self.densities] = self.read_file_and_header(input_files[0])
        [self.rxn_names, self.rxn_rates] = self.read_file_and_header(input_files[1])  
        [self.heating_labels, self.heating_rates] = self.read_file_and_header(input_files[2])          
            
        # find Te, Tg
        self.nonspec = []
        for s,spec in enumerate(self.species_names):
            if spec == 'time':
                self.it = s
                self.nonspec.append(s)
            if spec == 'e-':
                self.ie = s 
            if spec == 'Te':
                self.ite = s
                self.nonspec.append(s)
            if spec == 'Tg':
                self.itg = s 
                self.nonspec.append(s)
            if spec == 'photon_flux':
                self.iflux = s
                self.nonspec.append(s)
            if spec == 'volume':
                self.ivol = s
                self.nonspec.append(s)
            if spec == 'emom':
                self.iemom = s
                self.nonspec.append(s)
                
    def read_file_and_header(self, file):
        of = open(file,"r")
        line = of.readlines()[0]
        header = line.split(',')
        array = np.genfromtxt(file,skip_header = 1, delimiter = ",")
        # remove '/n'
        header.pop()
        array = np.delete(array, -1, 1)
        array = np.transpose(array)
        print(header)
        return [header, array]
        
def aml_fig_settings(scale,font):
    plt.rcParams['axes.linewidth'] = 1*scale
    plt.rcParams['xtick.major.size'] = 4*scale
    plt.rcParams['xtick.minor.size'] = 2*scale
    plt.rcParams['ytick.major.size'] = 4*scale
    plt.rcParams['ytick.minor.size'] = 2*scale
    plt.rcParams['xtick.major.width'] = 1*scale
    plt.rcParams['xtick.minor.width'] = 1*scale
    plt.rcParams['ytick.major.width'] = 1*scale
    plt.rcParams['ytick.minor.width'] = 1*scale
    plt.rcParams['lines.linewidth'] = 1*scale
    plt.rcParams['font.size'] = 8*scale*font
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['figure.figsize'] = [3.15*scale,3.15*scale]
    plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['axes.labelsize'] = 10*scale*font
    plt.rcParams["legend.frameon"] = False

def plot_densities(data,sim_name):
    
    # densities
    fig1, ax1 = plt.subplots()
    labels=[]
    for s,spec in enumerate(data.species_names):
        if s not in data.nonspec:
            labels.append(spec)
            if spec == 'e-':
                ax1.plot(data.densities[data.it],data.densities[s],color='purple',dashes=[1,1])  
            else:                
                ax1.plot(data.densities[data.it],data.densities[s])                  
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Density (m$\mathregular{^{-3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(sim_name+'_densities',dpi=600)  
    
    plt.legend(labels,'upper left',fontsize=4)
    plt.savefig(sim_name+'_densities_legend',dpi=600) 

    # volume
    fig1, ax1 = plt.subplots()
    labels=[]
    ax1.plot(data.densities[data.it],data.densities[data.ivol])
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Plasma Volume (m$\mathregular{^{3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(sim_name+'_volume',dpi=600)

    # inventory
    fig1, ax1 = plt.subplots()
    labels=[]
    for s,spec in enumerate(data.species_names):
        if s not in data.nonspec:
            labels.append(spec)
            nV = []
            for t in range(len(data.densities[data.it])):
                nV.append(data.densities[s][t]*data.densities[data.ivol][t])
            # if spec == 'Al':
                # nV = [x*0.1 for x in nV]
                # labels[-1] += 'x0.1'
            if spec == 'e-':
                ax1.plot(data.densities[data.it],nV,color='purple',dashes=[1,1])
            else:                
                ax1.plot(data.densities[data.it],nV)                  

    plt.yscale('log')
    plt.xscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Number of Particles')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(sim_name+'_inventories',dpi=600)  
    
    plt.legend(labels,'upper left',fontsize=4)
    plt.savefig(sim_name+'_inventories_legend',dpi=600) 
      
    # temperatures
    fig4, ax4 = plt.subplots()
    ax4.plot(data.densities[data.it],data.densities[data.ite])
    ax4.plot(data.densities[data.it],data.densities[data.itg])
    plt.legend(['T$_e$','T$_g$'])
    plt.ylabel('Temperature (K)')
    plt.xlabel('Time (s)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(sim_name+'_temperatures',dpi=600) 

   # temperatures + photon flux
    fig6, ax6 = plt.subplots()
    ax6.plot(data.densities[data.it],data.densities[data.ite])
    ax6.plot(data.densities[data.it],data.densities[data.itg])
    plt.ylabel('Temperature (K)')
    plt.xlabel('Time (s)')
    plt.xscale('log')
     
    ax7 = ax6.twinx()
    plot_flux = []
    for flux in data.densities[data.iflux]:
        plot_flux.append(flux/1e29)
    ax7.plot(data.densities[data.it],plot_flux,'r',dashes = [3,3])
    #plt.legend(['T$_e$','T$_g$','photon_flux'])
    plt.ylabel('Photon Flux (10$\mathregular{^{29}}$ m$\mathregular{^{-2}}$s$\mathregular{^{-1}}$)')
    plt.tight_layout()
    plt.savefig(sim_name+'_temperatures_ps',dpi=600) 
    
    # plasma frequency
    fig9, ax9 = plt.subplots()
    plasma_freq = []
    for t in range(len(data.densities[data.it])):
        plasma_freq.append((data.densities[data.ie][t]*(1.602e-19)**2/(9.11e-31*8.854e-12))**0.5/(2*3.14159))
    ax9.plot(data.densities[data.it],plasma_freq) 
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Plasma Frequency (1/s)')
    plt.xlabel('Time (s)')
    plt.tight_layout() 
    plt.savefig(sim_name+'_plasma_freq',dpi=600)
    
    # electron momentum
    fig1, ax1 = plt.subplots()
    labels=[]             
    ax1.plot(data.densities[data.it],data.densities[data.iemom])  
    plt.xscale('log')
    #plt.xlim(1e-8,1e-5)
    #plt.ylim(-1e20,0)
    plt.ylabel('Electron Momentum (1/m$\mathregular{^{2}}$s)')
    plt.xlabel('Time (s)')
    
    # electron drift velocity
    ax1_right = ax1.twinx()
    velocity = []
    for t in range(len(data.densities[data.it])):
        velocity.append(data.densities[data.iemom][t]/(data.densities[data.ie][t]))
    ax1_right.plot(data.densities[data.it],velocity,color='green') 
    #plt.yscale('log')
    #plt.ylim(-1e7,0)
    plt.xscale('log')
    plt.ylabel('Electron Drift Velocity (m/s)')
    plt.tight_layout()
    plt.savefig(sim_name+'_emom',dpi=600)  
    
def plot_rates(data,sim_name):

    ##visual
    myrect=[0,0,1,0.75]
    
    time_ns = [x/1e-9 for x in data.densities[data.it]]

    #split rates by maximum value
    sort_indices = np.argsort(np.amax(data.rxn_rates, axis=1))
    
    fig1, ax1 = plt.subplots()
    labels=[]
    # plot the largest rates first
    for i in range(len(sort_indices)-1,len(sort_indices)/2,-1):
        rxn = sort_indices[i]
        if rxn != 0:
            labels.append(data.rxn_names[rxn])
            ax1.plot(data.densities[data.it],data.rxn_rates[rxn])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Rate (m$\mathregular{^{-3}}$s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)
    # plt.savefig(sim_name+'_densities',dpi=600)

    fig2, ax2 = plt.subplots()
    labels=[]
    # plot the smaller rates first
    for i in range(len(sort_indices)/2,-1,-1):
        rxn = sort_indices[i]
        if rxn != 0:
            labels.append(data.rxn_names[rxn])
            ax2.plot(data.densities[data.it],data.rxn_rates[rxn])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Rate (m$\mathregular{^{-3}}$s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)  

    # photon processes
    fig3, ax3 = plt.subplots()
    labels=[]    
    found = False
    for r,rxn in enumerate(data.rxn_names):
        if 'hv' in rxn: 
            labels.append(data.rxn_names[r])
            ax3.plot(data.densities[data.it],data.rxn_rates[r])
            found = True
    if found:
        plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
        plt.yscale('log')
        plt.xscale('log')
        ylims = ax1.get_ylim()
        plt.ylim(max(ylims[0],1),ylims[1])
        plt.ylabel('Rate (m$\mathregular{^{-3}}$s$\mathregular{^{-1}}$)')
        plt.xlabel('Time (s)')
        plt.tight_layout(rect=myrect)      
        plt.savefig(sim_name+'_photon',dpi=600) 

    # photon rate*volume
    fig3, ax3 = plt.subplots()
    labels=[]    
    plotting_fac = 1
    for r,rxn in enumerate(data.rxn_names):
        if 'hv' in rxn: 
            kV = []
            [lhs,rhs] = data.rxn_names[r].split('->')
            for t in range(len(data.densities[data.it])):
                if lhs.count('hv') == 2:
                    kV.append((2*data.rxn_rates[r][t]*data.densities[data.ivol][t])/plotting_fac)
                else:
                    kV.append((data.rxn_rates[r][t]*data.densities[data.ivol][t])/plotting_fac)
            if max(kV) >= 2e14/plotting_fac:
                labels.append(data.rxn_names[r])
                ax3.plot(data.densities[data.it],kV)
        
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Photon Absorption Freq. (s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout()      
    plt.savefig(sim_name+'_photon_lin',dpi=600) 

    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1) 
    plt.tight_layout(rect=myrect)   
    plt.savefig(sim_name+'_photon_lin_legend',dpi=600)   
    
    # ionization
    fig4, ax4 = plt.subplots()
    labels=[]    
    for r in range(1,len(data.rxn_names)):
        [lhs,rhs] = data.rxn_names[r].split('->')
        if rhs.count('e-') > lhs.count('e-'): 
            labels.append(data.rxn_names[r])
            ax4.plot(data.densities[data.it],data.rxn_rates[r])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax4.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Ionization Rate (m$\mathregular{^{-3}}$s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)  
    plt.savefig(sim_name+'_ionization_rate',dpi=600)  

    # ionization rate * volume
    fig5, ax5 = plt.subplots()
    labels=[]    
    for r in range(1,len(data.rxn_names)):
        [lhs,rhs] = data.rxn_names[r].split('->')
        if rhs.count('e-') > lhs.count('e-'): 
            labels.append(data.rxn_names[r])
            kV = []
            for t in range(len(data.densities[data.it])):
                kV.append(data.rxn_rates[r][t]*data.densities[data.ivol][t])
            ax5.plot(data.densities[data.it],kV)
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.xscale('log')
    plt.ylabel('Ionization Frequency (s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)  
    plt.savefig(sim_name+'_ionization_lin',dpi=600)

    # ionization rate * volume, sorted
    # plot reaction with largest value of kV first
    kV_max=[0]*len(data.rxn_names)
    for r in range(1,len(data.rxn_names)):
        [lhs,rhs] = data.rxn_names[r].split('->')
        if rhs.count('e-') > lhs.count('e-'):
            kV_max[r] = np.amax(data.rxn_rates[r]*data.densities[data.ivol])
    print('kV_max =', kV_max)
    sort_indices = np.argsort(kV_max)
    print('sort_indicies', sort_indices)

    fig1, ax1 = plt.subplots()
    labels=[]
    # plot the largest rates first
    for i in range(len(sort_indices)-1,len(sort_indices)-8,-1):
        rxn = sort_indices[i]
        if rxn != 0:
            labels.append(data.rxn_names[rxn])
            kV = []
            for t in range(len(data.densities[data.it])):
                kV.append(data.rxn_rates[rxn][t]*data.densities[data.ivol][t])
            ax1.plot(data.densities[data.it],kV)
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Ionization Frequency (s$\mathregular{^{-1}}$)')
    plt.xlabel('Time')
    plt.tight_layout(rect=myrect)
    plt.savefig(sim_name+'_iz_sorted',dpi=600)
    
    # recombination rate * volume
    fig7, ax7 = plt.subplots()
    plotting_fac = 1e0
    labels=[]    
    for r in range(1,len(data.rxn_names)):
        [lhs,rhs] = data.rxn_names[r].split('->')
        if rhs.count('e-') < lhs.count('e-'): 
            kV = []
            for t in range(len(data.densities[data.it])):
                kV.append((data.rxn_rates[r][t]*data.densities[data.ivol][t])/plotting_fac)
            #if max(kV) >= 5e13/plotting_fac:
            labels.append(data.rxn_names[r])
            ax7.plot(data.densities[data.it],kV)
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax1.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Recombination Rate (s$\mathregular{^{-1}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout()  
    plt.savefig(sim_name+'_recombination',dpi=600)
   
    plt.legend(labels,fontsize=4, loc='lower right', ncol=1)
    plt.savefig(sim_name+'_recombination_legend',dpi=600)

def plot_heating(data,sim_name):   

    ##visual
    myrect=[0,0,1,0.8]

    #sort rates by maximum value
    sort_indices = np.argsort(np.amax(data.heating_rates, axis=1))

    # plot electron heating
    fig1, ax1 = plt.subplots()
    labels=[]    
    
    # plot the largest rates first
    #for i in range(len(sort_indices)-1,-1,-1):
    for i in range(len(sort_indices)-1,len(sort_indices)-15,-1):
        r = sort_indices[i]
        if '_Te' in data.heating_labels[r] or 'ohmic' in data.heating_labels[r]:
            if max(data.heating_rates[r])>0:
                labels.append(data.heating_labels[r][:-3])
                ax1.plot(data.densities[data.it],data.heating_rates[r])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    # plt.xlim(1.4e-8,2.1e-8)
    plt.ylim(1e2,1e20)
    plt.ylabel('Electron Heating Rate (W/m$\mathregular{^{3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)   
    
    plt.savefig(sim_name+'_e_heating',dpi=600)

    # plot ohmic heating
    fig1, ax1 = plt.subplots()
    labels=[]    
    
    # plot the largest rates first
    #for i in range(len(sort_indices)-1,-1,-1):
    for r in range(len(data.heating_rates)):
        if 'ohmic' in data.heating_labels[r]:
            print('found')        
            labels.append(data.heating_labels[r])
            ax1.plot(data.densities[data.it],data.heating_rates[r])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Electron Heating Rate (W/m$\mathregular{^{3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect)   
    
    plt.savefig(sim_name+'_e_heating_ohmic',dpi=600)

    # plot gas heating
    fig2, ax2 = plt.subplots()
    labels=[]    

    # plot the largest rates first
    #for i in range(len(sort_indices)-1,-1,-1):
    for i in range(len(sort_indices)-1,len(sort_indices)-20,-1):
        r = sort_indices[i]
        if '_Tg' in data.heating_labels[r] and max(data.heating_rates[r])>0:
            labels.append(data.heating_labels[r][:-3])
            ax2.plot(data.densities[data.it],data.heating_rates[r])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    plt.yscale('log')
    plt.xscale('log')
    ylims = ax2.get_ylim()
    plt.ylim(max(ylims[0],1),ylims[1])
    plt.ylabel('Gas Heating Rate (W/m$\mathregular{^{3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect) 

    plt.savefig(sim_name+'_gas_heating',dpi=600)    
    
    # plot electron cooling
    sort_indices = np.argsort(np.amin(data.heating_rates, axis=1))
    
    fig3, ax3 = plt.subplots()
    labels=[]  
    # plot the most negative rates first
    #for i in range(0,len(sort_indices),1):
    for i in range(0,10,1):
        r = sort_indices[i]
        if '_Te' in data.heating_labels[r] and min(data.heating_rates[r])<0:
            labels.append(data.heating_labels[r][:-3])
            ax3.plot(data.densities[data.it],data.heating_rates[r])
    plt.legend(labels,fontsize=4,bbox_to_anchor=(1, 1), loc='lower right', ncol=1)
    #plt.yscale('log')
    plt.xscale('log')
    #plt.ylim(1e10,1e16)
    plt.ylabel('Electron Energy Losses (W/m$\mathregular{^{3}}$)')
    plt.xlabel('Time (s)')
    plt.tight_layout(rect=myrect) 

    plt.savefig(sim_name+'_e_cooling',dpi=600)      
    
    #plt.show()
    
if __name__ == '__main__':

    sim_name = sys.argv[1]
    input_files = [sim_name + '_densities.out',sim_name + '_rates.out', sim_name + '_heating.out']
    divide = sys.argv[1].rfind('/')+1
    plot_name = sys.argv[1][:divide] + 'plot/' + sys.argv[1][divide:]
    print(plot_name)
    data = Dataset(input_files)
    
    aml_fig_settings(1,1.3)
    plot_densities(data,plot_name)      
    plot_rates(data,plot_name)
    plot_heating(data,plot_name)
    
