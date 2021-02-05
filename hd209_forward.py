# -*- coding: utf-8 -*-
"""
Created on Tuesday March 24 2020
@author: dario

Forward-modelling for hot jupiter HD209458b
Synthethic data is produced using ArielRad noise estimates
The output is the .txt datafile and plots
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from astropy.io import ascii
import numpy as np
import taurex.log
taurex.log.disableLogging()

#loading cross sections
from taurex.cache import OpacityCache,CIACache
OpacityCache().clear_cache()
#cross sections at R=7000 in the 0.3 to 15 micrometer range
OpacityCache().set_opacity_path("C:\\Users\dario\Desktop\TFG\Opacities")
CIACache().set_cia_path("C:\\Users\dario\Desktop\TFG\CIA")

from taurex.planet import Planet
from taurex.temperature import Isothermal
from taurex.pressure import SimplePressureProfile
from taurex.stellar import Star
from taurex.chemistry import TaurexChemistry,ConstantGas
from taurex.model import TransmissionModel
from taurex.contributions import AbsorptionContribution,CIAContribution
from taurex.contributions import RayleighContribution, SimpleCloudsContribution
from spectres import spectres

# LOAD OPACITIES
## h2o and co as minor gases
h2o_xsec = OpacityCache()['H2O']
ch4_xsec = OpacityCache()['CH4']
co_xsec = OpacityCache()['CO']
co2_xsec = OpacityCache()['CO2']
nh3_xsec = OpacityCache()['NH3']

###################### STELLAR PROPERTIES #####################
Ts = 6086 # K
rs = 1.17967 # R_sun
d = 47.45667 # pc
magk = 6.308
ms = 1.1753 # M_sun
star = Star(temperature=Ts, radius=rs, distance= d, 
            magnitudeK= magk, mass=ms)

###################### PLANET PROPERTIES #####################
## conversion to jupiter units
mp = 0.7113066929391265 #MJ or mp= 226.0734539*(1/317.82838) 
rp = 1.3541859096502 # Rj; rp= 15.17975462*(0.08921) 
a = 0.04783850140054166 # in AU or a=7156537947/1.495978707e11 # from meters to AU
b = 0.5 # b= a*cos(i)
P = 3.52473906 # days
alb = 0.038
T14 = 9624.538897 # s
planet = Planet(planet_mass=mp, planet_radius=rp,
                 planet_distance=a,
                 impact_param=b, orbital_period=P, albedo=alb,
                 transit_time=T14)

isothermal = Isothermal(T=1487) 
pressure_profile = SimplePressureProfile(nlayers = 20, atm_min_pressure = 1e-4,
                                         atm_max_pressure = 1e7)

###################### ATMOSPHERIC COMPOSITION ##################
#major gases
h2 = 0.9
he = 0.1
n2_to_h2 = 5.75e-5/h2
chemistry = TaurexChemistry(fill_gases=['H2','He','N2'],ratio=[he/h2,n2_to_h2])

#minor gases (in ppm) 
############################# SOLAR values ################3
h2o = 4.27e-4
#ch4 = 1e-9
co = 4.27e-4
co2 = 1.26e-7
#nh3 = 3.16e-10

chemistry.addGas(ConstantGas('H2O',mix_ratio=h2o))
#chemistry.addGas(ConstantGas('CH4',mix_ratio=ch4))
chemistry.addGas(ConstantGas('CO',mix_ratio=co))
chemistry.addGas(ConstantGas('CO2',mix_ratio=co2))
#chemistry.addGas(ConstantGas('NH3',mix_ratio=nh3))

# BUILD THE MODEL
tm = TransmissionModel(planet=planet,
                       temperature_profile=isothermal,
                       chemistry=chemistry,
                       star=star,
                       pressure_profile=pressure_profile)
# Contributions
tm.add_contribution(AbsorptionContribution())
tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
tm.add_contribution(RayleighContribution())

# TOP OF DECK CLOUD PRESSURE
# no clouds
p_cloud = 1e6
tm.add_contribution(SimpleCloudsContribution(clouds_pressure = p_cloud))

tm.build()
res = tm.model()




tm.build()
res = tm.model()

# =============================================================================
#                           ADD NOISE FROM ARIELRAD
# =============================================================================
    
data = ascii.read("0000_HD 209458 b.csv")
wlgrid0, noise0= (np.array([]) for k in range(2))
for i in range(len(data)):
    wlgrid0 = np.append(wlgrid0,data[i][1])
    noise0 = np.append(noise0, data[i][6]) # noise on transit floor

###################### NUMBER OF TRANSITS #######################
num_transits = 1
noise0 = noise0/np.sqrt(num_transits)  
#################################################################  
from taurex.binning import FluxBinner
wngrid3 = 10000/wlgrid0    
wngrid3 = np.sort(wngrid3)
bn = FluxBinner(wngrid=wngrid3)
bin_wn, bin_rprs3,_,_  = bn.bin_model(tm.model(wngrid=wngrid3))
x = np.flip(10000/bin_wn)
y = np.flip(bin_rprs3)
   
#%%
# =============================================================================
#                    BINDOWN FROM TIER3 TO TIER2
# =============================================================================
# Load the spectral data, the first column is wavelength, second flux density and third flux density uncertainty

tier2file = np.loadtxt('tier2-grid.txt')
wl = x
wlbinned = tier2file[:,0]

spectrum = y
noise = noise0

# Call the spectres function to resample the input spectrum or spectra to the new wavelength grid
'''
end1 = 15
end2 = 82
# slice the wavelength grid in three for each instrument
wl_ch0 = wl[:end1]
wl_ch1 = wl[end1:end2]
wl_ch2 = wl[end2:]

wl_bin0 = wlbinned[:11]
wl_bin1 = wlbinned[11:47]
wl_bin2 = wlbinned[47:]

tier2flux0, tier2noise0 = spectres(wl_bin0, wl_ch0, spectrum[:end1],spec_errs=noise[:end1], verbose=False)
tier2flux1, tier2noise1 = spectres(wl_bin1, wl_ch1, spectrum[end1:end2],spec_errs=noise[end1:end2], verbose=False)
tier2flux2, tier2noise2 = spectres(wl_bin2, wl_ch2, spectrum[end2:],spec_errs=noise[end2:], verbose=False)
tier2flux, tier2noise = sum(tier2flux0,tier2flux1,tier2flux2), sum(tier2noise0,tier2noise1,tier2noise2)
'''


tier2flux, tier2noise = spectres(wlbinned, wl, spectrum,spec_errs=noise, verbose=False)    
tier2grid = wlbinned       

tier2flux = np.delete(tier2flux, [10,43])
tier2grid = np.delete(tier2grid, [10,43])
tier2noise = np.delete(tier2noise, [10,43])
#%%           


# WRITE TIER3 SPECTRA IN FILES
filename = 'hd209_tier3.txt'
f = open(filename, 'w')
f.write('# Lambda (um)\t  (Rp/Rs)^2\t Error (noise) \n') 
f.write('#--------------------------------------------- \n')   

for i,j,k in zip(x, y, noise0):
    f.write('%8.4e \t %8.4e \t %8.4e \n' % (i,j,k)) # lambda, (rp/rs)^2, noise, bin-width
f.close()

# WRITE TIER2 SPECTRA IN FILES
filename = 'hd209_tier2.txt'
f = open(filename, 'w')
f.write('# Lambda (um)\t  (Rp/Rs)^2\t Error (noise) \n') 
f.write('#--------------------------------------------- \n')   

for i,j,k in zip(tier2grid, tier2flux, tier2noise):
    f.write('%8.4e \t %8.4e \t %8.4e \n' % (i,j,k)) # lambda, (rp/rs)^2, noise, bin-width
f.close()
  
######## Multiply by 100 to show in %
y = y*100
tier2flux = tier2flux*100
noise0 = noise0*100
tier2noise = tier2noise*100
#%% PLOT USING SUBPLOTS for nicer output

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([0.0, 0.5, 0.8, 0.4], xticklabels=[], xscale='log', )
ax2 = fig.add_axes([0.0, 0.1, 0.8, 0.4], xscale='log')
#ax3 = fig.add_axes([0.4, 0.5, 0.4, 0.4], xticklabels=[], yticklabels=[],xscale='log')

matplotlib.rcParams['text.usetex'] = False

color1 = '#2e4a34'
ax1.plot(x,y, label='Tier-3', color=color1,alpha=1)
ax1.fill_between(x,y-noise0,y+noise0, color=color1,alpha=0.2)

ax2.plot(tier2grid,tier2flux, label='Tier-2', c='red',alpha=1)
ax2.fill_between(tier2grid,tier2flux-tier2noise,tier2flux+tier2noise,alpha=0.2, color='red')


xlocator = [0.5, 1,2,3,4,5,6,7,8]
xlabels = ['0.5','1','2','3','4','5','6','7','8']
ax2.set_xticks(ticks=xlocator)
ax2.set_xticklabels(xlabels)
lim = [0.54, 7.75]
ax1.set_xlim(lim[0],lim[1])
ax2.set_xlim(lim[0],lim[1])
ax2.set_xlabel('$\lambda (\mu m)$', fontsize=14)
ax1.set_ylabel('Transit Depth (%)', fontsize=14), ax2.set_ylabel('Transit Depth (%)', fontsize=14)
ax1.set_title('HD209458b', fontsize=16)

ylocator = [1.450, 1.500, 1.550]
ax2.set_yticks(ticks=ylocator)

loc_legend =  (0.05,0.8)
font_legend = 14
ax1.legend(frameon=False, loc=loc_legend, fontsize = font_legend)
ax2.legend(frameon=False, loc= loc_legend, fontsize = font_legend)

ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)

#, ax2.legend(frameon=False, loc='upper left')
#ax1.set_title('K2-18b ({} transits)'.format(num_transits))
#ax1.text('Tier-3',bbox_transform=ax1.transAxes, bbox_to_anchor=((0.9,0.7)))
#ax2.text('Tier-2',bbox_transform=ax2.transAxes, bbox_to_anchor=((0.9,0.7)))


plt.savefig('hd209_tiers_sub.png',dpi=300, bbox_inches='tight')
plt.show()

