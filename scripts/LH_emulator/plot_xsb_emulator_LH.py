import matplotlib.pyplot as plt
import numpy             as np
import h5py
import math
from astropy.cosmology import Planck18 as cosmo
import os

import cgm_toolkit.cgm_profiles as cgm_prof

plt.rcParams.update({'font.size': 14})
plt.rc('legend',fontsize=14)

mp = 1.67e-24
Mpc = 3.0856e24
kpc = Mpc/1000.0
erg_to_keV = 6.242e+8
Zsun = 0.0127
Msun = 1.989e33 #g cm^-3
kb = 1.38e-16 # erg/K
erg_to_keV = 6.242e+8
K_to_keV = kb * erg_to_keV
XH = 0.76 #primordial hydrogen fraction
mu = 0.58824; # X=0.76 assumed 
mu_e = mue = 2.0/(1.0+XH); # X=0.76 assumed

def return_xsb_emulator (profile_type='xsb_median', feedback_type='ASN1', home='./X_emulator_profiles/', suite='SIMBA', interpolation_type='linear'):
    '''
    Inputs:
        A: feedback strength (float)
        z: redshift (float)
        logM: halo mass in log10 Msun
    Return:
        r: radial bins in Mpc (numpy array)
        profile: emulator profile values in log10 cgs units (numpy array)
    '''
    import x_helper_functions_LH as fs
    mass=fs.mass
    mass_str=fs.mass_str
    snap=fs.snap
    redshifts=fs.choose_redshift(suite)
    #vary,sims=fs.choose_vary(feedback_type)
    #samples=fs.cartesian_prod(vary,redshifts,mass)
    #nsamp=samples.shape[0]
    samples,radius,y,emulator=fs.build_xsb_emulator(home,suite,feedback_type,profile_type,interpolation_type)
    return radius, emulator

def save_emulator(filename, radius, emulator):

    import pickle

    with open(filename, 'wb') as f:
        pickle.dump((radius, emulator), f) 
    
def load_emulator(filename) :

    import pickle
    with open(filename, 'rb') as f:
        radius, emulator = pickle.load(f) 

    return radius, emulator

def emulated_xsb_profile(emulator, A, z, logM):
    params=[[A, z, logM]] #the order here is important- A, then z, then logM
    profile = emulator(params).ravel()
    return profile

def extract(simulation, snap, suite='IllustrisTNG', prof_dir='./Profiles/', radial_range=[2.0e-3, 2.0]):

    '''
    Return values of the CGM profiles from the CAMELS simulation

    Inputs:
      simulation: string, name of the simulation, e.g., 1P_0, LH_123, CV_12
      snap: string, number of the snapshot, from '000' to '033', '033' being the last snapshot corresponding to z=0

    Outputs:
      z: float, redshift
      r: np array, radial bin on kpc
      val_dens: np array, density profile in g/cm^3
      val_pres: np array, volume-weighted thermal pressure profile in erg/cm^3
      val_temp_mw: np array, mass-weighted temperature in K
      val_metals_mw: np array, mass-weighted metallcity in Zsun
      mh: np array, halo mass (M200c) in Msun
      rh: np array, halo radius (R200c) in kpc

    '''

    #data_file= data_dir+'/'+suite+'/'+simulation+'/snap_'+snap+'.hdf5'
    #profile_file = prof_dir+'/'+suite+'/'+simulation+'/'+suite+'_'+simulation+'_'+snap+'.hdf5'
    profile_file = prof_dir+'/'+simulation+'/'+suite+'_'+simulation+'_'+snap+'.hdf5'

    #b=h5py.File(data_file,'r')
    stacks=h5py.File(profile_file,'r')
    z=stacks['header'].attrs[u'Redshift']
    h=stacks['header'].attrs[u'HubbleParam']
    omegab=stacks['header'].attrs[u'OmegaBaryon']
    omegam=stacks['header'].attrs[u'Omega0']
    omegalam=stacks['header'].attrs[u'OmegaLambda']
    comoving_factor = (1.+z)
    
    density_conversion_factor = Msun*kpc**(-3) * 1e10 * h**2 * comoving_factor**3

    #convert pressure from 1e10Msol/h*(km/s)**2 ckpc^{-3} to keV cm^{-3}
    pressure_conversion_factor = density_conversion_factor * 1e10 * erg_to_keV
    temperature_conversion_factor = (1e5)**2 * kb * erg_to_keV

    val            = stacks['GasProfiles']
    val_gasdens    = np.array(val[0,:,:]) * density_conversion_factor/(mu*mp)  #density in cm^-3
    val_pres       = np.array(val[1,:,:]) * pressure_conversion_factor  #thermal pressure in keV cm^-3
    val_temp_mw    = np.array(val[2,:,:]) * temperature_conversion_factor #mass-weighted temperature in keV
    val_metals_mw  = np.array(val[3,:,:])/Zsun #mass-weighted metallicity in solar units
    val_gasdens2   = np.array(val[4,:,:]) * density_conversion_factor**2/ (mu*mp)**2   #density in cm^-6

    hot_val            = stacks['HotGasProfiles']
    hot_val_gasdens    = np.array(hot_val[0,:,:]) * density_conversion_factor/(mu*mp) #density in cm^-3
    hot_val_pres       = np.array(hot_val[1,:,:]) * pressure_conversion_factor  #thermal pressure in keV cm^-3
    hot_val_temp_mw    = np.array(hot_val[2,:,:]) * temperature_conversion_factor #mass-weighted temperature in keV
    hot_val_metals_mw  = np.array(hot_val[3,:,:])/Zsun #mass-weighted metallicity in solar units
    hot_val_gasdens2   = np.array(hot_val[4,:,:]) * density_conversion_factor**2/ (mu*mp)**2  #density in cm^-6
    
    dm_val         = stacks['DmProfiles']
    val_dmdens     = np.array(dm_val[0,:,:]) * density_conversion_factor #density in g cm^3
    star_val       = stacks['StarProfiles']
    val_stardens   = np.array(star_val[0,:,:]) * density_conversion_factor #density in g cm^3
 
    bins           = np.array(stacks['nbins']) #number of radial bins
    r              = np.array(stacks['r']) / h / comoving_factor #radial bins in comoving kpc
    nprofs         = np.array(stacks['nprofs']) #number of halos
    m200c          = np.array(stacks['Group_M_Crit200'])*1e10 / h #M200c in Msol
    r200c          = np.array(stacks['Group_R_Crit200']) / h / comoving_factor #R200c in kpc

    xsb = np.zeros([ len(m200c), len(r)])
    for i, m in enumerate(m200c):
        halo_prof = cgm_prof.HaloProfile(m, z, r, hot_val_pres[i,:], hot_val_gasdens[i,:], hot_val_metals_mw[i,:], temperature = hot_val_temp_mw[i,:])
        xsb[i,:] = halo_prof.projected_xray_surface_brightness_profile (r, etable='./cgm_toolkit/data/etable_05_2keV.hdf5')*kpc*kpc*2*4.0*math.pi

    profiles = {}
    profiles['pressure'] = val_pres
    profiles['temperature'] = val_temp_mw
    profiles['density'] = val_gasdens
    profiles['entropy'] = val_temp_mw / (val_gasdens)**(2./3.)
    profiles['metallicity'] = val_metals_mw
 
    profiles['hot_pressure'] = hot_val_pres
    profiles['hot_temperature'] = hot_val_temp_mw
    profiles['hot_density'] = hot_val_gasdens
    profiles['hot_entropy'] = hot_val_temp_mw / (hot_val_gasdens)**(2./3.)
    profiles['hot_metallicity'] = hot_val_metals_mw

    profiles['dm_density'] = val_dmdens
    profiles['star_density'] = val_stardens
    
    profiles['entropy'][profiles['entropy'] != profiles['entropy']] = 0.0
    profiles['hot_entropy'][profiles['hot_entropy'] != profiles['hot_entropy']] = 0.0

    profiles['xsb'] = xsb

    rcut_profiles = {}
    rmask = [ (r >= radial_range[0]*1000.0) & (r <= radial_range[1]*1000.0) ] 

    for k in profiles.keys():
        rcut_profiles[k] = np.zeros( [len(m200c), len(r[rmask])] )
    for i, m in enumerate(m200c):
        for k in profiles.keys():
            rcut_profiles[k][i,:] = profiles[k][i,:][rmask]
 
    r = r[rmask]

    return z, r, nprofs,  m200c, r200c, rcut_profiles

home='./X_emulator_profiles/' #point to your profiles
#suite='IllustrisTNG' #SIMBA or IllustrisTNG
suite='SIMBA' #SIMBA or IllustrisTNG
interpolation_type='linear' #this is the Rbf interpolation function


#---emulated profiles

feedback_type_list = ['ASN1', 'ASN2', 'AAGN1', 'AAGN2']
A_list = [0.5, 1.0, 2.0]
A = 1.0
xray_redshift = 0.04896

rerun_building_emulator = False

feedback_type = feedback_type_list[0]
emulator_file =suite+'_LH_xsb_emulator_'+feedback_type+'.pkl' 

if (not os.path.exists(emulator_file)) or rerun_building_emulator :

    radius_Mpc, x_emulator = return_xsb_emulator(feedback_type=feedback_type, profile_type='xsb_median', home=home, suite=suite)
    save_emulator(emulator_file , radius_Mpc, x_emulator)

else :
    radius_Mpc, x_emulator = load_emulator(emulator_file)

radius_kpc = radius_Mpc * 1000.0
#---simulated profiles

snap='030'
sim ='LH_0'

prof_dir = './Profiles/'+suite+'/'
z, r_sim, nprofs_sim, m200c_sim, r200c_sim, profiles_sim = extract(sim,snap,suite=suite,prof_dir=prof_dir)

xsb_sim = profiles_sim['xsb']

xsb_emu = np.zeros([len(m200c_sim), len(radius_kpc)])

for i, M200c in enumerate(m200c_sim):
    xsb_emu[i,:] = 10**emulated_xsb_profile(x_emulator, A, xray_redshift, np.log10(M200c) )

Mrange_list = [ (12.0, 12.5), (12.5, 13.0), (13.0, 13.5)]

for i, Mrange in enumerate(Mrange_list):
    
    print(Mrange)
    
    mask = ( np.log10(m200c_sim) < Mrange[1]) & ( np.log10(m200c_sim) > Mrange[0])
    
    fig,ax = plt.subplots(figsize=(5,5))
    
    ax.plot(radius_kpc, np.median((xsb_emu[mask]), axis=0), color='C1',label=r'emulator')
    ax.fill_between(radius_kpc, np.percentile((xsb_emu[mask]), 25, axis=0),np.percentile((xsb_emu[mask]), 75, axis=0), alpha=0.3, color='C1')

    ax.plot(r_sim, np.median((xsb_sim[mask]), axis=0), color='C3',label=r'simulation')
    ax.fill_between(r_sim, np.percentile((xsb_sim[mask]), 25, axis=0),np.percentile((xsb_sim[mask]), 75, axis=0), alpha=0.3, color='C3')
                                      
    ax.legend()
    ax.set_ylim(1e29,1e38)
    ax.set_xlim(10, 1000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_title(r'$\log_{10}M_{200c}/M_\odot \in ['+str(Mrange[0])+','+str(Mrange[1])+')$')

    ax.set_xlabel(r'Projected Radius $[{\rm kpc}]$')

    ax.set_ylabel(r'XSB Profile $[{\rm erg\,s^{-1}\,kpc^{-2}}]$')

    ax.legend()
    
    fig.savefig(suite+'_xsb_emulator_test_LH_'+str(Mrange[0])+'_'+str(Mrange[1])+'.png',facecolor='white', transparent=False)
    #fig.tight_layout()
    plt.close(fig)

