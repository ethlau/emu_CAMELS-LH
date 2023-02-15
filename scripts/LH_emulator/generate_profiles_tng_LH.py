import numpy as np
import cgm_toolkit.cgm_profiles as cgm_prof
import h5py
import math

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

def extract(simulation, snap, suite='IllustrisTNG', prof_dir='./Profiles/', radial_range=[0, 20.0]):

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
    #profile_file = prof_dir+'/'+simulation+'/'+suite+'_'+simulation+'_'+snap+'.hdf5'
    profile_file = prof_dir+'/'+suite+'/'+simulation+'/'+suite+'_'+simulation+'_'+snap+'.hdf5'

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
        xsb[i,:][ xsb[i,:] == 0 ] = 1e-30 

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

#-----------------------------------input section
suite='IllustrisTNG'
#suite='SIMBA'
emulator_type='general'  #CMASS, general
#for CMASS emulator use mopc_profiles.py. Gives correct radial range. Can still use this script to make plots though.

start_sim, end_sim = 0,999
#start_sim, end_sim = 0,0
#nums=np.linspace(22,65,44,dtype='int') #0,65,66 for all. 22,65,44
nums=np.linspace(start_sim,end_sim,end_sim-start_sim+1,dtype='int') #0,65,66 for all. 22,65,44
#simulations=['1P_'+str(n) for n in nums]
simulations=['LH_'+str(n) for n in nums]

#snap_arr=['033','032','031','030','029','028','027','026','025','024']
#mass_str_arr=['11-11.5','11.5-12','12-12.3','12.3-13.1','13.1-14.0']
#mh_low_arr=[10**11.,10**11.5+0.1,10**12.+0.1,10**12.3+0.1, 10**(13.1)+0.1]
#mh_high_arr=[10**11.5,10**12.,10**12.3,10**13.1,10**14]
#mh_low_pow_arr=[11,11.5,12,12.3,13.1]
#mh_high_pow_arr=[11.5,12,12.3,13.1,14.0]

snap_arr=['033','032','031']
mass_str_arr=['12.0-12.2','12.2-12.4','12.4-12.6','12.6-12.8','12.8-13.0']
mh_low_arr =[10**(12.0),10**(12.2), 10**(12.4), 10**(12.6), 10**(12.8)]
mh_high_arr=[10**(12.2),10**(12.4), 10**(12.6), 10**(12.8), 10**(13.0)]
mh_low_pow_arr =[12.0,12.2,12.4,12.6,12.8]
mh_high_pow_arr=[12.2,12.4,12.6,12.8,13.0]

mean_masses_uw={}
mean_masses_w={}
median_masses={}

def get_errors(arr):
    arr=np.array(arr,dtype='float')
    #percent_84=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],84),0,arr)
    #percent_50=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],50),0,arr)
    #percent_16=np.apply_along_axis(lambda v: np.percentile(v[np.nonzero(v)],16),0,arr)
    percent_84=np.percentile(arr, 84, axis=0)
    percent_50=np.percentile(arr, 50, axis=0)
    percent_16=np.percentile(arr, 16, axis=0)
    errup=percent_84-percent_50
    errlow=percent_50-percent_16

    std_arr=[]
    for i in range(arr.shape[1]): #for every radial bin
        std_arr.append(np.std(np.apply_along_axis(lambda v: np.log10(v[np.nonzero(v)]),0,arr[:,i])))
    std=np.array(std_arr,dtype='float')
    return errup,errlow,std



for j in np.arange(len(simulations)):
    sim=simulations[j]

    for k in np.arange(len(snap_arr)):
        snap=snap_arr[k]

        z, r, nprofs, m200c, r200c, profiles = extract(sim,snap,suite=suite)

        #z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw=extract(sim,snap)
        #omegab=0.049
        #h=0.6711
        #omegam,sigma8=np.loadtxt('/home/jovyan/Simulations/'+suite+'/'+simulations[j]+'/CosmoAstro_params.txt',usecols=(1,2),unpack=True)
        #omegalam=1.0-omegam
        #rhocrit=2.775e2
        #rhocrit_z=rhocrit*(omegam*(1+z)**3+omegalam)
            
        #mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=profile_functions.correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
        for m in np.arange(len(mh_low_arr)):
            mh_low=mh_low_arr[m]
            mh_high=mh_high_arr[m]
            mass_str=mass_str_arr[m]
            mh_low_pow=mh_low_pow_arr[m]
            mh_high_pow=mh_high_pow_arr[m]
            print(sim,snap,mass_str)

            mask = (m200c > mh_low ) & (m200c < mh_high) 

            #mstarm,mhm,rhm,sfrm,GroupFirstSubm,val_presm,val_densm,nprofsm,val_metal_gmwm,val_temp_gmwm=profile_functions.mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)

            r_mpc=r/1.e3
            
            #outer cut 20, inner cut 2e-3 for TNG, SIM can do 5e-4
            #r_mpc_cut,val_densm=profile_functions.outer_cut_multi(20,r_mpc,val_densm)
            #r_mpc_cut2,val_densm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_densm)
            #r_mpc_cut,val_presm=profile_functions.outer_cut_multi(20,r_mpc,val_presm)
            #r_mpc_cut2,val_presm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_presm)
            #r_mpc_cut,val_metal_gmwm=profile_functions.outer_cut_multi(20,r_mpc,val_metal_gmwm)
            #r_mpc_cut2,val_metal_gmwm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_metal_gmwm)
            #r_mpc_cut,val_temp_gmwm=profile_functions.outer_cut_multi(20,r_mpc,val_temp_gmwm)
            #r_mpc_cut2,val_temp_gmwm=profile_functions.inner_cut_multi(2.e-3,r_mpc_cut,val_temp_gmwm)
                        
            #mean_unnorm_densm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_densm)
            #mean_unnorm_presm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_presm)
            #median_unnorm_densm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_densm)
            #median_unnorm_presm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_presm)
            #mean_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_metal_gmwm)
            #mean_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_temp_gmwm)
            #median_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_gmwm)
            #median_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_gmwm)
   
            xsb = profiles['xsb']
            hot_density = profiles['hot_density']
            hot_temperature = profiles['hot_temperature']
            hot_metallicity = profiles['hot_metallicity']

            nprofsm = hot_density[mask].shape[0]
            #print(nprofsm)

            mhm = m200c[mask]

            mean_masses_uw[sim]=np.mean(mhm)
            median_masses[sim]=np.median(mhm)
 
            if (nprofsm > 0) :

                mean_xsb = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,xsb[mask])
                median_xsb = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,xsb[mask])
                mean_hot_density = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,hot_density[mask])
                median_hot_density = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,hot_density[mask])
                mean_hot_temperature = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,hot_temperature[mask])
                median_hot_temperature = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,hot_temperature[mask])
                mean_hot_metallicity = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,hot_metallicity[mask])
                median_hot_metallicity = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,hot_metallicity[mask])


                errup_xsb,errlow_xsb,std_xsb = get_errors(xsb[mask])
                errup_hot_density,errlow_hot_density,std_hot_density = get_errors(hot_density[mask])
                errup_hot_temperature,errlow_hot_temperature,std_hot_temperature = get_errors(hot_temperature[mask])
                errup_hot_metallicity,errlow_hot_metallicity,std_hot_metallicity = get_errors(hot_metallicity[mask])

                #errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm=profile_functions.get_errors(val_densm)
                #errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm=profile_functions.get_errors(val_presm)
                #errup_metal_gmw_unnormm,errlow_metal_gmw_unnormm,std_metal_gmw_unnormm=profile_functions.get_errors(val_metal_gmwm)
                #errup_temp_gmw_unnormm,errlow_temp_gmw_unnormm,std_temp_gmw_unnormm=profile_functions.get_errors(val_temp_gmwm)
            else :

                mean_xsb = 1e-30*r_mpc
                median_xsb = 1e-30*r_mpc
                errup_xsb = 1e-30*r_mpc
                errlow_xsb = 1e-30*r_mpc
                std_xsb = 1e-30*r_mpc
 
                mean_hot_density = 1e-30*r_mpc
                median_hot_density = 1e-30*r_mpc
                errup_hot_density = 1e-30*r_mpc
                errlow_hot_density = 1e-30*r_mpc
                std_hot_density = 1e-30*r_mpc
      
                mean_hot_temperature = 1e-30*r_mpc
                median_hot_temperature = 1e-30*r_mpc
                errup_hot_temperature = 1e-30*r_mpc
                errlow_hot_temperature = 1e-30*r_mpc
                std_hot_temperature = 1e-30*r_mpc

                mean_hot_metallicity = 1e-30*r_mpc
                median_hot_metallicity = 1e-30*r_mpc
                errup_hot_metallicity = 1e-30*r_mpc
                errlow_hot_metallicity = 1e-30*r_mpc
                std_hot_metallicity = 1e-30*r_mpc
 
            #header='R (Mpc), mean xsb , errup, errlow, std, median xsb\n nprofs %i, mean mh %f, median mh %f \n Mass range %.2f - %.1f'%(nprofsm,np.mean(mhm),np.median(mhm),mh_low_pow,mh_high_pow)
 
            header='R (Mpc), hot_dens, hot_temp, hot_metal, xsb (mean, errup, errlow, std, median) \n nprofs %i, mean mh %f, median mh %f \n Mass range %.2f - %.1f'%(nprofsm,np.mean(mhm),np.median(mhm),mh_low_pow,mh_high_pow)
            
            np.savetxt('./X_emulator_profiles/'+suite+'/'+suite+'_'+sim+'_'+snap+'_uw_%s.txt'%mass_str,np.c_[r_mpc, mean_hot_density, errup_hot_density, errlow_hot_density, std_hot_density,  median_hot_density, mean_hot_temperature, errup_hot_temperature, errlow_hot_temperature, std_hot_temperature,  median_hot_temperature, mean_hot_metallicity, errup_hot_metallicity, errlow_hot_metallicity, std_hot_metallicity,  median_hot_metallicity, mean_xsb, errup_xsb, errlow_xsb, std_xsb,  median_xsb ], header=header)

        
   

