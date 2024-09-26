# ------------------------------------------------
# Name: calc_roll_factor.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 12 September 2024
# Purpose: Calculate roll factor on lidar scans
# ------------------------------------------------

import numpy as np
import xarray as xr
import gzip
import os
import xrft
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from functions import *

# --------------------------
# settings

# date/time info of lidar scans
des_year = "2023"
des_mon = "10"
des_day = "07"
des_sys = "WG100-L0AD00004JP"
lidar_loc = "Hampton, VA" # Hampton, VA or Oklahoma Mobile
# directory storing 24 hours of lidar data
# directory = f"/data/arrcwx/robbyfrost/lidar_obs/{des_sys}/{des_year}/{des_mon}/{des_day}/"
directory = "/data/arrcwx/robbyfrost/lidar_obs/test_data/"
# directory for saving figues
# figdir = f"/home/robbyfrost/analysis/TurbTor_Lidar/figures/{des_sys}/{des_year}/{des_mon}/{des_day}/"
figdir = f"/home/robbyfrost/analysis/TurbTor_Lidar/figures/test_data/"
os.makedirs(figdir, exist_ok=True)

# flag to filter vort_z field
filter_vort = True

# desired elevation angle
des_elev = 0.
# offset in meters where data begins
range_offset = 1425

# --------------------------
# read in lidar files

# Initialize an empty list to store the datasets
lall = []
# Iterate through all files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith("PPI_1.nc.gz"):
        file_path = os.path.join(directory, filename)
        # Open each .nc.gz file and load it as an xarray Dataset
        with gzip.open(file_path, 'rb') as f:
            ds = xr.open_dataset(f)
        
            # check scan is good
            if (ds.ntime.size > 100) and (ds.nrange.size > 100) and (ds.elevation[0].data > des_elev-1. and ds.elevation[0].data < des_elev+1.):
                print(f"Reading {filename}")
                # oversampling_ratio = ds.attrs['oversampling_ratio']  
                start_point = np.argmin(abs(ds.ranges.data-range_offset))+1
                # set arrays for calculation simplicity
                r = ds.ranges[start_point:].data - range_offset
                az = ds.azimuth.data
                el = ds.elevation.data
                vr = ds.dpl[:,0,start_point:].data
                # calculate vertical vorticity
                vort_z = ( (vr[1:,:] - vr[:-1,:]) / (np.deg2rad(az[1:]) - np.deg2rad(az[:-1]))[:,np.newaxis] ) * (1 / r)
                
                # create dictionary to store lidar file
                lidar = {
                    # lidar information
                    'ob_method' : ds.attrs['observation_method'],
                    'serial_number' : ds.attrs['serial_number'],
                    'lat' : ds.latitude.data,
                    'lon' : ds.longitude.data,
                    'altitude' : ds.altitude.data,
                    # scan setting information
                    'rpm_azimuth' : ds.rpm_azimuth[start_point:].data,
                    # scan dimensions
                    'r' : r,
                    'az' : az,
                    'el' : el,
                    # important observations
                    'vr' : vr,
                    'vort_z' : vort_z,
                    'pwr' : ds.pwr[:,0,start_point:].data * 100,
                    'snr' : ds.snr[:,0,start_point:].data,
                    'sw' : ds.wth[:,0,start_point:].data,
                    # other observations
                    'noise_level' : ds.noise_level[:,0,start_point:].data,
                    # 'power_spectra' : ds.power_spectra[:,0,start_point:].data,
                    'doppler_velocity' : ds.doppler_velocity.data,
                    # scan time information
                    'record_start_time' : ds.start_time.data,
                    'start_time' : ds.attrs['start_time'][11:19],
                    'start_date' : ds.attrs['start_time'][:10],
                    'end_time' : ds.attrs['end_time'][11:19],
                    'end_date' : ds.attrs['end_time'][:10]
                        }
                
                lall.append(lidar)
                ds.close()

# --------------------------
# filter vertical vorticity field
if filter_vort:
    for i, l in enumerate(lall):
        # grab data
        vort = xr.DataArray(l['vort_z'], dims=("az","r"), coords={'az': l['az'][:-1], 'r': l['r']}).fillna(0)
        # Create uniform coordinates for azimuth and range
        az_uniform = np.linspace(vort.az.min().data, vort.az.max().data, vort.shape[0])
        r_uniform = np.linspace(vort.r.min().data, vort.r.max().data, vort.shape[1])
        # Interpolate data onto uniform grid
        vort_uniform = vort.interp(az=az_uniform, r=r_uniform)
        # take fft
        f_var = xrft.fft(vort_uniform, dim=('az','r'), true_phase=True, true_amplitude=True)
        # take cutoff wavenumber as 1/1000 /m
        fc = 1/250.

        # zero out x and y wavenumbers above this cutoff to get lowpass filter
        jrp = np.where(f_var.freq_r > fc)[0]
        jrn = np.where(f_var.freq_r < -fc)[0]
        jazp = np.where(f_var.freq_az > fc)[0]
        jazn = np.where(f_var.freq_az < -fc)[0]
        f_var[:,jrp] = 0
        f_var[:,jrn] = 0
        f_var[jazp,:] = 0
        f_var[jazn,:] = 0

        # take ifft
        var_filt = xrft.ifft(f_var, dim=('freq_az','freq_r'),
                            true_amplitude=True, true_phase=True,
                            lag=(f_var.freq_r.direct_lag, f_var.freq_az.direct_lag)).real
        l['vort_z_filt'] = var_filt
        print(var_filt.shape)

# --------------------------
# calculate the 2D autocorrelation of vorticity

autocorr_2D = []
if filter_vort:
    for i, l in enumerate(lall):
        # grab data
        vort_z_filt = l['vort_z_filt']
        # # Create uniform coordinates for azimuth and range
        # az_uniform = np.linspace(vort.az.min().data, vort.az.max().data, vort.shape[0])
        # r_uniform = np.linspace(vort.r.min().data, vort.r.max().data, vort.shape[1])
        # # Interpolate data onto uniform grid
        # vort_uniform = vort.interp(az=az_uniform, r=r_uniform)
        # subtract x,y mean
        dfluc = xrft.detrend(vort_z_filt, dim=("az","r"), detrend_type="constant")
        # normalize by standard deviation
        dnorm = dfluc / dfluc.std(dim=("az","r"))
        # calculate PSD using xrft
        PSD = xrft.power_spectrum(dnorm, dim=("az","r"), true_phase=True, 
                                    true_amplitude=True)
        # take real part of ifft to return ACF
        R = xrft.ifft(PSD, dim=("freq_az","freq_r"), true_phase=True, 
                        true_amplitude=True, lag=(0,0)).real
        # remove negative radial lags
        R = R.where(R.r >= 0, drop=True)

        # append to list
        autocorr_2D.append(R)
else:
    for i, l in enumerate(lall):
        # grab data
        vort = xr.DataArray(l['vort_z'], dims=("az","r"), coords={'az': l['az'][:-1], 'r': l['r']}).fillna(0)
        # Create uniform coordinates for azimuth and range
        az_uniform = np.linspace(vort.az.min().data, vort.az.max().data, vort.shape[0])
        r_uniform = np.linspace(vort.r.min().data, vort.r.max().data, vort.shape[1])
        # Interpolate data onto uniform grid
        vort_uniform = vort.interp(az=az_uniform, r=r_uniform)
        # subtract x,y mean
        dfluc = xrft.detrend(vort_uniform, dim=("az","r"), detrend_type="constant")
        # normalize by standard deviation
        dnorm = dfluc / dfluc.std(dim=("az","r"))
        # calculate PSD using xrft
        PSD = xrft.power_spectrum(dnorm, dim=("az","r"), true_phase=True, 
                                    true_amplitude=True)
        # take real part of ifft to return ACF
        R = xrft.ifft(PSD, dim=("freq_az","freq_r"), true_phase=True, 
                        true_amplitude=True, lag=(0,0)).real
        # remove negative radial lags
        R = R.where(R.r >= 0, drop=True)

        # append to list
        autocorr_2D.append(R)

# --------------------------
# plot vort and autocorr to test

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')

# max range
rmax = 10 # km
for i, (l,R) in enumerate(zip(lall,autocorr_2D)):
    XX, YY = dis_angle_to_2Dxy(l['r'],l['az'])
    # make fig
    fig, ax = plt.subplots(figsize=(20,10), ncols=2)
    # plot vort field
    vmin, vmax = -0.05, 0.05
    if filter_vort:
        pcm1 = ax[0].pcolormesh(XX.T[:-1]/1e3, YY.T[:-1]/1e3, l['vort_z_filt'],
                                vmin=vmin, vmax=vmax, cmap="RdBu")
    else:
        pcm1 = ax[0].pcolormesh(XX.T[:-1]/1e3, YY.T[:-1]/1e3, l['vort_z'],
                                vmin=vmin, vmax=vmax, cmap="RdBu")
    ax[0].set_title("Inferred Vertical Vorticity")
    cbar2 = fig.colorbar(pcm1, ax=ax[0], aspect=20, pad=0.03, fraction=0.06, shrink=0.81)
    cbar2.set_label("$\\zeta$ [s$^{-1}$]")
    cbar2.set_ticks(np.arange(vmin, vmax+0.0001, 0.025))
    ax[0].set_xlim(-rmax,rmax)
    ax[0].set_ylim(-rmax,rmax)
    ax[0].set_aspect('equal')
    ax[0].xaxis.set_major_locator(MultipleLocator(5))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_xlabel("Zonal Distance [km]")
    ax[0].yaxis.set_major_locator(MultipleLocator(5))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1))
    ax[0].set_ylabel("Meridional Distance [km]")
    # plot vort autocorr
    vmin, vmax = -0.5, 0.5
    pcm2 = ax[1].pcolormesh(R.az, R.r, R.T, 
                            vmin=vmin, vmax=vmax, 
                            cmap="seismic")
    ax[1].set_title("2D Autocorrelation of $\\zeta$")
    cbar1 = fig.colorbar(pcm2, ax=ax[1], aspect=20, pad=0.03, fraction=0.06, shrink=0.81)
    cbar1.set_label("$R_{\\zeta \\zeta}$")
    cbar1.set_ticks(np.arange(vmin, vmax+0.001, 0.1))
    aspect_ratio = (R.az.max() - R.az.min()) / (R.r.max() - R.r.min())
    ax[1].set_aspect(aspect_ratio)  # Set the calculated aspect ratio
    ax[1].xaxis.set_major_locator(MultipleLocator(60))
    ax[1].xaxis.set_minor_locator(MultipleLocator(5))
    ax[1].set_xlabel("$r_{\\theta}$ [$^{\\circ}$]")
    ax[1].yaxis.set_major_locator(MultipleLocator(1000))
    ax[1].yaxis.set_minor_locator(MultipleLocator(100))
    ax[1].set_ylabel("$r_R$ [m]")

    plt.suptitle(f"{lidar_loc} MetroWeather CDL {l['start_date']} {l['start_time']} UTC ({round(l['el'][0],1)}$^{{\\circ}}$ Elevation)", 
                fontweight="bold", fontsize=25, y=0.925)
    fig.tight_layout()
    
    if filter_vort:
        dout = f"{figdir}filtered_vort_vortautocorr_{l['start_date']}T{l['start_time']}.png"
    else:
        dout = f"{figdir}vort_vortautocorr_{l['start_date']}T{l['start_time']}.png"
    plt.savefig(dout)