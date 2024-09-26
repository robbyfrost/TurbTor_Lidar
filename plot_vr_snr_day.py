# ------------------------------------------------
# Name: plot_vr_snr_day.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 25 September 2024
# Purpose: Plotting radial velocity and signal to
# noise ratio from MetroWX CDL PPI scans for an 
# entire day
# ------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gzip
import os
from functions import *
import matplotlib.pyplot as plt
from matplotlib import rc
import pyart
from matplotlib.ticker import MultipleLocator
import zstandard as zstd

# import sys
# sys.path.append('/home/robbyfrost/Analysis/MetroWX_Lidar/plotting/')
# import colorlevel2 as cl

# --------------------------
# settings

# date/time info of lidar scans
des_year = "2024"
des_mon = "09"
des_day = "24"
des_sys = "ARRC_Truck" # WG100-L0AD00003JP, WG100-L0AD00004JP, or ARRC_Truck
lidar_loc = "ARRC Mobile" # Hampton, VA or Oklahoma Mobile
# directory storing 24 hours of lidar data
directory = f"/data/arrcwx/robbyfrost/lidar_obs/{des_sys}/{des_year}/{des_mon}/{des_day}/"
# directory for saving figues
figdir = f"/home/robbyfrost/Analysis/TurbTor_Lidar/figures/{des_sys}/{des_year}/{des_mon}/{des_day}/"
os.makedirs(figdir, exist_ok=True)

# desired elevation angle
des_elev = 5.
# offset in meters where data begins
range_offset = 1425
# max range for PPI plots in km
rmax = 10

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')

# --------------------------
# read in lidar files

# Initialize an empty list to store the datasets
lall = []

# Iterate through all files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".nc"):
        file_path = os.path.join(directory, filename)
        # Open each .nc.gz file and load it as an xarray Dataset
        # with gzip.open(file_path, 'rb') as f:
        ds = xr.open_dataset(file_path)
    
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
            
            # lall.append(lidar)
            ds.close()

            # --------------------------
            # make plots

            # iterate over lidar scans
            # for i, l in enumerate(lall):
            l = lidar
            # cartesian stuff for plotting
            XX, YY = dis_angle_to_2Dxy(l['r'], north0_to_arctheta(l['az']))
            cmap = snr_cmap()

            # create plot
            print("Making plot")
            fig, ax = plt.subplots(figsize=(20,10), ncols=2)
            # plot radial velocity
            vmin, vmax = -10, 10
            pcm1 = ax[0].pcolormesh(XX.T/1e3, YY.T/1e3, l['vr'],
                                    vmin=vmin, vmax=vmax,
                                    cmap="pyart_NWSVel")
            ax[0].set_title("Radial Velocity")
            cbar1 = fig.colorbar(pcm1, ax=ax[0], aspect=20, pad=0.03, fraction=0.06, shrink=0.81)
            cbar1.set_label("$V_r$ [m s$^{-1}$]")
            cbar1.set_ticks(np.arange(vmin, vmax+0.001, 5))
            # plot vertical vorticity
            vmin, vmax = -20, 5
            pcm2 = ax[1].pcolormesh(XX.T/1e3, YY.T/1e3, 10*np.log10(l['snr']-1),
                                    vmin=vmin, vmax=vmax,
                                    cmap=cmap)
            ax[1].set_title("Signal to Noise Ratio")
            cbar2 = fig.colorbar(pcm2, ax=ax[1], aspect=20, pad=0.03, fraction=0.06, shrink=0.81)
            cbar2.set_label("$SNR [dB]")
            cbar2.set_ticks(np.arange(vmin, vmax+0.0001, 5))
            # stuff for both axes
            for iax in ax:
                iax.set_aspect('equal')
                iax.set_xlim(-rmax,rmax)
                iax.xaxis.set_major_locator(MultipleLocator(5))
                iax.xaxis.set_minor_locator(MultipleLocator(1))
                iax.set_xlabel("Zonal Distance [km]")
                iax.set_ylim(-rmax,rmax)
                iax.yaxis.set_major_locator(MultipleLocator(5))
                iax.yaxis.set_minor_locator(MultipleLocator(1))
                iax.set_ylabel("Meridional Distance [km]")

            plt.suptitle(f"{lidar_loc} MetroWeather CDL {l['start_date']} {l['start_time']} UTC ({round(l['el'][0],1)}$^{{\\circ}}$ Elevation)", 
                        fontweight="bold", fontsize=25, y=0.925)
            fig.tight_layout()

            time_str = l['start_date'].replace("-","") + l['start_time'].replace(":","")
            dout = f"{figdir}vr_snr_PPI_el{int(l['el'][0])}_{time_str}.png"
            print(f"Saving to {dout}")
            plt.savefig(dout)