#%% ----------------------------------------------
# Name: vorticity_timeseries.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 10 September 2024
# Purpose: Plotting timeseries of horizontally 
# averaged vertical vorticity
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
from datetime import datetime
import matplotlib.dates as mdates

#%% --------------------------
# settings

# date/time info of lidar scans
des_year = '2024'
des_mon = '09'
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
# start and end azimuths for averaging
az_start, az_end = 293.925, 358.99

# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')

#%% --------------------------
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
        try:
            ds = ds.where(ds.elevation == des_elev, drop=True)
        except ValueError:
            continue
    
        # check scan is good
        if (ds.ntime.size > 100) and (ds.nrange.size > 100) and (ds.elevation[0].data > des_elev-1. and ds.elevation[0].data < des_elev+1.):
            print(f"Reading {filename}")
            # oversampling_ratio = ds.attrs['oversampling_ratio']
            # print(ds.ranges.shape)
            start_point = np.argmin(abs(ds.ranges[:,0].data-range_offset))+1
            # set arrays for calculation simplicity
            r = ds.ranges[start_point:,0].data - range_offset
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

#%% --------------------------
# create timeseries of averaged zeta

# arrays
vort_ts = np.empty(len(lall))
time_list = []
# enumerate 2D vorticity data
for i, l in enumerate(lall):
    # narrow down averaging area
    azs_idx, aze_idx = np.argmin(abs(l['az'] - az_start)), np.argmin(abs(l['az'] - az_end))
    bh = beam_height_2D(l['r'], l['el'])
    ridx = np.argmin(abs(bh[:,0]-250))
    l['vort_z_avg'] = l['vort_z'][azs_idx:aze_idx,10:ridx]
    # mean absolute value of vort_z
    vort_ts[i] = np.nanmean(abs(l['vort_z_avg']), axis=(0,1))
    # array of time from each timestep
    time_list.append(datetime.strptime(f"{l['start_date']} {l['start_time']}", '%Y-%m-%d %H:%M:%S'))
# convert time to array
time_array = np.array(time_list)
# filter out extreme zeta values above 0.1
# vort_ts = np.where(vort_ts > 1, np.nan, vort_ts) # TODO: Make this remove scans where small area of data are captured instead
print(vort_ts)
#%% --------------------------
# plot timeseries of averaged zeta

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(time_array, vort_ts)

ax.set_xlabel("Time [UTC]")
plt.xticks(rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_ylabel("$|\\zeta|$ [s$^{-1}$]")
ax.set_title(f"{lidar_loc} MetroWeather CDL {des_year}-{des_mon}-{des_day} ({round(l['el'][0],1)}$^{{\circ}}$ Elevation)",
             fontweight="bold")

plt.tight_layout()

plt.show()
plt.savefig(f"{figdir}{des_year}{des_mon}{des_day}_vort_timeseries.png")