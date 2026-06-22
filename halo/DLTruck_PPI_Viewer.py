#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:59:10 2024

@author: joshua.gebauer
"""

import numpy as np
import matplotlib.pyplot as plt
import pyart
import glob
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
from matplotlib.patches import Polygon
from matplotlib.patches import Wedge
from datetime import datetime



def get_radar_data(path, time,rname):
    
    files = []
    files = files + sorted(glob.glob(path + '/' + rname + '*' + '_V06'))
    
    dtime = np.array([(datetime.strptime(x,path + rname + '%Y%m%d_%H%M%S_V06') - datetime(1970,1,1)).total_seconds()
                      for x in files])
    
    
    dif = np.abs(dtime-time)
    
    foo = np.argmin(dif)
    
    r = pyart.io.read_nexrad_archive(files[foo])
    
    times = []
    snum = []
    for i in range(r.nsweeps):
        
        sweep = r.extract_sweeps([i])
        if (np.nanmean(sweep.elevation['data']) < .6):
            
            times.append(dtime[foo] + np.nanmean(sweep.time['data']))
            snum.append(i)
    
    times = np.array(times)
    
    dif = np.abs(times-time)

    foo = np.argmin(dif)

    sweep = r.extract_sweeps([foo])
    ref = sweep.get_field(0,'reflectivity')
    
    rlat,rlon,alt = sweep.get_gate_lat_lon_alt(0)
    
    return ref, rlat, rlon
    
    
fname = '/Users/joshua.gebauer/dltruckdlcsmDL1.b1.20240523.000000.cdf'

mm_name = '/Users/joshua.gebauer/LIDR_TRUCK_20240523_MM.nc'

radar = True
rpath = '/Users/joshua.gebauer/NEXRAD_Data/KFDR/'

f = Dataset(fname)

vr = f.variables['velocity'][:]
snum = f.variables['snum'][:]
az = f.variables['azimuth'][:]
el = f.variables['elevation'][:]
hour = f.variables['hour'][:]
heading = f.variables['heading'][:]
r = f.variables['range'][:]
snr = 10*np.log10(f.variables['intensity'][:]-1)
time = f.variables['base_time'][0] + f.variables['time_offset'][:]
f.close()

m = Dataset(mm_name)
m_heading = m.variables['gps_dir'][:]
c_heading = m.variables['compass_dir'][:]
mtime = m.variables['epochtime'][:]
lat = m.variables['lat'][:]
lon = m.variables['lon'][:]

#az = (az+heading) % 360
#az += 180
#foo = np.where((snr > 0) | (snr < -23))

#foo = np.where(heading < -800)[0]
#vr[foo] = np.nan
#az[foo] = np.nan
#el[foo] = np.nan


foo = np.where(r > 8)[0]
vr[:,foo] = np.nan

unique_snum = np.unique(snum)

d01 = ccrs.LambertConformal(central_longitude=-99.562789, central_latitude=34.584454,standard_parallels=(30,60))

lidar_position = True
for i in unique_snum:
    
    
    
    foo = np.where(i == snum)
    
    scan_el = el[foo]
    scan_az = az[foo]
    scan_hr = hour[foo]
    scan_vr = vr[foo[0],:]
    scan_time = time[foo]
    
    
    sort_index = np.argsort(scan_hr)
    
    
    scan_el = scan_el[sort_index]
    scan_az = scan_az[sort_index]
    scan_hr = scan_hr[sort_index]
    scan_vr = scan_vr[sort_index,:]
    scan_time = scan_time[sort_index]
    
    az_1 = []
    el_1 = []
    hr_1 = []
    vr_1 = []
    scan_time1 = []
    
    az_2 = []
    el_2 = []
    hr_2 = []
    vr_2 = []
    scan_time2 = []
    
    sign = 0
    switch = False
    for j in range(1,len(scan_az)):
        
        direction = np.sign(scan_az[j]-scan_az[j-1])
        if j == 1:
            sign = np.copy(direction)
            az_1.append(scan_az[j-1])
            el_1.append(scan_el[j-1])
            hr_1.append(scan_hr[j-1])
            vr_1.append(scan_vr[j-1,:])
            scan_time1.append(scan_time[j-1])
        
        else:
            if (sign == direction) and not switch:
                az_1.append(scan_az[j-1])
                el_1.append(scan_el[j-1])
                hr_1.append(scan_hr[j-1])
                vr_1.append(scan_vr[j-1,:])
                scan_time1.append(scan_time[j-1])
            elif (sign == direction) and switch:
                az_2.append(scan_az[j-1])
                el_2.append(scan_el[j-1])
                hr_2.append(scan_hr[j-1])
                vr_2.append(scan_vr[j-1,:])
                scan_time2.append(scan_time[j-1])
            else:
                if not switch:
                    switch = True
                    if abs(direction) > 0.1:
                        sign = np.copy(direction)
                    else:
                        sign = sign*-1
                    az_1.append(scan_az[j-1])
                    el_1.append(scan_el[j-1])
                    hr_1.append(scan_hr[j-1])
                    vr_1.append(scan_vr[j-1,:])
                    scan_time1.append(scan_time[j-1])
                else:
                    if scan_az[j]-scan_az[j-1] == 0:
                        continue
                    else:
                        # The direction should never switch twice
                        break
    
    
    az_1 = np.array(az_1)
    el_1 = np.array(el_1)
    hr_1 = np.array(hr_1)
    vr_1 = np.array(vr_1)
    scan_time1 =np.array(scan_time1)
    
    az_2 = np.array(az_2)
    el_2 = np.array(el_2)
    hr_2 = np.array(hr_2)
    vr_2 = np.array(vr_2)
    scan_time2 = np.array(scan_time2)
    
    
    
    for j in range(2):
        
        if j == 0:
            if len(az_1) == 0:
                continue
            
            foo = np.where((scan_time1[0] <= mtime) & (scan_time1[-1] >= mtime))
            scan1_heading = m_heading[foo]
            scan1_cheading = np.nanmean(c_heading[foo])
            scan_lat = np.nanmean(lat[foo])
            scan_lon = np.nanmean(lon[foo])
            
            
            
            az_1 = (az_1 + scan1_cheading ) % 360
            
            if lidar_position:
                lidar_x0, lidar_y0 = d01.transform_point(scan_lon,scan_lat,src_crs=ccrs.Geodetic())
                lidar_position = False
            
            if radar:
                ref, rlat,rlon = get_radar_data(rpath, np.nanmean(scan_time1), 'KFDR')
                temp = d01.transform_points(ccrs.Geodetic(),rlon,rlat)
                radar_x = temp[:,:,0]
                radar_y = temp[:,:,1]
            
            lidar_x, lidar_y = d01.transform_point(scan_lon,scan_lat,src_crs=ccrs.Geodetic())
            
            # lidar_x = 0
            # lidar_y = 0
            
            x = r[None,:]*np.sin(np.radians(az_1[:,None]))*np.cos(np.radians(np.nanmean(el_1)))*1000 + lidar_x
            y = r[None,:]*np.cos(np.radians(az_1[:,None]))*np.cos(np.radians(np.nanmean(el_1)))*1000 + lidar_y
            
            vort = ((vr_1[2:,:]-vr_1[:-2,:])/((np.deg2rad(az_1[2:])-np.deg2rad(az_1[:-2]))[:,None]))/(r[None,:]*1000)
            try:
                
                plt.figure(figsize=(18,18))
                ax1 = plt.subplot(121,projection = d01)
                #ax1 = plt.subplot(121,aspect='equal')
                plt.pcolormesh(x,y,vr_1,cmap='pyart_Carbone42',vmin=-38,vmax=38)
                cbar = plt.colorbar()
                cbar.set_label('[m/s]')
                plt.xlim(lidar_x0-8000,lidar_x0+1000)
                plt.ylim(lidar_y0-1000,lidar_y0+8000)
                
                # plt.xlim(lidar_x0-16000,lidar_x0+16000)
                # plt.ylim(lidar_y0-16000,lidar_y0+16000)
                #ax1.scatter(0,0,marker='*',color='r',s=75)
                plt.title('Radial Velocity')
            
                ax2 = plt.subplot(122,projection = d01)
                #ax2 = plt.subplot(122,aspect='equal')
                plt.pcolormesh(x[1:-1],y[1:-1],vort,cmap='RdBu_r',vmin=-0.05,vmax=0.05)
                cbar = plt.colorbar()
                cbar.set_label('[s^-1]')
                plt.xlim(lidar_x0-8000,lidar_x0+1000)
                plt.ylim(lidar_y0-1000,lidar_y0+8000)
                
                # plt.xlim(lidar_x0-16000,lidar_x0+16000)
                # plt.ylim(lidar_y0-16000,lidar_y0+16000)
                #ax2.scatter(0,0,marker='*',color='r',s=75)
                plt.title('Inferred Vorticity')
                
                if radar:
                    cmin = 20; cmax = 61; cint = 10; clevs = np.round(np.arange(cmin,cmax,cint),3)
                    ax1.contour(radar_x,radar_y,ref,clevs,colors = 'k')
                    ax2.contour(radar_x,radar_y,ref,clevs,colors = 'k')
            
                plt.suptitle(fname[-19:-11] + ' Hour: ' + str(np.round(np.nanmean(hr_1),2)) + ' EL: ' + str(np.nanmean(el_1)), fontsize = 24)
                plt.savefig('/Users/joshua.gebauer/DLTruck_PPIs/' + fname[-19:-11] + '/' + str(int(np.nanmean(hr_1*3600))) + '.png',dpi=300)
            
                plt.close()
            except:
                plt.close()
                continue
            
            
            
        else:
            if len(az_2) == 0:
                continue
            
            foo = np.where((scan_time2[0] <= mtime) & (scan_time2[-1] >= mtime))
            scan2_heading = m_heading[foo]
            scan2_cheading = np.nanmean(c_heading[foo])
            scan_lat = np.nanmean(lat[foo])
            scan_lon = np.nanmean(lon[foo])
            
            az_2 = (az_2 + scan2_cheading) % 360
            
            if lidar_position:
                lidar_x0, lidar_y0 = d01.transform_point(scan_lon,scan_lat,src_crs=ccrs.Geodetic())
                lidar_position = False
            
            if radar:
                ref, rlat,rlon = get_radar_data(rpath, np.nanmean(scan_time2), 'KFDR')
                temp = d01.transform_points(ccrs.Geodetic(),rlon,rlat)
                radar_x = temp[:,:,0]
                radar_y = temp[:,:,1]
                
                
            lidar_x, lidar_y = d01.transform_point(scan_lon,scan_lat,src_crs=ccrs.Geodetic())
            
            # lidar_x = 0
            # lidar_y = 0
            
            x = r[None,:]*np.sin(np.radians(az_2[:,None]))*np.cos(np.radians(np.nanmean(el_2)))*1000 + lidar_x
            y = r[None,:]*np.cos(np.radians(az_2[:,None]))*np.cos(np.radians(np.nanmean(el_2)))*1000 + lidar_y
            
            vort = ((vr_2[2:,:]-vr_2[:-2,:])/((np.deg2rad(az_2[2:])-np.deg2rad(az_2[:-2]))[:,None]))/(r[None,:]*1000)
            
            try:
                
                plt.figure(figsize=(18,18))
                #ax1 = plt.subplot(121,aspect='equal')
                ax1 = plt.subplot(121,projection = d01)
                plt.pcolormesh(x,y,vr_2,cmap='pyart_Carbone42',vmin=-38,vmax=38)
                cbar = plt.colorbar()
                cbar.set_label('[m/s]')
                plt.xlim(lidar_x0-8000,lidar_x0+1000)
                plt.ylim(lidar_y0-1000,lidar_y0+8000)
                
                # plt.xlim(lidar_x0-16000,lidar_x0+16000)
                # plt.ylim(lidar_y0-16000,lidar_y0+16000)
                #ax1.scatter(0,0,marker='*',color='r',s=75)
                plt.title('Radial Velocity')
            
                ax2 = plt.subplot(122,projection = d01)
                #ax2 = plt.subplot(122,aspect='equal')
                plt.pcolormesh(x[1:-1],y[1:-1],vort,cmap='RdBu_r',vmin=-0.05,vmax=0.05)
                cbar = plt.colorbar()
                cbar.set_label('[s^-1]')
                plt.xlim(lidar_x0-8000,lidar_x0+1000)
                plt.ylim(lidar_y0-1000,lidar_y0+8000)
                
                # plt.xlim(lidar_x0-16000,lidar_x0+16000)
                # plt.ylim(lidar_y0-16000,lidar_y0+16000)
                #ax2.scatter(0,0,marker='*',color='r',s=75)
                plt.title('Inferred Vorticity')
                
                if radar:
                    cmin = 20; cmax = 61; cint = 10; clevs = np.round(np.arange(cmin,cmax,cint),3)
                    ax1.contour(radar_x,radar_y,ref,clevs,colors = 'k')
                    ax2.contour(radar_x,radar_y,ref,clevs,colors = 'k')
                    
                plt.suptitle(fname[-19:-11] + ' Hour: ' + str(np.round(np.nanmean(hr_2),2)) + ' EL: ' + str(np.nanmean(el_2)), fontsize = 24)
                plt.savefig('/Users/joshua.gebauer/DLTruck_PPIs/' + fname[-19:-11] + '/' + str(int(np.nanmean(hr_2*3600))) + '.png',dpi=300)                                          
                plt.close()
            except:
                plt.close()
                continue
            
    
            
            
        
            
            
    