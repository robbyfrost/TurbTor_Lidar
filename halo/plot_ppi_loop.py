# ------------------------------------------------
# Name: plot_ppi_loop.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 22 May 2026
# Purpose: Plot a whole folder of halo lidar PPIs
# ------------------------------------------------
# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import xarray as xr
import pyart
from netCDF4 import Dataset
from lidar_utils import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# from metpy.plots import USCOUNTIES
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import glob
import os
import xarray as xr
# plotting set up
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')
# ------------------------------------------------
# params

# directory to output figures
dfig = "/home/robbyfrost/Figures/halo/20250425/"
# path to cfradial files
dlid = "/data/arrcwx/robbyfrost/halo/20250425/cfrada/"
# list of lidar files
lfiles = sorted(glob.glob(f"{dlid}*"))
# number of files
nt = len(lfiles)
# ------------------------------------------------
# loop over files
for jt in range(nt):
    # ----------------------
    # read cfradial
    ds = xr.open_dataset(lfiles[jt])
    # extract needed arrays
    vr = ds.vr.values
    az = ds.azimuth.values
    r = ds.range.values*1e3
    vort = ((vr[2:,:]-vr[:-2,:]) / ((np.deg2rad(az[2:])-np.deg2rad(az[:-2]))[:,None])) / (r[None,:])
    x, y = dis_angle_to_2Dxy(r, north0_to_arctheta(az))
    # time info
    tsstr = ds.attrs['time_coverage_start'][:10] + " " + ds.attrs['time_coverage_start'][11:19]
    tsdt = datetime.strptime(tsstr, '%Y-%m-%d %H:%M:%S')
    tsstr_out = tsdt.strftime("%Y%m%d_%H%M%S")
    testr = ds.attrs['time_coverage_end'][:10] + " " + ds.attrs['time_coverage_end'][11:19]
    tedt = datetime.strptime(testr, '%Y-%m-%d %H:%M:%S')

    # check for bad file
    if (vort.shape[0]==0) or (vort.shape[1]==0) \
    or (vort.shape[0]==1) or (vort.shape[1]==1):
        print(f"Skipping {tsdt}.\n")
        continue
    # ----------------------
    # plot 

    # make figure
    fig, axs = plt.subplots(figsize=(16,10.75),
                            ncols=2,
                            sharey=True,
                            constrained_layout=True,
                            dpi=200)
    # elevation
    elplt = float(ds.fixed_angle.values)
    # fig title
    fig.suptitle(f"NSSL Doppler Wind Lidar \nValid at {tsdt} UTC, El={elplt:.2f}$\\bf{{^{{\\circ}}}}$", 
                    fontsize=20, fontweight='bold')

    # radial velocity
    ax = axs[0]
    ax.set_title("Radial Velocity")
    pcm = ax.pcolormesh(x.T/1e3, 
                        y.T/1e3, 
                        vr,
                        cmap='Carbone42',
                        vmin=-38, 
                        vmax=38)
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", label="$V_r$ [m s$^{-1}$]", pad=0.03)
    ax.set_ylabel("Meridional Distance [km]")
    # zeta
    ax = axs[1]
    ax.set_title("Inferred Vertical Vorticity")
    pcm = ax.pcolormesh(x.T[1:-1,:]/1e3, 
                        y.T[1:-1,:]/1e3, 
                        vort,
                        cmap='RdBu_r',
                        vmin=-0.05, 
                        vmax=0.05)
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", label="$\\zeta$ [s$^{-1}$]", pad=0.03)

    for ax in axs:
        ax.set_aspect('equal')
        ax.grid(alpha=0.5)
        ax.set_xlabel("Zonal Distance [km]")
        ax.set_xlim(-8.5,0)
        ax.set_ylim(-2.25,6.75)

    dout = f"{dfig}{tsstr_out}_vr_zeta.png"
    plt.savefig(dout)
    plt.close(fig)
    print(f"Done with: {tsdt}")

print("Finished plotting sweeps\n")