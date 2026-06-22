# ------------------------------------------------
# Name: plot_ppi_loop.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 27 April 2026
# Purpose: Plot tons of lidar PPIs
# ------------------------------------------------
import io
import glob
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd
import xarray as xr
import pyart

sys.path.append("/home/robbyfrost/Analysis/MetroWX_Lidar/2026/")
from process import reprocess_moments
from plot_ppi import prep_for_plot, plot_ppi
from make_gif import create_gif

sys.path.append("/home/robbyfrost/Analysis/TurbTor_Radar/")
from Analysis.TurbTor_Lidar.lidar_functions import dis_angle_to_2Dxy, north0_to_arctheta, snr_cmap, prep_for_plot

# plotting set up
from matplotlib import rc
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
rc('font', family='sans-serif')
rc('font', weight='normal', size=15)
rc('figure', facecolor='white')
# ------------------------------------------------
# params

# date (YYYYMMDD)
date = '20260426'
# deployment number
deployment = 1
# azimuth offset (truck heading, degrees)
az_off = 90

# path to lidar data
dlid = f"/data/arrcwx/MetroWeather/ARRC_MCDL/2026/202604/{date}/"
# directory to output figures
dfig = f"/home/robbyfrost/Figures/MW/truck/{date}/"
os.makedirs(dfig) if not os.path.exists(dfig) else None

# scan type (CAPPI, Pointing, RHI, or PPI)
scan = "CAPPI"

# GIF naming convention
naming_con = f"{date}_D{deployment}"
# GIF frame duration
duration = 300

# bounds for plot
xbds = [-5,0]
ybds = [0,5]

# ------------------------------------------------
# read in data
files = sorted(glob.glob(f"{dlid}*{scan}*"))

# loop over time
for f in files[:6]:
    # read file
    dctx = zstd.ZstdDecompressor()
    with open(f, 'rb') as fi:
        raw = dctx.decompress(fi.read())
    ds = xr.open_dataset(io.BytesIO(raw), engine='h5netcdf')

    # ------------------------------------------------
    # prep for plot

    # sweep elevations
    fixed_angle = np.unique(ds.elevation.values)
    
    # number of sweeps
    nsweeps = len(fixed_angle)

    # loop over sweeps
    for swp in range(nsweeps):
        ds_swp = prep_for_plot(ds, swp, az_off)
        tplot = str(datetime.fromtimestamp(float(ds_swp.start_time[0,0].values) / 1e6, tz=timezone.utc))[:-13]

        # ------------------------------------------------
        # plot

        # make fig
        fig, axs = plt.subplots(
            figsize=(18,7.9),
            ncols=3,
            sharey=True,
            sharex=True,
            constrained_layout=True,
            dpi=200
        )
        fig.suptitle(f"{tplot} UTC (El={fixed_angle[swp]:.1f}$^{{\\circ}}$)",
                    fontsize=25,
                    fontweight='bold')
        # snr
        ax = axs[0]
        vmin, vmax = -15, -5
        ax.set_title("Signal to Noise")
        pcm = ax.pcolormesh(
            ds_swp.x/1e3,
            ds_swp.y/1e3,
            ds_swp.SNR_OU,
            cmap=snr_cmap(),
            vmin=vmin,
            vmax=vmax
        )
        cbar = plt.colorbar(pcm, ax=ax, label="SNR [dB]", orientation='horizontal')
        cbar.set_ticks(np.arange(vmin, vmax+1e-10, 2))
        ax.set_ylabel("Meridional Distance (km)")
        # radial velocity
        ax = axs[1]
        vmin, vmax = -20, 20
        ax.set_title("Radial Velocity")
        pcm = ax.pcolormesh(
            ds_swp.x/1e3,
            ds_swp.y/1e3,
            ds_swp.VRC_OU,
            cmap='pyart_Carbone42',
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(pcm, ax=ax, label="$V_r$ [m s$^{-1}$]", orientation='horizontal')
        # inferred vertical vorticity
        ax = axs[2]
        vmin, vmax = -0.05, 0.05
        ax.set_title("Inferred Vertical Vorticity")
        pcm = ax.pcolormesh(
            ds_swp.x/1e3,
            ds_swp.y/1e3,
            ds_swp.VORTS_OU,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax
        )
        cbar = plt.colorbar(pcm, ax=ax, label="$\\zeta_{inf}$ [s$^{-1}$]", orientation='horizontal')
        cbar.set_ticks(np.arange(vmin, vmax+1e-10, 0.02))

        # clean up
        for ax in axs:
            ax.set_xlabel("Zonal Distance (km)")
            ax.set_aspect('equal')
            ax.grid(alpha=0.5)
            ax.set_xlim(xbds)
            ax.set_ylim(ybds)

        # output
        dout = f"{dfig}{re.sub(r'\D', '', tplot)}_{fixed_angle[swp]:.1f}el.png"
        plt.savefig(dout)
        print(f"Saved to: {dout}\n")

# ------------------------------------------------
# make gif

create_gif(naming_con, dfig, dfig, duration, 1)
print(f"Output to: {dfig}{naming_con}.gif")