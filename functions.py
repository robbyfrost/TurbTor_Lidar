# ------------------------------------------------
# Name: functions.py
# Author: Robby M. Frost
# University of Oklahoma
# Created: 10 September 2024
# Purpose: Functions for using lidar data
# ------------------------------------------------

import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

# ------------------------------------------------
# functions
def north0_to_arctheta(theta):
    return  np.where(theta > 270, 450-theta, 90-theta)

def beam_height_2D(r_km,ele, instrumentheight=0.0):
    a = 6.371e3 # Earth's Radius
    ae = 4 / 3 * a # Effective Earth's radius
    bh = np.sqrt((r_km[:,np.newaxis])**2 + ae**2 + (2 * ae * r_km)[:,np.newaxis].dot(np.sin(ele * np.pi / 180)[np.newaxis,:])) - ae + instrumentheight
    return bh

def beam_range_2D(r_km,ele, instrumentheight=0.0):
    a = 6.371e3 # Earth's Radius
    ae = 4 / 3 * a # Effective Earth's radius
    h = beam_height_2D(r_km, ele, instrumentheight) # Beam Height
    br = ae * np.arcsin(r_km[:,np.newaxis].dot(np.cos(ele * np.pi / 180)[np.newaxis,:]) / (ae + h))
    return br,h

def dis_angle_to_2Dxy(r,theta):
    X = r[:,np.newaxis].dot(np.cos(theta[np.newaxis,:] * np.pi / 180))
    Y = r[:,np.newaxis].dot(np.sin(theta[np.newaxis,:] * np.pi / 180))
    return X,Y

# ------------------------------------------------
# colorbars functions

# snr
def snr_cmap():
    cdict11= {'red':  ((  0.0, 150/255, 150/255),
                    ( 2/19, 207/255, 207/255),
                    ( 6/19,  67/255,  67/255),
                    ( 7/19, 111/255, 111/255),
                    ( 8/19,  53/255,  17/255),
                    (11/19,   9/255,   9/255),
                    (12/19,     1.0,     1.0),
                    (14/19,     1.0,     1.0),
                    (16/19, 113/255,     1.0),
                    (17/19,     1.0,     1.0),
                    (18/19, 225/255, 178/255),
                    (  1.0,  99/255,  99/255)),

            'green': ((  0.0, 145/255, 145/255),
                    ( 2/19, 210/255, 210/255),
                    ( 6/19,  94/255,  94/255),
                    ( 7/19, 214/255, 214/255),
                    ( 8/19, 214/255, 214/255),
                    (11/19,  94/255,  94/255),
                    (12/19, 226/255, 226/255),
                    (14/19, 128/255,     0.0),
                    (16/19,     0.0,     1.0),
                    (17/19, 146/255, 117/255),
                    (18/19,     0.0,     0.0),
                    (  1.0,     0.0,     0.0)),

            'blue':  ((  0.0,  83/255,  83/255),
                    ( 2/19, 180/255, 180/255),
                    ( 4/19, 180/255, 180/255),
                    ( 6/19, 159/255, 159/255),
                    ( 7/19, 232/255, 232/255),
                    ( 8/19,  91/255,  24/255),
                    (12/19,     0.0,     0.0),
                    (16/19,     0.0,     1.0),
                    (17/19,     1.0,     1.0),
                    (18/19, 227/255,     1.0),
                    (  1.0, 214/255, 214/255))
            }
    cmap = colors.LinearSegmentedColormap(name='radar_NEXRAD_Zhh', segmentdata=cdict11)
    
    return cmap