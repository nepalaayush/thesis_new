#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:59:52 2024

@author: aayush
"""
import os 
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
import numpy as np 
import matplotlib.pylab as plt 


mytext_W = """

2024-03-04 14:06:40,536 pymri        INFO     02: Reconstructing Frame using 681 spokes for angle -37 deg. Avg slope: -15.79°/s +/- 6.32°/s
2024-03-04 14:06:49,619 pymri        INFO     03: Reconstructing Frame using 915 spokes for angle -35 deg. Avg slope: -14.29°/s +/- 4.22°/s
2024-03-04 14:06:59,863 pymri        INFO     04: Reconstructing Frame using 989 spokes for angle -33 deg. Avg slope: -13.81°/s +/- 3.81°/s
2024-03-04 14:07:10,957 pymri        INFO     05: Reconstructing Frame using 1008 spokes for angle -31 deg. Avg slope: -13.31°/s +/- 2.98°/s
2024-03-04 14:07:21,915 pymri        INFO     06: Reconstructing Frame using 1405 spokes for angle -29 deg. Avg slope: -9.61°/s +/- 3.04°/s
2024-03-04 14:07:35,470 pymri        INFO     07: Reconstructing Frame using 1424 spokes for angle -27 deg. Avg slope: -9.58°/s +/- 2.89°/s
2024-03-04 14:07:49,173 pymri        INFO     08: Reconstructing Frame using 1586 spokes for angle -25 deg. Avg slope: -8.70°/s +/- 3.26°/s
2024-03-04 14:08:03,713 pymri        INFO     09: Reconstructing Frame using 1327 spokes for angle -23 deg. Avg slope: -10.40°/s +/- 3.03°/s
2024-03-04 14:08:16,981 pymri        INFO     10: Reconstructing Frame using 1268 spokes for angle -21 deg. Avg slope: -10.92°/s +/- 2.66°/s
2024-03-04 14:08:29,528 pymri        INFO     11: Reconstructing Frame using 916 spokes for angle -19 deg. Avg slope: -14.76°/s +/- 4.75°/s
""" 


mytext_NW = """ 
2024-03-04 14:15:43,856 pymri        INFO     00: Reconstructing Frame using 77 spokes for angle -37 deg. Avg slope: -14.10°/s +/- 5.43°/s
2024-03-04 14:15:50,208 pymri        INFO     01: Reconstructing Frame using 537 spokes for angle -35 deg. Avg slope: -9.31°/s +/- 4.49°/s
2024-03-04 14:15:58,121 pymri        INFO     02: Reconstructing Frame using 1012 spokes for angle -33 deg. Avg slope: -9.66°/s +/- 4.03°/s
2024-03-04 14:16:09,287 pymri        INFO     03: Reconstructing Frame using 1263 spokes for angle -31 deg. Avg slope: -10.68°/s +/- 3.36°/s
2024-03-04 14:16:21,643 pymri        INFO     04: Reconstructing Frame using 1464 spokes for angle -29 deg. Avg slope: -9.45°/s +/- 2.92°/s
2024-03-04 14:16:35,369 pymri        INFO     05: Reconstructing Frame using 1424 spokes for angle -27 deg. Avg slope: -9.66°/s +/- 2.27°/s
2024-03-04 14:16:48,856 pymri        INFO     06: Reconstructing Frame using 1804 spokes for angle -25 deg. Avg slope: -7.65°/s +/- 2.21°/s
2024-03-04 14:17:04,556 pymri        INFO     07: Reconstructing Frame using 1848 spokes for angle -23 deg. Avg slope: -7.44°/s +/- 2.48°/s
2024-03-04 14:17:20,475 pymri        INFO     08: Reconstructing Frame using 1555 spokes for angle -21 deg. Avg slope: -8.83°/s +/- 3.03°/s
2024-03-04 14:17:35,124 pymri        INFO     09: Reconstructing Frame using 1102 spokes for angle -19 deg. Avg slope: -12.17°/s +/- 2.97°/s
""" 
#%%
def extract_data(text):
    angles = []
    avg_slopes = []
    std_devs = []

    for line in text.splitlines():
        if "angle" in line:
            angle_str = line.split("angle ")[1].split(" deg")[0]
            angles.append(int(angle_str))

        if "°/s" in line:
            slope_parts = line.split("Avg slope: ")[1].split(" ")
            # Extract the numerical part of the average slope, removing the trailing unit
            avg_slope_str = slope_parts[0].rstrip("°/s")
            avg_slopes.append(float(avg_slope_str))

            # Extract the numerical part of the standard deviation, removing the trailing unit
            std_dev_str = slope_parts[2].rstrip("°/s")
            std_devs.append(float(std_dev_str))

    return np.array(angles), np.array(avg_slopes), np.array(std_devs)

angles_NW_pos, avg_slopes_NW_pos, std_devs_NW_pos = extract_data(mytext_NW)

angles_W_pos, avg_slopes_W_pos, std_devs_W_pos = extract_data(mytext_W)

rom_W = (max(angles_W_pos) - min(angles_W_pos))
rom_NW = (max(angles_NW_pos) - min(angles_NW_pos))

t = 2 # the time required to cover one range either upwards or downwards. calculated by using 60 / 120 * 4 .. one rom in 4 beats with 120 beats / min 
omega_W = rom_W / t 
omega_NW = rom_NW / t 
#%%

def plot_the_slope (angle1,slope1,std1, angle2,slope2,std2): 
    
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,6))
    
    
    ax.plot(angle1, slope1, label='loaded', color='blue')
    ax.plot(angle2, slope2, label='unloaded', color='orange')
    ax.set_title('Average Rate of change of slope vs angle ')
    ax.set_xlabel('Angle at which a particular frame was reconstructed [degrees]')
    ax.set_ylabel('angular velocity of the tibia at a given angle [degree/sec]')
    #ax.set_xticks(angle1)
    ax.fill_between(angle1, 
                    slope1 - std1, 
                    slope1 + std1, 
                    color='blue', alpha=0.1)
    ax.fill_between(angle2, 
                    slope2 - std2, 
                    slope2 + std2, 
                    color='orange', alpha=0.1)
    ax.axhline(-omega_NW, color='k', linestyle='--', label='Theoretical Speed ')
    ax.grid()
    ax.legend()
    plt.savefig('AN_01.03_both_slope_vs_angle.svg')

plot_the_slope(angles_W_pos, avg_slopes_W_pos, std_devs_W_pos, angles_NW_pos, avg_slopes_NW_pos, std_devs_NW_pos)

#%%

def plot_the_std(angle1, std1, angle2, std2):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12,6))
    
    
    ax.plot(angle1, std1, label='loaded', color='blue')
    ax.plot(angle2, std2, label='unloaded', color='orange')
    ax.set_title('Plotting the standard deviation of slope vs angle. ')
    ax.set_xlabel('Angle at which a particular frame was reconstructed [degrees]')
    ax.set_ylabel('std of the angular velocity of the tibia at a given angle [degree/sec]')
    ax.set_xticks(angle1)
    
    ax.grid()
    ax.legend()
    plt.savefig('AN_01.03_both_std_vs_angle.svg')
    

plot_the_std(angles_W_pos, std_devs_W_pos, angles_NW_pos, std_devs_NW_pos)    