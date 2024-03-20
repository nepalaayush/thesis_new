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

024-03-01 18:11:57,488 pymri        INFO     00: Reconstructing Frame using 42 spokes for angle -47 deg. Avg slope: 9.58°/s +/- 5.09°/s
2024-03-01 18:12:04,801 pymri        INFO     01: Reconstructing Frame using 255 spokes for angle -45 deg. Avg slope: 9.55°/s +/- 5.25°/s
2024-03-01 18:12:10,969 pymri        INFO     02: Reconstructing Frame using 687 spokes for angle -43 deg. Avg slope: 13.58°/s +/- 6.74°/s
2024-03-01 18:12:19,703 pymri        INFO     03: Reconstructing Frame using 751 spokes for angle -41 deg. Avg slope: 18.14°/s +/- 5.86°/s
2024-03-01 18:12:29,013 pymri        INFO     04: Reconstructing Frame using 643 spokes for angle -39 deg. Avg slope: 21.46°/s +/- 3.66°/s
2024-03-01 18:12:37,691 pymri        INFO     05: Reconstructing Frame using 633 spokes for angle -37 deg. Avg slope: 21.72°/s +/- 3.57°/s
2024-03-01 18:12:46,144 pymri        INFO     06: Reconstructing Frame using 669 spokes for angle -35 deg. Avg slope: 20.71°/s +/- 3.34°/s
2024-03-01 18:12:54,938 pymri        INFO     07: Reconstructing Frame using 665 spokes for angle -33 deg. Avg slope: 20.60°/s +/- 3.97°/s
2024-03-01 18:13:03,586 pymri        INFO     08: Reconstructing Frame using 709 spokes for angle -31 deg. Avg slope: 19.55°/s +/- 3.39°/s
2024-03-01 18:13:12,619 pymri        INFO     09: Reconstructing Frame using 735 spokes for angle -29 deg. Avg slope: 18.80°/s +/- 4.10°/s
2024-03-01 18:13:21,833 pymri        INFO     10: Reconstructing Frame using 706 spokes for angle -27 deg. Avg slope: 19.48°/s +/- 3.43°/s
2024-03-01 18:13:30,654 pymri        INFO     11: Reconstructing Frame using 855 spokes for angle -25 deg. Avg slope: 16.10°/s +/- 4.15°/s
2024-03-01 18:13:40,495 pymri        INFO     12: Reconstructing Frame using 1125 spokes for angle -23 deg. Avg slope: 12.27°/s +/- 4.03°/s
2024-03-01 18:13:52,201 pymri        INFO     13: Reconstructing Frame using 1477 spokes for angle -21 deg. Avg slope: 9.35°/s +/- 3.38°/s
2024-03-01 18:14:06,013 pymri        INFO     14: Reconstructing Frame using 1598 spokes for angle -19 deg. Avg slope: 8.62°/s +/- 2.90°/s

""" 


mytext_NW = """ 
2024-03-04 13:27:55,953 pymri        INFO     00: Reconstructing Frame using 43 spokes for angle -47 deg. Avg slope: -9.26°/s +/- 3.90°/s
2024-03-04 13:28:01,944 pymri        INFO     01: Reconstructing Frame using 274 spokes for angle -45 deg. Avg slope: -8.84°/s +/- 4.87°/s
2024-03-04 13:28:08,368 pymri        INFO     02: Reconstructing Frame using 815 spokes for angle -43 deg. Avg slope: -11.48°/s +/- 5.33°/s
2024-03-04 13:28:18,336 pymri        INFO     03: Reconstructing Frame using 969 spokes for angle -41 deg. Avg slope: -14.00°/s +/- 4.05°/s
2024-03-04 13:28:29,148 pymri        INFO     04: Reconstructing Frame using 1008 spokes for angle -39 deg. Avg slope: -13.68°/s +/- 3.61°/s
2024-03-04 13:28:40,391 pymri        INFO     05: Reconstructing Frame using 968 spokes for angle -37 deg. Avg slope: -14.21°/s +/- 4.06°/s
2024-03-04 13:28:51,331 pymri        INFO     06: Reconstructing Frame using 1005 spokes for angle -35 deg. Avg slope: -13.79°/s +/- 3.72°/s
2024-03-04 13:29:02,191 pymri        INFO     07: Reconstructing Frame using 886 spokes for angle -33 deg. Avg slope: -15.52°/s +/- 3.56°/s
2024-03-04 13:29:12,788 pymri        INFO     08: Reconstructing Frame using 781 spokes for angle -31 deg. Avg slope: -17.81°/s +/- 4.18°/s
2024-03-04 13:29:22,466 pymri        INFO     09: Reconstructing Frame using 692 spokes for angle -29 deg. Avg slope: -19.88°/s +/- 4.21°/s
2024-03-04 13:29:31,446 pymri        INFO     10: Reconstructing Frame using 670 spokes for angle -27 deg. Avg slope: -20.41°/s +/- 5.35°/s
2024-03-04 13:29:40,140 pymri        INFO     11: Reconstructing Frame using 635 spokes for angle -25 deg. Avg slope: -21.85°/s +/- 5.22°/s
2024-03-04 13:29:48,693 pymri        INFO     12: Reconstructing Frame using 550 spokes for angle -23 deg. Avg slope: -25.21°/s +/- 4.52°/s
2024-03-04 13:29:56,937 pymri        INFO     13: Reconstructing Frame using 506 spokes for angle -21 deg. Avg slope: -27.11°/s +/- 3.79°/s
2024-03-04 13:30:04,942 pymri        INFO     14: Reconstructing Frame using 627 spokes for angle -19 deg. Avg slope: -22.24°/s +/- 4.40°/s
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
    #ax.axhline(-omega_NW, color='k', linestyle='--', label='Theoretical Speed ')
    ax.grid()
    ax.legend()
    plt.savefig('MK_01.03_pos_vs_neg_NW_slope_vs_angle_reps.svg')

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
    plt.savefig('MK_01.03_pos_vs_neg_NW_sd.svg')
    

plot_the_std(angles_W_pos, std_devs_W_pos, angles_NW_pos, std_devs_NW_pos)    