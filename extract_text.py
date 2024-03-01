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
2024-02-26 16:18:47,406 pymri        INFO     00: Reconstructing Frame using 38 spokes for angle -167 deg. Avg slope: 15.88°/s +/- 9.20°/s
2024-02-26 16:18:53,013 pymri        INFO     01: Reconstructing Frame using 295 spokes for angle -165 deg. Avg slope: 13.94°/s +/- 9.25°/s
2024-02-26 16:18:59,529 pymri        INFO     02: Reconstructing Frame using 928 spokes for angle -163 deg. Avg slope: 12.24°/s +/- 8.43°/s
2024-02-26 16:19:10,099 pymri        INFO     03: Reconstructing Frame using 1406 spokes for angle -161 deg. Avg slope: 16.21°/s +/- 9.52°/s
2024-02-26 16:19:23,572 pymri        INFO     04: Reconstructing Frame using 1273 spokes for angle -159 deg. Avg slope: 22.75°/s +/- 13.72°/s
2024-02-26 16:19:36,448 pymri        INFO     05: Reconstructing Frame using 934 spokes for angle -157 deg. Avg slope: 34.01°/s +/- 16.15°/s
2024-02-26 16:19:47,023 pymri        INFO     06: Reconstructing Frame using 686 spokes for angle -155 deg. Avg slope: 47.50°/s +/- 11.86°/s
2024-02-26 16:19:56,039 pymri        INFO     07: Reconstructing Frame using 613 spokes for angle -153 deg. Avg slope: 53.13°/s +/- 12.77°/s
2024-02-26 16:20:04,625 pymri        INFO     08: Reconstructing Frame using 611 spokes for angle -151 deg. Avg slope: 54.44°/s +/- 15.87°/s
2024-02-26 16:20:13,386 pymri        INFO     09: Reconstructing Frame using 600 spokes for angle -149 deg. Avg slope: 55.34°/s +/- 15.73°/s
2024-02-26 16:20:21,940 pymri        INFO     10: Reconstructing Frame using 613 spokes for angle -147 deg. Avg slope: 54.98°/s +/- 15.23°/s
2024-02-26 16:20:30,441 pymri        INFO     11: Reconstructing Frame using 623 spokes for angle -145 deg. Avg slope: 52.95°/s +/- 15.45°/s
2024-02-26 16:20:39,264 pymri        INFO     12: Reconstructing Frame using 666 spokes for angle -143 deg. Avg slope: 49.69°/s +/- 16.43°/s
2024-02-26 16:20:48,127 pymri        INFO     13: Reconstructing Frame using 733 spokes for angle -141 deg. Avg slope: 46.05°/s +/- 15.90°/s
2024-02-26 16:20:57,367 pymri        INFO     14: Reconstructing Frame using 797 spokes for angle -139 deg. Avg slope: 42.04°/s +/- 15.12°/s
2024-02-26 16:21:07,085 pymri        INFO     15: Reconstructing Frame using 827 spokes for angle -137 deg. Avg slope: 40.54°/s +/- 12.51°/s
2024-02-26 16:21:17,182 pymri        INFO     16: Reconstructing Frame using 955 spokes for angle -135 deg. Avg slope: 35.05°/s +/- 13.41°/s
2024-02-26 16:21:27,792 pymri        INFO     17: Reconstructing Frame using 1010 spokes for angle -133 deg. Avg slope: 33.05°/s +/- 12.11°/s
2024-02-26 16:21:38,908 pymri        INFO     18: Reconstructing Frame using 1121 spokes for angle -131 deg. Avg slope: 29.87°/s +/- 11.32°/s
2024-02-26 16:21:50,866 pymri        INFO     19: Reconstructing Frame using 1687 spokes for angle -129 deg. Avg slope: 19.54°/s +/- 11.92°/s
2024-02-26 16:22:05,869 pymri        INFO     20: Reconstructing Frame using 1747 spokes for angle -127 deg. Avg slope: 14.71°/s +/- 9.89°/s""" 


mytext_NW = """ 
2024-02-26 16:05:06,772 pymri        INFO     02: Reconstructing Frame using 709 spokes for angle -167 deg. Avg slope: 17.23°/s +/- 9.72°/s
2024-02-26 16:05:16,322 pymri        INFO     03: Reconstructing Frame using 1146 spokes for angle -165 deg. Avg slope: 20.69°/s +/- 11.99°/s
2024-02-26 16:05:28,859 pymri        INFO     04: Reconstructing Frame using 1007 spokes for angle -163 deg. Avg slope: 29.56°/s +/- 12.12°/s
2024-02-26 16:05:40,649 pymri        INFO     05: Reconstructing Frame using 938 spokes for angle -161 deg. Avg slope: 34.18°/s +/- 15.98°/s
2024-02-26 16:05:51,602 pymri        INFO     06: Reconstructing Frame using 714 spokes for angle -159 deg. Avg slope: 45.89°/s +/- 13.96°/s
2024-02-26 16:06:01,003 pymri        INFO     07: Reconstructing Frame using 656 spokes for angle -157 deg. Avg slope: 50.44°/s +/- 17.45°/s
2024-02-26 16:06:10,245 pymri        INFO     08: Reconstructing Frame using 550 spokes for angle -155 deg. Avg slope: 59.51°/s +/- 11.68°/s
2024-02-26 16:06:18,812 pymri        INFO     09: Reconstructing Frame using 522 spokes for angle -153 deg. Avg slope: 63.11°/s +/- 12.03°/s
2024-02-26 16:06:27,235 pymri        INFO     10: Reconstructing Frame using 515 spokes for angle -151 deg. Avg slope: 65.14°/s +/- 12.56°/s
2024-02-26 16:06:35,934 pymri        INFO     11: Reconstructing Frame using 505 spokes for angle -149 deg. Avg slope: 65.23°/s +/- 13.51°/s
2024-02-26 16:06:44,523 pymri        INFO     12: Reconstructing Frame using 516 spokes for angle -147 deg. Avg slope: 64.13°/s +/- 13.40°/s
2024-02-26 16:06:53,036 pymri        INFO     13: Reconstructing Frame using 531 spokes for angle -145 deg. Avg slope: 61.57°/s +/- 13.41°/s
2024-02-26 16:07:01,301 pymri        INFO     14: Reconstructing Frame using 600 spokes for angle -143 deg. Avg slope: 54.99°/s +/- 16.62°/s
2024-02-26 16:07:10,061 pymri        INFO     15: Reconstructing Frame using 689 spokes for angle -141 deg. Avg slope: 48.71°/s +/- 17.34°/s
2024-02-26 16:07:19,258 pymri        INFO     16: Reconstructing Frame using 749 spokes for angle -139 deg. Avg slope: 43.80°/s +/- 16.92°/s
2024-02-26 16:07:28,819 pymri        INFO     17: Reconstructing Frame using 796 spokes for angle -137 deg. Avg slope: 41.90°/s +/- 14.72°/s
2024-02-26 16:07:38,807 pymri        INFO     18: Reconstructing Frame using 907 spokes for angle -135 deg. Avg slope: 36.95°/s +/- 14.94°/s
2024-02-26 16:07:49,368 pymri        INFO     19: Reconstructing Frame using 1204 spokes for angle -133 deg. Avg slope: 28.77°/s +/- 15.49°/s
2024-02-26 16:08:01,844 pymri        INFO     20: Reconstructing Frame using 1592 spokes for angle -131 deg. Avg slope: 21.13°/s +/- 14.60°/s
2024-02-26 16:08:16,395 pymri        INFO     21: Reconstructing Frame using 1745 spokes for angle -129 deg. Avg slope: 16.33°/s +/- 11.89°/s
2024-02-26 16:08:31,924 pymri        INFO     22: Reconstructing Frame using 1159 spokes for angle -127 deg. Avg slope: 10.92°/s +/- 9.73°/s """ 
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
    ax.grid()
    ax.legend()
    plt.savefig('US_both_slope_vs_angle_pos_ai2.svg')

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
    plt.savefig('US_both_std_vs_angle_pos_ai2.svg')
    

plot_the_std(angles_W_pos, std_devs_W_pos, angles_NW_pos, std_devs_NW_pos)    