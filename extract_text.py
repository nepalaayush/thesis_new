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


mytext_W = """2024-02-23 15:16:35,185 pymri        INFO     00: Reconstructing Frame using 2044 spokes for angle -38 deg. Avg slope: 4.00°/s +/- 4.79°/s
2024-02-23 15:16:37,487 pymri        INFO     01: Reconstructing Frame using 1147 spokes for angle -37 deg. Avg slope: 12.00°/s +/- 4.00°/s
2024-02-23 15:16:39,086 pymri        INFO     02: Reconstructing Frame using 860 spokes for angle -36 deg. Avg slope: 16.12°/s +/- 3.73°/s
2024-02-23 15:16:40,449 pymri        INFO     03: Reconstructing Frame using 748 spokes for angle -35 deg. Avg slope: 18.43°/s +/- 3.99°/s
2024-02-23 15:16:41,793 pymri        INFO     04: Reconstructing Frame using 664 spokes for angle -34 deg. Avg slope: 20.59°/s +/- 4.14°/s
2024-02-23 15:16:43,055 pymri        INFO     05: Reconstructing Frame using 603 spokes for angle -33 deg. Avg slope: 22.44°/s +/- 4.18°/s
2024-02-23 15:16:44,584 pymri        INFO     06: Reconstructing Frame using 561 spokes for angle -32 deg. Avg slope: 24.14°/s +/- 4.08°/s
2024-02-23 15:16:46,077 pymri        INFO     07: Reconstructing Frame using 547 spokes for angle -31 deg. Avg slope: 25.33°/s +/- 3.85°/s
2024-02-23 15:16:47,659 pymri        INFO     08: Reconstructing Frame using 547 spokes for angle -30 deg. Avg slope: 25.17°/s +/- 3.94°/s
2024-02-23 15:16:49,307 pymri        INFO     09: Reconstructing Frame using 611 spokes for angle -29 deg. Avg slope: 22.52°/s +/- 5.46°/s
2024-02-23 15:16:50,719 pymri        INFO     10: Reconstructing Frame using 757 spokes for angle -28 deg. Avg slope: 18.27°/s +/- 6.72°/s
2024-02-23 15:16:52,079 pymri        INFO     11: Reconstructing Frame using 879 spokes for angle -27 deg. Avg slope: 15.67°/s +/- 6.25°/s
2024-02-23 15:16:53,583 pymri        INFO     12: Reconstructing Frame using 904 spokes for angle -26 deg. Avg slope: 15.28°/s +/- 5.11°/s
2024-02-23 15:16:55,166 pymri        INFO     13: Reconstructing Frame using 893 spokes for angle -25 deg. Avg slope: 15.42°/s +/- 4.08°/s
2024-02-23 15:16:56,650 pymri        INFO     14: Reconstructing Frame using 899 spokes for angle -24 deg. Avg slope: 15.34°/s +/- 3.52°/s
2024-02-23 15:16:58,132 pymri        INFO     15: Reconstructing Frame using 882 spokes for angle -23 deg. Avg slope: 15.63°/s +/- 3.41°/s
2024-02-23 15:16:59,559 pymri        INFO     16: Reconstructing Frame using 892 spokes for angle -22 deg. Avg slope: 15.45°/s +/- 3.87°/s
2024-02-23 15:17:00,980 pymri        INFO     17: Reconstructing Frame using 968 spokes for angle -21 deg. Avg slope: 14.28°/s +/- 4.23°/s
2024-02-23 15:17:02,443 pymri        INFO     18: Reconstructing Frame using 1081 spokes for angle -20 deg. Avg slope: 12.77°/s +/- 4.48°/s
2024-02-23 15:17:04,023 pymri        INFO     19: Reconstructing Frame using 1074 spokes for angle -19 deg. Avg slope: 12.85°/s +/- 4.40°/s
2024-02-23 15:17:05,659 pymri        INFO     20: Reconstructing Frame using 987 spokes for angle -18 deg. Avg slope: 14.00°/s +/- 4.41°/s
2024-02-23 15:17:07,124 pymri        INFO     21: Reconstructing Frame using 937 spokes for angle -17 deg. Avg slope: 14.71°/s +/- 4.85°/s
2024-02-23 15:17:08,563 pymri        INFO     22: Reconstructing Frame using 967 spokes for angle -16 deg. Avg slope: 14.23°/s +/- 5.56°/s
2024-02-23 15:17:10,042 pymri        INFO     23: Reconstructing Frame using 1053 spokes for angle -15 deg. Avg slope: 12.94°/s +/- 5.44°/s
2024-02-23 15:17:11,562 pymri        INFO     24: Reconstructing Frame using 1086 spokes for angle -14 deg. Avg slope: 10.99°/s +/- 5.08°/s
2024-02-23 15:17:13,158 pymri        INFO     25: Reconstructing Frame using 930 spokes for angle -13 deg. Avg slope: 9.21°/s +/- 4.60°/s
2024-02-23 15:17:14,607 pymri        INFO     26: Reconstructing Frame using 571 spokes for angle -12 deg. Avg slope: 8.79°/s +/- 4.24°/s
2024-02-23 15:17:16,148 pymri        INFO     27: Reconstructing Frame using 286 spokes for angle -11 deg. Avg slope: 8.09°/s +/- 4.04°/s
2024-02-23 15:17:17,498 pymri        INFO     28: Reconstructing Frame using 121 spokes for angle -10 deg. Avg slope: 6.89°/s +/- 3.29°/s""" 


mytext_NW = """ 2024-02-23 15:21:06,371 pymri        INFO     00: Reconstructing Frame using 1676 spokes for angle -37 deg. Avg slope: 7.68°/s +/- 7.45°/s
2024-02-23 15:21:08,683 pymri        INFO     01: Reconstructing Frame using 678 spokes for angle -36 deg. Avg slope: 20.36°/s +/- 3.05°/s
2024-02-23 15:21:10,023 pymri        INFO     02: Reconstructing Frame using 588 spokes for angle -35 deg. Avg slope: 23.42°/s +/- 2.76°/s
2024-02-23 15:21:11,292 pymri        INFO     03: Reconstructing Frame using 549 spokes for angle -34 deg. Avg slope: 25.07°/s +/- 2.79°/s
2024-02-23 15:21:12,654 pymri        INFO     04: Reconstructing Frame using 536 spokes for angle -33 deg. Avg slope: 25.47°/s +/- 3.05°/s
2024-02-23 15:21:13,938 pymri        INFO     05: Reconstructing Frame using 558 spokes for angle -32 deg. Avg slope: 24.67°/s +/- 3.63°/s
2024-02-23 15:21:15,241 pymri        INFO     06: Reconstructing Frame using 594 spokes for angle -31 deg. Avg slope: 23.39°/s +/- 4.15°/s
2024-02-23 15:21:16,557 pymri        INFO     07: Reconstructing Frame using 625 spokes for angle -30 deg. Avg slope: 22.06°/s +/- 4.40°/s
2024-02-23 15:21:17,871 pymri        INFO     08: Reconstructing Frame using 681 spokes for angle -29 deg. Avg slope: 20.26°/s +/- 4.74°/s
2024-02-23 15:21:19,250 pymri        INFO     09: Reconstructing Frame using 744 spokes for angle -28 deg. Avg slope: 18.56°/s +/- 4.94°/s
2024-02-23 15:21:20,626 pymri        INFO     10: Reconstructing Frame using 831 spokes for angle -27 deg. Avg slope: 16.62°/s +/- 5.50°/s
2024-02-23 15:21:22,014 pymri        INFO     11: Reconstructing Frame using 918 spokes for angle -26 deg. Avg slope: 15.00°/s +/- 5.39°/s
2024-02-23 15:21:23,471 pymri        INFO     12: Reconstructing Frame using 1010 spokes for angle -25 deg. Avg slope: 13.61°/s +/- 4.89°/s
2024-02-23 15:21:25,094 pymri        INFO     13: Reconstructing Frame using 1139 spokes for angle -24 deg. Avg slope: 12.15°/s +/- 4.26°/s
2024-02-23 15:21:26,845 pymri        INFO     14: Reconstructing Frame using 1154 spokes for angle -23 deg. Avg slope: 11.93°/s +/- 3.52°/s
2024-02-23 15:21:28,581 pymri        INFO     15: Reconstructing Frame using 1137 spokes for angle -22 deg. Avg slope: 12.13°/s +/- 3.35°/s
2024-02-23 15:21:30,305 pymri        INFO     16: Reconstructing Frame using 1141 spokes for angle -21 deg. Avg slope: 12.12°/s +/- 3.54°/s
2024-02-23 15:21:31,942 pymri        INFO     17: Reconstructing Frame using 1147 spokes for angle -20 deg. Avg slope: 12.04°/s +/- 3.95°/s
2024-02-23 15:21:33,570 pymri        INFO     18: Reconstructing Frame using 1148 spokes for angle -19 deg. Avg slope: 12.07°/s +/- 4.21°/s
2024-02-23 15:21:35,231 pymri        INFO     19: Reconstructing Frame using 1092 spokes for angle -18 deg. Avg slope: 12.65°/s +/- 4.26°/s
2024-02-23 15:21:36,978 pymri        INFO     20: Reconstructing Frame using 1019 spokes for angle -17 deg. Avg slope: 13.48°/s +/- 4.66°/s
2024-02-23 15:21:38,613 pymri        INFO     21: Reconstructing Frame using 948 spokes for angle -16 deg. Avg slope: 14.42°/s +/- 4.75°/s
2024-02-23 15:21:40,227 pymri        INFO     22: Reconstructing Frame using 925 spokes for angle -15 deg. Avg slope: 14.47°/s +/- 4.63°/s
2024-02-23 15:21:41,837 pymri        INFO     23: Reconstructing Frame using 950 spokes for angle -14 deg. Avg slope: 12.90°/s +/- 5.29°/s
2024-02-23 15:21:43,425 pymri        INFO     24: Reconstructing Frame using 900 spokes for angle -13 deg. Avg slope: 11.26°/s +/- 5.36°/s
2024-02-23 15:21:44,890 pymri        INFO     25: Reconstructing Frame using 713 spokes for angle -12 deg. Avg slope: 10.68°/s +/- 4.96°/s
2024-02-23 15:21:46,185 pymri        INFO     26: Reconstructing Frame using 460 spokes for angle -11 deg. Avg slope: 10.26°/s +/- 5.24°/s
2024-02-23 15:21:47,324 pymri        INFO     27: Reconstructing Frame using 226 spokes for angle -10 deg. Avg slope: 9.56°/s +/- 5.74°/s """ 
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

angles_NW_pos_ai1, avg_slopes_NW_pos_ai1, std_devs_NW_pos_ai1 = extract_data(mytext_NW)

angles_W_pos_ai1, avg_slopes_W_pos_ai1, std_devs_W_pos_ai1 = extract_data(mytext_W)
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
    plt.savefig('MK_both_slope_vs_angle_pos_ai1.svg')

plot_the_slope(angles_NW_pos_ai1, avg_slopes_NW_pos_ai1, std_devs_NW_pos_ai1, angles_W_pos_ai1, avg_slopes_W_pos_ai1, std_devs_W_pos_ai1 )

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
    plt.savefig('MK_both_std_vs_angle_pos_ai1.svg')
    

plot_the_std(angles_W_pos_ai1, std_devs_W_pos_ai1, angles_NW_pos_ai1, std_devs_NW_pos_ai1)    