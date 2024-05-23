#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:13:55 2024

@author: aayush
"""

import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

sns.set_context("talk")

#%%
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/master_df_point.pkl', 'rb') as file:
    master_df_point =  pickle.load(file)

#%%
def plot_binned_angle_data(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)

    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])['angle'].mean().reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)

    # Filter out a specific dataset if needed
    final_data = grouped[grouped['Dataset'] != 6]

    # Plotting the data
    plt.figure(figsize=(10, 6))
    lineplot = sns.lineplot(
        data=final_data,
        x='Bin_Center',
        y='angle',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci='sd'  # Uses standard deviation for the confidence intervals
    )
    handles, labels = lineplot.get_legend_handles_labels()
    new_labels = ['Unloaded', 'Loaded']
    lineplot.legend(handles, new_labels)
    #plt.axhline(y=180, color='gray', linestyle='--')  # Adds a horizontal line at y=0
    plt.xlim(-100,100)
    plt.xlabel("Percentage of Flexion [%]")
    plt.ylabel("Average Angle [°]")
    plt.title("Angle between the long axis of tibia and femur segments")
    plt.grid(True)
    #plt.savefig('resutl_angle_modification.svg', dpi=300)
    plt.tight_layout()
    plt.show()
    
    
# Example usage
plot_binned_angle_data(modified_angle_df, 10)

#%%
# lets split the data and plot side by side without aggregation 
plt.clf()
fg = sns.relplot(
    data=modified_angle_df, 
    x="Percent Flexed", 
    y="angle",
    #hue="Dataset", 
    col="Condition", 
    kind="line",
    facet_kws={'sharey': True, 'sharex': True},  # Adjust axis sharing as needed
    style="Dataset",
    markers =True
)

fg.refline(y=180)
fg.set_axis_labels("Percentage of flexion [%]", "Angle[°]")
plt.savefig('angle_non_agg.svg', dpi=300)


#%%
plt.clf()
fg = sns.relplot(
    data=master_df_point, 
    x="Percent Flexed", 
    y="Relative Norm",
    #hue="Dataset", 
    col="Condition", 
    kind="line",
    facet_kws={'sharey': True, 'sharex': True},  # Adjust axis sharing as needed
    style="Dataset",
    markers =True
)

fg.refline(y=0)
fg.set_axis_labels("Percentage of flexion [%]", "Euclidean Distance[mm]")
plt.savefig('norm_non_agg.svg', dpi=300)

#%%
# ok, trying to draw sequence diagram using python 
# for tutorial refer to this: https://mrsd.readthedocs.io/en/latest/flash/index.html
import matplotlib.pyplot 
import mrsd
import copy 
figure, plot = matplotlib.pyplot.subplots(figsize=(6,4), tight_layout=True)
# this creates an empty sequence diagram 
diagram = mrsd.Diagram(
    plot, ["RF", "$G_{slice}$", "$G_{phase}$", "$G_{readout}$", "Signal"])


pulse = mrsd.RFPulse(2, 1, center=0) # duration of 2, magnitude of 1 (always between -1 and 1) and center of 0
slice_selection = mrsd.Gradient(pulse.duration, 0.5, ramp=0.1, center=pulse.center) # amplitude is 0.5, centered on the pulse. this creates a flat top gradient

# once we create these two events, we add it to their respective channels : 
diagram.add("RF", pulse)
diagram.add("$G_{slice}$", slice_selection)

# this is to add objects to the diagram directly 
'''
pulse = diagram.rf_pulse("RF", 2, 1, center=0)
slice_selection = diagram.gradient(
    "$G_{slice}$", pulse.duration, 0.5, center=pulse.center)
'''

# we can create slice selective pulse directly too, because it is common. such a pulse is a rf pulse that is played concurrently with a gradient on the slice axis 
'''
pulse, slice_selection = diagram.selective_pulse(
    "RF", "$G_{slice}$", 2, gradient_amplitude=0.5, ramp=0.1, center=0)
'''
# the two things above are mutually exclusive, either we add them one by one, or we do the thing together. 

# now we add the readouts. readout comprises three events: the ADC being switched on and off, the Echo, and the readout gradient. 
#  Note that the ADC object takes an extra parameter, ec: this is passed to matplotlib and can be used to change the aspect of the drawn object (color, line style, transparency, etc.).
TE = 4 
d_readout = 2
adc = mrsd.ADC(d_readout, center=TE, ec="0.5")
echo = mrsd.Echo(d_readout, 1, center=adc.center)
readout = mrsd.Gradient(d_readout, 1, ramp=0.1, center=adc.center)


diagram.add("Signal", adc)
diagram.add("Signal", echo)
diagram.add("$G_{readout}$", readout)

# there are convenience functions to do this at once. but for now, this is wokring just fine. create diagram, create event variables and just add them.. 
# what is not working is the resolution 

# now to add the phase encoding 

d_encoding = 1

phase_encoding = mrsd.MultiGradient(d_encoding, 1, 0.1, end=readout.begin)
diagram.add("$G_{phase}$", phase_encoding)


#matplotlib.pyplot.savefig('pulse_diagram_test.png', dpi=300)

# upto here, what we have is a really simplified basic unit of the sequence diagram. but now we need to add multiple gradients fot the full sequence to actually work. 
# this means we need a spoiler on the slice gradient, as well as a thing on the readout. these are called Encoding gradients

# prephasing lobe of the readout gradient 
readout_prephasing = readout.adapt(d_encoding, -0.5, 0.1, end=readout.begin) # start at the phase encoding, end at the readout begin , negative half amp with a ramp of 0.1 
diagram.add("$G_{readout}$", readout_prephasing)

slice_rewinding = slice_selection.adapt(pulse.end, -0.5,0.1, center=phase_encoding.begin)
diagram.add("$G_{slice}$", slice_rewinding)


# now that the unit is completed in its fullest sense, we can now annotate and make copies 

TR = 10
diagram.interval(0,TE, -1.5, 'TE') # apparantly we can go beyond -1, i guess for these cases. it makes sense when we look at the -2.5 
diagram.interval(0, TR, -2.5, "TR")

diagram.annotate("RF", 0.2, 1, r"$\alpha = 8°$") # im guessing 0.2 is slightly after 0, so its the time point, and 1 would be the mag

#diagram.annotate("$G_{phase}$", phase_encoding.end, 0.5, r"$\uparrow$") # direction of phase encoding. leaving it off for now 

# now to create copies of the rf and gslice to show the start of next rep, by copying and moving them 
diagram.add("RF", copy.copy(pulse).move(TR))
diagram.add("$G_{slice}$", copy.copy(slice_selection).move(TR))

diagram.annotate("RF", TR+0.2, 1, r"$\alpha$")


slice_spoiler = slice_selection.adapt(1, 1,  ramp=0.1, end=7) # just a dumb trial and error i got this to look ok. 
diagram.add("$G_{slice}$", slice_spoiler)


diagram.annotate("$G_{slice}$", 6, 1.2, "$_{spoiler}$")

diagram.annotate("$G_{slice}$", slice_rewinding.begin, -0.9, "$_{rewinder}$")

plt.savefig("flash_seq.png", dpi=300)