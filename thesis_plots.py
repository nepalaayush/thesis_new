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
import pingouin as pg 

sns.set_context("talk")

#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/df_angle.pkl', 'rb') as file:
    df_angle =  pickle.load(file)

#%%
master_df_cut = master_df_point[master_df_point['Dataset'].isin([2,4,5,6,7])]

#%%
voxel_size = 0.48484848484848486
#%%
def plot_bone_data(df, bone, voxel_size):
    
    copy_df = df.copy()
    # Multiply 'Average Cost' by 'voxel_size'
    copy_df['Average Cost'] = copy_df['Average Cost'] * voxel_size
    

    # Group by Dataset, Condition, and Bones and calculate statistics
    grouped = copy_df.groupby(['Dataset', 'Condition', 'Bone'])
    mean_cost = grouped['Average Cost'].mean()
    std_cost = grouped['Average Cost'].std()
    count = grouped['Average Cost'].count()
    sem_cost = std_cost / np.sqrt(count)

    # Combine into a DataFrame
    stats_df = pd.DataFrame({
        'Mean': mean_cost,
        'STD': std_cost,
        'Count': count,
        'SEM': sem_cost
    }).reset_index()

    # Filter data for the selected bone
    bone_data = stats_df[stats_df['Bone'] == bone]

    # Sort the data to match the bar plot order
    bone_data = bone_data.sort_values(['Dataset', 'Condition'])
    
    # Print highest and lowest mean alignment errors
    highest_error = bone_data['Mean'].max()
    lowest_error = bone_data['Mean'].min()
    print(f"Highest mean alignment error: {highest_error} mm")
    print(f"Lowest mean alignment error: {lowest_error} mm")

    # Calculate overall averages for each condition
    averages = bone_data.groupby('Condition')['Mean'].mean()
    for condition, avg in averages.items():
        print(f"Average mean alignment error for {condition}: {avg} mm")
    
    
    
    # Find the row with the highest and lowest mean alignment errors
    highest_error_row = bone_data.loc[bone_data['Mean'].idxmax()]
    lowest_error_row = bone_data.loc[bone_data['Mean'].idxmin()]

    # Print highest and lowest mean alignment errors with SEM
    print(f"Highest mean alignment error: {highest_error_row['Mean']} mm ± {highest_error_row['SEM']} mm")
    print(f"Lowest mean alignment error: {lowest_error_row['Mean']} mm ± {lowest_error_row['SEM']} mm")

    # Calculate and print the overall averages for each condition with SEM
    condition_stats = bone_data.groupby('Condition').agg({'Mean': 'mean', 'SEM': 'mean'})
    for condition, row in condition_stats.iterrows():
        print(f"Average mean alignment error for {condition}: {row['Mean']} mm ± {row['SEM']} mm")
    
    
    
    
    
    
    # Plotting with error bars
    plt.figure(figsize=(10, 6))
    
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}
    
    ax = sns.barplot(data=bone_data, x='Dataset', y='Mean', hue='Condition', palette= custom_palette, ci=None)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=bone_data['SEM'], fmt='none', c='black', capsize=5)
    plt.title(f'Mean alignment error per point for all {bone} datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Alignment Error [mm]')
    custom_labels = [1, 2, 3, 4, 5]
    plt.xticks(ticks=range(5), labels=custom_labels, rotation=45)
    #plt.xticks(rotation=45)
    plt.legend(title='Condition')
    plt.tight_layout()
    #plt.savefig(f'bar_no_1_3_{bone}_both.png', dpi=300)
    plt.show()
    
plot_bone_data(master_df_cost[master_df_cost['Dataset'].isin([2,4,5,6,7])], 'Femur', voxel_size)

#%%

# trying to do some stats here: 
    
from scipy.stats import ttest_rel

def calculate_averages_and_test(df, bone, conditions=('Loaded', 'Unloaded')):
    # Filter by bone
    df = df[df['Bone'] == bone]
    
    # Compute mean of 'Average Cost' for each 'Dataset' and 'Condition'
    mean_costs = df.groupby(['Dataset', 'Condition'])['Average Cost'].mean().reset_index()
    
    # Pivot the data
    pivoted_means = mean_costs.pivot(index='Dataset', columns='Condition', values='Average Cost')
    
    # Drop any rows with missing values to ensure each dataset has both conditions
    pivoted_means.dropna(inplace=True)
    
    # Check if there are enough datasets to perform the test
    if pivoted_means.shape[0] >= 2:
        # Perform a paired t-test
        t_stat, p_value = ttest_rel(pivoted_means[conditions[0]], pivoted_means[conditions[1]])
        print(f"Paired t-test results -- T-statistic: {t_stat}, P-value: {p_value}")
    else:
        print("Not enough data to perform a paired t-test.")

# Example usage, assuming 'master_df_cost' is defined and contains the necessary structure.

calculate_averages_and_test(master_df_cost, 'Tibia')

#%%
#trying to redefine the angle.. 

df_angle['angle'] = 180 - df_angle['angle'] 
    
#%%
''' 
# !!! this is the code to plot the angle , given df_angle in the root directory 
''' 
def plot_binned_angle_data(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    #df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid) # to match with the other plot 
    
    
    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])['angle'].mean().reset_index()
    #grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
    
    # Filter out a specific dataset if needed
    #final_data = grouped[grouped['Dataset'] != 6]

    # Plotting the data
    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}
    
    
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='angle',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci='sd',# Uses standard deviation for the confidence intervals
        palette=custom_palette
    )
    # T-tests for each bin
    t_test_results = []
    for name, group in grouped.groupby('Custom_Bin'):
        loaded = group[group['Condition'] == 'Loaded']['angle']
        unloaded = group[group['Condition'] == 'Unloaded']['angle']
        t_test = pg.ttest(loaded, unloaded, paired=False)
        t_test_results.append((name, t_test['p-val'].values[0]))
    
    #plt.axhline(y=180, color='gray', linestyle='--')  # Adds a horizontal line at y=0
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Average Angle [°]")
    plt.title("Angle between the long axis of tibia and femur segments")
    plt.grid(True)
    # Custom text for indicating Extension and Flexion
    #plt.text(-50, -0.1, 'Extension', ha='center', va='top', fontsize=12, color='black', transform=plt.gca().get_xaxis_transform())
    #plt.text(50, -0.1, 'Flexion', ha='center', va='top', fontsize=12, color='black', transform=plt.gca().get_xaxis_transform())
    #plt.annotate('', xy=(-50, -0.08), xytext=(-100, -0.08), arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'), textcoords=plt.gca().get_xaxis_transform())
    #plt.annotate('', xy=(50, -0.08), xytext=(100, -0.08), arrowprops=dict(arrowstyle='<|-', lw=1.5, color='black'), textcoords=plt.gca().get_xaxis_transform())
    # Text for Extension and Flexion exactly below the x-axis ticks
    #plt.text(-50, -0.05, 'Extension', ha='center', va='top', fontsize=12, color='black', transform=plt.gca().get_xaxis_transform())
    #plt.text(50, -0.05, 'Flexion', ha='center', va='top', fontsize=12, color='black', transform=plt.gca().get_xaxis_transform())
    
   # Arrow for Extension and Flexion
   


   #plt.savefig('inverted_angle.png', dpi=300)
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(t_test_results, columns=['Bin', 'p-value'])
    
# Example usage
nn = plot_binned_angle_data(df_angle, 10)



#%%
# to do the statistical tests. 

def plot_and_test_binned_data(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that include the full range from -100 to 100
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percent Flexed' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)

    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])['Relative Norm'].mean().reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)

    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}

    # Plotting the data
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='Relative Norm',
        hue='Condition',
        marker="o",
        ci='sd',  # Uses standard deviation for the error bars
        palette=custom_palette
    )
    
    # T-tests for each bin
    t_test_results = []
    for name, group in grouped.groupby('Custom_Bin'):
        loaded = group[group['Condition'] == 'Loaded']['Relative Norm']
        unloaded = group[group['Condition'] == 'Unloaded']['Relative Norm']
        t_test = pg.ttest(loaded, unloaded, paired=False)
        t_test_results.append((name, t_test['p-val'].values[0]))

    # Display t-test results
    #for bin_center, p_val in t_test_results:
       # plt.text(bin_center.mid, 1, f'p={p_val:.3f}', color='red')  # Adjust positioning and color as necessary

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Euclidean Distance (mm)")
    plt.title("Variation of distance with respect to flexion-extension cycle")
    plt.grid(True)
    #plt.savefig('distance_no_13.png', dpi=300)
    plt.show()

    return pd.DataFrame(t_test_results, columns=['Bin', 'p-value'])

xx = plot_and_test_binned_data(master_df_cut, 10)

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
# adding the relative Y column from the Disp_Y column 
def calculate_relative_y(df):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Function to calculate relative Y displacement for each group
    def calculate_group_relative_y(group):
        reference_value = group.loc[group['Frame Number'] == 0, 'Disp_Y'].values[0]
        group['Relative_Y'] = group['Disp_Y'] - reference_value
        return group

    # Apply the function to each group and add the new column to the original DataFrame
    df_copy = df_copy.groupby(['Dataset', 'Condition']).apply(calculate_group_relative_y)
    
    return df_copy


master_df_cut = calculate_relative_y(master_df_cut)


#%%
# plotting the relative translation only not norm : can work with any column of choosing, which is nice. 

def plot_and_test_binned_data_2(df, bin_width, analysis_column):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that include the full range from -100 to 100
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percent Flexed' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)

    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])[analysis_column].mean().reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)

    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}

    # Plotting the data
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y=analysis_column,
        hue='Condition',
        marker="o",
        ci='sd',  # Uses standard deviation for the error bars
        palette=custom_palette
    )
    
    # T-tests for each bin
    t_test_results = []
    for name, group in grouped.groupby('Custom_Bin'):
        loaded = group[group['Condition'] == 'Loaded'][analysis_column]
        unloaded = group[group['Condition'] == 'Unloaded'][analysis_column]
        t_test = pg.ttest(loaded, unloaded, paired=False)
        t_test_results.append((name, t_test['p-val'].values[0]))

    # Display t-test results
    #for bin_center, p_val in t_test_results:
       # plt.text(bin_center.mid, 1, f'p={p_val:.3f}', color='red')  # Adjust positioning and color as necessary

    #plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Euclidean Distance (mm)")
    plt.title("Variation of distance with respect to flexion-extension cycle")
    plt.grid(True)
    #plt.savefig('distance_no_13.png', dpi=300)
    plt.show()

    return pd.DataFrame(t_test_results, columns=['Bin', 'p-value'])

# Example usage
disp_Y = plot_and_test_binned_data_2(master_df_cut, 10, 'Relative Norm')


#%%
# plot and test phases: 
    
def plot_and_test_phases(df, analysis_column):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define phases
    df_copy['Phase'] = pd.cut(df_copy['Percent Flexed'], bins=[-100, 0, 100], labels=['Extension', 'Flexion'], include_lowest=True)

    # Group by 'Condition' and 'Phase' to calculate means
    grouped = df_copy.groupby(['Condition', 'Phase'])[analysis_column].mean().reset_index()

    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}

    # Plotting the data
    sns.barplot(
        data=grouped,
        x='Phase',
        y=analysis_column,
        hue='Condition',
        ci='sd',  # Uses standard deviation for the error bars
        palette=custom_palette
    )
    
    # T-tests for each condition
    t_test_results = []
    for condition in ['Loaded', 'Unloaded']:
        phase_1 = df_copy[(df_copy['Condition'] == condition) & (df_copy['Phase'] == 'Extension')][analysis_column]
        phase_2 = df_copy[(df_copy['Condition'] == condition) & (df_copy['Phase'] == 'Flexion')][analysis_column]
        t_test = pg.ttest(phase_1, phase_2, paired=False)
        t_test_results.append((condition, t_test['p-val'].values[0]))

    # Display t-test results
    for condition, p_val in t_test_results:
        print(f"{condition} Condition: p-value = {p_val:.3f}")

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.ylabel("Mean Euclidean Distance (mm)")
    plt.title("Comparison of Mean Distance Between Different Phases")
    plt.grid(True)
    plt.savefig('compare_phase.png', dpi=300)
    plt.show()

    return pd.DataFrame(t_test_results, columns=['Condition', 'p-value'])

# Example usage
comp_phase = plot_and_test_phases(master_df_cut, 'Relative Norm')

    
#%%
# ok, trying to draw sequence diagram using python 
# for tutorial refer to this: https://mrsd.readthedocs.io/en/latest/flash/index.html
import matplotlib.pyplot 
import mrsd
import copy 
figure, plot = matplotlib.pyplot.subplots(figsize=(6,4), tight_layout=True)
# this creates an empty sequence diagram 
diagram = mrsd.Diagram(
    plot, ["RF", "$G_{SS}$", "$G_{PE}$", "$G_{FE}$", "Signal"])


pulse = mrsd.RFPulse(2, 1, center=0) # duration of 2, magnitude of 1 (always between -1 and 1) and center of 0
slice_selection = mrsd.Gradient(pulse.duration, 0.5, ramp=0.1, center=pulse.center) # amplitude is 0.5, centered on the pulse. this creates a flat top gradient

# once we create these two events, we add it to their respective channels : 
diagram.add("RF", pulse)
diagram.add("$G_{SS}$", slice_selection)

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
diagram.add("$G_{FE}$", readout)

# there are convenience functions to do this at once. but for now, this is wokring just fine. create diagram, create event variables and just add them.. 
# what is not working is the resolution 

# now to add the phase encoding 

d_encoding = 1

phase_encoding = mrsd.MultiGradient(d_encoding, 1, 0.1, end=readout.begin)
diagram.add("$G_{PE}$", phase_encoding)


#matplotlib.pyplot.savefig('pulse_diagram_test.png', dpi=300)

# upto here, what we have is a really simplified basic unit of the sequence diagram. but now we need to add multiple gradients fot the full sequence to actually work. 
# this means we need a spoiler on the slice gradient, as well as a thing on the readout. these are called Encoding gradients

# prephasing lobe of the readout gradient 
readout_prephasing = readout.adapt(d_encoding, -0.5, 0.1, end=readout.begin) # start at the phase encoding, end at the readout begin , negative half amp with a ramp of 0.1 
diagram.add("$G_{FE}$", readout_prephasing)

slice_rewinding = slice_selection.adapt(pulse.end, -0.5,0.1, center=phase_encoding.begin)
diagram.add("$G_{SS}$", slice_rewinding)


# now that the unit is completed in its fullest sense, we can now annotate and make copies 

TR = 10
diagram.interval(0,TE, -1.5, 'TE') # apparantly we can go beyond -1, i guess for these cases. it makes sense when we look at the -2.5 
diagram.interval(0, TR, -2.5, "TR")

diagram.annotate("RF", 0.2, 1, r"$\alpha^\circ$") # im guessing 0.2 is slightly after 0, so its the time point, and 1 would be the mag

#diagram.annotate("$G_{phase}$", phase_encoding.end, 0.5, r"$\uparrow$") # direction of phase encoding. leaving it off for now 

# now to create copies of the rf and gslice to show the start of next rep, by copying and moving them 
diagram.add("RF", copy.copy(pulse).move(TR))
diagram.add("$G_{SS}$", copy.copy(slice_selection).move(TR))

diagram.annotate("RF", TR+0.2, 1, r"$\alpha^\circ$")


slice_spoiler = slice_selection.adapt(1, 1,  ramp=0.1, end=7) # just a dumb trial and error i got this to look ok. 
diagram.add("$G_{SS}$", slice_spoiler)


#diagram.annotate("$G_{slice}$", 6, 1.2, "$_{spoiler}$")

#diagram.annotate("$G_{slice}$", slice_rewinding.begin, -0.9, "$_{rewinder}$")

plt.savefig("flash_seq.png", dpi=300)