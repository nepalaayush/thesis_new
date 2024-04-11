#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:55:40 2024

@author: aayush
"""

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("talk")

with open('/data/projects/ma-nepal-segmentation/data/data_20_03/angle_and_rel_df.pkl', 'rb') as file:
    angle_and_rel_df =  pickle.load(file)
angle_and_rel_df.head()
#%%
angle_and_rel_df["Dataset"] = pd.Categorical(angle_and_rel_df["Dataset"].apply(lambda x: x.split(" ")[1]))
angle_and_rel_df["Condition"] = pd.Categorical(angle_and_rel_df["Condition"])
angle_and_rel_df["Type"] = pd.Categorical(angle_and_rel_df["Type"])
angle_and_rel_df.head()
#%%
fg = sns.lmplot(data=angle_and_rel_df, x='Angles', y='Relative Translation', hue='Dataset', ci=None, col="Condition")
fg.fig.suptitle('Translation vs. Angles for Unloaded Condition', y=1.05)
for ax in fg.axes.flat:
    ax.set_xlabel('Angle (Degree)')
fg.axes[0,0].set_ylabel('Translation (mm)')
#%%

frames_to_remove = [16, 17, 15, 18, 14, 19]
equal_frame_df = angle_and_rel_df[
    ~(angle_and_rel_df['Frame Number'].isin(frames_to_remove) & angle_and_rel_df['Dataset'].isin(["2", "3"]))
].reset_index(drop=True)
equal_frame_df['Frame Number'] = equal_frame_df.groupby(['Dataset', 'Condition']).cumcount()


#%%
'''
the whole idea here is to remove dataset 7 as a categorical because simply filtering wont remove the legend  
'''
plot_df = is_df_1_7.copy()
filtered_data = plot_df[plot_df['Dataset'] != "7"]


filtered_data['Dataset'] = filtered_data['Dataset'].cat.remove_unused_categories()
#%%
''' this is a working code to plot the IS translation for all datasets w.r.t frame or percent flexed. the mean can be adjusted by commenting out hue 
'''  
fg = sns.relplot(
    filtered_data, 
    x="Percent Flexed",
    #x = 'Frame Number',
    y="Relative Translation", 
    col="Condition", 
    hue="Dataset", 
    kind="line", 
    #ci='sd', # one sd shows a bit more variation than the default 95% confidence interval 
#    facet_kws={"sharey":False}
)
fg.refline(y=0)


fg.fig.suptitle('Inferior(+ve)/Superior(-ve) translation of the tibia relative to femur', fontsize=16, ha='center')
fg.fig.subplots_adjust(top=0.86)


#%%
def print_percent_flexed_width(df):
    # Ensure the dataframe is sorted by 'Dataset' and then by 'Frame Number' or an equivalent
    df = df.sort_values(by=['Dataset', 'Frame Number'])

    # Initialize an empty dictionary to hold the width for each dataset
    width_per_dataset = {}

    # Iterate over each dataset
    for dataset in df['Dataset'].unique():
        # Filter the dataframe for the current dataset
        dataset_df = df[df['Dataset'] == dataset]

        # Calculate the differences between consecutive 'Percent Flexed' values
        # Assuming the data is sorted by some sort of frame or time order
        differences = dataset_df['Percent Flexed'].diff().dropna()

        # The width should be consistent, so we can take the first non-zero difference
        width = differences.abs().loc[differences != 0].iloc[0]

        # Store the width for the current dataset
        width_per_dataset[dataset] = width

    # Print the width for each dataset
    for dataset, width in width_per_dataset.items():
        print(f"Dataset {dataset}: Width of 'Percent Flexed' = {width}")

print_percent_flexed_width(is_df_1_7)
#%%
# Step 1: Filter out dataset '7'
is_df_1_6 = is_df_1_7[is_df_1_7['Dataset'] != '7']

# Step 2: Remove the unused category
is_df_1_6['Dataset'] = is_df_1_6['Dataset'].cat.remove_unused_categories()

#%%
ap_df_1_6 = ap_df_1_7[ap_df_1_7['Dataset'] != '7']

# Step 2: Remove the unused category
ap_df_1_6['Dataset'] = ap_df_1_6['Dataset'].cat.remove_unused_categories()
#%%

#%%

def add_bins(df, bin_width):

    # Define bin edges
    bin_edges = [-101, -100]  # Bin for -100
    bin_edges += list(range(-100 + bin_width, 0, bin_width))  # Bins from -100 to 0
    bin_edges += [0]  # Bin for 0
    bin_edges += list(range(bin_width, 100, bin_width))  # Bins from 0 to 100
    bin_edges += [100, 101]  # Bin for 100
    
    # Assign 'Percent Flexed' values to bins
    df['Custom_Bin'] = pd.cut(df['Percent Flexed'], bins=bin_edges, include_lowest=True)
    
    return df 

# Check the binning results
add_bins(ap_df_1_6, bin_width=5) 

#%%


# Make sure to include 'Condition' in the groupby
narrow_binned_means = is_df_1_6.groupby(['Condition', 'Custom_Bin', 'Dataset'])['Relative Translation'].mean().reset_index()

# Calculate the bin centers
narrow_binned_means['Bin_Center'] = narrow_binned_means['Custom_Bin'].apply(lambda x: x.mid)
zero_bin_data = narrow_binned_means[narrow_binned_means['Custom_Bin'].apply(lambda x: 0 in x)]
#print(zero_bin_data)
# Plotting
fg = sns.relplot(
    data=narrow_binned_means, 
    x="Bin_Center", 
    y="Relative Translation", 
    col="Condition", 
    #hue="Condition", 
    kind="line"
)

# Add a reference line at y=0
fg.refline(y=0)
fg.set_axis_labels("Bin Centre (% flexed)", "Relative Translation (mm)")
# Adjust the layout and display the plot
plt.subplots_adjust(top=0.9)
plt.show()
# %%
# Assuming is_df_1_6 has a column named 'Angles' which you want to plot against 'Relative Translation'
# Plotting
fg = sns.relplot(
    data=is_df_1_6, 
    x="Angles", 
    y="Relative Translation", 
    col="Condition", 
    #hue="Dataset",  # Uncomment if you have multiple datasets and want to differentiate them
    kind="line"
)

# Add a reference line at y=0
#fg.refline(y=0)

# Adjust the layout and display the plot
plt.subplots_adjust(top=0.9)
plt.show()
# %% 
from scipy.interpolate import interp1d


# Assuming is_df_1_6 is your DataFrame
# Group by dataset to handle each one separately
grouped = is_df_1_6.groupby('Dataset')

# Define common angle range
common_angles = np.arange(0, 180, 2)  # Define your range and step

# Store interpolated translations
interpolated_translations = []

for name, group in grouped:
    # Create interpolation function for the current dataset
    interp_func = interp1d(group['Angles'], group['Relative Translation'], kind='linear', fill_value='extrapolate')
    
    # Interpolate the translation data for the common angle range
    interp_translations = interp_func(common_angles)
    
    # Store the results with the common angles in a DataFrame
    interpolated_df = pd.DataFrame({'Angles': common_angles, 'Relative Translation': interp_translations, 'Dataset': name, 'Condition': condition })
    interpolated_translations.append(interpolated_df)

# Concatenate all interpolated results
interpolated_data = pd.concat(interpolated_translations)

# Now you can plot using Seaborn as you intended
fg = sns.relplot(
    data=interpolated_data, 
    x="Angles", 
    y="Relative Translation", 
    col="Condition", 
    kind="line"
)

plt.subplots_adjust(top=0.9)
plt.show()

#%%
 
NBIN = 10
# equal_frame_df["Angles binned"] = pd.cut(equal_frame_df["Angles"], NBIN)

#%%
equal_frame_df_agg = (
 equal_frame_df
 .assign(Angles_binned=pd.cut(equal_frame_df["Angles"], NBIN))
 .groupby(["Dataset", "Condition", "Angles_binned"])
 [["Relative Translation", "Angles"]]
 .agg({"Angles": "mean", "Relative Translation" : ["mean", "std"]})
 .dropna()
)
# %%
fg = sns.relplot(
    data=equal_frame_df_agg, 
    x=("Angles", "mean"), 
    y=("Relative Translation", "mean"), 
    hue="Dataset", 
    row="Condition"
)
for (ds, cond), df_ in equal_frame_df_agg.groupby(["Dataset", "Condition"]):
    fg.axes_dict[cond].fill_between(
        df_[("Angles", "mean")], 
        df_[("Relative Translation", "mean")] - df_[("Relative Translation", "std")], 
        df_[("Relative Translation", "mean")] + df_[("Relative Translation", "std")], 
        alpha=0.5,
        color=f"C{int(ds)-1}"
    )
#%%
equal_frame_df_agg = (
 equal_frame_df
 .assign(Angles_binned=pd.cut(equal_frame_df["Angles"], NBIN))
 .groupby(["Condition", "Angles_binned"])
 [["Relative Translation", "Angles"]]
 .agg({"Angles": "mean", "Relative Translation" : ["mean", "std"]})
 .dropna()
)
# %%
fg = sns.relplot(
    data=equal_frame_df_agg, 
    x=("Angles", "mean"), 
    y=("Relative Translation", "mean"), 
    row="Condition"
)
for cond, df_ in equal_frame_df_agg.groupby(["Condition"]):
    fg.axes_dict[cond].fill_between(
        df_[("Angles", "mean")], 
        df_[("Relative Translation", "mean")] - df_[("Relative Translation", "std")], 
        df_[("Relative Translation", "mean")] + df_[("Relative Translation", "std")], 
        alpha=0.5,
    )