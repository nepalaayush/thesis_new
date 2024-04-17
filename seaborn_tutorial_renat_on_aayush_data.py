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
import numpy as np 
sns.set_context("talk")
#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/master_df_point.pkl', 'rb') as file:
    master_df_point =  pickle.load(file)

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
add_bins(master_df_point, bin_width=6) 

#%%


# Make sure to include 'Condition' in the groupby
narrow_binned_means = master_df_point.groupby(['Condition', 'Custom_Bin', 'Dataset'])['Relative Norm'].mean().reset_index()

# Calculate the bin centers
narrow_binned_means['Bin_Center'] = narrow_binned_means['Custom_Bin'].apply(lambda x: x.mid)
zero_bin_data = narrow_binned_means[narrow_binned_means['Custom_Bin'].apply(lambda x: 0 in x)]
#print(zero_bin_data)
# Plotting
fg = sns.relplot(
    data=narrow_binned_means, 
    x="Bin_Center", 
    y="Relative Norm", 
    #col="Condition", 
    hue="Condition", 
    kind="line"
)

# Add a reference line at y=0
fg.refline(y=0)
fg.set_axis_labels("Bin Centre (% flexed)", "Relative Norm (mm)")
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


#%%
ref_points = viewer.layers['AN_NW_fem_shape'].data[0][:,1:3]
#%%
applied_transformation = apply_transformations_new(ref_points, t_matrices_fem, 0)    
viewer.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='white')

    
#%%
''' Here we have the point analysis things  '''

def create_points_arrays(fem_NW_name, tib_NW_name, fem_W_name, tib_W_name, fem_index, tib_index): 
    fem_shape = viewer.layers[fem_NW_name].data
    
    tib_shape = viewer.layers[tib_NW_name].data
    
    fem_points_NW = np.array ( [item[fem_index] for item in fem_shape] ) 
    
    tib_points_NW = np.array ( [item[tib_index] for item in tib_shape] )
    
    
    fem_shape_W = viewer.layers[fem_W_name].data
    
    tib_shape_W = viewer.layers[tib_W_name].data
    
    fem_points_W = np.array ( [item[fem_index] for item in fem_shape_W] ) 
    
    tib_points_W = np.array ( [item[tib_index] for item in tib_shape_W] )
    
    return fem_points_NW, tib_points_NW, fem_points_W, tib_points_W

fem_points_NW, tib_points_NW, fem_points_W, tib_points_W = create_points_arrays('AN_NW_fem_shape', 'AN_NW_tib_shape', 'AN_W_fem_shape', 'AN_W_tib_shape',38,1)


#%%

voxel_size_y = 0.7272727272727273 / 1.5  # Adjusted voxel size for y
voxel_size_x = 0.7272727272727273 / 1.5  # Adjusted voxel size for x


# Function to create DataFrame for a condition
def create_condition_df(points_fem, points_tib, condition):
    return pd.DataFrame({
        'Frame': points_fem[:, 0].astype(int),
        'Femur_Y': points_fem[:, 1] * voxel_size_y,
        'Femur_X': points_fem[:, 2] * voxel_size_x,
        'Tibia_Y': points_tib[:, 1] * voxel_size_y,
        'Tibia_X': points_tib[:, 2] * voxel_size_x,
        'Condition': condition,
    })

# Create DataFrames for each condition
df_NW = create_condition_df(fem_points_NW, tib_points_NW, 'Unloaded')
df_W = create_condition_df(fem_points_W, tib_points_W, 'Loaded')

# Combine DataFrames
AN_point_df_3 = pd.concat([df_NW, df_W], ignore_index=True)

#%%
AN_point_df_3['Dataset'] = 3
#%%
def add_norm(df):
    df['Norm'] = np.sqrt(
    (df['Femur_X'] - df['Tibia_X'])**2 +
    (df['Femur_Y'] - df['Tibia_Y'])**2
)
    
add_norm(AN_point_df_3)    
#%%
fg = sns.relplot(
    AN_point_df_3, 
    x="Frame",
    #x = 'Frame Number',
    y="Norm", 
    #col="Condition", 
    hue="Condition", 
    kind="line", 
    #ci='sd', # one sd shows a bit more variation than the default 95% confidence interval 
#    facet_kws={"sharey":False}
)
#fg.refline(y=0)
#%%
master_df_point = pd.concat([AN_point_df_5, JL_point_df_7, MK_point_df, MK_point_df_4, HS_point_df ], ignore_index=True)
#%%

def add_percent_flexed(df):
    df = df.copy()  # To avoid modifying the original dataframe
    percent_flexed_values = {}  # To store first half values for mirroring

    for dataset_id, group_data in df.groupby('Dataset'):
        total_frames = group_data['Frame'].max() + 1  # Total number of frames
        # Define halfway_point for the calculation, not for mirroring
        halfway_calculation_point = (total_frames // 2) - 1

        # Calculate 'Percent Flexed' for each frame
        for index, row in group_data.iterrows():
            frame = row['Frame']
            if frame <= halfway_calculation_point:
                # Before or at the halfway calculation point, scale from -100% to 0%
                percent_flexed = ((frame / halfway_calculation_point) * 100) - 100
                percent_flexed_values[dataset_id, frame] = percent_flexed
            else:
                # After the halfway point, mirror the value from the first half
                # Check if it's the peak frame
                if frame == halfway_calculation_point + 1:
                    percent_flexed = 0
                else:
                    # Calculate mirror frame
                    mirror_frame = halfway_calculation_point - (frame - halfway_calculation_point - 1)
                    percent_flexed = -percent_flexed_values[dataset_id, mirror_frame]
            df.at[index, 'Percent Flexed'] = percent_flexed

    return df
AN_point_df_3 = add_percent_flexed(AN_point_df_3)
#%%
def add_relative_norm_column(df):
    """
    Adds a 'Relative Norm' column to the dataframe where the norm value is adjusted
    to start from 0 for the first frame of each condition within each dataset.

    Parameters:
    df (pandas.DataFrame): The dataframe containing at least 'Norm', 'Dataset', and 'Condition' columns.

    Returns:
    pandas.DataFrame: The modified dataframe with a new 'Relative Norm' column.
    """
    # Calculating the relative norm
    def calculate_relative_norm(group):
        first_norm = group.iloc[0]['Norm']  # First 'Norm' value in the group
        group['Relative Norm'] = group['Norm'] - first_norm
        return group

    # Applying the function to each group and returning the modified dataframe
    return df.groupby(['Dataset', 'Condition']).apply(calculate_relative_norm)

# Example usage
# Assume 'master_df_point' is your existing DataFrame loaded elsewhere in your code.
# master_df_point = pd.read_csv('path_to_your_data.csv')  # If you need to load it from a CSV file

# Adding the 'Relative Norm' column
AN_point_df_3 = add_relative_norm_column(AN_point_df_3)


#%%
fg = sns.relplot(
    AN_point_df_3, 
    x="Percent Flexed",
    #x = 'Frame Number',
    y="Relative Norm", 
    col="Condition", 
    #hue="Dataset", 
    kind="line", 
    #ci='sd', # one sd shows a bit more variation than the default 95% confidence interval 
#    facet_kws={"sharey":False}
)
fg.refline(y=0)
fg.set_axis_labels("% Flexed", "Relative Norm (mm)")
#%%
with open('AN_point_df_3.pkl', 'wb') as f:
    pickle.dump(AN_point_df_3, f)  
