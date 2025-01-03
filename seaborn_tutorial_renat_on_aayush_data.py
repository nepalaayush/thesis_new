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
''' we need to find a better way to define the bins. in a way that is the most repeatable '''
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

print_percent_flexed_width(master_df_point) # what this does is give us the width for each of the datasets .. to decide on the bin width, we will go with the smallest width
#%%

sns.histplot(master_df_point['Percent Flexed'], bins=20, kde=False, color="blue")

def add_bins(df):
    # Bin edges
    bin_edges = [-101] 
    # Smallest bin width
    bin_width = 4.545
    
    # Generate edges from -100 to 0
    bin_edges += list(np.arange(-100 + bin_width, 0, bin_width))
    
    bin_edges += [0]
    
    bin_edges += list(np.arange(0 + bin_width, 100, bin_width))
    
    bin_edges += [100, 101]
    
    bin_labels = [f"{round(bin_edges[i], 2)} to {round(bin_edges[i+1], 2)}" for i in range(len(bin_edges)-1)]
     
    # Binning the data
    master_df_point['Binned Percent Flexed'] = pd.cut(master_df_point['Percent Flexed'], bins=bin_edges, labels=bin_labels, include_lowest=True)

add_bins(master_df_point)
#%%

def add_bins(df, bin_width):

    # Define bin edges
    #bin_edges = [-101, -100]  # Bin for -100
    #bin_edges += list(range(-100 + bin_width, 0, bin_width))  # Bins from -100 to 0
    #bin_edges += [0]  # Bin for 0
    #bin_edges += list(range(bin_width, 100, bin_width))  # Bins from 0 to 100
    #bin_edges += [100, 101]  # Bin for 100
    
    
    # doing a simple bin without the fancy exact bins for the three extreme values: 
        
    bin_edges = list(range(-100,100,bin_width))    
    # Assign 'Percent Flexed' values to bins
    df['Custom_Bin_10'] = pd.cut(df['Percent Flexed'], bins=bin_edges, include_lowest=True)
    
    return df 

# Check the binning results
add_bins(master_df_point, bin_width=10) 


#%%
''' ! ! !  saved in the latex folder latest attempt at this plot  ''' 
def plot_binned_data(df, bin_width):
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
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}

    # Plotting the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='Relative Norm',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        #err_style="bars",  # Shows error bars instead of a band
        ci='sd',  # Uses standard deviation for the error bars
        palette = custom_palette
    )
    plt.axhline(y=0, color='gray', linestyle='--')  # Adds a horizontal line at y=0
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Euclidean Distance (mm)")
    plt.title("Variation of distance with respect to flexion-extension cycle")
    plt.savefig('distance_stickman.png', dpi=300)
    plt.show()

# Example usage
plot_binned_data(df_point, 10)

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
    kind="line",
    errorbar='sd',
    err_style='bars'
)

# Add a reference line at y=0
fg.refline(y=0)
fg.set_axis_labels("Bin Centre (% flexed)", "Euclidean Distance (mm)")
# Adjust the layout and display the plot
plt.subplots_adjust(top=0.9)
plt.show()

#%%

narrow_binned_means = master_df_point.groupby(['Condition', 'Binned Percent Flexed', 'Dataset'])['Relative Norm'].mean().reset_index()

# Calculate the bin centers from the custom labels
narrow_binned_means['Bin_Center'] = narrow_binned_means['Binned Percent Flexed'].apply(
    lambda x: (float(x.split(' to ')[0]) + float(x.split(' to ')[1])) / 2
)

# Filter data where bin center is zero, if necessary (it seems you might have used this to focus on specific ranges)
zero_bin_data = narrow_binned_means[narrow_binned_means['Bin_Center'] == 0]

# Plotting with Seaborn
fg = sns.relplot(
    data=narrow_binned_means, 
    x="Bin_Center", 
    y="Relative Norm", 
    hue="Condition", 
    kind="line",
    facet_kws={'sharey': False, 'sharex': False}  # Adjust axis sharing as needed
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

fem_points_NW, tib_points_NW, fem_points_W, tib_points_W = create_points_arrays('MK_NW_fem_shape_stiched', 'MK_NW_tib_shape_stiched', 'MK_NW_fem_shape_stiched', 'MK_NW_tib_shape_stiched',23,3)


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
US_point_df_8 = pd.concat([df_NW, df_W], ignore_index=True)

#%%
US_point_df_8['Dataset'] = 8
#%%
def add_norm(df):
    df['Norm'] = np.sqrt(
    (df['Femur_X'] - df['Tibia_X'])**2 +
    (df['Femur_Y'] - df['Tibia_Y'])**2
)
    
add_norm(US_point_df_8)    
#%%
fg = sns.relplot(
    US_point_df_8, 
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

MM_point_df_2 = add_percent_flexed(MM_point_df_2)
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
MM_point_df_2 = add_relative_norm_column(MM_point_df_2)


#%%
fg = sns.relplot(
    MM_point_df_2, 
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
def get_bin_center(bin_range):
    bounds = bin_range.strip('()[]').split(',')
    return (float(bounds[0]) + float(bounds[1])) / 2

# Calculating bin centers
df['Bin Center'] = df['Bin'].apply(get_bin_center)

# Filtering for "No Weight" condition
df_filtered = df[df['Condition'] == 'No Weight']

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_filtered, x='Bin Center', y='angle', hue='Dataset', style='Dataset', palette='deep')
plt.title('Angle vs. Bin Center for No Weight Condition Across Datasets')
plt.xlabel('Bin Center')
plt.ylabel('Angle')
plt.grid(True)
plt.legend(title='Dataset')
plt.show()


#%%

def get_bin_center(bin_range):
    bounds = bin_range.strip('()[]').split(',')
    return (float(bounds[0]) + float(bounds[1])) / 2

# Calculating bin centers
df['Bin Center'] = df['Bin'].apply(get_bin_center)

# Filtering for "No Weight" condition
df_filtered = df[df['Condition'] == 'No Weight']

# Grouping by Bin Center and calculating average
average_per_bin = df_filtered.groupby('Bin Center')['angle'].mean().reset_index()

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=df[df['Dataset']!=6], x='Bin Center', y='angle', ci='sd', marker='o', estimator='mean', hue='Condition')
plt.title('Angle between the long axis of tibia and femur segments')
plt.xlabel('Percentage of flexion [%]')
plt.ylabel('Average Angle [°]')
plt.grid(True)
#plt.savefig('results_angle.png', dpi=300)
plt.show()

#%%
''' ! ! ! ! the figure for Results saved in \thesis_new\angle_datasets folder ''' 
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
import pandas as pd

def print_binned_angle_stats(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)

    # Group by 'Condition' and 'Custom_Bin' to calculate means and standard deviations
    grouped = df_copy.groupby(['Condition', 'Custom_Bin'])['angle'].agg(['mean', 'std']).reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)

    # Print the mean and standard deviation values for each condition
    for condition in grouped['Condition'].unique():
        print(f"Condition: {condition}")
        condition_data = grouped[grouped['Condition'] == condition]
        for _, row in condition_data.iterrows():
            print(f"Bin Center: {row['Bin_Center']}, Mean Angle: {row['mean']}, Std Dev: {row['std']}")

# Example usage
print_binned_angle_stats(modified_angle_df, 10)

#%%
with open('df_angle.pkl', 'wb') as f:
    pickle.dump(df_angle, f) 
    
    
#%%
# runnning some statistical tests here 
# First, an independent t-test to just see if the loaded and unloaded are significantly different
from scipy import stats
#%%
loaded_values = master_df_point[master_df_point['Condition'] == 'Loaded']['Relative Norm']
unloaded_values = master_df_point[master_df_point['Condition'] == 'Unloaded']['Relative Norm']
#%%
# Perform the independent t-test
t_stat, p_value = stats.ttest_ind(loaded_values, unloaded_values)

# Print the t-statistic and p-value
print("T-statistic: ", t_stat)
print("P-value: ", p_value)
#%%
# next up, a paired t test per dataset 
    
# to start off, we want to get a p value per dataset, so, we have to loop 7 times. this is how we initialize it: 
for dataset_id in master_df_point['Dataset'].unique():
    # the first task is, for each dataset, we have to filter the loaded and unloaded so that they are their own separate dataframes 
    loaded_df = master_df_point[( master_df_point['Dataset']== dataset_id ) & (master_df_point['Condition'] == 'Loaded') ]
    unloaded_df = master_df_point[( master_df_point['Dataset']== dataset_id ) & (master_df_point['Condition'] == 'Unloaded') ]
    
    # now we can perform the paired t test on these dataframes
    t_stat, p_value = stats.ttest_rel(unloaded_df['Relative Norm'], loaded_df['Relative Norm'])
    print(f"Dataset {dataset_id}: T-statistic = {t_stat}, P-value = {p_value}")

''' Dataset 5: T-statistic = 3.4519449261178754, P-value = 0.0016295750222419518
Dataset 7: T-statistic = -5.091645863362112, P-value = 6.775722723124217e-06
Dataset 1: T-statistic = -0.1609477431684528, P-value = 0.8733327885155148
Dataset 4: T-statistic = -0.9540618389405038, P-value = 0.34851656628719707
Dataset 6: T-statistic = -4.922799137043911, P-value = 1.7902255002853064e-05
Dataset 3: T-statistic = 3.306558581022953, P-value = 0.0025238134250468685
Dataset 2: T-statistic = -1.7992376045913905, P-value = 0.08113185788721555 '''        

#%%
# now to do a paired t test but for bins .. using the aggregate 
results = [] # an empty list to store the results, we expect as many results as there are unique bins. in the case of 5 width bin (including the custom three bins), we have 41 unique values. . 
significant_results = []
significance_level = 0.05

for bin_label in master_df_point['Custom_Bin'].unique():
    # two distinct things are happening in the line below. what we want, is to basically check if there is any significant difference between the relative norm values 
    # for the unique bins. to do that we need to first loop through all the unique bins, then, for each bin, we need to calculate the mean. now the decision needs to be made 
    # whether or not we again do this dataset-wise or do it across all. but our objective is to findout the overall effect of loading, so lets not do groupby dataset. # no .groupby('Dataset') 
    loaded_data = master_df_point[(master_df_point['Condition']=='Loaded') & (master_df_point['Custom_Bin']== bin_label)]['Relative Norm']
    
    unloaded_data = master_df_point[(master_df_point['Condition']=='Unloaded') & (master_df_point['Custom_Bin']== bin_label)]['Relative Norm']
   
    t_stat, p_value = stats.ttest_ind(loaded_data, unloaded_data)
    results.append((bin_label, t_stat, p_value))
    
    if p_value < significance_level:
            significant_results.append((bin_label, t_stat, p_value))
    
for result in results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

print("\nSignificant Results (p < 0.05):")
for result in significant_results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")
    
#%%

# the above code did return some sort of result .. but the bins are dubios as i do not see anything from 0 to 5, this is mainly due to custom bin where i insisted on having the three things.. 
# so now, just creating a normal pd.cut where we just take 5 width range. 

master_binned = master_df_point.copy() 
bin_edges = np.arange(-100, 111, 10)

master_binned['Bin'] = pd.cut(master_binned['Percent Flexed'], bins=bin_edges, right=False, include_lowest=True)

results = []
significant_results = []
significance_level = 0.05

# Group by the newly created 'Bin' column and perform t-tests
for bin_label, group in master_binned.groupby('Bin'):
    loaded_data = group[group['Condition'] == 'Loaded']['Relative Norm']
    unloaded_data = group[group['Condition'] == 'Unloaded']['Relative Norm']

    # Only perform the t-test if both conditions have enough data
    if len(loaded_data) > 1 and len(unloaded_data) > 1:
        t_stat, p_value = stats.ttest_ind(loaded_data, unloaded_data)
        results.append((bin_label, t_stat, p_value))
        # Check significance
        if p_value < significance_level:
            significant_results.append((bin_label, t_stat, p_value))
    else:
        results.append((bin_label, 'Not enough data', 'N/A'))

# Output results
print("All Results:")
for result in results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

print("\nSignificant Results (p < 0.05):")
for result in significant_results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

#%%
'''  ''' 
master_binned = master_df_point.copy() 
bin_edges = np.arange(-100, 106, 5)

master_binned['Bin'] = pd.cut(master_binned['Percent Flexed'], bins=bin_edges, right=False, include_lowest=True)

results = []
significant_results = []
significance_level = 0.05

# Group by the newly created 'Bin' column and perform t-tests
for bin_label, group in master_binned.groupby('Bin'):
    loaded_data = group[group['Condition'] == 'Loaded']['angle']
    unloaded_data = group[group['Condition'] == 'Unloaded']['angle']

    # Only perform the t-test if both conditions have enough data
    if len(loaded_data) > 1 and len(unloaded_data) > 1:
        t_stat, p_value = stats.ttest_ind(loaded_data, unloaded_data)
        results.append((bin_label, t_stat, p_value))
        # Check significance
        if p_value < significance_level:
            significant_results.append((bin_label, t_stat, p_value))
    else:
        results.append((bin_label, 'Not enough data', 'N/A'))

# Output results
print("All Results:")
for result in results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

print("\nSignificant Results (p < 0.05):")
for result in significant_results:
    print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")


#%%
''' this function is used to get the p value for the results using t test  '''
def perform_t_tests(df, bin_width):
    # Create a copy of the DataFrame
    master_binned = df.copy() 

    # Define bin edges the same way as in plotting
    bin_edges = np.arange(-100, 101, bin_width)  # Note: To include 100, range goes to 101

    # Create bins
    master_binned['Bin'] = pd.cut(master_binned['Percent Flexed'], bins=bin_edges, right=True, include_lowest=True)

    # Initialize lists to store results
    results = []
    significant_results = []
    significance_level = 0.05

    # Group by the newly created 'Bin' column and perform t-tests
    for bin_label, group in master_binned.groupby('Bin'):
        loaded_data = group[group['Condition'] == 'Loaded']['Relative Norm']
        unloaded_data = group[group['Condition'] == 'Unloaded']['Relative Norm']

        # Only perform the t-test if both conditions have enough data
        if len(loaded_data) > 1 and len(unloaded_data) > 1:
            t_stat, p_value = stats.ttest_ind(loaded_data, unloaded_data)
            results.append((bin_label, t_stat, p_value))
            # Check significance
            if p_value < significance_level:
                significant_results.append((bin_label, t_stat, p_value))
        else:
            results.append((bin_label, 'Not enough data', 'N/A'))

    # Output results
    print("All Results:")
    for result in results:
        print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

    print("\nSignificant Results (p < 0.05):")
    for result in significant_results:
        print(f"Bin {result[0]}: T-statistic = {result[1]}, P-value = {result[2]}")

# Example usage
perform_t_tests(master_df_point[master_df_point['Dataset'].isin([1,2,5,6,7])], 10) 
#%%
import pingouin as pg 
# adding bins once again 
bin_edges = np.arange(-100, 101, 10)
master_df_point['Bin'] = pd.cut(master_df_point['Percent Flexed'], bins=bin_edges, right=True, include_lowest=True)
aov = pg.rm_anova(dv='Relative Norm', within=['Condition', 'Bin'], subject='Dataset', data=master_df_point)

# Print the summary of the ANOVA results


significant_bins = aov[aov['p-unc'] < 0.05]
print(significant_bins)



#%%
'''
Bin (0.0, 10.0]: T-statistic = 2.581347448142401, P-value = 0.021752348033689917
Bin (10.0, 20.0]: T-statistic = 2.651870913157143, P-value = 0.015303907958726073
 ''' 
 
#%%
mk_df_point  =master_df_point[master_df_point['Dataset'].isin([1, 4])]
#%%
# trying to compare the same thing , the distance, but now, showing the dataset on different days. 
def plot_distance_days(df, bin_width):
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
    default_palette = sns.color_palette()
    custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}

    # Create a FacetGrid to facet by 'Condition'
    g = sns.FacetGrid(grouped, col='Condition', hue='Dataset', palette=default_palette, height=6, aspect=1.5)
    g.map(sns.lineplot, 'Bin_Center', 'Relative Norm', marker="o", ci='sd').add_legend()

    # Add horizontal and vertical lines
    for ax in g.axes.flat:
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--')

    g.set_axis_labels("Flexion percentage [%]", "Euclidean Distance (mm)")
    g.set_titles(col_template="{col_name} Condition")
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Variation of distance with respect to flexion-extension cycle", fontsize=16)
    plt.savefig('distance_stickman.png', dpi=300)
    plt.show()

# Example usage
plot_distance_days(master_df_point[master_df_point['Dataset'].isin([3, 5])], 10)
 
 