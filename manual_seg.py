# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:56:48 2024

@author: Aayush
"""

import pickle
import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import napari 
import time 
from scipy import ndimage
import matplotlib.pylab as plt 

import pandas as pd 

from utils import (path_to_image, apply_canny, apply_remove, apply_skeleton, points_for_napari,
                   boolean_to_coords, apply_label, find_tibia_edges, find_array_with_min_n, downsample_points,
                   combined_consecutive_transform, coords_to_boolean, process_frame, sample_points_in_polygon,
                   fit_pca_line, get_uv_from_pca, find_edges_nnew, find_intersection, show_stuff)

import pingouin as pg 
import seaborn as sns
sns.set_context("talk")

#%%
# apparantly there is a backwards compatibility issue with pickle.load.. so following another solution from the net: 
    
master_df_inverted = pd.read_pickle('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/master_df_inverted.pkl')

#%%
master_df = master_df_point[~master_df_point['Dataset'].isin([1, 3])]
#%%
master_df = master_df.drop(columns=['Binned Percent Flexed', 'Custom_Bin'])

#%%
master_df['Dataset'] = pd.factorize(master_df['Dataset'])[0] + 1

#%%
df = pd.concat([master_df, modified_angle_df], ignore_index=True)
#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds1_angle_df_new.pkl', 'rb') as file:
    df1 = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds2_angle_df_new.pkl', 'rb') as file:
    df2 = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds3_angle_df_new.pkl', 'rb') as file:
    df3 = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds4_angle_df_new.pkl', 'rb') as file:
    df4 = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds5_angle_df_new.pkl', 'rb') as file:
    df5 = pickle.load(file)        
#%%
# Step 1: load the image from directory and normalize it
path_neg = 'C:/Users/Aayush/Documents/thesis_files/manual_segmentation_datasets/MM_2_dataset1/MM_NW_ai2_tgv_5e-2_neg_ngn.nii'
path_pos = 'C:/Users/Aayush/Documents/thesis_files/manual_segmentation_datasets/MM_2_dataset1/MM_NW_ai2_tgv_5e-2_pos_ngn.nii'
#%%
image_neg = path_to_image(path_neg)[2:]
image_pos = path_to_image(path_pos)[2:]
#%%
# since our image goes from extened to flexed.. the direction means, pos is going down.. and neg is coming up 
# which means. if we want to present our data as going up then coming down .. we have to reverse the neg, put it at the first half. 
image_neg = image_neg[::-1]
#%%
full_image = np.concatenate( (image_neg, image_pos) , axis=0)

#%%
#add the original image to napari
viewer = napari.view_image(full_image,  name='ds1_NW_full')

#%%

#this is just to create the mosaic. need to modify to do both shapes at once. 
total_frames = len(full_image) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer = viewer.layers["tib"].to_labels(full_image.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
#xrange=slice(150,480)
xrange = slice(80,350)
yrange=slice(140,400)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(full_image[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    #ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
     
plt.tight_layout()
plt.savefig('ds1_tib.svg')
#%%
shapes_data = viewer.layers['tib']  # need to reverse if last frame is extended (or in the future, simply reverse the source image) was .data 
binary_frame = ( shapes_data.to_labels(full_image.shape) == 1 ) [0]
#binary_frame = ( shapes_data.to_labels(((30, 528, 528))) == 1 ) [0]
binary_coords = np.column_stack(np.where(binary_frame))


#%%

def calculate_angle_between_bones(bone1, bone2, axis='long'):
    """
    Calculate the angle between two bones based on their coordinate system.

    :param bone1: Dictionary with bone data (femur or tibia).
    :param bone2: Dictionary with bone data (femur or tibia).
    :param axis: 'long' or 'short' to choose which axis to compare.
    :return: Angle in degrees between the two bones.
    """
    def get_axis_vector(bone, axis_type):
        """Extract the vector for the specified axis from the bone data."""
        points = bone[f'points_{axis_type}_axis']
        return points[1] - points[0]  # Vector from first point to second point

    # Get vectors for the specified axis
    vector1 = get_axis_vector(bone1, axis)
    vector2 = get_axis_vector(bone2, axis)

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle in radians and then convert to degrees
    #using arctan to see if we get negative values and a continuous curve or not 
    angle_radians = np.arctan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))
    #angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_degrees = np.degrees(angle_radians)
    #if angle_degrees > 90:
     #   angle_degrees =  180 - angle_degrees

    return angle_degrees



def calculate_and_plot_angles_between_bones(bone1, bone2, axis='long', name='', new_figure=True):
    """
    Calculate the angles between two bones for each frame and plot the angles.

    :param bone1: Dictionary with bone data (femur or tibia) for multiple frames.
    :param bone2: Dictionary with bone data (femur or tibia) for multiple frames.
    :param axis: 'long' or 'short' to choose which axis to compare.
    """
    angles = []
    frames = []
    
    for frame in bone1.keys():
        if frame in bone2:
            angle = calculate_angle_between_bones(bone1[frame], bone2[frame], axis)
            print(f"Frame {frame}: Angle = {angle} degrees")
            angles.append(angle)
            frames.append(frame)
    #if new_figure:
     #   plt.figure(figsize=(10, 6))
    #plt.figure(figsize=(10, 6))
    plt.plot(frames, angles, marker='o', label=name)
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title(f'Angle between Bones over Frames ({axis.capitalize()} Axis)')
    plt.grid(True)
    plt.legend()
    if new_figure:
        plt.savefig(f'{name}_Angle_betwn_long_axes.svg')
    

    return np.array(angles)

#%%

df['angle'] = 180 - df['angle']

with open('master_df_inverted.pkl', 'wb') as f:
    pickle.dump(df, f)   


#%%
def process_single_frame(binary_coords):
    # Calculate PCA line points
    line_points = fit_pca_line(binary_coords)
    
    # Return only the necessary data
    results = {
        "points_long_axis": line_points
    }
    
    return results

def process_multi_frame(shape_layer, image_shape):
    """
    Process each frame in the shape layer and create a dictionary
    with the results for each frame.
    
    :param shape_layer: The shape layer containing frame data (napari Shapes layer).
    :param image_shape: The shape of the full image.
    :return: A dictionary with processed data for each frame.
    """
    total_shapes = len(shape_layer.data)
    frame_dict = {}
    
    for frame_number in range(total_shapes):
        # Generate binary mask for the current shape
        binary_frame = (shape_layer.to_labels(image_shape) == (frame_number + 1))
        binary_coords = np.column_stack(np.where(binary_frame))
        
        # Process the binary coordinates to get points_long_axis
        result = process_single_frame(binary_coords)
        
        # Store the result in the dictionary
        frame_dict[frame_number] = result
    
    return frame_dict

tib_info = process_multi_frame(viewer.layers['tib'], full_image.shape)

fem_info = process_multi_frame(viewer.layers['fem'], full_image.shape)

#%%
ds1_angles_nw_new = calculate_and_plot_angles_between_bones(tib_info, fem_info, name='ds_1_NW_new')

#%%
with open('tib_info_ds1_new.pkl', 'wb') as f:
    pickle.dump(tib_info, f)   

with open('fem_info_ds1_new.pkl', 'wb') as f:
    pickle.dump(fem_info, f)   

with open('ds1_angles_nw_new.pkl', 'wb') as f:
    pickle.dump(ds1_angles_nw_new, f)   


#%%
# create dataframe from the angle array.. but the angles have not been modified, and neither have the 
def create_angle_dataframe(angle_array, dataset_id):
    """
    Create a dataframe from an angle array with specified columns.
    
    :param angle_array: Array of angle values.
    :param dataset_id: Identifier for the dataset.
    :return: Dataframe with columns ['Frame', 'Dataset', 'angle', 'Condition'].
    """
    # Create a dataframe
    df = pd.DataFrame({
        'Frame': range(len(angle_array)),  # Frame numbers as index
        'Dataset': dataset_id,             # Dataset identifier
        'angle': angle_array,              # Angle values
        'Condition': 'Manual'              # Condition set to 'Manual'
    })
    
    return df


ds1_angle_df_new = create_angle_dataframe(ds1_angles_nw_new, 1)

#%%
with open('manual_segmentation/manual_point_master.pkl', 'wb') as f:
    pickle.dump(manual_point_master, f)   
#%%
# now that all the datasets are here.. i need to create a man_seg df that contains all of them .. 
dfs = [df1, df2, df3, df4, df5]
combined_df = pd.concat(dfs, ignore_index=True)

#%%
# now need to add the percent flexed column 
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

manual_master_df = add_percent_flexed(manual_master_df)

#%%
# the original plotting functon used in the thesis 
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
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])['angle'].mean().reset_index() # this was the old line without defining mean and std 
    #grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)
    
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
    
    # Filter out a specific dataset if needed
    #final_data = grouped[grouped['Dataset'] != 6]
    
    # Plotting the data
    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    custom_palette = {'Manual': default_palette[1], 'Unloaded': default_palette[0]}
    
    
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='angle',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci='sd',# Uses standard deviation for the confidence intervals
        palette=custom_palette
    )
    
    
    # Manually change the labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['Auto' if label == 'Unloaded' else label for label in labels]
    plt.legend(handles=handles, labels=new_labels)
    
    
    
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Average Angle [°]")
    plt.title("Angle between the long axis of tibia and femur segments")
    plt.grid(True)
    plt.savefig('ds_ismrm\man_vs_auto_angle_agg.svg', dpi=500)
    
    return 

plot_binned_angle_data(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'  ], 10)

#%%
# the code that plots two columns \ this still bins the data into intervals of 10 flexion % .. 
def plot_individual_angle(df, bin_width, datasets):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    
    for i, dataset in enumerate(datasets):
        # Filter data for the current dataset
        dataset_df = df_copy[df_copy['Dataset'] == dataset]
        
        # Group by 'Condition' and 'Custom_Bin' to calculate means
        grouped = dataset_df.groupby(['Condition', 'Custom_Bin']).agg(
            angle_mean=('angle', 'mean'),
            angle_std=('angle', 'std')
        ).reset_index()
        grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
        
        # Plot the data
        sns.lineplot(
            data=grouped,
            x='Bin_Center',
            y='angle_mean',
            hue='Condition',
            marker="o",  # Adds markers to each data point
            ax=axes[i],
        )
        
        axes[i].axvline(x=0, color='gray', linestyle='--')
        axes[i].set_xlabel("Flexion percentage [%]")
        axes[i].set_title(f"Dataset {dataset}")
        axes[i].grid(True)
        
        if i == 0:
            axes[i].set_ylabel("Average Angle [°]")
        else:
            axes[i].get_legend().remove()  # Remove individual legends
    # Create a single legend for the entire figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, title='Condition', loc='upper center', ncol=2)
    
    # Manually change the labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['Auto' if label == 'Unloaded' else label for label in labels]
    plt.legend(handles=handles, labels=new_labels)
    
    plt.tight_layout()
    #plt.savefig('ds1_5_man_v_auto_angle.svg',dpi=300)
    plt.show()

# Example usage
datasets_to_plot = [1, 2, 3, 4, 5]
plot_individual_angle(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], 10, datasets_to_plot)


#%%
# !!! do this for comparing just two.  here is a function that does not do the aggregation.. since we are only interested in individual datasets.. plotting the y values directly from the x values no binns 
def plot_individual_angle_no_binning(df, datasets): 
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    for i, dataset in enumerate(datasets):
        # Filter data for the current dataset and each condition
        dataset_df = df[df['Dataset'] == dataset]

        # Plot the data directly without binning
        sns.lineplot(
            data=dataset_df,
            x='Percent Flexed',
            y='angle',
            hue='Condition',
            marker="o",
            ax=axes[i]
        )

        axes[i].axvline(x=0, color='gray', linestyle='--')
        axes[i].set_xlabel("Flexion percentage [%]")
        axes[i].set_title(f"Dataset {dataset}")
        axes[i].grid(True)  # Include grid for better visualization
        if i == 0:
            axes[i].set_ylabel("Angle [°]")
        else:
            axes[i].get_legend().remove()  # Remove individual legends from other plots to avoid clutter

    plt.tight_layout()
    plt.savefig('direct_angle_all_datasets.svg', dpi=300)
    plt.show()

# Example usage
datasets_to_plot = [1,2,3,4, 5]
plot_individual_angle_no_binning(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], datasets_to_plot)

#%% 
# to plot each dataset manual vs auto in two rows 

def plot_individual_angle_no_binning(df, datasets):
    # Define rows and columns for the subplot grid
    rows = 2
    columns = 3  # Keeping it consistent with three columns for visual symmetry

    fig, axes = plt.subplots(rows, columns, figsize=(18, 12), sharey=True)
    axes = axes.ravel()  # Flatten the axes array for easier iteration

    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset]
        
        # Use seaborn lineplot to plot the data
        sns.lineplot(
            data=dataset_df,
            x='Percent Flexed',
            y='angle',
            hue='Condition',
            marker="o",
            ax=axes[i]
        )
        
        # Add vertical line at x=0
        axes[i].axvline(x=0, color='gray', linestyle='--')
        axes[i].set_xlabel("Flexion percentage [%]")
        axes[i].set_title(f"Dataset {dataset}")
        axes[i].grid(True)
        
        # Set y-axis label for all subplots
        axes[i].set_ylabel("Angle [°]")

        # Handle the legend for each individual subplot
        handles, labels = axes[i].get_legend_handles_labels()
        new_labels = ['Auto' if label == 'Unloaded' else label for label in labels]
        
        # Add the legend for the first subplot, but not the others
        if i == 0:
            axes[i].legend(handles=handles, labels=new_labels)
        else:
            axes[i].get_legend().remove()  # Remove legends for other subplots

    # Turn off unused subplot axes if there are fewer datasets than subplots
    for j in range(i + 1, rows * columns):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('man_vs_auto_angles_nonagg.svg', dpi=500)
    plt.show()

# Example usage
datasets_to_plot = [1, 2, 3, 4, 5]
plot_individual_angle_no_binning(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], datasets_to_plot)



#%%
# !! !  now plotting the derivative of the graph above: 
# when we are comparing just two use this code: 
def plot_derivative_by_condition(df, datasets):

    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset].sort_values(by='Percent Flexed')  # Ensure data is sorted
        
        # Plotting derivatives for each condition within the dataset
        conditions = dataset_df['Condition'].unique()  # Get unique conditions
        for condition in conditions:
            condition_df = dataset_df[dataset_df['Condition'] == condition]
            x = condition_df['Percent Flexed'].values
            y = condition_df['angle'].values
            
            # Calculate slopes and midpoints
            slopes = ( y[1:] - y[:-1] ) / (x[1:] - x[:-1] )  # np.abs ( (y[1:] - y[:-1]) / (x[1:] - x[:-1]) )  
            midpoints = (x[:-1] + x[1:]) / 2

            # Label conditions appropriately
            label = 'Auto' if condition == 'Unloaded' else 'Manual'
            
            # Plotting the derivative
            axes[i].plot(midpoints, slopes, marker='o', label=f'{label} - Dataset {dataset}')
            axes[i].set_xlabel("Flexion Percentage [%]")
            axes[i].set_ylabel("Rate of Change of Angle [°/%]")
            axes[i].set_title(f"Rate of Change by Condition in Dataset {dataset}")
            axes[i].grid(True)
        
        axes[i].legend()

    plt.tight_layout()
    #plt.savefig('derivate_angle_1_5.svg', dpi=300)
    plt.show()

# Example usage
datasets_to_plot = [1,2]
plot_derivative_by_condition(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], datasets_to_plot)


#%% 
# !!! when we are comparing all the datasets in one, use this codew: 
def plot_derivative_by_condition(df, datasets):
    # Determine the number of rows and columns for the subplots based on the number of datasets
    rows = 2
    columns = 3  # Use 3 columns to keep the figure width consistent

    fig, axes = plt.subplots(rows, columns, figsize=(18, 12), sharey=True)
    axes = axes.ravel()  # Flatten the axes array for easier iteration
    
    
    all_slopes_data = []  # List to store slope data for DataFrame
    
    std_dev_data = []
    
    # Loop through each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset].sort_values(by='Percent Flexed')
        conditions = dataset_df['Condition'].unique()
        
        for condition in conditions:
            condition_df = dataset_df[dataset_df['Condition'] == condition]
            x = condition_df['Percent Flexed'].values
            y = condition_df['angle'].values
            
            # Calculate slopes and midpoints
            slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            midpoints = (x[:-1] + x[1:]) / 2
            
            
            for slope in slopes:
                all_slopes_data.append({
                    'Dataset': dataset,
                    'Condition': condition,
                    'Slope': slope
                })

            
            
            # Split data into two halves around the 0% flexion point
            negative_half_indices = midpoints < 0
            positive_half_indices = midpoints > 0
            
            negative_half_slopes = slopes[negative_half_indices]
            positive_half_slopes = slopes[positive_half_indices]

            # Calculate standard deviation for each half
            std_neg_half = np.std(negative_half_slopes)
            std_pos_half = np.std(positive_half_slopes)

            # Print out standard deviations
            condition_label = 'Auto' if condition == 'Unloaded' else 'Manual'
            print(f"Dataset {dataset}, Condition {condition_label}:")
            print(f"  Standard deviation of slopes from -100% to 0%: {std_neg_half:.3f}")
            print(f"  Standard deviation of slopes from 0% to 100%: {std_pos_half:.3f}")
            # Set label based on condition
            label = 'Auto' if condition == 'Unloaded' else 'Manual'
            
            # Plotting
            axes[i].plot(midpoints, slopes, marker='o', label=f'{label} - Dataset {dataset}')
            axes[i].set_xlabel("Flexion Percentage [%]")
            axes[i].set_ylabel("Rate of Change of Angle [°/%]")
            axes[i].set_title(f" Dataset {dataset}")
            axes[i].grid(True)
            axes[i].legend()

    # If there are fewer datasets than subplots, turn off the unused axes
    for j in range(i + 1, rows * columns):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('ds_ismrm\derivate_angle_all_datasets.svg', dpi=500)
    plt.show()
    
    # Create DataFrame from collected data
    slope_df = pd.DataFrame(all_slopes_data)
    
    
    return slope_df

# Example usage
datasets_to_plot = [1, 2, 3, 4, 5]
slope_df = plot_derivative_by_condition(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], datasets_to_plot)



#%%

def plot_derivative_by_condition(df, datasets):
    rows = 2
    columns = 3  # Use 3 columns to keep the figure width consistent

    fig, axes = plt.subplots(rows, columns, figsize=(18, 12), sharey=True)
    axes = axes.ravel()  # Flatten the axes array for easier iteration
    
    all_slopes_data = []  # List to store slope data for DataFrame
    std_dev_data = []  # List to store standard deviation data
    
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset].sort_values(by='Percent Flexed')
        conditions = dataset_df['Condition'].unique()
        
        for condition in conditions:
            condition_df = dataset_df[dataset_df['Condition'] == condition]
            x = condition_df['Percent Flexed'].values
            y = condition_df['angle'].values
            
            slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            midpoints = (x[:-1] + x[1:]) / 2
            
            for slope in slopes:
                all_slopes_data.append({
                    'Dataset': dataset,
                    'Condition': condition,
                    'Slope': slope
                })
            
            negative_half_indices = midpoints < 0
            positive_half_indices = midpoints > 0
            
            negative_half_slopes = slopes[negative_half_indices]
            positive_half_slopes = slopes[positive_half_indices]

            std_neg_half = np.std(negative_half_slopes)
            std_pos_half = np.std(positive_half_slopes)

            condition_label = 'Auto' if condition == 'Unloaded' else 'Manual'
            std_dev_data.append({
                'Dataset': dataset,
                'Condition': condition_label,
                'Half': 'Negative',
                'Std Dev': std_neg_half
            })
            std_dev_data.append({
                'Dataset': dataset,
                'Condition': condition_label,
                'Half': 'Positive',
                'Std Dev': std_pos_half
            })

            axes[i].plot(midpoints, slopes, marker='o', label=f'{condition_label} - Dataset {dataset}')
            axes[i].set_xlabel("Flexion Percentage [%]")
            axes[i].set_ylabel("Rate of Change of Angle [°/%]")
            axes[i].set_title(f"Rate of Change by Condition in Dataset {dataset}")
            axes[i].grid(True)
            axes[i].legend()

    for j in range(i + 1, rows * columns):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
    slope_df = pd.DataFrame(all_slopes_data)
    std_dev_df = pd.DataFrame(std_dev_data)
    
    return slope_df, std_dev_df

# Example usage
datasets_to_plot = [1, 2, 3, 4, 5]
slope_df, std_dev_df = plot_derivative_by_condition(master_df_inverted[master_df_inverted['Condition'] != 'Loaded'], datasets_to_plot)

# Plotting the standard deviations
fig, ax = plt.subplots(figsize=(12, 8))
std_dev_df.pivot_table(index=['Dataset', 'Condition'], columns='Half', values='Std Dev').plot(kind='bar', ax=ax)
ax.set_ylabel('Standard Deviation of Slopes')
ax.set_title('Standard Deviation of Slopes by Condition and Dataset')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%

slope_df['Slope'].replace([-np.inf, np.inf], np.nan, inplace=True)


df_clean = slope_df.dropna()

std_dev = slope_df.groupby(['Dataset', 'Condition'])['Slope'].std().reset_index()

print(std_dev)


#%%
#
#
#
#
#
#
# the distance stuff comes below this 

#%%
master_df_angle_both = pd.concat([modified_angle_df, df_angle ])
#%%
# the function modified to just print out the standard deviations 
def plot_binned_angle_data_print(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
    
    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means and std deviations
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset']).agg(
        angle_mean=('angle', 'mean'),
        angle_std=('angle', 'std')
    ).reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)

    # Calculate the overall mean and std deviation for each bin and condition
    overall_stats = grouped.groupby(['Condition', 'Custom_Bin']).agg(
        angle_mean=('angle_mean', 'mean'),
        angle_std=('angle_mean', 'std'),
        ci_lower=('angle_mean', lambda x: x.mean() - x.std()),
        ci_upper=('angle_mean', lambda x: x.mean() + x.std())
    ).reset_index()
    overall_stats['Bin_Center'] = overall_stats['Custom_Bin'].apply(lambda x: x.mid)
    
    # Print the standard deviations to quantify which condition is more 'stable'
    print("Standard Deviations for Each Condition and Bin:")
    print(overall_stats.pivot(index='Bin_Center', columns='Condition', values='angle_std'))

    # Plotting the data
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=overall_stats,
        x='Bin_Center',
        y='angle_mean',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci=None,  # We've calculated the CI manually
    )
    
    for condition in overall_stats['Condition'].unique():
        condition_data = overall_stats[overall_stats['Condition'] == condition]
        plt.fill_between(
            condition_data['Bin_Center'],
            condition_data['ci_lower'],
            condition_data['ci_upper'],
            alpha=0.2
        )

    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Average Angle [°]")
    plt.title("Angle between the long axis of tibia and femur segments")
    plt.grid(True)

    plt.show()

    return 

# Example usage
plot_binned_angle_data_print(master_df_angle, 10)

#%%
#the functino that plots bottom and top the standard deviation bar chart and the main graph 
def plot_binned_angle_data(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
    
    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means and std deviations
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset']).agg(
        angle_mean=('angle', 'mean'),
        angle_std=('angle', 'std')
    ).reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
    
    # Calculate the overall mean and std deviation for each bin and condition
    overall_stats = grouped.groupby(['Condition', 'Custom_Bin']).agg(
        angle_mean=('angle_mean', 'mean'),
        angle_std=('angle_mean', 'std'),
        ci_lower=('angle_mean', lambda x: x.mean() - x.std()),
        ci_upper=('angle_mean', lambda x: x.mean() + x.std())
    ).reset_index()
    overall_stats['Bin_Center'] = overall_stats['Custom_Bin'].apply(lambda x: x.mid)
    
    # Print the standard deviations to quantify which condition is more 'stable'
    std_devs = overall_stats.pivot(index='Bin_Center', columns='Condition', values='angle_std')
    print("Standard Deviations for Each Condition and Bin:")
    print(std_devs)
    
    # Plotting the data
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Plot the angle means with confidence intervals
    sns.lineplot(
        data=overall_stats,
        x='Bin_Center',
        y='angle_mean',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci=None,  # We've calculated the CI manually
        ax=axes[0]
    )
    
    for condition in overall_stats['Condition'].unique():
        condition_data = overall_stats[overall_stats['Condition'] == condition]
        axes[0].fill_between(
            condition_data['Bin_Center'],
            condition_data['ci_lower'],
            condition_data['ci_upper'],
            alpha=0.2
        )

    axes[0].axvline(x=0, color='gray', linestyle='--')
    axes[0].set_xlabel("Flexion percentage [%]")
    axes[0].set_ylabel("Average Angle [°]")
    axes[0].set_title("Angle between the long axis of tibia and femur segments")
    axes[0].grid(True)

    # Plot the standard deviations
    std_devs.plot(kind='bar', ax=axes[1])
    axes[1].set_xlabel("Flexion percentage [%]")
    axes[1].set_ylabel("Standard Deviation [°]")
    axes[1].set_title("Standard Deviations of Angles by Condition and Bin")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    return 

plot_binned_angle_data(master_df_angle, 10)

#%%
# function that just plots the bar chart of standard deviations 
def plot_standard_deviations(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means and std deviations
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset']).agg(
        angle_mean=('angle', 'mean'),
        angle_std=('angle', 'std')
    ).reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Calculate the overall mean and std deviation for each bin and condition
    overall_stats = grouped.groupby(['Condition', 'Custom_Bin']).agg(
        angle_mean=('angle_mean', 'mean'),
        angle_std=('angle_mean', 'std')
    ).reset_index()
    overall_stats['Bin_Center'] = overall_stats['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Print the standard deviations to quantify which condition is more 'stable'
    std_devs = overall_stats.pivot(index='Bin_Center', columns='Condition', values='angle_std')
    std_devs = std_devs.rename(columns={'Unloaded': 'Auto'})
    print("Standard Deviations for Each Condition and Bin:")
    print(std_devs)
    
    # Plot the standard deviations
    ax = std_devs.plot(kind='bar', figsize=(10, 6))
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Standard Deviation [°]")
    plt.title("Standard Deviations of Angles by Condition and Bin")
    plt.grid(True)
    
    # Adjust the legend to be horizontal and placed at the upper left
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Condition', loc='upper left', ncol=2)
    
    # Adjust the x-tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('bar_chart.svg', dpi=300)
    plt.show()
    

# Example usage
plot_standard_deviations(master_df_angle[master_df_angle['Condition'] != 'Loaded'], 10)

#%%
#using the exact shape from the auto datasets .. the first task is to extract just the first frame from the dragged and dropped shape. 

tib = viewer.layers['MM_NW_tib_shape'].data[0]

viewer.add_shapes(tib, shape_type='polygon', name='tib')

fem = viewer.layers['MM_NW_fem_shape'].data[0]

viewer.add_shapes(fem, shape_type='polygon', name='fem')

#%%

# Now doing the same thing but for the distance dataframe 

# first thing is to clean up the dataset 

# cleaned up above .. 

# now to nmodify the creating function, because the previous one used the W column # the code can be found in the seaborn tutorial python file 

def create_points_arrays(fem_NW_name, tib_NW_name, fem_index, tib_index): 
    fem_shape = viewer.layers[fem_NW_name].data
    tib_shape = viewer.layers[tib_NW_name].data
    
    fem_points_NW = np.array([item[fem_index] for item in fem_shape]) 
    tib_points_NW = np.array([item[tib_index] for item in tib_shape])
    
    return fem_points_NW, tib_points_NW


fem_points_NW, tib_points_NW = create_points_arrays('ds5_fem_frm_auto', 'ds5_tib_frm_auto',39,2)

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
df5_point = create_condition_df(fem_points_NW, tib_points_NW, 'Manual')

df5_point['Dataset'] = 5

#%%
def add_norm(df):
    df['Norm'] = np.sqrt(
    (df['Femur_X'] - df['Tibia_X'])**2 +
    (df['Femur_Y'] - df['Tibia_Y'])**2
)
    
add_norm(df5_point)  


#%%
manual_point_master  = pd.concat([manual_master_df, master_df ], ignore_index= True)

#%%
# before adding the relative norm column as well as the percent flex3ed, it might be worth it to first concat all the manual point dfs, and then do this, it will be much more efficient 
# afterwards, we can then concatenate this and the master df for plotting 

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
manual_master_df = add_relative_norm_column(manual_master_df)


#%%
# now time to do the plot 

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
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset'])['Relative Norm'].mean().reset_index()
    #grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: (x.left + x.right) / 2)
    
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
    
    # Filter out a specific dataset if needed
    #final_data = grouped[grouped['Dataset'] != 6]
    
    # Plotting the data
    # Prepare to plot
    plt.figure(figsize=(10, 6))
    default_palette = sns.color_palette()
    #custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}
    
    
    # Plotting the data
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='Relative Norm',
        hue='Condition',
        marker="o",
        ci='sd',  # Uses standard deviation for the error bars

    )
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Euclidean Distance (mm)")
    plt.title("Variation of distance with respect to flexion-extension cycle")
    plt.grid(True)
    plt.savefig('distance_manual_compare.png', dpi=300)
    plt.show()
    
    return 

#plot_binned_angle_data(master_df_angle[master_df_angle['Condition'] != 'Loaded'  ], 10)
plot_binned_angle_data(master_df_point[master_df_point['Condition'] != 'Loaded'  ], 10)

#%%
# now doing the standard deviation 

# function that just plots the bar chart of standard deviations 
def plot_standard_deviations_dist(df, bin_width):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Group by 'Condition', the new 'Custom_Bin', and 'Dataset' to calculate means and std deviations
    grouped = df_copy.groupby(['Condition', 'Custom_Bin', 'Dataset']).agg(
        dist_mean=('Relative Norm', 'mean'),
        dist_std=('Relative Norm', 'std')
    ).reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Calculate the overall mean and std deviation for each bin and condition
    overall_stats = grouped.groupby(['Condition', 'Custom_Bin']).agg(
        dist_mean=('dist_mean', 'mean'),
        dist_std=('dist_mean', 'std')
    ).reset_index()
    overall_stats['Bin_Center'] = overall_stats['Custom_Bin'].apply(lambda x: round(x.mid, 1))
    
    # Print the standard deviations to quantify which condition is more 'stable'
    std_devs = overall_stats.pivot(index='Bin_Center', columns='Condition', values='dist_std')
    std_devs = std_devs.rename(columns={'Unloaded': 'Auto'})
    print("Standard Deviations for Each Condition and Bin:")
    print(std_devs)
    
    # Plot the standard deviations
    ax = std_devs.plot(kind='bar', figsize=(10, 6))
    plt.xlabel("Flexion percentage [%]")
    plt.ylabel("Standard Deviation [mm]")
    plt.title("Standard Deviations of Distances by Condition and Bin")
    plt.grid(True)
    
    # Adjust the legend to be horizontal and placed at the upper left
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Condition', loc='upper left', ncol=2)
    
    # Adjust the x-tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('bar_chart_dist.svg', dpi=300)
    plt.show()
    

# Example usage
plot_standard_deviations_dist(master_df_point[master_df_point['Condition'] != 'Loaded'], 10)


#%%

# the code that plots two columns 
def plot_individual_angle(df, bin_width, datasets):
    # Make a copy of the DataFrame to ensure the original remains unchanged
    df_copy = df.copy()

    # Define bin edges that cover the entire expected range of flexion percentages
    bin_edges = list(range(-100, 101, bin_width))
    
    # Bin 'Percentage of Flexion' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
    
    # Create subplots for each dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(18, 6), sharey=True)
    
    for i, dataset in enumerate(datasets):
        # Filter data for the current dataset
        dataset_df = df_copy[df_copy['Dataset'] == dataset]
        
        # Group by 'Condition' and 'Custom_Bin' to calculate means
        grouped = dataset_df.groupby(['Condition', 'Custom_Bin']).agg(
            angle_mean=('Relative Norm', 'mean'),
            angle_std=('Relative Norm', 'std')
        ).reset_index()
        grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
        
        # Plot the data
        sns.lineplot(
            data=grouped,
            x='Bin_Center',
            y='angle_mean',
            hue='Condition',
            marker="o",  # Adds markers to each data point
            ax=axes[i],
        )
        
        axes[i].axvline(x=0, color='gray', linestyle='--')
        axes[i].set_xlabel("Flexion percentage [%]")
        axes[i].set_title(f"Dataset {dataset}")
        axes[i].grid(True)
        
        if i == 0:
            axes[i].set_ylabel("Euclidean Distance [mm]")
        else:
            axes[i].get_legend().remove()  # Remove individual legends
    # Create a single legend for the entire figure
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, title='Condition', loc='upper center', ncol=2)
    
    plt.tight_layout()
    #plt.savefig('ds3_4_man_v_auto_distance.svg',dpi=250)
    plt.show()

# Example usage
datasets_to_plot = [3, 4]
plot_individual_angle(master_df_point[master_df_point['Condition'] != 'Loaded'], 10, datasets_to_plot)

#%%
automatic_cv = [22.607229, 46.055132, 32.701069, 32.41297, 36.419106  ]
manual_cv = [56.035579, 58.6315, 52.060096, 70.945118, 59.842318 ]


#%%

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Create the box plots without outlier points
positions = [1, 1.7]  # Adjust these values to bring boxes closer
box_plot = ax.boxplot([automatic_cv, manual_cv], positions=positions, widths=0.5, 
                      patch_artist=True, showfliers=False, whis=[0,100])

# Color the boxes
colors = ['#1f77b4', '#ff7f0e']  # Blue for automatic, orange for manual
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)  # Add some transparency

# Add individual data points
for i, dataset in enumerate([automatic_cv, manual_cv]):
    x = np.random.normal(positions[i], 0.04, len(dataset))  # Add some x-jitter
    ax.scatter(x, dataset, alpha=0.7, c=colors[i], edgecolor='black', s=50)

# Add labels and title
ax.set_xticks(positions)
ax.set_xticklabels(['Automatic', 'Manual'])
ax.set_ylabel('Coefficient of Variation (%)')
#ax.set_title('Comparison of CV between Automatic and Manual Segmentation Methods')

# Set y-axis to start from 0
ax.set_ylim(0, max(max(automatic_cv), max(manual_cv)) + 3)

# Set x-axis limits to center the boxes
ax.set_xlim(0.5, 2.2)

# Add a grid for better readability
ax.yaxis.grid(True)

# Show the plot
plt.tight_layout()

plt.savefig('manuscript\\boxplot_cv.png', dpi=600, bbox_inches='tight', format='png')

plt.show()



#%%


automatic_cv = [22.607229, 46.055132, 32.701069, 32.41297, 36.419106]
manual_cv = [56.035579, 58.6315, 52.060096, 70.945118, 59.842318]

def print_stats(data, name):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    print(f"{name} stats:")
    print(f"Min: {min(data):.2f}, Max: {max(data):.2f}")
    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower whisker: {max(min(data), q1 - 1.5*iqr):.2f}")
    print(f"Upper whisker: {min(max(data), q3 + 1.5*iqr):.2f}\n")

print_stats(automatic_cv, "Automatic")
print_stats(manual_cv, "Manual")

fig, ax = plt.subplots()
box_plot = ax.boxplot([automatic_cv, manual_cv], labels=['Automatic', 'Manual'])
plt.show()
