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
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/modified_angle_df.pkl', 'rb') as file:
    modified_angle_df_df = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds2_angles_df.pkl', 'rb') as file:
    ds2_angles_nw = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds3_angles_df.pkl', 'rb') as file:
    ds3_angles_nw = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds4_angles_df.pkl', 'rb') as file:
    ds4_angles_nw = pickle.load(file)

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/ds5_angles_df.pkl', 'rb') as file:
    ds5_angles_nw = pickle.load(file)        
#%%
# Step 1: load the image from directory and normalize it
path_neg = 'C:/Users/Aayush/Documents/thesis_files/manual_segmentation_datasets/JL_7_dataset5/JL_NW_ai2_5e-2_neg_ngn.nii'
path_pos ='C:/Users/Aayush/Documents/thesis_files/manual_segmentation_datasets/JL_7_dataset5/JL_NW_ai2_5e-2_pos_ngn.nii'
#%%
image_neg = path_to_image(path_neg)[1:]
image_pos = path_to_image(path_pos)[1:] 
#%%
# since our image goes from extened to flexed.. the direction means, pos is going down.. and neg is coming up 
# which means. if we want to present our data as going up then coming down .. we have to reverse the neg, put it at the first half. 
image_neg = image_neg[::-1]
#%%
full_image = np.concatenate( (image_neg, image_pos) , axis=0)

#%%
#add the original image to napari
viewer = napari.view_image(full_image,  name='ds5_NW_full')

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

#%%

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
dataset_7_NW_angles = calculate_and_plot_angles_between_bones(JL_NW_fem_info_s, JL_NW_tib_info_s, name='dataset_7_NW')



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
ds5_angles_nw = calculate_and_plot_angles_between_bones(tib_info, fem_info, name='ds_5_NW')

#%%
with open('tib_info_ds5.pkl', 'wb') as f:
    pickle.dump(tib_info, f)   

with open('fem_info_ds5.pkl', 'wb') as f:
    pickle.dump(fem_info, f)   

with open('ds5_angles_nw.pkl', 'wb') as f:
    pickle.dump(ds5_angles_nw, f)   


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


ds5_angle_df = create_angle_dataframe(ds5_angles_nw, 5)

#%%
with open('master_df_angle_both.pkl', 'wb') as f:
    pickle.dump(master_df_angle_both, f)   
#%%
# now that all the datasets are here.. i need to create a man_seg df that contains all of them .. 
dfs = [ds1_angles_nw, ds2_angles_nw, ds3_angles_nw, ds4_angles_nw, ds5_angles_nw]
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

combined_df = add_percent_flexed(combined_df)
#%%
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
    #custom_palette = {'Loaded': default_palette[1], 'Unloaded': default_palette[0]}
    
    
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y='angle',
        hue='Condition',
        marker="o",  # Adds markers to each data point
        ci='sd',# Uses standard deviation for the confidence intervals
        #palette=custom_palette
    )

    return 

plot_binned_angle_data(master_df_angle_both, 10)

#%%
master_df_angle_both = pd.concat([modified_angle_df, df_angle ])
