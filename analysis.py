#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:47:23 2024

@author: aayush
"""
import numpy as np 
import matplotlib.pylab as plt 
from shapely.geometry import LineString, MultiPoint
import napari

from utils import (shapes_for_napari, apply_transformations_new, coords_to_boolean, process_frame, show_stuff)

#%%
def plot_phi_changes(transformation_matrices, reference_frame_index):
    # Extract phi angles and convert to degrees
    phis = [np.rad2deg(transformation[2]) for transformation in transformation_matrices]
    
    # Adjust phis based on the reference frame
    reference_phi = phis[reference_frame_index]
    adjusted_phis = [phi - reference_phi for phi in phis]
    
    # Calculate cumulative phis from the reference frame
    if reference_frame_index == 0:
        # If the reference frame is the first frame, calculate cumulative sum directly
        cumulative_phis = np.cumsum(adjusted_phis)
    else:
        # If the reference frame is not the first, reverse the list before cumulative sum
        cumulative_phis = np.cumsum(adjusted_phis[::-1])[::-1]
    
    # Generate the theoretical perfect line with 1-degree increments
    if reference_frame_index == 0:
        perfect_line = np.arange(0, -len(cumulative_phis), -1)
    else:
        #perfect_line = np.arange(len(cumulative_phis) - 1, -1, -1)
        first_phi_value = cumulative_phis[0]
        perfect_line = np.arange(first_phi_value, first_phi_value - len(cumulative_phis), -1)
    
    # Calculate the residuals
    residuals = cumulative_phis - perfect_line
    
    # Plotting the original data
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_phis, marker='o', label='Measured Data')
    plt.plot(perfect_line, color='red', linestyle='--', label='Perfect 1-degree Line')
    plt.xticks(ticks=range(len(cumulative_phis)), labels=(range(len(cumulative_phis)) if reference_frame_index == 0 else range(len(cumulative_phis) - 1, -1, -1)))
    plt.title(f"Measured Data vs. Perfect 1-degree Line (Using frame {reference_frame_index} as reference)")
    plt.xlabel("Rotary angle encoder")
    plt.ylabel("Rotation angle of tibia (in degrees)")
    plt.grid(True)
    plt.legend()
    
    # Plotting the residuals
    plt.subplot(2, 1, 2)
    plt.plot(residuals, marker='o', color='green')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks(ticks=range(len(cumulative_phis)), labels=(range(len(cumulative_phis)) if reference_frame_index == 0 else range(len(cumulative_phis) - 1, -1, -1)))
    plt.title("Deviation from Perfect 1-degree Line")
    plt.xlabel("Rotary angle encoder")
    plt.ylabel("Deviation (degrees)")
    plt.grid(True)
    
    # Show the plot with both the original data and the residuals
    plt.tight_layout()
    plt.show()
    
    # Print out the mean squared error of the residuals
    mse = np.mean(residuals**2)
    print(f"Mean Squared Error of the deviation: {mse:.4f}")


plot_phi_changes(transformation_matrices_last[1:], -1)

#%%
def plot_cost_values(values):
    # Ensure the input is a list or a numpy array
    if not isinstance(values, (list, np.ndarray)):
        raise ValueError("Input should be a list or numpy array of numerical values.")

    # Generate the item numbers (indices) for the x-axis
    item_numbers = range(len(values))
    
    # Create the plot
    plt.figure(figsize=(10, 6))  # Adjust the size to your preference
    plt.plot(item_numbers, values, marker='o', linestyle='-')  # Plot values vs. item numbers
    
    # Adding labels and title
    plt.title("Optimized overlap distance per frame")
    plt.xlabel("Rotary Angle Encoder")
    plt.ylabel("Minimized Cost function Value")
    plt.xticks(ticks=item_numbers)
    #plt.minorticks_on()
    # Adding grid for better readability
    plt.grid(True)
    
    # Show the plot
    plt.show()
    print('The sum of all the cost function values is:', np.sum(values))
plot_cost_values(cost_values_last)

#%%
# use a unblurred image 
path1 = '/data/projects/ma-nepal-segmentation/data/Singh^Udai/2023-09-11/72_MK_Radial_NW_CINE_60bpm_CGA/aw2_rieseling_admm_tgv_5e-3.nii'
image1 = open_nii(path1)
image1 = normalize(image)
#image1 = np.moveaxis(image, 1, 0)[1:]
#%%
viewer1 = napari.view_image(image)
#%%
# add the reference points and manually segment the reference frame 
viewer1.add_shapes(new_tib_coords_last[-1], shape_type='polygon')
#%%
# rename it to expanded_shape and then store it as ref_points variable 
ref_points = viewer1.layers['expanded_shape'].data[0]
#%%
applied_transformation = apply_transformations_new(ref_points, transformation_matrices_last, 24)    
viewer1.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='red')

#%%
tib_label = coords_to_boolean(new_tib_coords_last, image.shape)

total_frames = len(new_tib_coords_last) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer = viewer1.layers["Shapes"].to_labels(image.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(100,410)
yrange=slice(150,400)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(image[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('test.svg')
#%%
# After manually segmenting, find the info of the shapes. 
tib_info = process_frame(viewer1.layers['Shapes'].data)
#%%
show_stuff(tib_info, 'tib_nowt', viewer1)    
#%%
def track_origin(all_frame_info, bone_name):
    # Extract x and y coordinates of the origin for each frame
    x_coords = [all_frame_info[frame]['origin'][0] for frame in sorted(all_frame_info)]
    y_coords = [all_frame_info[frame]['origin'][1] for frame in sorted(all_frame_info)]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_coords, x_coords, c=sorted(all_frame_info), cmap='viridis', s=50)
    plt.plot(y_coords, x_coords, '-o', markersize=5, alpha=0.6)
    plt.colorbar(label='Frame Number')
    plt.title(f'Movement of {bone_name} Origin Over Frames')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()
    
def calculate_angle(vector_a, vector_b):
    """Calculate angle in degrees between two vectors."""
    cos_theta = np.dot(vector_a, vector_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Assuming femur_info and tibia_info have the same keys (frames)
def plot_angle_vs_frame(femur_info , tibia_info, label):
    frames = sorted(femur_info.keys())
    angles = []
    
    for frame in frames:
        femur_vector = femur_info[frame]['U']
        tibia_vector = tibia_info[frame]['U']
        angle = calculate_angle(femur_vector, tibia_vector)
        angles.append(angle)
    angles = (180 - np.array(angles) ) 
    # Plot
    plt.plot(frames, angles, label=f'{label}')
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Between long axes of Femur and Tibia Over Frames")
    plt.grid(True)
    plt.legend()
    plt.show()

#%%
track_origin(tib_info, 'tibia_NW_last_frame')
#%%
plot_angle_vs_frame(fem_info, tib_info, 'NW_US')