#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:47:23 2024

@author: aayush
"""
import pickle
import os 
#os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import matplotlib.pylab as plt 
import napari
from sklearn.metrics import mean_absolute_error

from utils import (path_to_image, shapes_for_napari, boolean_to_coords, apply_transformations_new, process_frame, show_stuff, dict_to_array, reconstruct_dict)


#%%
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/AN/01.03.24/AN_NW_fem_info.pkl', 'rb') as file:
    fem_info_NW =  pickle.load(file)
    
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/AN/01.03.24/AN_NW_tib_info.pkl', 'rb') as file:
    tib_info_NW =  pickle.load(file)    

#%%    
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/AN/01.03.24/AN_W_tib_info_using_NW_ref.pkl', 'rb') as file:
    tib_info_W =  pickle.load(file)
    
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/AN/01.03.24/AN_W_fem_info_using_NW_ref.pkl', 'rb') as file:
    fem_info_W =  pickle.load(file)
#%%

def plot_transformations_and_calculate_MAE(transformation_matrices, offset, angle_increment, reference_index, condition, residuals_color, ax=None):
    # Extract phi values from transformation matrices and convert to degrees
    phis = [np.rad2deg(transformation[2]) for transformation in transformation_matrices]
    if np.sum(phis) < 0:
        phis = [-1 * value for value in phis]
    # Adjust phis and calculate cumulative_phis based on the reference frame
    if reference_index == -1:  # Reference frame is last
        reversed_phis_excluding_last = phis[:-1][::-1]
        cumulative_phis_from_end_with_offset = [offset]
        for phi in reversed_phis_excluding_last:
            next_value = cumulative_phis_from_end_with_offset[-1] + phi
            cumulative_phis_from_end_with_offset.append(next_value)
        cumulative_phis_corrected_order = np.array(cumulative_phis_from_end_with_offset[::-1])
        cumulative_phis = cumulative_phis_corrected_order
        perfect_line = np.arange(offset + (len(phis) - 1) * angle_increment, offset - angle_increment, -angle_increment)
        print(cumulative_phis, 'cumulative_phis')
    elif reference_index == 0:  # Reference frame is first
        cumulative_phis = np.cumsum(phis)
        cumulative_phis = cumulative_phis - cumulative_phis[0] + offset
        perfect_line = np.arange(offset, offset + len(phis) * angle_increment, angle_increment)
    else:
        raise ValueError("reference_index must be 0 or -1")
    
    # Adjusting x-axis to start from the specified offset
    x_values_shifted = np.arange(offset, offset + len(cumulative_phis) * angle_increment, angle_increment)

    residuals = cumulative_phis - perfect_line
    mae = mean_absolute_error(cumulative_phis, perfect_line)

    # Check if an axis is provided, if not, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))
        ax1, ax2 = ax
        created_new_figure = True
    else:
        ax1, ax2 = ax
        created_new_figure = False
    
    # Adjusting labels depending on the condition
    cumulative_label = f'Cumulative Phi Data for {condition}'
    perfect_line_label = 'Perfect two degree line '
    residuals_label = f'Residuals for {condition}'
    
    
    # Plotting on the provided or new axes
    ax1.plot(x_values_shifted, cumulative_phis, label=cumulative_label, marker='o')
    ax1.plot(x_values_shifted, perfect_line, label=perfect_line_label, linestyle='--')
    ax1.set_title(f'Deviation from Perfect Line using {("last" if reference_index == -1 else "first")} frame as reference')
    ax1.set_xlabel("Rotary angle encoder")
    ax1.set_ylabel('Rotation angle of tibia (in degrees)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x_values_shifted)
    ax1.set_yticks(np.arange(np.min(perfect_line), np.max(perfect_line), 1))

    ax2.plot(x_values_shifted, residuals, label=residuals_label, marker='o', color=residuals_color)
    ax2.set_title('Residuals at Each Frame')
    ax2.set_xlabel('Frame (Starting from ' + str(offset) + ')')
    ax2.set_ylabel('Residual Value')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x_values_shifted)
    ax2.set_yticks(np.arange(np.floor(np.min(residuals)), np.ceil(np.max(residuals)) + 1, 1))
    #ax2.text(x=min(x_values_shifted), y=max(residuals), s=f"Mean Absolute Error (MAE): {mae:.2f}", color=residuals_color, fontsize=10)
    
    if condition == 'original_matrices':
        # For the original matrices, place text at minimum x and maximum y
        ax2.text(x=min(x_values_shifted), y=max(residuals), 
                s=f"MAE: {mae:.2f}", color=residuals_color, fontsize=10)
    else:  # Assuming this means it's the modified_matrices or any other condition
        # For the modified matrices, place text at maximum x and minimum y
        ax2.text(x=max(x_values_shifted) - 1, y=min(residuals), 
                s=f"MAE: {mae:.2f}", color=residuals_color, fontsize=10)
    
    if created_new_figure:
        plt.tight_layout()
        plt.savefig(f'matrix_angles_{condition}.svg') 
        plt.show()
    print(f"Mean Absolute Error (MAE): {mae}")

# Example usage:
#%%
# For a single plot
plot_transformations_and_calculate_MAE(transformation_matrices_first, offset=5, angle_increment= 2, reference_index=0, residuals_color='red', condition='US_W', ax=None)
#%%
# For overlaying multiple plots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
plot_transformations_and_calculate_MAE(tib_matrices, offset=5, angle_increment=2, reference_index=0, residuals_color='blue', condition='original_matrices', ax=ax)
plot_transformations_and_calculate_MAE(new_tibia_transforms, offset=5, angle_increment=2,reference_index=0,residuals_color='red', condition='modified_matrices', ax=ax)
plt.tight_layout()
plt.savefig('original_vs_modified_tib_matrices.svg')
plt.show()

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
    plt.savefig('cost_values_tib.svg')
    plt.show()
    print('The sum of all the cost function values is:', np.sum(values))
plot_cost_values(cost_values_first)
#%%
# use a unblurred image 
path1 = 'C:/Users/Aayush/Documents/thesis_files/_first_march_data/01.03/AN_W_ai2_tgv_5e-2_neg_ngn.nii'
image1 = path_to_image(path1)[2:]
#%%
viewer1 = napari.view_image(image1)
#%%
'''
attempt to do animation but isnt working  too well . or cant figure out how to use it properly 
from napari_animation import Animation
from skimage import io

# Create an animation object from the viewer
animation = Animation(viewer1)

# The number of frames in your animation
number_of_frames = 14  # Adjust this to match your animation's frame count

# Directory to save the frames
save_directory = "C:/Users/Aayush/Documents/thesis_files"  # Change this to your desired directory

# Iterate through each frame and save it
for frame_number in range(number_of_frames):
    # Set the viewer state to the state at the current frame
    animation.set_viewer_state(animation.key_frames[frame_number].viewer_state)

    # Capture the current viewer state as an image
    image = viewer1.screenshot()

    # Save the image
    file_path = f"{save_directory}/frame_{frame_number}.png"
    io.imsave(file_path, image)
'''
#%%
# add the reference points and manually segment the reference frame 
viewer1.add_shapes(reference_frame_first, shape_type='polygon')
#%%
# rename it to expanded_shape and then store it as ref_points variable 
#ref_points = viewer1.layers['expanded_fem'].data[0]
ref_points = viewer1.layers['AN_NW_fem_shape'].data[0][:,1:3]
#%%
applied_transformation = apply_transformations_new(ref_points, transformation_matrices_first, 0)    
viewer1.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='green')

#%%
# tib_label = coords_to_boolean(new_tib_coords_first, image1.shape)
tib_label = final_label

total_frames = len(tib_label) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer = viewer1.layers["fem_W"].to_labels(image1.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(70,350)
yrange=slice(150,350)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(image1[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    #ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('AN_W_segmented_fem.svg')

#%%
shapes_data = viewer1.layers['AN_NW_fem_shape'].data  # need to reverse if last frame is extended (or in the future, simply reverse the source image)


def process_and_transform_shapes(shapes_data, transformation_matrices, ref_index):
    # Process the reference frame (assuming the last one in the shapes data)
    single_shape_info = process_frame([shapes_data[0]])

    # Convert the dictionary into an array for transformation
    shape_info_array = dict_to_array(single_shape_info)

    # Apply transformations to the shape data
    transformed_info = apply_transformations_new(shape_info_array, transformation_matrices, ref_index)

    # Reconstruct the dictionary for each transformed frame
    transformed_dicts = {}
    for i, arr in enumerate(transformed_info):
        transformed_dicts.update(reconstruct_dict(i, arr))

    return transformed_dicts

fem_info_NW_test = process_and_transform_shapes(shapes_data, transformation_matrices_first, 0)
#%%
'''
needs a lot more work so just leave it as is 
shapes_data_test = viewer1.layers['tibia_NW copy'].data.astype(bool)
shape_data_coords = boolean_to_coords(shapes_data_test)
single_frame_true_values = shapes_for_napari(shape_data_coords)[0]
single_tib_binary = process_frame([single_frame_true_values])
'''
#%%
show_stuff(tib_info_W_using_NW_ref, 'tib_W', viewer1)
#%%
show_stuff(fem_info_W, 'fem_W', viewer1)

#%%
screenshots = []

axis_index = 0 
number_of_frames = len(image1)

# Loop through your frames and take screenshots
for frame_index in range(number_of_frames):
    viewer1.dims.set_point(axis_index, frame_index)  # Navigate to the frame
    screenshot = viewer1.screenshot()  # Take screenshot
    screenshots.append(screenshot)
#%%
def create_mosaic_matplotlib(screenshots,total_frames, rows=2, columns=3, figsize=(14,12)):
    """
    Create a mosaic image from the given list of screenshots using matplotlib.
    
    Parameters:
    screenshots (list): A list of numpy arrays of shape (H, W, C).
    rows (int): Number of rows in the mosaic.
    columns (int): Number of columns in the mosaic.
    figsize (tuple): Size of the figure for the mosaic.
    
    Returns:
    str: File path of the saved mosaic image.
    """
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=figsize, facecolor='black')
    # Calculate frame indices to include the first and last frames and others equally spaced
    #frame_indices = [0] + list(np.linspace(1, total_frames - 2, columns * rows - 2, dtype=int)) + [total_frames - 1]
    frame_indices = np.linspace(0, total_frames - 1, 6, dtype=int)
    print(frame_indices)
    plt.subplots_adjust(wspace=0, hspace=0)
    # Flatten the axes array for easy iteration and fill each subplot
    for ax, frame_idx in zip(axes.flatten(), frame_indices):
        ax.imshow(screenshots[frame_idx])
        ax.set_title(f"Frame {frame_idx}", color='white')
    # Remove any excess plots if screenshots are less than grid size
    #for i in range(len(screenshots), rows*columns):
     #   fig.delaxes(axes.flatten()[i])

    plt.tight_layout()

    # Save the mosaic image to a file
    output_path = 'mosaic_AN_W_both_bones.svg'
    
    plt.savefig(output_path, format='svg', facecolor=fig.get_facecolor())

    return output_path

mosaic_path = create_mosaic_matplotlib(screenshots, total_frames=len(image1))


#%%
def track_origin(all_frame_info, point_name, bone_name, new_figure,  marker, label):
    # Extract x and y coordinates of the origin for each frame
    x_coords = [all_frame_info[frame][f'{point_name}'][0] for frame in sorted(all_frame_info)]
    y_coords = [all_frame_info[frame][f'{point_name}'][1] for frame in sorted(all_frame_info)]
    
    total_distance = sum(
        ((x_coords[i] - x_coords[i-1])**2 + (y_coords[i] - y_coords[i-1])**2)**0.5 
        for i in range(1, len(x_coords))
    )
    print(total_distance)
    
    #plt.clf()
    # Plot
    if new_figure:
        plt.figure(figsize=(10, 8))

    plt.scatter(y_coords, x_coords, c=sorted(all_frame_info), cmap='viridis', s=50, marker=marker, label=f'{label} : {total_distance:.2f} (Total Distance)')
    plt.plot(y_coords, x_coords, markersize=5, alpha=0.6)
    
    plt.title(f'Movement of {bone_name} {point_name} Over Frames')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    
    plt.grid(True)
    plt.legend()
    return total_distance
    
def calculate_angle(vector_a, vector_b):
    """Calculate angle in degrees between two vectors."""
    cos_theta = np.dot(vector_a, vector_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_distance_betwn_origins (fem_info, tib_info, point_name, label):
    indices = sorted(fem_info)
    origin_distances = [ np.linalg.norm(fem_info[ind][f'{point_name}']  -  tib_info[ind][f'{point_name}']) for ind in indices  ]
    total_distance = np.sum(origin_distances)
    
    x_values = range(len(origin_distances))
    label_with_distance = f'{label}: Total distance {total_distance:.2f}'
    plt.plot(origin_distances, label=label_with_distance)
    plt.scatter(x_values, origin_distances, marker='x', color='k')
    plt.xlabel('Frame number')
    plt.ylabel('Distances')
    plt.grid(True)
    plt.title(f'Distance between femur and tibia {point_name} across the frames')
    #plt.text(0, np.max(origin_distances), f'The total distance is: {total_distance:.2f}')
    plt.legend()
    plt.savefig(f'dist_betwn_{point_name}.svg')
    
    return origin_distances
    
def normalize_vector(vector):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:  # Avoid division by zero
        return vector
    return vector / norm    
    
    
# Assuming femur_info and tibia_info have the same keys (frames)
def plot_angle_vs_frame(femur_info , tibia_info, label):
    frames = sorted(femur_info.keys())
    angles = []
    
    for frame in frames:
        femur_vector = femur_info[frame]['V']
        tibia_vector = tibia_info[frame]['V']
        angle = calculate_angle(normalize_vector(femur_vector), normalize_vector(tibia_vector) )
        angles.append(angle)
    angles = (np.array(angles) ) 
    # Plot
    plt.scatter(frames, angles, marker='x', color='k')
    plt.plot(frames, angles, label=f'{label}')
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Between long axes of Femur and Tibia Over Frames")
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{label}angle_between_bones.svg')
    plt.show()

#%%

total_centroid_W = track_origin(tib_info_W_using_NW_ref, 'centroid',  bone_name = 'tibia', marker='x', new_figure=True, label='loaded')

total_centroid_NW = track_origin(tib_info_NW,  'centroid',  bone_name = 'tibia', marker='o', new_figure=False, label='unloaded')
plt.gca().invert_yaxis()  # Invert the y-axis to align with image coordinates
plt.colorbar(label='Frame Number')
plt.savefig('AN_centroid_track_tibia_both_cases.svg')

#%%
total_origin_W = track_origin(tib_info_W_using_NW_ref, 'origin',  bone_name = 'tibia', marker='x', new_figure=True, label='loaded')

total_origin_NW = track_origin(tib_info_NW,  'origin',  bone_name = 'tibia', marker='o', new_figure=False, label='unloaded')
plt.gca().invert_yaxis()
plt.colorbar(label='Frame Number')
plt.savefig('AN_origin_track_tibia_both_cases.svg')

#%%
total_centroid_W_fem = track_origin(fem_info_W, 'centroid',  bone_name = 'femur', marker='x', new_figure=True, label='loaded')

total_centroid_NW_fem = track_origin(fem_info_NW,  'centroid',  bone_name = 'femur', marker='o', new_figure=False, label='unloaded')
plt.colorbar(label='Frame Number')
plt.savefig('MK_centroid_track_femur_both_cases.svg')

#%%
total_origin_W_fem = track_origin(fem_info_W, 'origin',  bone_name = 'femur', marker='x', new_figure=True, label='loaded')

total_origin_NW_fem = track_origin(fem_info_NW,  'origin',  bone_name = 'femur', marker='o', new_figure=False, label='unloaded')
plt.colorbar(label='Frame Number')
plt.savefig('MK_origin_track_femur_both_cases.svg')

#%%
centroid_dist_W = calculate_distance_betwn_origins(tib_info_W, fem_info_W, 'centroid', label='loaded')
centroid_dist_NW = calculate_distance_betwn_origins(tib_info_NW, fem_info_NW, 'centroid', label='unloaded' )

#%%
origin_dist_W = calculate_distance_betwn_origins(tib_info_W, fem_info_W, 'origin', label='loaded')
origin_dist_NW = calculate_distance_betwn_origins(tib_info_NW, fem_info_NW, 'origin', label='unloaded')

#%%
voxel_size = [0.7272727272727273, 0.7272727272727273]
def plot_translations(all_frame_info, point_name, label, plot_type):
    # Sort frames for consistent ordering
    sorted_frames = sorted(all_frame_info)
    
    # Extract x and y coordinates of the point for each frame
    y_coords = [all_frame_info[frame][point_name][0] for frame in sorted_frames]
    x_coords = [all_frame_info[frame][point_name][1] for frame in sorted_frames]
    
    translations_mm = []
      
    # Plot A-P translations
    if plot_type== 'AP':
        ap_translations = [x - x_coords[0] for x in x_coords]
        translations_mm = [ap * voxel_size[1] for ap in ap_translations]
        plt.plot(sorted_frames, translations_mm, label=f'A-P Translation {label}')
        plt.axhline(0, color='gray', linewidth=0.5)  # Zero line for reference
        plt.xlabel('Frame Number')
        plt.ylabel('Translation (mm)')
        plt.title('Anterior-Posterior Translation over Time')
        plt.legend()
        plt.grid(True)
    #plt.show()
    
    # Plot I-S translations
    if plot_type== 'IS':
        is_translations = [y_coords[0] - y  for y in y_coords] # so that positive means decrease in y value, positive is superior 
        translations_mm = [is_ * voxel_size[0] for is_ in is_translations]
        #plt.figure(figsize=(10, 5))
        plt.plot(sorted_frames, translations_mm, label=f'I-S Translation {label}')
        plt.axhline(0, color='gray', linewidth=0.5)  # Zero line for reference
        plt.xlabel('Frame Number')
        plt.ylabel('Translation (mm)')
        plt.title('Inferior-Superior Translation over Time')
        plt.legend()
        plt.grid(True)
    #plt.show()
    return np.array ( translations_mm ) 

ap_W_tib = plot_translations(tib_info_W_using_NW_ref, 'centroid', label='loaded', plot_type='AP')
ap_NW_tib = plot_translations(tib_info_NW, 'centroid', label='unloaded', plot_type='AP')


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
    angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_degrees = np.degrees(angle_radians)
    if angle_degrees > 90:
        angle_degrees =  180 - angle_degrees

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
    

    return angles
#%%
plt.figure(figsize=(10, 6))
angles_W = calculate_and_plot_angles_between_bones(fem_info_W, tib_info_W, name='MK_W', new_figure=False)
angles_NW = calculate_and_plot_angles_between_bones(fem_info_NW, tib_info_NW, name='MK_NW',new_figure=True )
plt.show()
#%%
def calculate_and_plot_angles_with_theoretical_line(bone1, bone2, axis='long'):
    angles = []
    frames = []
    for frame in bone1.keys():
        if frame in bone2:
            angle = calculate_angle_between_bones(bone1[frame], bone2[frame], axis)
            print(f"Frame {frame}: Angle = {angle} degrees")
            angles.append(angle)
            frames.append(frame)

    # Assuming linear increment for theoretical line
    theoretical_angles = np.array([angles[0] + 2 * i for i in range(len(frames))])

    # Calculating residuals
    residuals = np.array(angles) - theoretical_angles

    # Calculating Mean Absolute Error
    mae = np.mean(np.abs(residuals))

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot angles and theoretical line
    plt.subplot(2, 1, 1)
    plt.plot(frames, angles, marker='o', label='Actual Angles')
    plt.plot(frames, theoretical_angles, linestyle='--', color='grey', label='Theoretical Line')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title(f'Actual vs Theoretical Angles ({axis.capitalize()} Axis)')
    plt.legend()
    plt.grid(True)

    # Plot residuals
    plt.subplot(2, 1, 2)
    plt.plot(frames, residuals, marker='o', color='red', label='Residuals')
    plt.xlabel('Frame')
    plt.ylabel('Residual Value')
    plt.title('Residuals at Each Frame')
    plt.legend()
    plt.grid(True)
    mae_text = f"Mean Absolute Error (MAE): {mae:.2f}"
    plt.text(x=min(frames), y=max(residuals), s=mae_text, color='blue', fontsize=10)
    plt.tight_layout()
    
    plt.savefig('AN_NW_mae_angle_between_bones.svg') 
    plt.show()

    # Print the Mean Absolute Error
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

calculate_and_plot_angles_with_theoretical_line(fem_info_NW, tib_info_NW, axis='long')

    
#%%
def plot_angles_vs_dist(angles, distances, point_name, new_figure=True, label=None):
    # Only create a new figure if requested
    if new_figure:
        plt.figure(figsize=(10, 6))
   
    plt.scatter(angles, distances, marker='x')
    plt.plot(angles, distances, label=f'{label}' if label else None)
    plt.xlabel('Angle between the long axes of femur and tibia')
    plt.ylabel(f'Distance between {point_name}s')
    plt.title(f'Distance between {point_name}s measured at each angle')
    plt.grid(True)
   
    # Only add a legend if a label is provided
    if label:
        plt.legend()
        
        

#%%
plot_angles_vs_dist(angles_NW, origin_dist_NW, 'origin', new_figure=True, label='unloaded')

plot_angles_vs_dist(angles_W, origin_dist_W, 'origin', new_figure=False, label='loaded')

# Now, show the combined plot
plt.savefig('Angle_vs_origin_both_bones.svg')
plt.show()

#%%
plot_angles_vs_dist(angles_NW, centroid_dist_NW, 'centroid', new_figure=True, label='unloaded')

plot_angles_vs_dist(angles_W, centroid_dist_W, 'centroid', new_figure=False, label='loaded')

# Now, show the combined plot
plt.savefig('Angle_vs_centroid_both_bones.svg')

plt.show()
#%%
def nullify_fem(femur_matrices, tibia_matrices):
    new_tibia_transforms = []

    for femur_matrix, tibia_matrix in zip(femur_matrices, tibia_matrices):
        # Inverting femur transformation
        inverted_femur_x, inverted_femur_y, inverted_femur_phi = -femur_matrix[0], -femur_matrix[1], -femur_matrix[2]

        # Combining with tibia transformation
        combined_x = inverted_femur_x + tibia_matrix[0]
        combined_y = inverted_femur_y + tibia_matrix[1]
        combined_phi = inverted_femur_phi + tibia_matrix[2]

        # Append the combined transformation to the new list
        new_tibia_transforms.append([combined_x, combined_y, combined_phi])

    return new_tibia_transforms

# Use the function with your transformation matrices
new_tibia_transforms = nullify_fem(fem_matrices, tib_matrices)

#%%
modified_tib_info= process_and_transform_shapes(viewer1.layers['tibia_NW'].data, new_tibia_transforms)

#%%
# Saving the dictionary to a file
with open('AN_W_tib_ap.pkl', 'wb') as f:
    pickle.dump(ap_W_tib, f)
#%%    
with open('AN_W_fem_info_using_NW_ref.pkl', 'wb') as f:
    pickle.dump(fem_info_W, f)
#%%
with open('MK_NW_fem_info.pkl', 'wb') as f:
    pickle.dump(transformed_dicts_fem, f)
''' to load do: 
    with open('my_dict.pkl', 'rb') as f:
    my_dict_loaded = pickle.load(f)'''    
    
#%%
# what follows below is an attempt to plot the tibia angle w.r.t the femur reference frame. first, load the info dicts 
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/new_analysis_all/AN/01.03.24/AN_NW_t_matrices_tib.pkl', 'rb') as file:
    NW_t_matrices_tib = pickle.load(file)    

#%%

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/new_analysis_all/AN/AN_W_tib_info.pkl', 'rb') as file:
    tib_info = pickle.load(file)  
    
#%%
