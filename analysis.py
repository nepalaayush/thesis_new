#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:47:23 2024

@author: aayush
"""
import pickle
import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import matplotlib.pylab as plt 
from shapely.geometry import LineString, MultiPoint
import napari
from sklearn.metrics import mean_absolute_error

from utils import (open_nii, normalize, shapes_for_napari, apply_transformations_new, coords_to_boolean, process_frame, show_stuff, dict_to_array, reconstruct_dict)


#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/26.01.24/MK_NW/MK_NW_t_matrices_last_tib.pkl', 'rb') as file:
    t_matrices_NW=  pickle.load(file)

#%%

def plot_transformations_and_calculate_MAE(transformation_matrices, offset, angle_increment, reference_index, condition, ax=None):
    # Extract phi values from transformation matrices and convert to degrees
    phis = [np.rad2deg(transformation[2]) for transformation in transformation_matrices]
    
    if reference_index == -1:
        # Reverse phis for reference at the end
        phis.reverse()

    if reference_index == 0:
        cumulative_phis = np.cumsum(phis) - offset
        print(cumulative_phis)
    else:
       # Reverse the cumulative sum to reflect decreasing trend
        cumulative_phis = np.cumsum(phis[::-1])[::-1]
        # Adjust for the offset
        cumulative_phis = cumulative_phis - cumulative_phis[-1] + offset
        print(cumulative_phis)
    # Adjusting x-axis to start from the specified offset
    x_values_shifted = np.arange(offset, offset + len(cumulative_phis) * angle_increment, angle_increment)

    
    if reference_index == 0:
        # Perfect increment line, starting from -offset and decreasing by angle_increment
        perfect_increment = np.arange(-offset, -offset - len(cumulative_phis) * angle_increment, -angle_increment)
    else:  
        # For reference at the end, start high and decrease towards the offset
        num_steps = len(cumulative_phis)
        high_value = offset + angle_increment * (num_steps - 1)
        perfect_increment = np.linspace(high_value, offset, num_steps)
    # Calculating residuals
    residuals = cumulative_phis - perfect_increment

    # Calculating Mean Absolute Error (MAE)
    mae = mean_absolute_error(cumulative_phis, perfect_increment)

    # Check if an axis is provided, if not, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))
        ax1, ax2 = ax
        created_new_figure = True
    else:
        ax1, ax2 = ax
        created_new_figure = False

    # Plotting on the provided or new axes
    ax1.plot(x_values_shifted, cumulative_phis, label=f'Cumulative Phi Data Ref {reference_index}', marker='o')
    ax1.plot(x_values_shifted, perfect_increment, label=f'Perfect Increment Ref {reference_index}', linestyle='--')
    ax1.set_title(f'Deviation from Perfect theoretical Line using frame {reference_index} as reference')
    ax1.set_xlabel("Rotary angle encoder")
    ax1.set_ylabel('Rotation angle of tibia (in degrees)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x_values_shifted)
    ax1.set_yticks(np.arange(np.max(perfect_increment), np.min(perfect_increment), -1))

    ax2.plot(x_values_shifted, residuals, label=f'Residuals Ref {reference_index}', marker='o', color='red')
    ax2.set_title('Residuals at Each Frame')
    ax2.set_xlabel('Frame (Starting from ' + str(offset) + ')')
    ax2.set_ylabel('Residual Value')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x_values_shifted)
    ax2.set_yticks(np.arange(np.floor(np.min(residuals)), np.ceil(np.max(residuals)) + 1, 1))
    ax2.text(x=min(x_values_shifted), y=max(residuals), s=f"Mean Absolute Error (MAE): {mae:.2f}", color='blue', fontsize=10)

    if created_new_figure:
        plt.tight_layout()
        plt.show()

    print(f"Mean Absolute Error (MAE): {mae}")

# Example usage:
#%%
# For a single plot
plot_transformations_and_calculate_MAE(t_matrices_first, offset=5, angle_increment=2, reference_index=0, ax=None)
#%%
# For overlaying multiple plots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))
plot_transformations_and_calculate_MAE(transformation_matrices_last, offset=5, angle_increment=2, reference_index=-1, ax=ax)
plot_transformations_and_calculate_MAE(t_matrices_NW, offset=5, angle_increment=2, reference_index=-1, ax=ax)
plt.tight_layout()
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
    plt.savefig('26.01.24/cost_values_tibia_0.svg')
    plt.show()
    print('The sum of all the cost function values is:', np.sum(values))
plot_cost_values(cost_values_first)
#%%
# use a unblurred image 
path1 = 'C:/Users/Aayush/Documents/thesis_files/thesis_new/26.01.24/MK_NW/MK_NW_ai2_tgv_5e-2_pos.nii'
image1 = open_nii(path1)
image1 = normalize(image1)
image1 = np.moveaxis(image1, 1, 0)
#%%
viewer1 = napari.view_image(image1)
#%%

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

#%%
# add the reference points and manually segment the reference frame 
viewer1.add_shapes(reference_frame_last, shape_type='polygon')
#%%
# rename it to expanded_shape and then store it as ref_points variable 
ref_points = viewer1.layers['expanded_shape_femur'].data[0]
#%%
applied_transformation = apply_transformations_new(ref_points, transformation_matrices_last, -1)    
viewer1.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='green')

#%%
# tib_label = coords_to_boolean(new_tib_coords_first, image1.shape)
tib_label = final_label

total_frames = len(tib_label) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer = viewer1.layers["Shapes"].to_labels(image1.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(0,450)
yrange=slice(150,400)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(image1[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('MK_NW_segmented_femur.svg')

#%%
shapes_data = viewer1.layers['tibia_NW'].data
#%%
# After manually segmenting, find the info of the shapes.
#do the pca on just the reference frame 
single_tib_info = process_frame([shapes_data[-1]])
#%%
# convert the dictionary into an array to later on transform it using the same t matrix 
tib_info_array = dict_to_array(single_tib_info)
#%%
transformed_info_tibia  = apply_transformations_new(tib_info_array, transformation_matrices_last, -1)
#%%
transformed_dicts_tib = {}

for i, arr in enumerate(transformed_info_tibia):
    transformed_dicts_tib.update(reconstruct_dict(i, arr))


#%%
def process_and_transform_shapes(shapes_data, transformation_matrices):
    # Process the reference frame (assuming the last one in the shapes data)
    single_shape_info = process_frame([shapes_data[-1]])

    # Convert the dictionary into an array for transformation
    shape_info_array = dict_to_array(single_shape_info)

    # Apply transformations to the shape data
    transformed_info = apply_transformations_new(shape_info_array, transformation_matrices, -1)

    # Reconstruct the dictionary for each transformed frame
    transformed_dicts = {}
    for i, arr in enumerate(transformed_info):
        transformed_dicts.update(reconstruct_dict(i, arr))

    return transformed_dicts

modified_tib_info= process_and_transform_shapes(shapes_data, new_tibia_transforms)
#%%
fem_shapes_data = viewer1.layers['femur_NW'].data

#%%
single_fem_info = process_frame([fem_shapes_data[-1]])
#%%

fem_info_array = dict_to_array(single_fem_info)
#%%
transformed_info_fem = apply_transformations_new(fem_info_array, transformation_matrices_last, -1)
#%%
transformed_dicts_fem = {}

for i, arr in enumerate(transformed_info_fem):
    transformed_dicts_fem.update(reconstruct_dict(i, arr))


#%%
show_stuff(transformed_dicts_tib, 'tib_wt', viewer1)
    
#%%
show_stuff(transformed_dicts_fem, 'fem_wt', viewer1)
#%%
screenshot = viewer1.screenshot()
viewer1.add_image(screenshot, rgb=True, name='screenshot')  

from skimage.io import imsave
imsave('screenshot.png', screenshot)

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
imsave('mosaic.png', mosaic_image)

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
    frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)
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
    output_path = 'mosaic_MK_NW_both_axes.svg'
    
    plt.savefig(output_path, format='svg', facecolor=fig.get_facecolor())

    return output_path

mosaic_path = create_mosaic_matplotlib(screenshots, total_frames=len(image1))


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
    plt.savefig(f'origin_track_{bone_name}.svg')
    plt.show()
    
def calculate_angle(vector_a, vector_b):
    """Calculate angle in degrees between two vectors."""
    cos_theta = np.dot(vector_a, vector_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_distance_betwn_origins (fem_info, tib_info):
    indices = sorted(fem_info)
    origin_distances = [ np.linalg.norm(fem_info[ind]['origin']  -  tib_info[ind]['origin']) for ind in indices  ]
    plt.plot(origin_distances)
    plt.xlabel('Frame number')
    plt.title('Distance between femur and tibia origins across the frames')
    

def calculate_distance_betwn_centroids (fem_info, tib_info):
    indices = sorted(fem_info)
    origin_distances = [ np.linalg.norm(fem_info[ind]['centroid']  -  tib_info[ind]['centroid']) for ind in indices  ]
    plt.plot(origin_distances)
    plt.xlabel('Frame number')
    plt.title('Distance between femur and tibia centroids across the frames')

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
track_origin(transformed_dicts_tib, 'MK_NW_tib')

#%%
track_origin(transformed_dicts_fem, 'MK_NW_fem')
#%%
plot_angle_vs_frame(transformed_dicts_fem, transformed_dicts_tib, 'NW_ai2')

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



def calculate_and_plot_angles_between_bones(bone1, bone2, axis='long'):
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
    
    plt.figure(figsize=(10, 6))
    plt.plot(frames, angles, marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title(f'Angle between Bones over Frames ({axis.capitalize()} Axis)')
    plt.grid(True)
    plt.show()



calculate_and_plot_angles_between_bones(transformed_dicts_fem, transformed_dicts_tib)

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
    
    plt.savefig('MK_NW_mae_angle_between_bones_wrt_fem.svg') 
    plt.show()

    # Print the Mean Absolute Error
    print(f"Mean Absolute Error (MAE): {mae:.2f}")

calculate_and_plot_angles_with_theoretical_line(transformed_dicts_fem, modified_tib_info, axis='long')


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
new_tibia_transforms = nullify_fem(transformation_matrices_last, t_matrices_tib)

#%%
# for plotting multiple transformation matrices on the same plot: 


#%%
# Saving the dictionary to a file
with open('MK_NW_tib_info_modified.pkl', 'wb') as f:
    pickle.dump(modified_tib_info, f)
#%%    
with open('MK_NW_fem_info.pkl', 'wb') as f:
    pickle.dump(transformed_dicts_fem, f)

''' to load do: 
    with open('my_dict.pkl', 'rb') as f:
    my_dict_loaded = pickle.load(f)'''    
    
#%%
# what follows below is an attempt to plot the tibia angle w.r.t the femur reference frame. first, load the info dicts 
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/26.01.24/MM_W_tib_info_ai2.pkl', 'rb') as file:
    tib_info = pickle.load(file)    



with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/26.01.24/MM_W_fem_info_ai2.pkl', 'rb') as file:
    fem_info = pickle.load(file)  
    
#%%
