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
from shapely.geometry import LineString, MultiPoint
import napari
from sklearn.metrics import mean_absolute_error

from utils import (open_nii, normalize, shapes_for_napari, apply_transformations_new, coords_to_boolean, process_frame, show_stuff, dict_to_array, reconstruct_dict)

    # Extract phi angles and convert to degrees
#%%
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/26.01.24/MM_W/MM_W_tib_info_ai2.pkl', 'rb') as file:
    tib_info = pickle.load(file)
#%%
def plot_transformations(transformation_matrices, index):
    # Extracting the rotation angles from the transformation matrices and converting to degrees
    phis = [np.rad2deg(transformation[2]) for transformation in transformation_matrices]

    # Creating the plot
    plt.figure(figsize=(10, 6))
    if index==0:
        plt.scatter(np.arange(1, len(phis[1:]) + 1), phis[1:], color='blue')  # Plotting the points
        plt.plot(np.arange(1, len(phis[1:]) + 1), phis[1:], linestyle='dotted', color='red')  # Connecting the points with a dotted line
    else:
        plt.scatter(np.arange(1, len(phis[:-1]) + 1), phis[:-1], color='blue')
        plt.plot(np.arange(1, len(phis[:-1]) + 1), phis[:-1], linestyle='dotted', color='red')  # Connecting the points with a dotted line
    
    plt.grid(True)  # Adding grid
    plt.xlabel('Transformation Index')
    plt.ylabel('Rotation Angle (Degrees)')
    plt.title(f'Rotation Angles from Transformation Matrices using {index} frame as reference')
    plt.savefig(f'phi_plot_{index}.svg') 
    plt.show()
    
   
plot_transformations(t_matrices_last_ai2, -1)
#%%

def plot_transformations_and_calculate_MAE(transformation_matrices, offset, angle_increment, reference_index):
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

    # Creating plots
    plt.figure(figsize=(10, 12))

    # Adjusted graph (Top Graph)
    plt.subplot(2, 1, 1)
    plt.plot(x_values_shifted, cumulative_phis, label='Cumulative Phi Data', marker='o')
    plt.plot(x_values_shifted, perfect_increment, label='Perfect ' + str(angle_increment) + '-Degree Increment (Start ' + str(-offset) + ')', linestyle='--')
    plt.title(f'Deviation from Perfect theoretical Line when using frame {reference_index} as reference')
    plt.xlabel("Rotary angle encoder")
    plt.ylabel('Rotation angle of tibia (in degrees)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xticks(x_values_shifted)
    plt.yticks(np.arange(np.max(perfect_increment), np.min(perfect_increment), -1))
    # Residuals plot (Bottom Graph)
    plt.subplot(2, 1, 2)
    plt.plot(x_values_shifted, residuals, label='Residuals', marker='o', color='red')
    plt.title('Residuals at Each Frame')
    plt.xlabel('Frame (Starting from ' + str(offset) + ')')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xticks(x_values_shifted)
    plt.yticks(np.arange(np.floor(np.min(residuals)), np.ceil(np.max(residuals)) + 1, 1))
    
    plt.tight_layout()
    plt.savefig(f'angle_residuals_ai2_{reference_index}.svg') 
    plt.show()

    # Print the Mean Absolute Error
    print(f"Mean Absolute Error (MAE): {mae}")    
plot_transformations_and_calculate_MAE(transformation_matrices_last, offset=5, angle_increment=2, reference_index=-1)

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
path1 = '/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/26.01.24/MM_W/MM_W_ai2_tgv_5e-2_neg_r0_r40.nii'
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
ref_points = viewer1.layers['expanded_shape'].data[0]
#%%
applied_transformation = apply_transformations_new(ref_points, transformation_matrices_last, 14)    
viewer1.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='green')

#%%
# tib_label = coords_to_boolean(new_tib_coords_first, image1.shape)
tib_label = final_label

total_frames = len(tib_label) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer = viewer1.layers["Shapes"].to_labels(image1.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(150,450)
yrange=slice(150,400)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(image1[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('NW_MM_segmented_tibia.svg')

#%%
shapes_data = viewer1.layers['tibia_W'].data
#%%
# After manually segmenting, find the info of the shapes. 
single_tib_info = process_frame([shapes_data[0]])
#%%
info_array = dict_to_array(single_tib_info)
#%%
transformed_info = applied_transformation = apply_transformations_new(info_array, t_matrices_first, 0)
#%%
transformed_dicts = {}

for i, arr in enumerate(transformed_info):
    transformed_dicts.update(reconstruct_dict(i, arr))

#%%
fem_info = process_frame(viewer1.layers['Shapes'].data)
#%%
show_stuff(transformed_dicts, 'tib_wt', viewer1)
    
#%%
show_stuff(fem_info, 'fem_wt', viewer1)
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
    frame_indices = [ 0, 3, 6, 9 , 12, 13]
    print(frame_indices)
    plt.subplots_adjust(wspace=0, hspace=0)
    # Flatten the axes array for easy iteration and fill each subplot
    for ax, (screenshot, frame_idx) in zip(axes.flatten(), zip(screenshots, frame_indices)):
        ax.imshow(screenshot)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Frame {frame_idx}", color='white')
    # Remove any excess plots if screenshots are less than grid size
    #for i in range(len(screenshots), rows*columns):
     #   fig.delaxes(axes.flatten()[i])

    plt.tight_layout()

    # Save the mosaic image to a file
    output_path = 'mosaic_matplotlib.svg'
    
    plt.savefig(output_path, format='svg', facecolor=fig.get_facecolor())

    return output_path

mosaic_path = create_mosaic_matplotlib(screenshots, total_frames=len(image1))

#%%
fig, axes = plt.subplots( 2, 3, figsize= (14,12), facecolor='black')
# Adjust subplot parameters
#plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

print(frame_indices)
total_frames = len(image1) 
desired_frames = 6
frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)
xrange=slice(50,350)
yrange=slice(150,400)
for ax, indi in zip(axes.flatten(), frame_indices):
    ax.imshow(screenshots[indi])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {indi}", color='white')

plt.tight_layout()
plt.savefig('simple_mosaic.svg')
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
    plt.scatter(frames, angles, marker='x', color='k')
    plt.plot(frames, angles, label=f'{label}')
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Between long axes of Femur and Tibia Over Frames")
    plt.grid(True)
    plt.legend()
    plt.savefig('angle_between_bones.svg')
    plt.show()

#%%
track_origin(tib_info, 'W_0_ai2_tibia')

#%%
track_origin(fem_info, 'NW_0_ai2_femur')
#%%
plot_angle_vs_frame(fem_info, tib_info, 'W_ai2')

#%%
# Saving the dictionary to a file
with open('MM_W_tib_info_ai2.pkl', 'wb') as f:
    pickle.dump(tib_info, f)
    
with open('MM_W_fem_info_ai2.pkl', 'wb') as f:
    pickle.dump(fem_info, f)

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
