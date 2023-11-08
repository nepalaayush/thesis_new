#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 08:14:03 2023

@author: aayush
"""

import numpy as np
import napari
import nibabel as nib 
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from skimage.measure import label, regionprops
from skimage.feature import canny
from skimage.morphology import skeletonize, remove_small_objects
#import cv2
from scipy.ndimage import gaussian_filter
import time 
from scipy.interpolate import CubicSpline

#%%
def open_nii(path):
    ''' Input: Path of nifti file (.nii) Output: pixelarray  ''' 
    nifti_img = nib.load(path)
    #pixelarray = np.transpose ( nifti_img.get_fdata(), (2,0,1)) # have to transpose because of the way nifti loads things 
    pixelarray = nifti_img.get_fdata()
    return np.moveaxis(pixelarray, -1,0)

def normalize (image):
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    normalized_img = ( image - min_val ) / (max_val - min_val) 
    scaled_img = (normalized_img * 255)
    
    return scaled_img 

def gradify(pixelarray):
    dy, dx = np.gradient(pixelarray, axis=(1, 2))

    # Magnitude of the gradient for each frame
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    gradient_direction = np.arctan2(dy, dx)

    return gradient_magnitude, gradient_direction

def apply_canny(pixelarray, low, high, sigma):
    canny_edge = np.zeros_like(pixelarray)
    
    for i in range(pixelarray.shape[0]):
        canny_image = canny(pixelarray[i], low_threshold= low, high_threshold=high, sigma= sigma )    
        canny_edge[i] = canny_image
    return canny_edge.astype(dtype=bool)

def apply_remove(pixelarray, size, connectivity):
    removed_3d = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        removed_image = remove_small_objects(pixelarray[i], min_size=size, connectivity=connectivity)
        removed_3d[i] = removed_image
    return removed_3d

def apply_skeleton(pixelarray):
    skeletonized = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        skel_frame = skeletonize(pixelarray[i])
        skeletonized[i] = skel_frame
    return skeletonized


def apply_transformation(matrix, points):
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    return transformed_points[:, :2]  # Convert back to Cartesian coordinates

def apply_transformation_all_frames_consecutive(initial_set, transformation_matrices):
    transformed_subsets = []
    
    # Initialize with the initial set of points
    current_points = initial_set
    transformed_subsets.append(current_points)
    
    # Apply each transformation matrix in sequence
    for matrix in transformation_matrices[1:]:  # Skip the first identity matrix
        transformed_points = apply_transformation(matrix, current_points)
        transformed_subsets.append(transformed_points)
        
        # Update current_points for the next iteration
        current_points = transformed_points
        
    return transformed_subsets

def procrustes(X, Y):
    X = X.astype(float)  # Ensure X is float
    Y = Y.astype(float)
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X -= centroid_X
    Y -= centroid_Y
    U, _, Vt = np.linalg.svd(np.dot(Y.T, X))
    R = np.dot(U, Vt)
    t = centroid_Y - np.dot(R, centroid_X)
    return R, t


def calculate_transform_matrices_procrustes_consecutive(all_coords):
    num_frames = len(all_coords)
    transformation_matrices = []
    
    # Initialize the identity matrix for the first frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])
    
    # Add the identity matrix for the first frame
    transformation_matrices.append(identity_matrix)

    for i in range(1, num_frames):
        # Estimate the transformation matrix using Procrustes
        # between the i-th frame and the (i-1)-th frame
        R, t = procrustes(all_coords[i-1], all_coords[i])
        transformation_matrix = np.hstack([R, t.reshape(-1, 1)])  # Combine into a 2x3 matrix
        transformation_matrices.append(transformation_matrix)
        
    return transformation_matrices

def points_for_napari(list_points):
    all_points = []

    for i, subset in enumerate(list_points):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_points.append(frame_subset)

    all_points = np.vstack(all_points)
    return all_points 

def show_order(curve):
    plt.scatter(curve[:, 0], curve[:, 1])

    # Annotate each point with its index
    for i, (x, y) in enumerate(curve):
        plt.annotate(str(i), (x, y))

    plt.show()

def boolean_to_coords(boolean_array):
    all_coordinates = []
    ''' Input: a 3d boolean array, like a label image 
    Output: the coordinates where array is true, as a list, each list is a frame'''
    # Loop through each frame and grab the coordinates
    for frame in boolean_array:
        coords = np.argwhere(frame)
        all_coordinates.append(coords)
    return all_coordinates
    
def coords_to_boolean(sorted_coordinates, shape):
    # Initialize the 3D array
    new_array = np.zeros(shape, dtype=bool)
    
    # Go through each frame
    for frame_index, frame_coords in enumerate(sorted_coordinates):
        for y, x in frame_coords:
            try:
                new_array[frame_index, int(y), int(x)] = True
            except IndexError as e:
                print(f"IndexError at frame {frame_index} with coordinates ({x}, {y}). Shape is {shape}")
                raise e
    
    return new_array

def check_integrity(list_of_cords): 
    print ( [ np.var ( pairwise_distances(i) )  for i in list_of_cords ] ) 

def pairwise_distances(points):
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances


def sort_points_single_frame(points):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    starting_point = points[np.argmax(points[:, 0])]  # Highest row value
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        

        # # Check if the distance is much larger than average
        # if len(sorted_points) > 1:
        #     avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
        #     large_jump = np.linalg.norm(np.array(next_point) - np.array(current_point)) > 2 * avg_distance
        #     if large_jump:
        #         print(f"Warning: Large distance jump detected in frame {frame_number} between point {current_point} and point {next_point}")  # Modified print statement
        #         break

        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)

    return np.array(sorted_points)

def sort_points_all_frames(list_of_points):
    sorted_list = []
    for i in list_of_points:
        sorted_list.append(sort_points_single_frame(i))
    return sorted_list 

def equalize_lengths(points_list):
    # Find the length of the smallest (n,2) array
    min_length = min([len(points) for points in points_list])

    # Trim each array in the list to have the same length as the smallest array
    equalized_list = [points[-min_length:] for points in points_list]

    return equalized_list
#%%
# Step 1: load the image from directory and normalize it
image = open_nii('C:/Users/Aayush/Documents/thesis_files/more_data/aw1_rieseling_admm_tgv_5e-2.nii')
image = normalize(image)
image = np.moveaxis(image, 1, 0)[1:]
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='aw1_rieseling_admm_tgv_5e-2.nii')
#%%
viewer.add_image(image)
#%%
# Step 2: apply gaussian blur to the original image and add it in napari. 
smooth_image = gaussian_filter(image, 2)
viewer.add_image(smooth_image , name='smooooth')

#%%
smooth_image = image # when using regularized, it is already smooth
# Step 3: take the gradient of the smooth image, both magnitude as well as direction
grad_smooth = gradify(smooth_image)[0]
grad_direction = gradify(smooth_image)[1]
viewer.add_image(grad_smooth, name='gradient_smooth')
viewer.add_image(grad_direction, name='direction')
#%%
inverted = 255 - image 
grad_inverted = gradify(inverted)[0]
viewer.add_image(grad_inverted, name='inverted_ori')
#%%
# Step 4: find the best suitable low and high range for edge detection
start_time = time.time() 

def apply_canny_multiple_thresholds(pixelarray, low_range, high_range, num_steps, sigma):
    ''' Note, values less than low are ignored, values greater than high are considered edges. Pixels with a gradient magnitude between low and high are conditionally accepted if they're connected to strong edges. ''' 
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    
    # Initialize a 4D array to store results
    canny_multi_edge = np.zeros((num_steps, *pixelarray.shape), dtype=bool)
    
    for j, (low, high) in enumerate(zip(low_values, high_values)):
        canny_edge = apply_canny(pixelarray, low, high, sigma)
        canny_multi_edge[j] = canny_edge
    
    return canny_multi_edge

low_range = (11,15) # 
high_range = (15,18 ) # 
num_steps = 15
sigma = 2
print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(inverted, low_range, high_range, num_steps, sigma)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")

#%%
# add the 4d image to a new viewer
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='inverted_sigma2')
#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[0]
viewer.add_image(tib_canny, name='after_edge_detection_sigma_2')
#%%
#Step 6: Use remove small objects at various ranges to find the most suitable
def apply_remove_multiple_sizes(pixelarray, size_range, num_steps, connectivity):
    size_values = np.linspace(size_range[0], size_range[1], num_steps).astype(int)
    print(size_values) 
    # Initialize a 4D array to store results
    removed_multi_3d = np.zeros((num_steps, *pixelarray.shape), dtype=bool)
    
    for i, size in enumerate(size_values):
        removed_3d = apply_remove(pixelarray, size, connectivity)
        removed_multi_3d[i] = removed_3d
    
    return removed_multi_3d

# Example usage
size_range = (55, 75)  # 100 min and 200 max size for femur 
num_steps = 20  # Number of steps for size parameter
connectivity = 2  # Fixed connectivity value
print(np.linspace(size_range[0],size_range[1], num_steps))
# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(tib_canny, size_range, num_steps, connectivity)


# add it to the 4d viewer
viewer3.add_image(removed_4d, name='multi_remove_small')
#%%
# pick the right index
bone_canny = removed_4d[19] 
viewer.add_image(bone_canny, name='after_remove_small')
#%%
bone_canny[:,:270] = False # this was necessary to ease the labelling process 
viewer.add_image(bone_canny, name='after_remove_small')

#%%
# Step 7 find labels of connected regions from the edge image
labeled_image, num_features = label(bone_canny, return_num=True, connectivity=2)

viewer.add_labels(labeled_image, name='labeled_tib')    
#%%
# pick a suitable label that represents one long edge of the bone
tib_label = labeled_image == 3                                                                                               
viewer.add_image(tib_label, name='one_label')
#%%
label_skeleton = apply_skeleton(tib_label)
viewer.add_image(label_skeleton, name='skeleton')

#%%
unsorted_skeleton = boolean_to_coords(label_skeleton)

zeroth_frame = sort_points_single_frame(unsorted_skeleton[0])
#%%
viewer.add_points(zeroth_frame)
#%%
def equidistant_points(points, n):
    # Calculate pairwise distances between points
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    
    # Calculate cumulative distances
    cumulative_distances = np.cumsum(distances)
    total_distance = cumulative_distances[-1]
    
    # Calculate desired spacing
    desired_spacing = total_distance / (n - 1)
    
    # Select equidistant points
    new_points = [points[0]]  # Start with the first point
    current_dist = 0

    for i in range(1, n - 1):  # We already have the starting point, and we'll manually add the endpoint
        current_dist += desired_spacing
        # Find the two original points which the current_dist is between
        idx = np.searchsorted(cumulative_distances, current_dist)
        weight = (current_dist - cumulative_distances[idx - 1]) / (cumulative_distances[idx] - cumulative_distances[idx - 1])
        # Linearly interpolate between these two points
        point = points[idx - 1] + weight * (points[idx] - points[idx - 1])
        new_points.append(point)
    
    new_points.append(points[-1])  # End with the last point
    return np.array(new_points)

equidistant_zeroth = equidistant_points(zeroth_frame,30)
#%%
viewer.add_points(equidistant_zeroth, size=1, face_color='red') 
#%%
def adjust_downsampled_points(downsampled, original_curve):
    """
    Adjust the positions of downsampled points to make them equidistant 
    while ensuring they remain on the original curve.

    Parameters:
    - downsampled: np.array of shape (30,2) representing downsampled points.
    - original_curve: np.array of shape (100,2) representing the original curve.

    Returns:
    - np.array of shape (30,2) representing the adjusted downsampled points.
    """

    # Compute the desired equidistant length
    pairwise_distances = [np.linalg.norm(downsampled[i] - downsampled[i - 1]) for i in range(1, len(downsampled))]
    desired_distance = sum(pairwise_distances) / len(pairwise_distances)

    # Cubic spline interpolation of the original curve
    t = np.linspace(0, 1, len(original_curve))
    cs_x = CubicSpline(t, original_curve[:, 0])
    cs_y = CubicSpline(t, original_curve[:, 1])

    # Adjust the downsampled points
    adjusted_points = [downsampled[0]]  # Start with the first point as anchor
    t_last = 0  # To keep track of the last position on t to avoid backtracking
    for i in range(1, len(downsampled)):
        # Search along the curve for the next position using a fine resolution
        search_t = np.linspace(t_last, 1, 1000)
        for ti in search_t:
            potential_point = np.array([cs_x(ti), cs_y(ti)])
            if np.linalg.norm(potential_point - adjusted_points[-1]) >= desired_distance:
                adjusted_points.append(potential_point)
                t_last = ti
                break

    return np.array(adjusted_points)

zeroth_adjusted = adjust_downsampled_points(equidistant_zeroth, zeroth_frame)
#%%
viewer.add_points(zeroth_adjusted, size=1, face_color='blue') 
#%%
unsorted_skeleton[0] = zeroth_adjusted
#%%
sorted_skeleton = sort_points_all_frames(unsorted_skeleton)
#%%
def find_equidistant_subset(template_frame, target_frame, avg_gap):
    # Initialize Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_frame)
    
    # Start from the last point in the template
    start_point = template_frame[-1]
    
    # Find the closest point in the target frame
    _, initial_indices = nbrs.kneighbors([start_point])
    print("Initial indices:", initial_indices, type(initial_indices))
    current_point = target_frame[initial_indices[0][0]]
    
    # Initialize the list of selected points
    selected_points = [current_point]
    
    # Loop to find equidistant points
    while True:
        # Find all points upstream (lower index)
        upstream_points = [point for i, point in enumerate(target_frame) if i < initial_indices[0][0]]
        
        # If we've reached the start of the list, break
        if not upstream_points:
            break
        
        # Find the point closest to avg_gap away from current_point
        distances = [np.linalg.norm(np.array(current_point) - np.array(point)) for point in upstream_points]
        closest_point = upstream_points[np.argmin(np.abs(np.array(distances) - avg_gap))]
        
        # Update current_point and add to selected points
        current_point = closest_point
        selected_points.append(current_point)
        
        # Update initial_indices to the index of the newly found point
        initial_indices = np.array([np.where((target_frame == current_point).all(axis=1))])
        #print("Initial indices:", initial_indices, type(initial_indices))
        if initial_indices.size == 0:
            print("No matching point found for current_point in target_frame.")
            break

    return np.array( selected_points ) 

#%%
avg_gap = np.mean(pairwise_distances(sorted_skeleton[0]))
testing = find_equidistant_subset(sorted_skeleton[0], sorted_skeleton[1], avg_gap)

#%%
def find_all_equidistant_subsets(sorted_skeleton_list):
    # Initialize list to store subsets
    all_subsets = []
    
    # Your initial template frame
    template_frame = sorted_skeleton_list[0]
    
    all_subsets.append(template_frame)
    # Calculate the initial avg_gap
    avg_gap = np.mean(pairwise_distances(template_frame))
    
    
    # Loop through all target frames
    for target_frame in sorted_skeleton_list[1:]:
        # Find the subset for the current target frame based on the template frame
        single_subset = find_equidistant_subset(template_frame, target_frame, avg_gap)
    
        # Add this subset to the list of all subsets
        all_subsets.append(single_subset)
        
        # Update the template frame for the next iteration
        template_frame = sort_points_single_frame(single_subset)

    return all_subsets

result_subsets = find_all_equidistant_subsets(sorted_skeleton)

viewer.add_points(points_for_napari(result_subsets), name='result_subsets', size=2, face_color='orange')
#%%
result_sorted = sort_points_all_frames(result_subsets)
equalized_result = equalize_lengths(result_sorted)
viewer.add_points(points_for_napari(equalized_result), name='equalized_result', size=2, face_color='green')

''' went from 30 points to 17. almost halved because of one particular frame. but that was due to the short length of that particular edge.  ''' 
#%%
matrices_list_result = calculate_transform_matrices_procrustes_consecutive(equalized_result)
post_transformation_result = apply_transformation_all_frames_consecutive(equalized_result[0], matrices_list_result)

viewer.add_points(points_for_napari(post_transformation_result), face_color='red', size=1) 

#%%
def find_angles_from_matrices(matrices_list):
    angles = [0]
    # Extract the angles from the transformation matrices
    for matrix in matrices_list[1:]:  # Skip the first identity matrix
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        angles.append(np.degrees(angle))  # Appending the angle in degrees 
        
        
    plt.scatter(range (1, len(matrices_list) + 1 ), angles )
    plt.xlabel('frame number')
    plt.ylabel('angle between first frame')
    plt.grid() 
#%%
def to_global_transformations(consecutive_transformations):
    # Identity matrix in 3x3 homogenous form
    global_transformation = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

    # This will hold the global transformations
    global_transformations = [global_transformation[:2, :].copy()]

    for trans in consecutive_transformations[1:]:  # Skip the first identity matrix
        # Convert (2,3) to (3,3) by appending [0, 0, 1]
        trans_3x3 = np.vstack((trans, [0, 0, 1]))
        # Multiply the last global transformation by the new transformation
        global_transformation = global_transformation.dot(trans_3x3)
        # Append the new transformation to the list, converting back to (2,3)
        global_transformations.append(global_transformation[:2, :])

    return global_transformations

global_transformations = to_global_transformations(matrices_list_result)

find_angles_from_matrices(global_transformations)
#%%
def find_angles_and_ratios_from_matrices(matrices_list):
    angles = [0]
    ratios = [np.nan]  # Start with NaN because there is no ratio for the first identity matrix

    # Extract the angles and ratios from the transformation matrices
    for matrix in matrices_list[1:]:  # Skip the first identity matrix
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        angles.append(np.degrees(angle))  # Appending the angle in degrees

        delta_x = matrix[0, 2]
        delta_y = matrix[1, 2]
        
        # Avoid division by zero, set ratio to None or some large value if delta_y is 0
        #ratio = delta_x / delta_y if delta_y != 0 else np.nan
        ratio = delta_y
        ratios.append(ratio)
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('frame number')
    ax1.set_ylabel('angle (degrees)', color=color)
    ax1.scatter(range(1, len(matrices_list) + 1), angles, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('delta y', color=color)  # we already handled the x-label with ax1
    ax2.scatter(range(1, len(matrices_list) + 1), ratios, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # to make sure that the labels don't get cut off
    plt.show()
    
find_angles_and_ratios_from_matrices(global_transformations)
