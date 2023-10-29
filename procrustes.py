#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:15:32 2023

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

def apply_canny(pixelarray, low, high):
    canny_edge = np.zeros_like(pixelarray)
    
    for i in range(pixelarray.shape[0]):
        canny_image = canny(pixelarray[i], low_threshold= low, high_threshold=high )    
        canny_edge[i] = canny_image
    return canny_edge.astype(dtype=bool)

def apply_remove(pixelarray, size, connectivity):
    removed_3d = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        removed_image = remove_small_objects(pixelarray[i], min_size=size, connectivity=connectivity)
        removed_3d[i] = removed_image
    return removed_3d

def points_for_napari(list_points):
    all_points = []

    for i, subset in enumerate(list_points):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_points.append(frame_subset)

    all_points = np.vstack(all_points)
    return all_points   

def shapes_for_napari(list_shapes):
    all_shapes = []

    for i, subset in enumerate(list_shapes):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_shapes.append(frame_subset)

    return all_shapes

def show_order(curve):
    plt.scatter(curve[:, 0], curve[:, 1])

    # Annotate each point with its index
    for i, (x, y) in enumerate(curve):
        plt.annotate(str(i), (x, y))

    plt.show()
    
def pairwise_distances(points):
    """
    Calculate the distances between consecutive pairs of points in the given array.
    
    Args:
    - points (numpy.ndarray): An array of shape (N, 2) representing the points.
    
    Returns:
    - distances (numpy.ndarray): An array of shape (N-1,) representing the distances between consecutive points.
    """
    # Calculate the Euclidean distances between consecutive points
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances

def extract_coords_from_frames(label_frames):
    all_coords = []  # This list will hold the coordinates for each frame
    for frame in label_frames:
        coords = np.argwhere(frame)  # Find the coordinates of True pixels
        all_coords.append(coords)
    return all_coords
#%%
# Step 1: load the image from directory and normalize it
image = open_nii('C:/Users/Aayush/Documents/thesis_files/data_zf1_admm_tgv=1e-1/data_zf1_admm_tgv=1e-1.nii')
image = normalize(image)
image = np.moveaxis(image, 1, 0)[1:]
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='riesleing_e-1')
#%%
viewer.add_image(image, name='reco_riseling_non')
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
# apply canny edge directly
canny_direction = apply_canny(grad_direction, low=2.5, high=3.2)
viewer.add_image(canny_direction, name='edge_direction')
canny_edge = apply_canny(grad_smooth, low=0, high=1) # 
viewer.add_image(canny_edge, name= 'canny')
#%%
inverted = 255 - image
#%%
viewer.add_image(inverted)
#%%
# Step 4: find the best suitable low and high range for edge detection
start_time = time.time() 

def apply_canny_multiple_thresholds(pixelarray, low_range, high_range, num_steps):
    ''' Note, values less than low are ignored, values greater than high are considered edges. Pixels with a gradient magnitude between low and high are conditionally accepted if they're connected to strong edges. ''' 
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    
    # Initialize a 4D array to store results
    canny_multi_edge = np.zeros((num_steps, *pixelarray.shape), dtype=bool)
    
    for j, (low, high) in enumerate(zip(low_values, high_values)):
        canny_edge = apply_canny(pixelarray, low, high)
        canny_multi_edge[j] = canny_edge
    
    return canny_multi_edge

low_range = (0,10) # 
high_range = (10, 30) # 
num_steps = 20

print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(image, low_range, high_range, num_steps)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")

#%%
# add the 4d image to a new viewer
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='high_res_2')
#%%

#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[11]
viewer.add_image(tib_canny, name='tgv_1e-1')
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
bone_canny = removed_4d[5] 
viewer.add_image(bone_canny, name='after_remove_small') #%%
#%%
np.save('after_remove_small', bone_canny)
#%%
# 
# Step 7 find labels of connected regions from the edge image
labeled_image, num_features = label(bone_canny, return_num=True, connectivity=2)

viewer.add_labels(labeled_image, name='labeled_tib')    
#%%
# pick a suitable label that represents one long edge of the bone
tib_label = labeled_image == 7                                                                                               
viewer.add_image(tib_label, name='one_label')
#%%
np.save('tib_label_1', tib_label)

#%%
tib_label = tib_label
viewer.add_image(tib_label, name='one_label')
#%%
def find_corres(setA, setB):
    if len(setA) == len(setB):
        return setB  # or return setA, since they are of equal length

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setB)
    distances, indices = nbrs.kneighbors(setA)
    B_corresponding = setB[indices.flatten()]
    return B_corresponding

''' for the record this function works.. but for template frame having more points, it just returns the target frame as is
the issue being we cannot get H when this happens. so below, this function is modified to choose the least points frame by default ''' 
def find_corres_for_all_frames(fem_label):
    template_index = np.argmin([np.sum(frame) for frame in fem_label])
    print('template is frame: ', template_index)
    # Get the coordinates for the template
    template_set = fem_label[template_index]
    template_props = regionprops(template_set.astype(int))
    template_cords = template_props[0].coords

    all_subsets = []  # This list will hold all the subsets for each frame

    # Loop over all frames in fem_label
    for i in range(len(fem_label)):
        # Skip the template frame itself
        if i == template_index:
            all_subsets.append(template_cords)
            continue  # skip the rest of the loop for this iteration

        test_set = fem_label[i]
        test_props = regionprops(test_set.astype(int))
        test_cords = test_props[0].coords

        if len(template_cords) >= len(test_cords):
            all_subsets.append(test_cords)
        else:
            # Use your find_corres function to find the corresponding points
            subset_cords = find_corres(template_cords, test_cords)

            # Add these corresponding points to all_subsets list
            all_subsets.append(subset_cords)

    return all_subsets

all_subsets = find_corres_for_all_frames(tib_label)
viewer.add_points(points_for_napari(all_subsets), name='all_subsets', size=1, face_color='brown')
#%%
''' this sorts the points '''
def sort_curve_points_manually(points, frame_number):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    starting_point = points[np.argmax(points[:, 0])]  # Highest row value
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        
        # Check if the distance is much larger than average
        if len(sorted_points) > 1:
            avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
            large_jump = np.linalg.norm(np.array(next_point) - np.array(current_point)) > 5 * avg_distance
            if large_jump:
               print(f"Warning: Large distance jump detected in frame {frame_number} between point {current_point} and point {next_point}")  # Modified print statement
               #break
        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)

    return np.array(sorted_points)

def sort_curve_for_all_frames2(frames):
    sorted_frames = []
    for frame_number, frame in enumerate(frames):  # Enumerate to get the frame number
        sorted_frame = sort_curve_points_manually(frame, frame_number)  # Pass frame number
        sorted_frames.append(sorted_frame)
    return sorted_frames

sorted_subsets = sort_curve_for_all_frames2(all_subsets)
#%%
viewer.add_points(points_for_napari(sorted_subsets), name='sorted_subsets', size=1, face_color='blue')

#%%
def resample_curve(curve, n_points=25):
    # Calculate the total length of the curve
    diff = np.diff(curve, axis=0)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    total_length = np.sum(dists)
    
    # Calculate step distance
    step = total_length / (n_points - 1)
    
    # Initialize variables
    new_curve = [curve[0]]  # Start with the first point
    dist_covered = 0.0
    j = 0  # Index for the original curve
    
    for _ in range(1, n_points - 1):
        dist_needed = step
        
        while dist_needed > 0:
            segment_remaining = dists[j] - dist_covered
            if segment_remaining > dist_needed:
                # Get the next point from the current segment
                ratio = dist_needed / dists[j]
                next_point = curve[j] + ratio * (curve[j+1] - curve[j])
                new_curve.append(next_point)
                
                # Update the distance covered on the current segment
                dist_covered += dist_needed
                dist_needed = 0.0
            else:
                # Move to the next segment
                dist_needed -= segment_remaining
                dist_covered = 0.0
                j += 1
                
    new_curve.append(curve[-1])  # End with the last point
    return np.array(new_curve)

# For each frame in all_subsets, apply the function
all_subsets_resampled = [resample_curve(curve,50) for curve in sorted_subsets]
#%%
viewer.add_points(points_for_napari(all_subsets_resampled), name='resampled_subsets', size=2, face_color='red') 

#%%
''' Oct 29: applying the all_subsets_resampled directly yielded terrible results. Perhaps fixing them a bit will be better ''' 
from scipy.interpolate import CubicSpline

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

#%%
adjusted_1 = adjust_downsampled_points(all_subsets_resampled[0], sorted_subsets[0])
viewer.add_points(adjusted_1, name='adjusted_frame_0', size=2, face_color='violet' )
#%%
def apply_transformation(matrix, points):
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    return transformed_points[:, :2]  # Convert back to Cartesian coordinates

def apply_transformation_all_frames(reference_set, transformation_matrices):
    transformed_subsets = []
# Assuming `transformation_matrices` is your list of 2x3 transformation matrices
# and `reference_coords` is the coordinates of your reference frame
    for matrix in transformation_matrices:
        transformed_points = apply_transformation(matrix, reference_set)
        print(transformed_points.shape) 
        transformed_subsets.append(transformed_points)
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

def calculate_transform_matrices_procrustes(all_coords, reference_index):
    num_frames = len(all_coords)
    transformation_matrices = []

    # Initialize the identity matrix for the reference frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])

    # Get the coordinates of the reference frame
    reference_coords = all_coords[reference_index]

    for i in range(num_frames):
        if i == reference_index:
            # Add the identity matrix for the reference frame
            transformation_matrices.append(identity_matrix)
        else:
            # Estimate the transformation matrix using Procrustes
            R, t = procrustes(reference_coords, all_coords[i])
            transformation_matrix = np.hstack([R, t.reshape(-1, 1)])  # Combine into a 2x3 matrix
            transformation_matrices.append(transformation_matrix)

    return transformation_matrices
#%%
matrices_list = calculate_transform_matrices_procrustes(all_subsets_resampled, 0)
post_transformation = apply_transformation_all_frames(all_subsets_resampled[0], matrices_list)
#%%
viewer.add_points(points_for_napari(post_transformation), face_color='green', size=2) 

#%%
ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
transformed_expanded = apply_transformation_all_frames(ref_points, matrices_list) 
transformed_shapes = shapes_for_napari(transformed_expanded)
viewer.add_shapes(transformed_shapes, shape_type='polygon') 

#%%
shortest_length = min(np.sum(np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))) for curve in sorted_subsets)
#%%
all_subsets_resampled = [resample_curve4(curve, 50, shortest_length) for curve in sorted_subsets]
viewer.add_points(points_for_napari(all_subsets_resampled), name='resampled_target_length', size=2, face_color='green') 
#%%
def print_total_lengths(resampled_curves):
    for i, curve in enumerate(resampled_curves):
        # Calculate the total length of the curve
        diff = np.diff(curve, axis=0)
        dists = np.sqrt((diff ** 2).sum(axis=1))
        total_length = np.sum(dists)
        
        print(f"Total length of curve in frame {i}: {total_length}")

# Assuming all_subsets_resampled is your list of resampled curves
# Call the function like this:
print_total_lengths(all_subsets_resampled)

#%%
def resample_curve6(curve, n_points, target_length):
    # Calculate the total length of the curve
    diff = np.diff(curve, axis=0)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    total_length = np.sum(dists)
    
    if total_length < target_length:
        raise ValueError("Total length of the curve is less than the target length.")
    
    # Calculate step distance
    step = target_length / (n_points - 1)
    
    # Initialize variables
    new_curve = [curve[0]]  # Start with the first point
    dist_covered = 0.0
    j = 0  # Index for the original curve
    
    for _ in range(1, n_points):
        dist_needed = step
        
        while dist_needed > 0:
            if j >= len(dists):  # Exit condition
                break
            
            segment_remaining = dists[j] - dist_covered
            if segment_remaining > dist_needed:
                # Get the next point from the current segment
                ratio = dist_needed / dists[j]
                next_point = curve[j] + ratio * diff[j]
                new_curve.append(next_point)
                
                # Update the distance covered on the current segment
                dist_covered += dist_needed
                dist_needed = 0.0
            else:
                # Move to the next segment
                dist_needed -= segment_remaining
                dist_covered = 0.0
                j += 1
                
    return np.array(new_curve)

# Calculate the shortest length among all curves
shortest_length = min(
    np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
    for curve in sorted_subsets
)

# Resample each curve to have the same number of points and the same total length
all_subsets_resampled = [resample_curve6(curve, 50, shortest_length) for curve in sorted_subsets]
#%%
print_total_lengths(all_subsets_resampled)
viewer.add_points(points_for_napari(all_subsets_resampled), name='resampled_target_length', size=2, face_color='green') 
#%%
def scale_to_length(curve, target_length):
    # Calculate the current total length of the curve
    diff = np.diff(curve, axis=0)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    total_length = np.sum(dists)
    
    # Calculate the scaling factor and scale the curve
    scaling_factor = target_length / total_length
    scaled_curve = curve * scaling_factor
    
    return scaled_curve

# Post-process to ensure each curve has exactly the same total length
all_subsets_resampled_scaled = [scale_to_length(curve, shortest_length) for curve in all_subsets_resampled]
#%%
print_total_lengths(all_subsets_resampled_scaled)
viewer.add_points(points_for_napari(all_subsets_resampled_scaled), name='scaled', size=2, face_color='orange') 
#%%


new_list = extract_coords_from_frames(new_label)
#%%
''' now that we have resampled frame 1, convert the tib_label to coordinates and then stick this in.  '''
new_list[0] = adjusted_points
viewer.add_points(points_for_napari(new_list))
