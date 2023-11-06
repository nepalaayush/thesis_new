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
image = open_nii('/data/projects/ma-nepal-segmentation/data/Singh^Udai/2023-09-11/72_MK_Radial_NW_CINE_60bpm_CGA/aw1_rieseling_admm_tgv_5e-3.nii')
image = normalize(image)
image = np.moveaxis(image, 1, 0)[1:]
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='zero-1')
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

low_range = (0,1) # 
high_range = (2,7 ) # 
num_steps = 20
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
viewer3.add_image(canny_multi_edge, name='inverted_sigma1')
#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[19]
viewer.add_image(tib_canny, name='after_edge_detection_non_blurreed_sigma_1')