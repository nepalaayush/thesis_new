#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:15:32 2023

@author: aayush
"""

import numpy as np
import napari
import nibabel as nib 
#import matplotlib.pyplot as plt 
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
#%%
image = open_nii('C:/Users/Aayush/Documents/thesis_files/CINE_HighRes.nii')
image = normalize(image)
#%%
viewer = napari.view_image(image)
#%%
smooth_image = gaussian_filter(image, 3)

viewer.add_image(smooth_image, name='smooth_image3')
#%%
grad_smooth = gradify(smooth_image)[0]
grad_direction = gradify(smooth_image)[1]
viewer.add_image(grad_smooth, name='gradient_smooth')
viewer.add_image(grad_direction, name='direction')
#%%
canny_direction = apply_canny(grad_direction, low=0, high=2)
viewer.add_image(canny_direction, name='edge_direction')
#%%
canny_edge = apply_canny(grad_smooth, low=0, high=1) # 
viewer.add_image(canny_edge, name= 'canny')
#%%
from scipy.ndimage import binary_dilation, binary_erosion

dilated = binary_dilation(canny_edge, iterations=2) 
opened = binary_erosion(dilated, iterations=2)
viewer.add_image(opened, name='morph_opened')
#%%
skeleton = skeletonize(opened)
viewer.add_image(skeleton, name='skeleton')
#%%

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

low_range = (1,2) # 3,4 for fem
high_range = (2.5, 3) # 4,4.2 for fem 
num_steps = 25

print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(grad_direction, low_range, high_range, num_steps)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")

#%%
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='4d_canny_s_level_3')
#%%
fem_canny = canny_multi_edge[1]
#%%
tib_canny = canny_multi_edge[5]
viewer.add_image(tib_canny, name='edge_smooth_5')
#%%
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
size_range = (10, 50)  # 100 min and 200 max size for femur 
num_steps = 50  # Number of steps for size parameter
connectivity = 2  # Fixed connectivity value
print(np.linspace(size_range[0],size_range[1], num_steps))
# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(tib_canny, size_range, num_steps, connectivity)
#%%
viewer3.add_image(removed_4d, name='multi_remove_small')

#%%
tib_canny = removed_4d[0] 
viewer.add_image(tib_canny, name='tibia_edge_image') 
#%%
labeled_image, num_features = label(tib_canny, return_num=True, connectivity=2)

viewer.add_labels(labeled_image, name='labeled_tib')    
#%%
tib_label = labeled_image == 17                                                                                                 
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
all_subsets_resampled = [resample_curve(curve) for curve in all_subsets]
#%%
resampled_napari = points_for_napari(all_subsets_resampled) 
viewer.add_points(resampled_napari) 


