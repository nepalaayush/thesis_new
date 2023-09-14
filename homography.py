#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:59:12 2023

@author: aayush
"""

import numpy as np
import napari
import nibabel as nib 
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from skimage.measure import label, regionprops
from skimage.feature import canny
import cv2
from scipy.ndimage import gaussian_filter
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
    return gradient_magnitude

def apply_canny(pixelarray, low, high):
    canny_edge = np.zeros_like(pixelarray)
    
    for i in range(pixelarray.shape[0]):
        canny_image = canny(pixelarray[i], low_threshold= low, high_threshold=high )    
        canny_edge[i] = canny_image
    return canny_edge.astype(dtype=bool)

#%%
image = open_nii('/data/projects/ma-nepal-segmentation/data/CINE_HighRes.nii')
image = normalize(image)
#%%
viewer = napari.view_image(image)
#%%
smooth_image = gaussian_filter(image, 2)
#viewer.add_image(smooth_image, name='smooth_image')
#%%
grad_smooth = gradify(smooth_image)
viewer.add_image(grad_smooth, name='gradient_smooth')
#%%
canny_edge = apply_canny(grad_smooth, low=0.1, high=1) # 
viewer.add_image(canny_edge, name= 'canny')
#%%
single_grad_smooth = grad_smooth[0]
single_canny = canny_edge[0]
single_smooth = gaussian_filter(image[0], 2)
#%%
def apply_canny_multiple_thresholds(pixelarray, low_range, high_range, num_steps):
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    
    # Initialize a 4D array to store results
    canny_multi_edge = np.zeros((num_steps, *pixelarray.shape), dtype=bool)
    
    for j, (low, high) in enumerate(zip(low_values, high_values)):
        canny_edge = apply_canny(pixelarray, low, high)
        canny_multi_edge[j] = canny_edge
    
    return canny_multi_edge

low_range = (1, 1)
high_range = (2, 10)
num_steps = 10


canny_multi_edge = apply_canny_multiple_thresholds(grad_smooth, low_range, high_range, num_steps)
#%%
viewer.add_image(canny_multi_edge, name='4d_canny')
#%%
fem_canny = canny_multi_edge[6]
#%%
viewer.add_image(fem_canny, name='fem_canny')
#%%
labeled_image, num_features = label(fem_canny, return_num=True, connectivity=1)
#%%
viewer.add_image(labeled_image, name='labeled_fem')
#%%
viewer.add_image((labeled_image == 18), name='one_label')
#%%
fem_label = labeled_image == 18 
z, y, x = np.where(fem_label)

# Stack them into an N x 3 NumPy array
coords = np.column_stack((z, y, x))    
#%%
props = regionprops(fem_label.astype(int))
coords1 = props[0].coords 
#%%
def find_corrs_list(grouped_coords, choice):
    # Set your 'choice' frame as setA
    setA = grouped_coords[choice][:, 1:]

    # Initialize an empty list to hold the corresponding coordinate sets for each frame
    corrs_list = []

    for setB in grouped_coords:
        # Only keep the y and x coordinates (assuming z, y, x)
        setB = setB[:, 1:]

        # Find corresponding points
        A_corresponding = find_corres(setA, setB)

        corrs_list.append(A_corresponding)

    return corrs_list

def find_corres(setA, setB):
    if len(setA) > len(setB):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setA)
        distances, indices = nbrs.kneighbors(setB)
        A_corresponding = setA[indices.flatten()]
        return A_corresponding
    else:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setB)
        distances, indices = nbrs.kneighbors(setA)
        B_corresponding = setB[indices.flatten()]
        return B_corresponding

#%%
def group_the_coords(image, coords):
    num_frames = image.shape[0]

    # Initialize an empty list to store the grouped arrays
    grouped_coords = []

    # Loop through each frame to group the coordinates
    for frame in range(num_frames):
        grouped_array = coords[coords[:, 0] == frame]
        grouped_coords.append(grouped_array)
    
    return grouped_coords
        
grouped_coords = group_the_coords(fem_label, coords)
#%%
fem_cords = find_corrs_list(grouped_coords, 10)
#%%
final_points = []
# Initialize an empty list to hold the 3D coordinates
for t, arr in enumerate(fem_cords):
    frame_annotated = np.hstack([arr, np.full((arr.shape[0], 1), t)])
    final_points.append(frame_annotated)

# Stack all the annotated points
final_points = np.vstack(final_points)
#%%
viewer0 = napari.Viewer()
#%%
transposed_points = np.transpose(final_points)

viewer0.add_points(transposed_points)
#%%
dummy = np.array([[ 13.        , 265.04950248, 281.19690652],
       [ 13.        , 291.33595552, 273.68649136],
       [ 13.        , 322.87969918, 287.95628016],
       [ 13.        , 316.87136705, 318.7489823 ],
       [ 12.        , 319.87553311, 358.55418263],
       [ 12.        , 205.71722561, 257.91461954],
       [ 12.        , 283.82554323, 351.04376747],
       [ 12.        , 204.96618409, 321.75314836]])

#%%
''' this works  ''' 
final_points = []

# Iterate through each frame, getting the frame number and the array of points for that frame
for frame_number, points_in_frame in enumerate(fem_cords):
    # Create an array of the same length as points_in_frame, filled with the frame_number
    frame_numbers = np.full((len(points_in_frame), 1), frame_number)
    
    # Horizontally stack the frame_numbers array and the points_in_frame array
    annotated_points = np.hstack([frame_numbers, points_in_frame])
    
    # Append the annotated points to final_points
    final_points.append(annotated_points)

# Vertically stack all the annotated points to create a single array
final_array = np.vstack(final_points)
#%%
viewer.add_points(final_array)