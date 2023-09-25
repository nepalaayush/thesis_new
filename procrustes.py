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
image = open_nii('C:/Users/MSI/Documents/thesis/data/CINE_HighRes.nii')
image = normalize(image)
#%%
viewer = napari.view_image(image)
#%%
smooth_image = gaussian_filter(image, 2)
viewer.add_image(smooth_image, name='smooth_image')
#%%
grad_smooth = gradify(smooth_image)

viewer.add_image(grad_smooth, name='gradient_smooth')
#%%
canny_edge = apply_canny(smooth_image, low=1, high=10) # 
viewer.add_image(canny_edge, name= 'canny')
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

low_range = (4,10) # 3,4 for fem
high_range = (15, 17) # 4,4.2 for fem 
num_steps = 5

print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(smooth_image, low_range, high_range, num_steps)
#%%
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='4d_canny_s')
#%%
fem_canny = canny_multi_edge[1]
#%%
tib_canny = canny_multi_edge[1]
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
size_range = (60, 65)  # 100 min and 200 max size for femur 
num_steps = 5  # Number of steps for size parameter
connectivity = 2  # Fixed connectivity value
print(np.linspace(size_range[0],size_range[1], num_steps))
# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(canny_edge, size_range, num_steps, connectivity)
#%%
