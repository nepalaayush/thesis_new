#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:08:58 2024

@author: aayush
"""

import pickle
import os 
#os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')

import numpy as np 
import napari 
import time 
from scipy import ndimage

from utils import (path_to_image, apply_canny, apply_remove, apply_skeleton, points_for_napari,
                   boolean_to_coords, apply_label, find_tibia_edges, find_array_with_min_n, downsample_points,
                   combined_consecutive_transform, coords_to_boolean)
#%%
# Step 1: load the image from directory and normalize it
path = '/data/projects/dfg-berlin-kneedynamics/data/Schulz^Helena/2024-09-13/33_MK_UTE_12bpm_S96_HighRes_W/HS_3d_W_riesling_pos.nii'

image = path_to_image(path)

viewer = napari.view_image(image,  name='3d_ute')

#%%
# Step 2: Reorient the image for sagittal view
# Assuming the original shape is (19, 252, 50, 210)
# We want to rearrange it to (19, 50, 210, 252)
image_sagittal = np.transpose(image, (0, 2, 1, 3))

# Step 3: Display the image in napari
viewer = napari.view_image(image_sagittal, name='3d_ute_sagittal')

# Optional: Add axis labels for clarity
viewer.dims.axis_labels = ['Frame', 'Sagittal Slice', 'IS', 'AP' ]  # not sure what this does as of yet. 

#%%
slice_mid = image_sagittal[:,20,:,:]


canny_one_slice = apply_canny(slice_mid, 0,5,2)

viewer2 = napari.view_image(canny_one_slice)

#%%
slice_2d = slice_mid[0]

# trying out some other techniques to do the edges better: 
from skimage import filters, segmentation, morphology, measure, exposure
from skimage.feature import canny 
import matplotlib.pylab as plt 
from skimage import filters, feature


slice_norm = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))


exposed = exposure.equalize_adapthist(slice_norm)

slice_smooth = filters.gaussian(exposed, sigma=0.1, preserve_range=True)

plt.imshow(slice_smooth)

#%%

def detect_edges(slice_2d):
    # Sobel edge detection
    edges_sobel = filters.sobel(slice_2d)
    # Thresholding
    edges_binary = edges_sobel > filters.threshold_otsu(edges_sobel)
    return edges_binary

edge_smooth = detect_edges(slice_smooth)
plt.imshow(edge_smooth)

#%%
def compare_edge_detection(slice_2d):
    # Sobel
    edges_sobel = filters.sobel(slice_2d)
    
    # Scharr
    edges_scharr = filters.scharr(slice_2d)
    
    # Prewitt
    edges_prewitt = filters.prewitt(slice_2d)
    
    # Roberts
    edges_roberts = filters.roberts(slice_2d)
    
    # Canny
    edges_canny = feature.canny(slice_2d)

    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(slice_2d, cmap='gray')
    axes[0].set_title('Original')

    axes[1].imshow(edges_sobel, cmap='gray')
    axes[1].set_title('Sobel')

    axes[2].imshow(edges_scharr, cmap='gray')
    axes[2].set_title('Scharr')

    axes[3].imshow(edges_prewitt, cmap='gray')
    axes[3].set_title('Prewitt')

    axes[4].imshow(edges_roberts, cmap='gray')
    axes[4].set_title('Roberts')

    axes[5].imshow(edges_canny, cmap='gray')
    axes[5].set_title('Canny')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return edges_sobel, edges_scharr, edges_prewitt, edges_roberts, edges_canny

# Use your preprocessed image
edges_sobel, edges_scharr, edges_prewitt, edges_roberts, edges_canny = compare_edge_detection(slice_smooth)

#%%
canny_exposed = canny(exposed, low_threshold= 0, high_threshold=0.1, sigma=1.5 ) 
plt.imshow(canny_exposed)
canny_normal = canny(slice_2d, low_threshold= 0, high_threshold=5, sigma=1.5 ) 
plt.imshow(canny_normal)