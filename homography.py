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
