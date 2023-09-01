#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:53:30 2023

@author: aayush
"""

import numpy as np
import napari
import nibabel as nib 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize, binary_erosion
#%%
def open_nii(path):
    ''' Input: Path of nifti file (.nii) Output: pixelarray  ''' 
    nifti_img = nib.load(path)
    #pixelarray = np.transpose ( nifti_img.get_fdata(), (2,0,1)) # have to transpose because of the way nifti loads things 
    pixelarray = nifti_img.get_fdata()
    return np.moveaxis(pixelarray, -1,0)

def log_transform(image):
    """Applies a logarithm transform to an image.

    Args:
        image (numpy.ndarray): The image to be transformed.

    Returns:
        numpy.ndarray: The transformed image.
    """

    return np.log1p(image)


def bwskel(image):
    """Performs a skeletonization of an image.

    Args:
        image (numpy.ndarray): The image to be skeletonized.

    Returns:
        numpy.ndarray: The skeletonized image.
    """

    import skimage.morphology as sm

    return sm.skeletonize(image)
#%%
image = open_nii('/data/projects/ma-nepal-segmentation/data/CINE_HighRes.nii')[0]

#%%
viewer = napari.view_image(image)
#%%

def gaussian_smoothing(image, sigma):
    
    # Separate magnitude and phase
    magnitude = np.abs(freq_img)
    phase = np.angle(freq_img)

    # Apply Gaussian smoothing to the magnitude
    magnitude_blur = gaussian_filter(magnitude, sigma)

    # Combine the smoothed magnitude with the original phase
    freq_blur = magnitude_blur * np.exp(1j * phase)
    
    return freq_blur

def fft(image):
    return np.fft.fftshift( np.fft.fft2(image) ) 


def ifft(freq_image):
    return np.fft.ifft2( np.fft.ifftshift(freq_image))     


def normalize_array(array):
  """Normalizes an array so that the values are between 0 and 1.

  Args:
    array: The array to be normalized.

  Returns:
    The normalized array.
  """

  min_value = np.min(array)
  max_value = np.max(array)

  normalized_array = (array - min_value) / (max_value - min_value)

  return normalized_array
#%%
freq_img = fft(image)
#%%
viewer.add_image(np.real(freq_img), name='freq_mag')
#%%

freq_blur = gaussian_smoothing(freq_img, 30)
# Get the spatial domain image
spatial_blur = ifft(freq_blur)
viewer.add_image(np.real(spatial_blur), name = '30 sigma')

#%%
binary_image = np.real(spatial_blur) < -1e-9
viewer.add_image(binary_image)
#%%

ede_img = skeletonize(binary_image)
viewer.add_image(ede_img, name='skeleton binary')
#%%

viewer.add_image(gaussian_filter(ede_img,3), name='smooth skeleton')

#%%
from scipy.ndimage import label, binary_dilation

labeled_lines, num_features = label(ede_img) # np.real(spatial_blur) < -1e-9

bone1 = [6927, 4993, 6197]
bone2 = [9202,8185,8735]

se = np.array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]])

bone1_template = binary_dilation(np.isin(labeled_lines, bone1), structure=se).astype(float)
bone2_template = binary_dilation(np.isin(labeled_lines, bone2), structure=se).astype(float)

#%%
viewer.add_image(labeled_lines) 