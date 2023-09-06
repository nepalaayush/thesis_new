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

def normalize (image):
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    normalized_img = ( image - min_val ) / (max_val - min_val) 
    scaled_img = (normalized_img * 255)
    
    return scaled_img 

def log_transform(image):
    """Applies a logarithm transform to an image.

    Args:
        image (numpy.ndarray): The image to be transformed.

    Returns:
        numpy.ndarray: The transformed image.
    """

    return np.log1p(image)

def create_low_pass_filter(image_shape, sigma=80):

    """

    Create a Gaussian low-pass filter.

    

    Parameters:

        image_shape (tuple): Shape of the image (height, width).

        sigma (float): Sigma value for the Gaussian function.

    

    Returns:

        numpy.ndarray: The high-pass filter.

    """

    M, N = image_shape

    #M, N = 2 * M + 1, 2 * N + 1  # Extend dimensions as in the Matlab code



    # Create meshgrid

    X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))

    centerX, centerY = np.ceil(N / 2), np.ceil(M / 2)



    # Gaussian numerator

    gaussian_numerator = (X - centerX)**2 + (Y - centerY)**2


otsu
    # Create high-pass filter using Gaussian equation

    H = np.exp(-gaussian_numerator / (2 * sigma ** 2))

    H = 1 - H  # High-pass filter

    H = np.fft.fftshift(H)  # Fourier shift



    return H  

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
image = open_nii('/data/projects/ma-nepal-segmentation/data/CINE_HighRes.nii')

image = normalize(image)

#%%
viewer = napari.view_image(image)
#%%
log_transformed_image = log_transform(image)

viewer.add_image(log_transformed_image, name='Log Transformed Image')

#%%
# Test function with a dummy image shape

image_shape = (256, 256)  # Replace with your actual image shape

high_pass_filter = create_high_pass_filter(image_shape)

actual_high = 1 - high_pass_filter

viewer.add_image(actual_high, name='High-Pass Filter')


#%%
def apply_high_pass_filter(image, low_pass_filter):
    # Perform Fourier Transform on the image
    image_fft = np.fft.fft2(image)
    #image_fft_shifted = np.fft.fftshift(image_fft)  # FFT shift
    #viewer.add_image(np.abs(image_fft), name='Non Shifted image fft')
    # Convert the low-pass filter to a high-pass filter
    high_pass_filter = 1 - low_pass_filter
    #viewer.add_image(high_pass_filter, name = 'high pass filter')
    # Apply the high-pass filter
    filtered_fft = image_fft * high_pass_filter
    
    # Perform inverse Fourier Transform to get the filtered image back
    filtered_image = np.fft.ifft2(filtered_fft)
    
    return np.abs(filtered_image)



low_pass_filter = create_low_pass_filter(image.shape, sigma=80)
# Apply the high-pass filter to the image
filtered_image = apply_high_pass_filter(image, low_pass_filter)
viewer.add_image(filtered_image, name='Filtered Image')

#%%
def apply_gaussian_smoothing(image, sigma=20):
    """
    Apply Gaussian smoothing to the image.
    
    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        sigma (float): The standard deviation of the Gaussian kernel.
        
    Returns:
        numpy.ndarray: The smoothed image.
    """
    return gaussian_filter(image, sigma)

# Assuming 'filtered_image' is the image obtained after high-pass filtering
smoothed_image = apply_gaussian_smoothing(filtered_image, sigma=2)

viewer.add_image(smoothed_image)

#%%

def apply_skeletonization_and_thresholding(image, threshold):
    """
    Apply thresholding and skeletonization to the image.
    
    Parameters:
        image (numpy.ndarray): 2D array representing the image.
        threshold (float): Threshold value for binarization.
        
    Returns:
        numpy.ndarray: The skeletonized image.
    """
    # Apply thresholding
    binary_image = image < threshold
    
    
    viewer.add_image(binary_image, name = 'binary_image')
    
    mask = image [binary_image]
    
    # Apply skeletonization
    skeletonized_image = skeletonize(mask)
    
    return skeletonized_image

# Assuming 'smoothed_image' is your final smoothed image
# You can tune the threshold based on your observations
skeletonized_image = apply_skeletonization_and_thresholding(smoothed_image, threshold=60)

viewer.add_image(skeletonized_image, name= 'skeletonized_image')

#%%
from skimage.segmentation import active_contour

ini_contours = np.squeeze( viewer.layers['Shapes'].data ) 

snake = [ active_contour(image, ini_contours) ] 

#%%

#%%

smooth_array = np.zeros_like(image)
low_pass_filter = create_low_pass_filter(image[0].shape, sigma=80)
for i in range(image.shape[0]):
    filtered_image = apply_high_pass_filter(image[i], low_pass_filter)
    smoothed_image = gaussian_filter(image[i], 2)
    smooth_array[i] = smoothed_image
    
#%%
viewer = napari.view_image(smooth_array)
#%%
viewer.add_image(smooth_array)

#%%
def gradify(pixelarray):
    dy, dx = np.gradient(pixelarray, axis=(1, 2))

    # Magnitude of the gradient for each frame
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    return gradient_magnitude

gradient_smooth = gradify(smooth_array)

viewer.add_image(gradient_smooth, name='gradient_smooth')
#%%
ini_contours = np.squeeze( viewer.layers['Shapes'].data ) 
ini_contours = ini_contours[:,1:]
#%%
snake = [ active_contour(image, ini_contours) ] 
#%%
# Create an empty shapes layer for the automated contours
#auto_shapes_layer = viewer.add_shapes(name='Automated Contours', shape_type='polygon')
auto_contours_list = []
# Loop through each frame in the image stack
for i, frame in enumerate(image):
    # Convert the frame to grayscale if it's not already (skip this if your image is already grayscale)
    
    # Apply the active contour algorithm to the frame using the initial contour
    snake = active_contour(frame, ini_contours)
    snake_with_frame = np.hstack([np.full((snake.shape[0], 1), i), snake])
    # Add the resulting contour to the new shapes layer
    
    auto_contours_list.append(snake_with_frame)


viewer.add_shapes(auto_contours_list, shape_type='polygon', name='Automated Contours')

#%%
from skimage.filters import threshold_otsu

thresh_value = 0.7
bin_array = gradient_smooth < thresh_value 

viewer.add_image(bin_array, name='thresholded')
#%%
viewer.add_image(gradient_smooth, name='gradient_smooth')
#%%
''' looking at canny on the gradient smooth ''' 
single_frame = gradient_smooth[0]

from skimage import feature 

canny_img = feature.canny(single_frame, sigma=3, low_threshold=0.4, high_threshold=2.17)

viewer.add_image(canny_img)