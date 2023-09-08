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
from scipy.ndimage import gaussian_filter, zoom
from skimage.morphology import skeletonize, binary_erosion, square, erosion, dilation, remove_small_objects
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


def gradify(pixelarray):
    dy, dx = np.gradient(pixelarray, axis=(1, 2))

    # Magnitude of the gradient for each frame
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    return gradient_magnitude

def apply_filter_to_kspace(image):
    image_fft = np.fft.fft2(image)
    viewer.add_image(np.abs(image_fft), name='image_fft')



    filtered_fft = image_fft * low_pass_filter
    viewer.add_image(np.abs(filtered_fft), name='filtered_fft') 


    filtered_image = np.fft.ifft2(filtered_fft)
    viewer.add_image(np.abs(filtered_image), name='filtered_image')

#%%
image = open_nii('/data/projects/ma-nepal-segmentation/data/CINE_HighRes.nii')

image = normalize(image)

#%%
viewer = napari.view_image(image)

#%%
smooth_image = gaussian_filter(image, 2)
viewer.add_image(smooth_image)
#%%

low_pass_filter = create_low_pass_filter(image[0].shape, sigma=40)

viewer.add_image(low_pass_filter, name='low pass filter_40')

#%%

grad_smooth = gradify(smooth_image)
viewer.add_image(grad_smooth, name='gradient_smooth')
#%%
apply_filter_to_kspace(grad_smooth)
#%%
#import cv2
#resampled_image = cv2.resize(grad_smooth, grad_smooth.shape, interpolation=cv2.INTER_CUBIC)

#resampled_image = zoom(grad_smooth, zoom=1, order=3) # this doesnt do much. 
#%%
#viewer.add_image(resampled_image, name='resampled')

#%%
binary_image = grad_smooth < 0.8

viewer.add_image(binary_image, name='binary')

#%%
from skimage import feature 

def apply_canny(pixelarray, low, high):
    canny_edge = np.zeros_like(grad_smooth)
    
    for i in range(grad_smooth.shape[0]):
        canny_image = feature.canny(grad_smooth[i], low_threshold= low, high_threshold=high )    
        canny_edge[i] = canny_image
    return canny_edge.astype(dtype=bool)


canny_edge = apply_canny(grad_smooth, low=0.1, high=1)
#%%
viewer.add_image(canny_edge, name='canny_direct')
#%%
from skimage.measure import label

labeled_image, num_features = label(canny_edge, return_num=True, connectivity=1)
viewer.add_image(labeled_image, name='con_1_label')

#%%

fem_label = labeled_image == 173 

viewer.add_image(fem_label)

#%%
# extract the coordinates from the fem_label 

from skimage.measure import regionprops

props = regionprops(fem_label.astype(int))
coords = props[0].coords  # since `fem_label` contains only one label, we take the first item

#%%
# another way to get coords.. which is potentially better:

z, y, x = np.where(fem_label)

# Stack them into an N x 3 NumPy array
coords = np.column_stack((z, y, x))    
#%%
#to get the coords in a napari shape format:
num_frames = fem_label.shape[0]

# Initialize an empty list to store the grouped arrays
grouped_coords = []

# Loop through each frame to group the coordinates
for frame in range(num_frames):
    grouped_array = coords[coords[:, 0] == frame]
    grouped_coords.append(grouped_array)
#%%
viewer.add_shapes(grouped_coords, shape_type='polygon')
#%%
import cv2
''' a slightly different stuff of finding the transformation between sets of points.  ''' 
setA = grouped_coords[0]
setB = grouped_coords[1]

setA = setA[:,1:]
setB = setB[:,1:]


H, mask = cv2.findHomography(setA, setB, cv2.RANSAC)
#%%
homogeneous_A = np.hstack([setA, np.ones((setA.shape[0], 1))])

# Transform the points
transformed_A_homogeneous = homogeneous_A @ H.T  # Note the transpose of H

# Convert back to 2D coordinates
transformed_A = transformed_A_homogeneous[:, :2] / transformed_A_homogeneous[:, 2, np.newaxis]

print("Transformed A:")
print(transformed_A)
#%%

viewer1 = napari.view_image(grad_smooth[0]) 
#%%
viewer1.add_shapes(transformed_A, shape_type='polygon', name='H.dot.A')
#%%
# now we tery to do it for when set A and B do NOT have the same number of points 

setB = grouped_coords[20][:,1:]
#%%
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setA)
distances, indices = nbrs.kneighbors(setB)

# Extract the corresponding points
A_corresponding = setA[indices.flatten()]
#%%
H, _ = cv2.findHomography(A_corresponding, setB, cv2.RANSAC)

homogeneous_A = np.hstack([setA, np.ones((setA.shape[0], 1))])

# Transform the points
transformed_A_homogeneous = homogeneous_A @ H.T  # Note the transpose of H

# Convert back to 2D coordinates
transformed_A = transformed_A_homogeneous[:, :2] / transformed_A_homogeneous[:, 2, np.newaxis]
#%%

viewer.add_points(transformed_A, name='transformed_A')
#%%
dilated_image = dilation(canny_edge[0], square(3))
eroded_image = erosion(dilated_image, square(3))

#viewer.add_image(eroded_image, name='eroded_image_3')
#%%
def apply_remove_small(pixelarray, min_size):
    removed_small = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        removed_small[i] = remove_small_objects(pixelarray[i], min_size)
    return removed_small

removed_small = apply_remove_small(canny_edge, min_size=5)

viewer.add_image(removed_small, name = 'min_size_5')

#%%
from skimage import measure, io 

labeled_image, num_features = measure.label(canny_edge, return_num=True, connectivity=1)
#%%
viewer.add_image(labeled_image , name='labeled_image_connectivity_1')

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
    #smoothed_image = gaussian_filter(image[i], 2)
    smooth_array[i] = filtered_image
    
#%%
viewer = napari.view_image(smooth_array)
#%%
viewer.add_image(smooth_array)

#%%


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