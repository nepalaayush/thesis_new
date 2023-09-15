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
from skimage.morphology import skeletonize
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
canny_edge = apply_canny(grad_smooth, low=1, high=10) # 
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

low_range = (0.1, 1)
high_range = (1, 10)
num_steps = 10


canny_multi_edge = apply_canny_multiple_thresholds(grad_smooth, low_range, high_range, num_steps)
#%%
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='4d_canny')
#%%
fem_canny = canny_multi_edge[0]
#%%
viewer = napari.Viewer()
#%%
viewer.add_image(fem_canny, name='fem_canny')
#%%
fem_canny = canny_edge
#%%

labeled_image, num_features = label(fem_canny, return_num=True, connectivity=1)

viewer.add_image(labeled_image, name='labeled_fem')                                                                        
#%%
fem_label = labeled_image == 27                                                                                                 
viewer.add_image(fem_label, name='one_label')
#%%
one_frame_uint8 = (fem_canny[0] * 255).astype('uint8')
kernel = np.ones((1, 1), np.uint8)
#%%
# Apply dilation
dilated_frame = cv2.dilate(one_frame_uint8, kernel, iterations=1)
viewer.add_image(dilated_frame, name='dilated')
#%%
eroded_frame = cv2.erode(one_frame_uint8, kernel, iterations=1)
viewer.add_image(eroded_frame, name='eroded' )
#%%
sk = skeletonize(fem_canny)
viewer.add_image(sk, name= 'skeleton')

#%%
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

all_subsets = find_corres_for_all_frames(fem_label)

#%%
viewer0 = napari.view_image(fem_label)
#%%
''' all_points is just for visual purposes to show in naapari. the H is obtained from the list 'all_subsets' ''' 
all_points = []
frame_ids = []

for i, subset in enumerate(all_subsets):
    frame_id_column = np.full((subset.shape[0], 1), i)
    frame_subset = np.hstack([frame_id_column, subset])
    all_points.append(frame_subset)

all_points = np.vstack(all_points)
#%%
viewer0.add_points(all_points, size=1)

#%%
def calculate_homography_matrices(all_subsets, template_index):
    all_H_list = []
    
    template_set = all_subsets[template_index]
    
    for i, target_set in enumerate(all_subsets):
        if i == template_index:
            # Add the identity matrix for the template index
            all_H_list.append(np.eye(3))
            continue
            
        H, _ = cv2.findHomography(template_set, target_set, cv2.RANSAC)
        
        all_H_list.append(H)
    
    return all_H_list

# Example usage:
all_H_list = calculate_homography_matrices(all_subsets, 23)
#%%
def transform_points_cv2(points, H):
    points_cv2_format = np.array([points], dtype=np.float32)
    transformed_points_cv2_format = cv2.perspectiveTransform(points_cv2_format, H)
    transformed_points = transformed_points_cv2_format[0]
    return transformed_points

# Assuming all_H_list contains your homography matrices
# and template_set contains the points of your template set

transformed_all_subsets = []

template_set = all_subsets[23]

for H in all_H_list:
    transformed_set = transform_points_cv2(template_set, H)
    transformed_all_subsets.append(transformed_set)
    
#%%
def points_in_napari(list_points):
    all_points = []

    for i, subset in enumerate(list_points):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_points.append(frame_subset)

    all_points = np.vstack(all_points)
    return all_points    
#%%
transformed_points = points_in_napari(transformed_all_subsets)

#%%
viewer0.add_points(transformed_points, size=1, face_color='yellow')
#%%
viewer1 = napari.view_image(grad_smooth)
#viewer1.add_shapes(all_subsets[23], shape_type='polygon')
#%%
viewer1.add_points(all_subsets[23], size=1)                                    

#%%
''' after using the template points to draw the actual segmented bone '''
segmented_points = viewer1.layers['Points'].data
#%%
def transform_segmented_points(segmented_points, all_H_list):
    # Function to transform points using cv2.perspectiveTransform
    def transform_points_cv2(points, H):
        points_cv2_format = np.array([points], dtype=np.float32)
        transformed_points_cv2_format = cv2.perspectiveTransform(points_cv2_format, H)
        transformed_points = transformed_points_cv2_format[0]
        return transformed_points

    transformed_all_points = []
    for H in all_H_list:
        transformed_points = transform_points_cv2(segmented_points, H)
        transformed_all_points.append(transformed_points)
        
    return transformed_all_points

transformed_all_points = transform_segmented_points(segmented_points, all_H_list)

#%%
transformed_points_in_napari = points_in_napari(transformed_all_points)

viewer1.add_points(transformed_points_in_napari, name='ultimate_check', size=1)
#%%
def transform_segmented_points_manual(segmented_points, all_H_list):
    transformed_all_points = []
    
    # Loop over all H matrices
    for H in all_H_list:
        # Convert points to homogeneous coordinates
        homogeneous_points = np.column_stack([segmented_points, np.ones(segmented_points.shape[0])])
        
        # Perform the transformation in homogeneous coordinates
        transformed_homogeneous_points = np.dot(H, homogeneous_points.T).T
        
        # Convert back to Cartesian coordinates
        transformed_points = transformed_homogeneous_points[:, :2] / transformed_homogeneous_points[:, 2, np.newaxis]
        
        # Add the transformed points to the list
        transformed_all_points.append(transformed_points)
        
    return transformed_all_points

transformed_all_points1 = transform_segmented_points_manual(segmented_points, all_H_list)

#%%
''' homography did not work  ''' 
def calculate_euclidean_matrices(all_subsets, template_index):
    all_E_list = []  # This list will hold all the Euclidean transformation matrices
    
    template_set = all_subsets[template_index]
    
    for i, target_set in enumerate(all_subsets):
        if i == template_index:
            # Add the identity matrix for the template index
            all_E_list.append(np.eye(3))
            continue
            
        E, _ = cv2.estimateAffinePartial2D(template_set, target_set)
        
        # Convert to a 3x3 matrix for consistency
        E = np.vstack([E, [0, 0, 1]])

        all_E_list.append(E)
    
    return all_E_list
# Sample usage
# fem_label is your 3D array with shape (26, y, x)
# template_index is the index of the template frame you want to use
all_E_list = calculate_euclidean_matrices(all_subsets, template_index=23)
#%%

def transform_segmented_points_manual(segmented_points, all_E_list):
    transformed_all_points = []
    
    # Loop over all E matrices
    for E in all_E_list:
        # Convert points to homogeneous coordinates
        homogeneous_points = np.column_stack([segmented_points, np.ones(segmented_points.shape[0])])
        
        # Perform the transformation in homogeneous coordinates
        transformed_homogeneous_points = np.dot(E, homogeneous_points.T).T  # E is already a 3x3 matrix including rotation and translation
        
        # Convert back to Cartesian coordinates
        transformed_points = transformed_homogeneous_points[:, :2] / transformed_homogeneous_points[:, 2, np.newaxis]
        
        # Add the transformed points to the list
        transformed_all_points.append(transformed_points)
        
    return transformed_all_points

transformed_points_list = transform_segmented_points_manual(segmented_points, all_E_list)
#%%
ultimate_test = points_in_napari(transformed_points_list)
#%%
viewer1.add_points(ultimate_test, size=1)