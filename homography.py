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

def points_in_napari(list_points):
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


canny_multi_edge = apply_canny_multiple_thresholds(smooth_image, low_range, high_range, num_steps)
#%%
print(np.linspace(4,10, num_steps), '________________')
print(np.linspace(15,17,num_steps)) 
#%%
viewer3 = napari.Viewer() 
#%%
viewer3.add_image(canny_multi_edge, name='4d_canny_s')
#%%
fem_canny = canny_multi_edge[1]
#%%
tib_canny = canny_multi_edge[1]
#%%
viewer4 = napari.Viewer()
#%%
viewer.add_image(tib_canny, name='tib_canny')

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

# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(tib_canny, size_range, num_steps, connectivity)
#%%
viewer3.add_image(removed_4d, name='tib_remove')
#%%
fem_removed = removed_4d[2]

viewer.add_image(fem_removed, name='fem_removed') 
#%%
tib_removed = removed_4d[2]

viewer.add_image(tib_removed, name='tib_removed') 
#%%
labeled_image, num_features = label(fem_removed, return_num=True, connectivity=2)

viewer.add_labels(labeled_image, name='labeled_fem2')                                                                        

#%%
labeled_image, num_features = label(tib_removed, return_num=True, connectivity=2)

viewer.add_labels(labeled_image, name='labeled_tib')                                                                        

#%%
fem_label = labeled_image == 2                                                                                                 
viewer.add_image(fem_label, name='one_label')
#%%
tib_label = labeled_image == 11                                                                                                 
viewer.add_image(tib_label, name='one_label')


#%%
# try dilation 
# try erosion 
#
#
#
#
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
tib_label_napari = points_in_napari(all_subsets)

viewer.add_points(tib_label_napari, name='tib_subsets', size = 2, face_color='red')
#%%
one_frame = all_subsets[2]
#%%
step_size = len(one_frame) // 30 
coarse_sample = one_frame[::step_size] 

#%%
viewer4.add_shapes(coarse_sample, name='single_frame', shape_type='polygon') 
#%%
viewer4.add_image(grad_smooth) 
#%%
fem_label_napari = points_in_napari(all_subsets)
#%%
viewer.add_points(fem_label_napari, name='fem_subsets', size = 2, face_color='red')
#%%
def calculate_transform_matrices(all_coords, reference_index):
    num_frames = len(all_coords)
    transformation_matrices = []

    # Initialize the identity matrix for the reference frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])

    # Get the coordinates of the reference frame
    reference_coords = all_coords[reference_index]

    for i in range(num_frames):
        if i == reference_index:
            # Add the identity matrix for the reference frame
            transformation_matrices.append(identity_matrix)
        else:
            # Estimate the transformation matrix for the current frame
            mtx,ret = cv2.estimateAffinePartial2D(reference_coords, all_coords[i])
            transformation_matrices.append(mtx)

    return transformation_matrices

from skimage.transform import estimate_transform
def calculate_transform_matrices_skimage(all_coords, reference_index):
    num_frames = len(all_coords)
    transformation_matrices = []

    # Initialize the identity matrix for the reference frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])

    # Get the coordinates of the reference frame
    reference_coords = all_coords[reference_index]

    for i in range(num_frames):
        if i == reference_index:
            # Add the identity matrix for the reference frame
            transformation_matrices.append(identity_matrix)
        else:
            # Estimate the transformation matrix for the current frame
            tform = estimate_transform('similarity', reference_coords, all_coords[i])
            transformation_matrices.append(tform.params[:2, :])  # Extract the 2x3 matrix

    return transformation_matrices


transformation_matrices = calculate_transform_matrices_skimage(all_subsets, 2)

#%%
def apply_transformation(matrix, points):
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    return transformed_points[:, :2]  # Convert back to Cartesian coordinates

def apply_transformation_all_frames(reference_set, transformation_matrices):
    transformed_subsets = []
# Assuming `transformation_matrices` is your list of 2x3 transformation matrices
# and `reference_coords` is the coordinates of your reference frame
    for matrix in transformation_matrices:
        transformed_points = apply_transformation(matrix, reference_set)
        print(transformed_points.shape) 
        transformed_subsets.append(transformed_points)
    return transformed_subsets
# Now `transformed_subsets` should contain the transformed points for each 

def remove_scaling_from_affine_matrices(affine_matrices):
    new_matrices = []
    for affine_matrix in affine_matrices:
        # Extract rotation and scaling by performing SVD on the upper-left 2x2 matrix
        upper_left = affine_matrix[:, :2]
        U, S, Vt = np.linalg.svd(upper_left)
        
        # Reconstruct the rotation matrix
        R = np.dot(U, Vt)
        
        # Extract translation
        T = affine_matrix[:, 2]
        
        # Create a new 2x3 matrix with only rotation and translation
        rot_trans_matrix = np.zeros((2, 3))
        rot_trans_matrix[:, :2] = R
        rot_trans_matrix[:, 2] = T
        
        new_matrices.append(rot_trans_matrix)
        
    return new_matrices
#%%
ref_points = viewer.layers['Points'].data[:,1:]
#%%
ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
#%%
transformed_subsets = apply_transformation_all_frames(ref_points, transformation_matrices)
#%%
shapes_in_napari = shapes_for_napari(transformed_subsets)

viewer.add_shapes(shapes_in_napari, shape_type='polygon') 
#%%
np.save('all_subsets', all_subsets) 
#
# now that we have obtained and checked the transformation matrices, it is time to test it out. 
#
#
#




#%%


#%%
grad_smooth = gradify(smooth_image)
viewer.add_image(grad_smooth, name='gradient_image') 
#%%
reference_shape = all_subsets[22] 
viewer.add_shapes(reference_shape, shape_type='polygon')

#%%
ref_shape = viewer.layers['Shapes'].data[0][:,1:] 
#%%
viewer.add_points(ref_shape) 
#%%
#%%

final_output = apply_transformation_all_frames(ref_points)

final_napari = points_in_napari(final_output) 

viewer.add_points(final_napari, name='final_output')
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

