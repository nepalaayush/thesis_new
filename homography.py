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
#import cv2
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
image = open_nii('C:/Users/Aayush/Documents/thesis_files/CINE_HighRes.nii')
image_norm = normalize(image)
padded_fft = np.load('C:/Users/Aayush/Documents/thesis_files/padded_fft.npy')
#%%
padded_fft = np.moveaxis(padded_fft, -1, 0)
padded_real = np.fft.ifft2(padded_fft).real
#%%
def zero_pad_fft(image, padding_factor):
    """
    Zero-pads the FFT of an image and returns the image from the padded FFT.

    Parameters:
        image (numpy.ndarray): Input MRI image as a 2D NumPy array.
        padding_factor (int): Factor by which to pad the FFT. A factor of 2 doubles the dimensions.

    Returns:
        numpy.ndarray: Zero-padded image.
    """
    # Compute the 2D FFT of the image
    fft_image = np.fft.fft2(image)

    # Calculate the new dimensions for zero-padding
    padded_shape = (image.shape[0] * padding_factor, image.shape[1] * padding_factor)

    # Create an array of zeros with the new shape
    zero_padded_fft = np.zeros(padded_shape, dtype=complex)

    # Copy the FFT data into the zero-padded array
    zero_padded_fft[:image.shape[0], :image.shape[1]] = fft_image

    # Compute the inverse FFT to get the zero-padded image
    zero_padded_image = np.fft.ifft2(zero_padded_fft).real

    return zero_padded_image

two_pad = zero_pad_fft(image[0], 2)

#%%
norm_two_pad = normalize(two_pad)
#%%
viewer = napari.view_image(image[0])
#%%
viewer0 = napari.view_image(two_pad)
#%%
smooth_image = gaussian_filter(image, 2)
viewer.add_image(smooth_image, name='smooth_image')
#%%
grad_smooth = gradify(smooth_image)

viewer.add_image(grad_smooth, name='gradient_smooth')
#%%
viewer0.add_image(norm_two_pad)
#%%
canny_edge = canny(two_pad, low_threshold=50, high_threshold=150) # 

viewer0.add_image(canny_edge, name= 'canny')

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
removed_4d = apply_remove_multiple_sizes(canny_edge, size_range, num_steps, connectivity)
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
tib_label = labeled_image == 16                                                                                                 
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
#%%
def extract_cords_from_boolean(boolean_array):
    ''' Input a 3d boolean array obtained from choosing a label from label array. Output: a list that stores the coordinates for each frame  ''' 
    all_subsets = []
    for frame in boolean_array:
        coords = np.argwhere(frame) 
        all_subsets.append(coords) 
    return all_subsets 
all_subsets = extract_cords_from_boolean(tib_label)
#%%
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
            tform = estimate_transform('euclidean', reference_coords, all_coords[i])
            transformation_matrices.append(tform.params[:2, :])  # Extract the 2x3 matrix

    return transformation_matrices

#%%
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

#%%
ref_points = viewer.layers['Points'].data[:,1:]
#%%
ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
#%%
ref_points = all_subsets[2]
#%%
transformed_subsets = apply_transformation_all_frames(ref_points, transformation_matrices)
#%%
shapes_in_napari = shapes_for_napari(transformed_subsets)

viewer.add_shapes(shapes_in_napari, shape_type='polygon', face_color='red') 
#%%
points_in_napari = points_for_napari(transformed_subsets) 
viewer.add_points(points_in_napari, face_color='green', size= 1) 

#%%
np.save('all_subsets', all_subsets) 
#
# now that we have obtained and checked the transformation matrices, it is time to test it out. 
#
#
#




#%%
''' using subsequent frames   ''' 
def calculate_transform_matrices_subsequent(all_coords):
    num_frames = len(all_coords)
    transformation_matrices = []

    # Initialize the identity matrix for the first frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])
    transformation_matrices.append(identity_matrix)

    for i in range(1, num_frames):
        # Find corresponding points between frame i-1 and frame i
        coords_i_minus_1 = all_coords[i-1]
        coords_i = all_coords[i]
        
        if len(coords_i_minus_1) >= len(coords_i):
            corres_coords_i = coords_i
            corres_coords_i_minus_1 = find_corres(coords_i, coords_i_minus_1)
        else:
            corres_coords_i = find_corres(coords_i_minus_1, coords_i)
            corres_coords_i_minus_1 = coords_i_minus_1

        # Estimate the transformation matrix for the current frame relative to the previous frame
        tform = estimate_transform('euclidean', corres_coords_i_minus_1, corres_coords_i)
        transformation_matrices.append(tform.params[:2, :])  # Extract the 2x3 matrix

    return transformation_matrices

def apply_transformation_all_frames_subsequent(initial_set, transformation_matrices):
    transformed_subsets = [initial_set]  # Start with the initial set
    current_set = initial_set

    for matrix in transformation_matrices[1:]:  # Skip the identity matrix
        transformed_points = apply_transformation(matrix, current_set)
        transformed_subsets.append(transformed_points)
        current_set = transformed_points  # Update the current set for the next iteration

    return transformed_subsets

def extract_coords_from_frames(label_frames):
    all_coords = []  # This list will hold the coordinates for each frame
    for frame in label_frames:
        coords = np.argwhere(frame)  # Find the coordinates of True pixels
        all_coords.append(coords)
    return all_coords

#%%
tib_subsets = extract_coords_from_frames(tib_label)
#%%
transformation_matrices_subsequent = calculate_transform_matrices_subsequent(tib_subsets)
#%%
transformed_subsets_subsequent = apply_transformation_all_frames_subsequent(tib_subsets[0], transformation_matrices_subsequent)
#%%
tib_subsets_napari = points_for_napari(tib_subsets) 
transformed_tib_subsets_napari = points_for_napari(transformed_subsets_subsequent) 
viewer.add_points(tib_subsets_napari, face_color='blue', size= 1) 
viewer.add_points(transformed_tib_subsets_napari, face_color='yellow', size=1) 
''' verdict, once again the drifting is very prominent ''' 

#%%
mini_tib = tib_subsets[15:] 
mini_matrices = calculate_transform_matrices_subsequent(mini_tib)
mini_transformed = apply_transformation_all_frames_subsequent(mini_tib[0], mini_matrices)
#%%
viewer = napari.Viewer() 
#%%
label_array = np.load('C:/Users/MSI/Documents/new_cloud/Jena/Books_For_MedPho/thesis/label_array.npy') 
#%%
viewer.add_image(label_array) 

#%%
tibia_array = label_array == 2
viewer.add_image(tibia_array) 
#%%
tib_subsets = find_corres_for_all_frames(tibia_array)
#%%
tib_label_napari = points_for_napari(tib_subsets)
viewer.add_points(tib_label_napari, name='tib_subsets', size = 2, face_color='red')
#%%
from scipy.ndimage import binary_erosion

def find_outline_3d(pixelarray):
    # Initialize an empty array to store the outlines
    outline_array = np.zeros_like(pixelarray, dtype=bool)
    
    # Define a 3x3x3 structuring element for 3D erosion
    selem = np.ones((3, 3), dtype=bool)
    
    # Loop through each 2D frame in the 3D array
    for i in range(pixelarray.shape[0]):
        # Erode the current frame
        eroded = binary_erosion(pixelarray[i], structure=selem)
        
        # Find the outline by subtracting the eroded image from the original image
        outline = pixelarray[i] & ~eroded
        
        # Store the result in the corresponding slice of the output array
        outline_array[i] = outline

    return outline_array


# Find the outline of the tibia
tibia_outline = find_outline_3d(tibia_array)
#%%
tib_subsets = find_corres_for_all_frames(tibia_outline)
tib_label_napari = points_for_napari(tib_subsets)
viewer.add_points(tib_label_napari, name='tib_subsets', size = 2, face_color='red')
#%%
transformation_matrices = calculate_transform_matrices_skimage(tib_subsets, 15) 
#%%
transformed_subsets = apply_transformation_all_frames(tib_subsets[-1], transformation_matrices)
#%%
points_in_napari = points_for_napari(transformed_subsets) 
viewer.add_points(points_in_napari, face_color='green', size= 1)

#%%
def procrustes(X, Y):
    X = X.astype(float)  # Ensure X is float
    Y = Y.astype(float)
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X -= centroid_X
    Y -= centroid_Y
    U, _, Vt = np.linalg.svd(np.dot(Y.T, X))
    R = np.dot(U, Vt)
    t = centroid_Y - np.dot(R, centroid_X)
    return R, t

def calculate_transform_matrices_procrustes(all_coords, reference_index):
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
            # Estimate the transformation matrix using Procrustes
            R, t = procrustes(reference_coords, all_coords[i])
            transformation_matrix = np.hstack([R, t.reshape(-1, 1)])  # Combine into a 2x3 matrix
            transformation_matrices.append(transformation_matrix)

    return transformation_matrices
#%%
transformation_matrices_proscrutes = calculate_transform_matrices_procrustes(tib_subsets, 15)
#%%
transformed_subsets_proscrutes = apply_transformation_all_frames(tib_subsets[15], transformation_matrices_proscrutes)
#%%
points_in_napari = points_for_napari(transformed_subsets_proscrutes) 
viewer.add_points(points_in_napari, face_color='green', size= 1)
#%%
''' trying to do this with shapes layer. .. proscrutes seems kinda promising  ''' 
shape_data = viewer.layers['Shapes'].data
new_array_list = [arr[:, 1:] for arr in shape_data]
#%%
transformation_matrices_proscrutes = calculate_transform_matrices_procrustes(new_array_list, 0)
#%%
transformed_subsets_proscrutes = apply_transformation_all_frames(new_array_list[0], transformation_matrices_proscrutes)
#%%
points_in_napari = points_for_napari(transformed_subsets_proscrutes) 
viewer.add_points(points_in_napari, face_color='green', size= 1)
''' for a self chosen coarsely sampled points, it is perfect  '''
#%%
ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
transformed_expanded = apply_transformation_all_frames(ref_points, transformation_matrices_proscrutes) 
transformed_shapes = shapes_for_napari(transformed_expanded)
viewer.add_shapes(transformed_shapes, shape_type='polygon') 


#%%

''' using the all_subsets from way up above, the real data for a labeled segment. The approach below tries to impose a metric to the points obtained  '''

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
#%%
viewer.add_points(resampled_napari) 
#%%
original_napari = points_for_napari(all_subsets) 
viewer.add_points(original_napari, edge_color='red') 
#%%
transformation_matrices_proscrutes = calculate_transform_matrices_procrustes(all_subsets_resampled, 0)
transformed_resampled = apply_transformation_all_frames(all_subsets_resampled[0], transformation_matrices_proscrutes)
#%%
resampled_trans_napari = points_for_napari(transformed_resampled) 
viewer.add_points(resampled_trans_napari, edge_color = 'green' ) 

''' -- -- surprisingly enough, by not using nearest neighbors, instead resampling, it worked really good for a few frames. like 3 out of 26. 
What follows is an attempt to integrate the nrs too  ''' 
#%%
all_subsets = find_corres_for_all_frames(tib_label)
#%%
nrs_resampled = [resample_curve(curve) for curve in all_subsets]
#%%
nrs_resampled_napari = points_for_napari(nrs_resampled) 

viewer.add_points(nrs_resampled_napari, face_color='blue') 
#%%
matrices_list = calculate_transform_matrices_procrustes(nrs_resampled, 2)
post_transformation = apply_transformation_all_frames(nrs_resampled[2], matrices_list)
#%%
viewer.add_points(points_for_napari(post_transformation), face_color='orange') 

''' ---- fantastic results! what follows is applying this transformation to handmade segmentation  ''' 
#%%
viewer = napari.Viewer() 
#%%
viewer.add_image(grad_smooth) 
#%%
viewer.add_shapes(nrs_resampled[2], shape_type='polygon') 
#%%
ref_points = viewer.layers['expanded_shape'].data[0]
transformed_expanded = apply_transformation_all_frames(ref_points, matrices_list) 
transformed_shapes = shapes_for_napari(transformed_expanded)
viewer.add_shapes(transformed_shapes, shape_type='polygon') 
#%%
viewer = napari.view_image(image)
#%%
viewer.add_image(bone_canny)
#%%
viewer.add_image(tib_label)
#%%
viewer.add_points(resampled_napari) 
#%%
viewer.add_shapes(transformed_shapes, shape_type='polygon')
#%%
viewer.add_points(points_for_napari(all_subsets), name = 'label_points')
#%%
all_coordinates = []

# Loop through each frame and grab the coordinates
for frame in tib_label:
    coords = np.argwhere(frame)
    all_coordinates.append(coords)
#%%
viewer.add_points(points_for_napari(all_coordinates), name='all_coords')
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
    new_curve = np.array(new_curve)  # Convert list to NumPy array for easy calculations
    resampled_diff = np.diff(new_curve, axis=0)
    resampled_dists = np.sqrt((resampled_diff ** 2).sum(axis=1))
    print("Distances between resampled points:", resampled_dists)
    return np.array(new_curve)
#%%
# For each frame in all_subsets, apply the function
all_coordinates_resampled = [resample_curve(curve) for curve in all_coordinates]
#%%
viewer.add_points(points_for_napari(all_coordinates_resampled), name='direct resampling', face_color='red', size=5)
#%%
def sort_and_isolate(points):
    # Sort points based on y-coordinate in descending order
    sorted_points = points[points[:, 1].argsort()[::-1]]
    
    # Initialize list to hold the isolated segment
    isolated_points = []
    
    # Start with the point having the highest y-coordinate
    last_y = sorted_points[0, 1]
    isolated_points.append(sorted_points[0])
    
    # Traverse, stop when y starts to increase
    for point in sorted_points[1:]:
        curr_y = point[1]
        if curr_y > last_y:
            break
        isolated_points.append(point)
        last_y = curr_y
    
    return np.array(isolated_points)

all_coordinates = []
isolated_coordinates = []

for frame in tib_label:
    coords = np.argwhere(frame)
    all_coordinates.append(coords)
    isolated_coords = sort_and_isolate(coords) 
    isolated_coordinates.append(isolated_coords)
#%%
viewer.add_points(points_for_napari(isolated_coordinates), name='isolated')
#%%
sorted_curve_resampled= [resample_curve(curve) for curve in isolated_coordinates]
viewer.add_points(points_for_napari(sorted_curve_resampled), name='sorted resampling', face_color='blue', size=5)
#%%
def resample_curve_with_original_points(curve, n_points=25):
    # Calculate the total length of the curve
    diff = np.diff(curve, axis=0)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    total_length = np.sum(dists)
    
    # Calculate the approximate distance between points
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
                # Find the point in the original curve that's closest to the target distance
                closest_point = curve[j] if abs(dists[j] - dist_needed) < abs(dists[j+1] - dist_needed) else curve[j+1]
                new_curve.append(closest_point)
                
                # Update the distance covered
                dist_covered += dist_needed
                dist_needed = 0.0
            else:
                # Move to the next segment
                dist_needed -= segment_remaining
                dist_covered = 0.0
                j += 1
    
    new_curve.append(curve[-1])  # End with the last point
    return np.array(new_curve)

sorted_curve_resampled= [resample_curve_with_original_points(curve) for curve in isolated_coordinates]
#%%
viewer.add_points(points_for_napari(sorted_curve_resampled), name='method_3')
#%%


# Calculate the differences between consecutive points
diffs = np.diff(all_coordinates[8], axis=0)

# Calculate the distances between consecutive points
distances = np.sqrt((diffs ** 2).sum(axis=1))

print("Distances between consecutive points:", distances)
#%%
curve = all_coordinates[8]  # Replace with the actual curve you want to check

plt.scatter(curve[:, 0], curve[:, 1])

# Annotate each point with its index
for i, (x, y) in enumerate(curve):
    plt.annotate(str(i), (x, y))

plt.show()

#%%
skeleton = skeletonize(tib_label)
#%%
all_coordinates = []

# Loop through each frame and grab the coordinates
for frame in skeleton:
    coords = np.argwhere(frame)
    all_coordinates.append(coords)
#%%
viewer.add_points(points_for_napari(all_coordinates), name='skeleton')
viewer.add_image(skeleton, name='skeletonized')

#%%
disp_layer = viewer.layers["transformed_shapes"].to_labels(image.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(100,410)
yrange=slice(150,400)
for ax, idi in zip(axes.flatten(), range(0,12,2)):
    ax.imshow(image[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('test.svg')