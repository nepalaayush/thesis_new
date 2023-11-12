# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:50:27 2023

@author: Aayush
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
import time 
from scipy.interpolate import CubicSpline
import scipy.optimize
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
    gradient_direction = np.arctan2(dy, dx)

    return gradient_magnitude, gradient_direction

def apply_canny(pixelarray, low, high, sigma):
    canny_edge = np.zeros_like(pixelarray)
    
    for i in range(pixelarray.shape[0]):
        canny_image = canny(pixelarray[i], low_threshold= low, high_threshold=high, sigma= sigma )    
        canny_edge[i] = canny_image
    return canny_edge.astype(dtype=bool)

def apply_remove(pixelarray, size, connectivity):
    removed_3d = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        removed_image = remove_small_objects(pixelarray[i], min_size=size, connectivity=connectivity)
        removed_3d[i] = removed_image
    return removed_3d

def apply_skeleton(pixelarray):
    skeletonized = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        skel_frame = skeletonize(pixelarray[i])
        skeletonized[i] = skel_frame
    return skeletonized


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

def apply_transformation_all_frames_consecutive(initial_set, transformation_matrices):
    transformed_subsets = []
    
    # Initialize with the initial set of points
    current_points = initial_set
    transformed_subsets.append(current_points)
    
    # Apply each transformation matrix in sequence
    for matrix in transformation_matrices[1:]:  # Skip the first identity matrix
        transformed_points = apply_transformation(matrix, current_points)
        transformed_subsets.append(transformed_points)
        
        # Update current_points for the next iteration
        current_points = transformed_points
        
    return transformed_subsets

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

def calculate_transform_matrices_procrustes_consecutive(all_coords):
    num_frames = len(all_coords)
    transformation_matrices = []
    
    # Initialize the identity matrix for the first frame
    identity_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]])
    
    # Add the identity matrix for the first frame
    transformation_matrices.append(identity_matrix)

    for i in range(1, num_frames):
        # Estimate the transformation matrix using Procrustes
        # between the i-th frame and the (i-1)-th frame
        R, t = procrustes(all_coords[i-1], all_coords[i])
        transformation_matrix = np.hstack([R, t.reshape(-1, 1)])  # Combine into a 2x3 matrix
        transformation_matrices.append(transformation_matrix)
        
    return transformation_matrices

def points_for_napari(list_points):
    all_points = []

    for i, subset in enumerate(list_points):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_points.append(frame_subset)

    all_points = np.vstack(all_points)
    return all_points 

def show_order(curve):
    plt.scatter(curve[:, 0], curve[:, 1])

    # Annotate each point with its index
    for i, (x, y) in enumerate(curve):
        plt.annotate(str(i), (x, y))

    plt.show()

def boolean_to_coords(boolean_array):
    all_coordinates = []
    ''' Input: a 3d boolean array, like a label image 
    Output: the coordinates where array is true, as a list, each list is a frame'''
    # Loop through each frame and grab the coordinates
    for frame in boolean_array:
        coords = np.argwhere(frame)
        all_coordinates.append(coords)
    return all_coordinates
    
def coords_to_boolean(sorted_coordinates, shape):
    # Initialize the 3D array
    new_array = np.zeros(shape, dtype=bool)
    
    # Go through each frame
    for frame_index, frame_coords in enumerate(sorted_coordinates):
        for y, x in frame_coords:
            try:
                new_array[frame_index, int(y), int(x)] = True
            except IndexError as e:
                print(f"IndexError at frame {frame_index} with coordinates ({x}, {y}). Shape is {shape}")
                raise e
    
    return new_array

def check_integrity(list_of_cords): 
    print ( [ np.var ( pairwise_distances(i) )  for i in list_of_cords ] ) 

def pairwise_distances(points):
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances


def sort_points_single_frame(points):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    starting_point = points[np.argmin(points[:, 0])]  # Highest row value
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        

        # # Check if the distance is much larger than average
        # if len(sorted_points) > 1:
        #     avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
        #     large_jump = np.linalg.norm(np.array(next_point) - np.array(current_point)) > 2 * avg_distance
        #     if large_jump:
        #         print(f"Warning: Large distance jump detected in frame {frame_number} between point {current_point} and point {next_point}")  # Modified print statement
        #         break

        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)

    return np.array(sorted_points)

def sort_points_all_frames(list_of_points):
    sorted_list = []
    for i in list_of_points:
        sorted_list.append(sort_points_single_frame(i))
    return sorted_list 

def equalize_lengths(points_list):
    # Find the length of the smallest (n,2) array
    min_length = min([len(points) for points in points_list])

    # Trim each array in the list to have the same length as the smallest array
    equalized_list = [points[-min_length:] for points in points_list]

    return equalized_list


def equidistant_points(points, n):
    # Calculate pairwise distances between points
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    
    # Calculate cumulative distances
    cumulative_distances = np.cumsum(distances)
    total_distance = cumulative_distances[-1]
    
    # Calculate desired spacing
    desired_spacing = total_distance / (n - 1)
    
    # Select equidistant points
    new_points = [points[0]]  # Start with the first point
    current_dist = 0

    for i in range(1, n - 1):  # We already have the starting point, and we'll manually add the endpoint
        current_dist += desired_spacing
        # Find the two original points which the current_dist is between
        idx = np.searchsorted(cumulative_distances, current_dist)
        weight = (current_dist - cumulative_distances[idx - 1]) / (cumulative_distances[idx] - cumulative_distances[idx - 1])
        # Linearly interpolate between these two points
        point = points[idx - 1] + weight * (points[idx] - points[idx - 1])
        new_points.append(point)
    
    new_points.append(points[-1])  # End with the last point
    return np.array(new_points)

def adjust_downsampled_points(downsampled, original_curve):
    """
    Adjust the positions of downsampled points to make them equidistant 
    while ensuring they remain on the original curve.

    Parameters:
    - downsampled: np.array of shape (30,2) representing downsampled points.
    - original_curve: np.array of shape (100,2) representing the original curve.

    Returns:
    - np.array of shape (30,2) representing the adjusted downsampled points.
    """

    # Compute the desired equidistant length
    pairwise_distances = [np.linalg.norm(downsampled[i] - downsampled[i - 1]) for i in range(1, len(downsampled))]
    desired_distance = sum(pairwise_distances) / len(pairwise_distances)

    # Cubic spline interpolation of the original curve
    t = np.linspace(0, 1, len(original_curve))
    cs_x = CubicSpline(t, original_curve[:, 0])
    cs_y = CubicSpline(t, original_curve[:, 1])

    # Adjust the downsampled points
    adjusted_points = [downsampled[0]]  # Start with the first point as anchor
    t_last = 0  # To keep track of the last position on t to avoid backtracking
    for i in range(1, len(downsampled)):
        # Search along the curve for the next position using a fine resolution
        search_t = np.linspace(t_last, 1, 1000)
        for ti in search_t:
            potential_point = np.array([cs_x(ti), cs_y(ti)])
            if np.linalg.norm(potential_point - adjusted_points[-1]) >= desired_distance:
                adjusted_points.append(potential_point)
                t_last = ti
                break

    return np.array(adjusted_points)
#%%
# Step 1: load the image from directory and normalize it
image = open_nii('C:/Users/Aayush/Documents/thesis_files/more_data/aw2_rieseling_admm_tgv_5e-3.nii')
image = normalize(image)
image = np.moveaxis(image, 1, 0)[1:]
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='aw2')
#%%
# using previously obtained label 
tib_label = tib_label
viewer.add_image(tib_label)
#%%
tib_coords = boolean_to_coords(tib_label) 
#%%
zeroth_frame = sort_points_single_frame(tib_coords[0])
zeroth_nonadjusted = equidistant_points(zeroth_frame,30)
zeroth_adjusted = adjust_downsampled_points(zeroth_nonadjusted, zeroth_frame)
viewer.add_points(zeroth_adjusted, face_color='blue', size =1)
#%%
new_tib_coords = tib_coords 
new_tib_coords[0] = zeroth_adjusted
#%%
def coords_distance_sum(coords1, coords2):
    dist = []
    for p in coords1:
        dist.append(np.min(np.sqrt(np.power(coords2[:,0] - p[0],2) + np.power(coords2[:,1] - p[1],2))))
    return np.sum(np.array(dist))

def transform(coords, x, y, phi):
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    shift_vec = np.array([x, y])
    new_coords = []
    for p in coords:
        new_coords.append(np.matmul(p, rot_mat) + shift_vec)
    return np.array(new_coords)
      
def match_coords(coords1, coords2, x0=[0, 0, 0]): # was using -np.deg2rad(2) as guess before 
    cost_fcn = lambda x: coords_distance_sum(transform(coords1, x[0], x[1], x[2]), coords2)
    fr = scipy.optimize.fmin(func=cost_fcn, x0=x0, retall=False, disp=False, ftol=1e-8, maxiter=1e3, maxfun=1e3, xtol=1e-8)
    min_cost = cost_fcn(fr)
    return fr, min_cost
#%%
def plot_frame(coords, ax, **kwargs):
    ax.plot(coords[:,1], coords[:,0], '.', **kwargs)
#data = boolean_to_coords(tib_label)
data = new_tib_coords
id_ref = 0
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(14,14))
fr = [0, 0, -np.deg2rad(2)]
giant_list = [] 

for ida, ax in enumerate(axes.flatten()):
    plot_frame(data[id_ref], ax, label=f"{id_ref}")
    plot_frame(data[ida], ax, label=f"{ida}")

    fr = match_coords(data[id_ref], data[ida], x0 = fr)
    #print(fr)
    plot_frame(transform(data[id_ref], fr[0], fr[1], fr[2]), ax, label=f"{id_ref} to {ida}")
    giant_list.append( transform(data[id_ref], fr[0], fr[1], fr[2]) ) 
    ax.axis('equal')
    ax.invert_yaxis()
    ax.legend()
#%%
viewer.add_points(points_for_napari(giant_list), size=2, face_color='orange')
#%%

#data = sort_points_all_frames(new_tib_coords)
data = new_tib_coords
# a non plot version of this 
# Initialization of lists to store results
transformation_matrices = []
giant_list = []
id_ref= 0 
# Main loop to calculate transformation matrices
for ida in range(len(data)):
    fr = match_coords(data[id_ref], data[ida], x0=[0, 0, 0])
    transformation_matrices.append(fr)  # Store the transformation matrix
    transformed_data = transform(data[id_ref], fr[0], fr[1], fr[2])
    giant_list.append(transformed_data)  # Store the transformed coordinates    
    
viewer.add_points(points_for_napari(giant_list), size=2, face_color='red')

#%%
def consecutive_transform(data):
    # Initialize the list to store transformation matrices and transformed data
    transformation_matrices = [np.array([0, 0, 0])]  # Identity transformation for the first frame
    giant_list = [data[0]]  # The first frame is the reference
    cost_values = [0]
    # Starting with the first frame as the reference frame
    reference_data = data[0]

    # Loop over each frame in the dataset starting from the second frame
    for ida in range(1, len(data)):
        # Calculate the transformation from the current reference to the next frame
        fr, cost = match_coords(reference_data, data[ida], x0=[0, 0, 0])
        transformation_matrices.append(fr)
        cost_values.append(cost)
        # Apply the transformation to the current reference frame
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        giant_list.append(transformed_data)

        # Update the reference frame to the transformed data for the next iteration
        reference_data = transformed_data

    return transformation_matrices, giant_list, cost_values

transformation_matrices, giant_list, cost_values_no_update = consecutive_transform(new_tib_coords)
viewer.add_points(points_for_napari(giant_list), size=2, face_color='yellow', name='consecutive_transform')

#%%
def consecutive_transform_with_guess_update(data):
    # Initialize the list to store transformation matrices and transformed data
    transformation_matrices = [np.array([0, 0, 0])]  # Identity transformation for the first frame
    giant_list = [data[0]]  # The first frame is the reference
    cost_values = [0]
    # Starting with the first frame as the reference frame
    reference_data = data[0]

    # Set the initial guess to zero transformation for the first frame
    x0 = np.array([0, 0, 0])

    # Loop over each frame in the dataset starting from the second frame
    for ida in range(1, len(data)):
        # Calculate the transformation from the current reference to the next frame
        fr, cost = match_coords(reference_data, data[ida], x0=x0)
        transformation_matrices.append(fr)
        cost_values.append(cost)
        # Apply the transformation to the current reference frame
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        giant_list.append(transformed_data)

        # Update the reference data and the initial guess for the next iteration
        reference_data = transformed_data
        x0 = fr  # Update the initial guess to the last found transformation parameters

    return transformation_matrices, giant_list, cost_values

transformation_matrices, giant_list, cost_values = consecutive_transform_with_guess_update(new_tib_coords)
viewer.add_points(points_for_napari(giant_list), size=2, face_color='green', name='consecutive_transform_with_guess_update')

#%%
''' to see the behaviour of cost function  '''
dx_range = np.linspace(-30, 25, 100)  # Example range for dx
dy_range = np.linspace(-20,25, 100)  # Example range for dy
dphi_constant = -0.02  # Constant value for dphi taken from the transform_matrices list, index 19 

# Create a meshgrid for dx and dy
DX, DY = np.meshgrid(dx_range, dy_range)

# Initialize an array to hold the cost function values
cost_values = np.zeros_like(DX)

# Get the coordinates for frame 20
coords1 = data[19]  # Assuming data is zero-indexed
coords2 = data[20]

# Compute the cost function values over the grid
for i in range(DX.shape[0]):
    for j in range(DX.shape[1]):
        # Apply the transformation with the current dx, dy and the constant dphi
        transformed_coords = transform(coords1, DX[i, j], DY[i, j], dphi_constant)
        
        # Compute the cost function value
        cost_values[i, j] = coords_distance_sum(transformed_coords, coords2)

# Plot the cost function values using a colormap
plt.figure(figsize=(8, 6))
plt.contourf(DX, DY, cost_values, levels=50, cmap='viridis')
plt.colorbar(label='Cost function value')
plt.xlabel('dx')
plt.ylabel('dy')
plt.title('Cost Function Landscape for Frame 20 (dphi constant)')
plt.show()
#%%
''' doing a 3d version below   ''' 

# Define ranges for dx, dy, and dphi
dx_range = np.linspace(-30, 25, 50)  # Smaller sample for computational feasibility
dy_range = np.linspace(-20, 25, 50)
dphi_range = np.linspace(-0.1, 0.1, 50)  # Example range for dphi

# Create a 3D meshgrid for dx, dy, and dphi
DX, DY, DPHI = np.meshgrid(dx_range, dy_range, dphi_range, indexing='ij')

# Initialize a 3D array to hold the cost function values
cost_values_3d = np.zeros_like(DX)

# Get the coordinates for frame 20
coords1 = data[19]  # Assuming data is zero-indexed
coords2 = data[20]

# Compute the cost function values over the grid
for i in range(DX.shape[0]):
    for j in range(DX.shape[1]):
        for k in range(DX.shape[2]):
            # Apply the transformation with the current dx, dy, and dphi
            transformed_coords = transform(coords1, DX[i, j, k], DY[i, j, k], DPHI[i, j, k])
            
            # Compute the cost function value
            cost_values_3d[i, j, k] = coords_distance_sum(transformed_coords, coords2) 
            
            
viewer.add_image(cost_values_3d, )