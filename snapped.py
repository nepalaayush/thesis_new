# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:26:52 2023

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

#%%
tib_label = np.load('C:/Users/Aayush/Documents/thesis_files/tib_label.npy')[:19]
viewer = napari.Viewer() 
viewer.add_image(tib_label)
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

def boolean_to_coords(boolean_array):
    all_coordinates = []
    ''' Input: a 3d boolean array, like a label image 
    Output: the coordinates where array is true, as a list, each list is a frame'''
    # Loop through each frame and grab the coordinates
    for frame in tib_label:
        coords = np.argwhere(frame)
        all_coordinates.append(coords)
    return all_coordinates

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


def sort_points_single_frame(points):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    starting_point = points[np.argmax(points[:, 0])]  # Highest row value
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        

        # Check if we're in the last 5 points and the distance is unusually large
        if len(remaining_points) <= 5:
            avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
            if np.linalg.norm(np.array(next_point) - np.array(current_point)) > 2 * avg_distance:
                remaining_points.remove(next_point)
                continue  # Skip to the next iteration without adding the point

        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)

    return np.array(sorted_points)
    
def sort_all_frames(all_subsets):
    sorted_list = []
    for array in all_subsets:
        sorted_list.append(sort_points_single_frame(array))
    return sorted_list

def pairwise_distances(points):
    """
    Calculate the distances between consecutive pairs of points in the given array.
    
    Args:
    - points (numpy.ndarray): An array of shape (N, 2) representing the points.
    
    Returns:
    - distances (numpy.ndarray): An array of shape (N-1,) representing the distances between consecutive points.
    """
    # Calculate the Euclidean distances between consecutive points
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances

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
tib_label_chords = boolean_to_coords(tib_label)
viewer.add_points(points_for_napari(tib_label_chords), name='label_points', face_color='white', size=1)
#%%
print(viewer.layers['Points'].data)
#%%
s_tib_label_chords = sort_all_frames(tib_label_chords)
#%%
viewer.add_points(points_for_napari(s_tib_label_chords), name='sorted_label_points', face_color='white', size=1)

#%%
def compute_distances(frames):
    distances_per_frame = []
    for frame in frames:
        distances = [np.linalg.norm(frame[i+1] - frame[i]) for i in range(len(frame)-1)]
        total_distance = sum(distances)
        distances_per_frame.append(total_distance)
    return distances_per_frame

distances = compute_distances(s_tib_label_chords)

#%%
def cumulative_distances(points):
    # Calculate pairwise distances between points
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    
    # Calculate cumulative distances
    cumulative_dists = np.insert(np.cumsum(distances), 0, 0)  # Insert a 0 at the beginning to represent the distance from the first point to itself

    return cumulative_dists
#%%
def equidistant_points_min(points, n):
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

new_frame = equidistant_points(s_tib_label_chords[0], 30 )
viewer.add_points(new_frame, size=1, face_color='green')
#%%
def cut_frame_at_distance(frame, desired_distance):
    # Reverse the frame
    reversed_frame = frame[::-1]
    
    # Calculate pairwise distances between points of reversed frame
    distances = np.linalg.norm(reversed_frame[1:] - reversed_frame[:-1], axis=1)
    
    # Calculate cumulative distances
    cumulative_distances = np.cumsum(distances)
    
    # Find where the cumulative distance exceeds the desired distance
    idx = np.searchsorted(cumulative_distances, desired_distance)
    
    # Cut off the frame
    truncated_frame = reversed_frame[:idx+1][::-1]  # Take up to the exceeding index and reverse back to original order
    
    return truncated_frame

truncated_frame = cut_frame_at_distance(new_frame, min(distances)) 
viewer.add_points(truncated_frame, size=1, face_color='orange')
#%%
''' So using the basic distances min somehow failed. even though the sampled curve has min distance, it looks clearly larger than the smallest curve. 
so, now the idea is to actually resample all the frames, using all their points. nbut i say how many points i want.  ''' 
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
def equidistant_points_list(points_list, n):
    return [equidistant_points(points, n) for points in points_list]

equidistant_list = equidistant_points_list(s_tib_label_chords, 60)
viewer.add_points(points_for_napari(equidistant_list), name='sorted_label_points', face_color='orange', size=1)
#%%
image = open_nii('C:/Users/Aayush/Documents/thesis_files/CINE_HighRes.nii')[:19]
viewer.add_image(image)

#%%
'''  kind of out of ideas. but after simple downsampling, lets find the matrix. using all points first, and then using a cropped version.  '''
# using all points: 
matrices_list = calculate_transform_matrices_procrustes(equidistant_list, 0)
post_transformation = apply_transformation_all_frames(equidistant_list[0], matrices_list)
#%%
viewer.add_points(points_for_napari(post_transformation), face_color='orange', size=2) 
''' worst result. ''' 
#%%
def last_n_points_list(points_list, n=15):
    return [points[-n:] for points in points_list]

truncated_list = last_n_points_list(equidistant_list, n=15)
viewer.add_points(points_for_napari(truncated_list), name='truncated_list', face_color='brown', size=1)
#%%
matrices_list = calculate_transform_matrices_procrustes(truncated_list, 2)
post_transformation = apply_transformation_all_frames(truncated_list[2], matrices_list)
#%%
viewer.add_points(points_for_napari(post_transformation), face_color='blue', size=2) 
''' did not change anything  ''' 
#%%
''' trying something totally new now  '''

def find_correspondences(A, B):
    """Find closest points in B for each point in A."""
    correspondences = []
    for point in A:
        distances = np.linalg.norm(B - point, axis=1)
        min_index = np.argmin(distances)
        correspondences.append(B[min_index])
    return np.array(correspondences)

def compute_transformation(A, B):
    """Compute rigid transformation from A to B."""
    # Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center the points
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Compute rotation matrix using SVD
    W = np.dot(A_centered.T, B_centered)
    U, _, VT = np.linalg.svd(W)
    
    # Ensure proper rotation
    V = VT.T
    D = np.identity(2)
    D[1, 1] = np.sign(np.linalg.det(np.dot(U, V.T)))
    
    R = np.dot(np.dot(U, D), VT)
    
    # Compute translation
    t = centroid_B - np.dot(centroid_A, R)
    
    return R, t


def apply_transformation1(A, R, t):
    """Apply transformation to points."""
    return np.dot(A, R.T) + t

def ICP(A, B, max_iterations=100, tolerance=1e-6):
    """Iterative Closest Point."""
    prev_error = 0
    for i in range(max_iterations):
        # Step 1: Find Correspondences
        B_matched = find_correspondences(A, B)
        
        # Step 2: Compute Transformation
        R, t = compute_transformation(A, B_matched)
        
        # Step 3: Apply Transformation
        A = apply_transformation1(A, R, t)
        
        # Check for convergence
        mean_error = np.mean(np.linalg.norm(A - B_matched, axis=1))
        if abs(prev_error - mean_error) < tolerance:
            break
        
        prev_error = mean_error
        
    return R, t, A 

def swap_coordinates(data):
    """Swap x and y coordinates."""
    return np.column_stack([data[:, 1], data[:, 0]])

A = swap_coordinates(equidistant_list[0])
B = swap_coordinates(equidistant_list[1])

R, t, A_transformed = ICP(equidistant_list[0], equidistant_list[1])
#%%
viewer.add_points(A_transformed)
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
    return np.array(new_curve)

# For each frame in all_subsets, apply the function
all_subsets_resampled = [resample_curve(curve,50) for curve in equidistant_list]
viewer.add_points(points_for_napari(all_subsets_resampled), name='resampled_subsets', size=1, face_color='blue') 
#%%
matrices_list = calculate_transform_matrices_procrustes(all_subsets_resampled, 2)
post_transformation = apply_transformation_all_frames(all_subsets_resampled[2], matrices_list)
#%%
viewer.add_points(points_for_napari(post_transformation), face_color='blue', size=2) 
#%%
from sklearn.linear_model import RANSACRegressor
#from scipy.spatial.transform import Rotation as R

def estimate_rigid_transform(A, B):
    """Estimate rigid transformation between A and B."""
    # Estimate translation
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Estimate rotation using SVD
    H = np.dot(A_centered.T, B_centered)
    U, S, Vt = np.linalg.svd(H)
    R_ = np.dot(Vt.T, U.T)

    # Ensure it's a right handed system
    if np.linalg.det(R_) < 0:
       Vt[-1,:] *= -1
       R_ = np.dot(Vt.T, U.T)

    # Compute translation
    t_ = -np.dot(R_, centroid_A.T) + centroid_B.T

    return R_, t_

def find_transformation(set1, set2, residual_threshold=0.01):
    """Find rigid transformation between two point sets using RANSAC."""
    
    # RANSAC to handle potential outliers
    model = RANSACRegressor(residual_threshold=residual_threshold)
    model.fit(set1, set2)
    A_inliers = set1[model.inlier_mask_]
    B_inliers = set2[model.inlier_mask_]

    # Estimate the rigid transformation using inliers
    R_, t_ = estimate_rigid_transform(A_inliers, B_inliers)
    
    return R_, t_
A = equidistant_list[0]
B = equidistant_list[1]
R_, t_ = find_transformation(B, A)

def apply_RANSAC(A, R_, t_):
    """Applies the rigid transformation to a set of points."""
    # Rotate the points
    A_rotated = np.dot(A, R_.T)
    
    # Translate the points
    A_transformed = A_rotated + t_
    
    return A_transformed

# Apply the transformation to A
A_transformed = apply_RANSAC(B, R_, t_)

viewer.add_points(A_transformed, size=2)
#%%
viewer.add_points(points_for_napari(equidistant_list), name='equidistant_list', size=1, face_color='blue') 
#%%
B_resampled = equidistant_points(B, 50)
viewer.add_points(B_resampled, name='B_resampled', face_color='red', size=1)
#%%
#%%
def rigid_transform_2D(A, B):
    """
    Computes the optimal rigid transformation (rotation and translation) 
    that aligns point set A to point set B.
    
    Parameters:
        A (np.array): Source 2D point set with shape (N, 2).
        B (np.array): Target 2D point set.
    
    Returns:
        R (np.array): 2x2 rotation matrix.
        t (np.array): 2x1 translation vector.
    """
    
    # Ensure both point sets have the same number of points
    assert A.shape == B.shape, "Both point sets must have the same number of points."
    
    # Compute centroids of A and B
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Move both sets to origin
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # Compute the cross-covariance matrix
    H = A_centered.T @ B_centered
    
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Ensure a proper rotation (handle reflection case)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation vector
    t = centroid_B.T - R @ centroid_A.T
    
    return R, t

# Example usage:
R, t = rigid_transform_2D(A, A)
transformed_A = (R @ A.T).T + t

viewer.add_points(transformed_A, size=1, face_color='pink')
