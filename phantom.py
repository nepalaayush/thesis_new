#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:10:14 2023

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
import time 
from scipy.interpolate import CubicSpline
#%%
def apply_transformation(matrix, points):
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    return transformed_points[:, :2]  # Convert back to Cartesian coordinates

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

def pairwise_distances(points):
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances


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


#%%
sorted_skeleton = sorted_skeleton
#%%
digital_phantom = np.zeros((13,480,480), dtype=bool)
viewer = napari.Viewer() 
viewer.add_image(digital_phantom)
#%%
shape_0 = viewer.layers['Shapes'].data

shape_0_array = shape_0[0]

shape_0_sliced = shape_0_array[:,1:]
#%%
# so i drew a shape, and now i will transform it using previously obtained, matrices_list. 
post_transformation = apply_transformation_all_frames_consecutive(shape_0_sliced, matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='green', size=2) 
#%%
matrices_list_phantom = calculate_transform_matrices_procrustes_consecutive(post_transformation)
post_transformation_phantom = apply_transformation_all_frames_consecutive(post_transformation[0], matrices_list_phantom)

viewer.add_points(points_for_napari(post_transformation), face_color='yellow', size=1) 


''' self consitency works. Now to test it using manual phantom  - didnt work, couldnt move it. '''
#%%
''' moving on, back to the main work. sampling.  ''' 

viewer.add_points(points_for_napari(sorted_skeleton), face_color='white', size=3) 
#%%
def calculate_relative_distance_and_angle(pointA, pointB, pointC):
    vecAB = np.array(pointB) - np.array(pointA)
    vecBC = np.array(pointC) - np.array(pointB)

    magAB = np.linalg.norm(vecAB)
    magBC = np.linalg.norm(vecBC)
    
    relative_distance = magAB / magBC if magBC != 0 else 0
    
    cos_angle = np.dot(vecAB, vecBC) / (magAB * magBC) if magAB != 0 and magBC != 0 else 0
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return relative_distance, angle

def find_corres_advanced(setA, setB):
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(setB)
    
    # Initialize the first two points using simple nearest neighbors
    distances, indices = nbrs.kneighbors(setA[0:2])
    B_corresponding = [setB[i[0]] for i in indices]
    
    # Iterate from the third point
    for i in range(2, len(setA)):
        distances, indices = nbrs.kneighbors([setA[i]])
        candidate_points = setB[indices[0]]

        # Calculate relative distance and angle for the template points
        relative_distance, angle = calculate_relative_distance_and_angle(setA[i-2], setA[i-1], setA[i])
        
        best_candidate = None
        best_score = float('inf')
        
        for candidate in candidate_points:
            # Calculate relative distance and angle for the candidate points
            candidate_relative_distance, candidate_angle = calculate_relative_distance_and_angle(B_corresponding[-2], B_corresponding[-1], candidate)
            
            # Score the candidate based on how closely these metrics match
            #score = abs(relative_distance - candidate_relative_distance) + abs(angle - candidate_angle)
            score = abs(relative_distance - candidate_relative_distance) # commented out the angle condition, just to check 
            if score < best_score:
                best_score = score
                best_candidate = candidate
                
        B_corresponding.append(best_candidate)
        
    return np.array(B_corresponding)

def find_corres_consecutive(coords_list):
   
    # Set the first frame's coordinates as the template
    template_cords = coords_list[0]
    
    all_subsets = [template_cords]  # Initialize with the template cords so that it remains untouched
    for i in range(1,len(coords_list)):
        test_cords = coords_list[i]
        subset_cords = find_corres_advanced(template_cords, test_cords)
        all_subsets.append(subset_cords)

        template_cords = subset_cords  # Set the found subset as the new template

    return all_subsets

#phantom_bool = coords_to_boolean(sorted_skeleton, (13,480,480))

phantom_found = find_corres_consecutive(sorted_skeleton)
#%%
viewer.add_points(points_for_napari(phantom_found), name='phantom_found', size=2, face_color='green')
#%%
# tp check the integrity 
def check_integrity(list_of_cords): 
    print ( [ np.var ( pairwise_distances(i) )  for i in list_of_cords ] ) 
# after checking, it is seen that my pairwise_distances function is sensitive to ordering. so need to sort the result after find it. doesnt affect how it looks in the napari 

phantom_sorted = sort_points_all_frames(phantom_found)

''' after checking integrity .. it is clearly evident that the integrity decreases as we increase frame number. For one thing, why is the first frame touched at all??  '''

# an interesting poitn is that now... the integrity does not necesarrily increase with frame.. not quite but still there is a trend 
#%%
''' implementing a new search.. based on only distance. takes list of cooridnates, gives back the subsets  ''' 
def find_equidistant_subset(template_frame, target_frame, avg_gap):
    # Initialize Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_frame)
    
    # Start from the last point in the template
    start_point = template_frame[-1]
    
    # Find the closest point in the target frame
    _, initial_indices = nbrs.kneighbors([start_point])
    print("Initial indices:", initial_indices, type(initial_indices))
    current_point = target_frame[initial_indices[0][0]]
    
    # Initialize the list of selected points
    selected_points = [current_point]
    
    # Loop to find equidistant points
    while True:
        # Find all points upstream (lower index)
        upstream_points = [point for i, point in enumerate(target_frame) if i < initial_indices[0][0]]
        
        # If we've reached the start of the list, break
        if not upstream_points:
            break
        
        # Find the point closest to avg_gap away from current_point
        distances = [np.linalg.norm(np.array(current_point) - np.array(point)) for point in upstream_points]
        closest_point = upstream_points[np.argmin(np.abs(np.array(distances) - avg_gap))]
        
        # Update current_point and add to selected points
        current_point = closest_point
        selected_points.append(current_point)
        
        # Update initial_indices to the index of the newly found point
        initial_indices = np.array([np.where((target_frame == current_point).all(axis=1))])
        #print("Initial indices:", initial_indices, type(initial_indices))
        if initial_indices.size == 0:
            print("No matching point found for current_point in target_frame.")
            break

    return np.array( selected_points ) 

avg_gap = np.mean(pairwise_distances(sorted_skeleton[0]))

single_subset = find_equidistant_subset(sorted_skeleton[0], sorted_skeleton[1], avg_gap)

#%%
def find_all_equidistant_subsets(sorted_skeleton_list):
    # Initialize list to store subsets
    all_subsets = []
    
    # Your initial template frame
    template_frame = sorted_skeleton_list[0]
    
    all_subsets.append(template_frame)
    # Calculate the initial avg_gap
    avg_gap = np.mean(pairwise_distances(template_frame))
    
    
    # Loop through all target frames
    for target_frame in sorted_skeleton_list[1:]:
        # Find the subset for the current target frame based on the template frame
        single_subset = find_equidistant_subset(template_frame, target_frame, avg_gap)
    
        # Add this subset to the list of all subsets
        all_subsets.append(single_subset)
        
        # Update the template frame for the next iteration
        template_frame = sort_points_single_frame(single_subset)

    return all_subsets

result_subsets = find_all_equidistant_subsets(sorted_skeleton)

viewer.add_points(points_for_napari(result_subsets), name='result_subsets', size=2, face_color='blue')

''' rray([8.60232527, 8.60232527, 8.48528137, 8.06225775, 9.        ,
       8.94427191, 8.94427191, 8.54400375, 8.06225775, 8.24621125,
       8.24621125, 8.24621125, 8.24621125, 9.05538514, 9.05538514,
       9.21954446, 8.24621125, 8.24621125, 8.54400375, 8.54400375,
       8.94427191, 8.54400375, 8.94427191, 8.54400375, 8.54400375,
       8.54400375, 8.54400375, 8.54400375, 8.94427191, 2.23606798])
    
    
now i can confidently say that i have sampled the curve as best as i can. just look at these numbers. 
sure, the last poitn is a weird one, but extremely simple to solve. checking the transform as is:     '''

#%%
# hang on, regardless, need to equalize the points anyways. 
result_sorted = sort_points_all_frames(result_subsets)
equalized_result = equalize_lengths(result_sorted)
#%%
viewer.add_points(points_for_napari(equalized_result), name='equalized_result', size=2, face_color='orange')
#%%
''' intentional or not, the equalization even made the problematic distal points better: 
variance :  [0.003551925576947973, 0.08641084252353012, 0.12416403854714428, 0.11022105131953783, 0.09034912924395656, 0.07734488271682008, 0.07961852434792746, 0.09979643990879117, 0.08894373701025575, 0.08833153819120776, 0.11959510750159295, 0.07147644870857463, 0.0784626479888453]   
    '''

matrices_list_result = calculate_transform_matrices_procrustes_consecutive(equalized_result)
post_transformation_result = apply_transformation_all_frames_consecutive(equalized_result[0], matrices_list_result)

viewer.add_points(points_for_napari(post_transformation_result), face_color='red', size=1) 
#%%
def find_angles_from_matrices(matrices_list):
    angles = [0]
    # Extract the angles from the transformation matrices
    for matrix in matrices_list[1:]:  # Skip the first identity matrix
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        angles.append(np.degrees(angle))  # Appending the angle in degrees 
        
        
    plt.scatter(range (1, len(matrices_list) + 1 ), angles )
    plt.xlabel('frame number')
    plt.ylabel('angle between first frame')
    plt.grid() 
#%%
def to_global_transformations(consecutive_transformations):
    # Identity matrix in 3x3 homogenous form
    global_transformation = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

    # This will hold the global transformations
    global_transformations = [global_transformation[:2, :].copy()]

    for trans in consecutive_transformations[1:]:  # Skip the first identity matrix
        # Convert (2,3) to (3,3) by appending [0, 0, 1]
        trans_3x3 = np.vstack((trans, [0, 0, 1]))
        # Multiply the last global transformation by the new transformation
        global_transformation = global_transformation.dot(trans_3x3)
        # Append the new transformation to the list, converting back to (2,3)
        global_transformations.append(global_transformation[:2, :])

    return global_transformations

global_transformations = to_global_transformations(matrices_list)

find_angles_from_matrices(global_transformations)
