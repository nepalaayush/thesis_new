#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:24:36 2024

@author: aayush
"""
import os 
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt 
from scipy import ndimage
from scipy.interpolate import CubicSpline
import scipy.optimize
from skimage.feature import canny
from skimage.morphology import skeletonize, remove_small_objects
from sklearn.decomposition import PCA
from shapely.geometry import LineString, MultiPoint
from matplotlib.path import Path

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

def path_to_image(path):
    image = open_nii(path)
    image = normalize(image)
    image = np.moveaxis(image, 1, 0)
    return image

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


def sort_points_single_frame(points, bone_type='tibia'):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    if bone_type == 'femur':
        starting_point = points[np.argmin(points[:, 0])]
    else:
        starting_point = points[np.argmax(points[:, 0])]  # Highest row value or lowest depending on argmax or arg min 
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        

        # # Check if the distance is much larger than average
        if len(sorted_points) > 1:
             avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
             large_jump = np.linalg.norm(np.array(next_point) - np.array(current_point)) > 2 * avg_distance
             if large_jump:
                 break

        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)
    
    # reverse the order or sorted_points if we are using np.argmax 
    sorted_points.reverse()
    
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


def apply_label(pixelarray):
    structuring_element = ndimage.generate_binary_structure(2, 2)
    labelized = []
    for i in range(pixelarray.shape[0]):
        label_frame, num_features = ndimage.label(pixelarray[i], structure=structuring_element, output=None)
        labelized.append(label_frame)
    return np.stack(labelized), num_features



''' the portion below attempts to find the automatic edge of interest, using a manually selected point '''

def move_and_find_label(frame_labels, start_coord, direction, stop_label=None):
    row, col = start_coord
    label = None

    if direction == 'east':
        # Move east until a non-zero label is found
        for i in range(col, frame_labels.shape[1]):
            if frame_labels[row, i] != 0:
                label = frame_labels[row, i]
                break

    elif direction == 'north':
        # Move north until a non-zero label is found
        for i in range(row, -1, -1):
            if frame_labels[i, col] != 0:
                if stop_label is not None and frame_labels[i, col] == stop_label:
                    # If we encounter the stop_label, return None to indicate we should not continue searching
                    return None
                label = frame_labels[i, col]
                break

    return label


def find_tibia_edges(label_image, start_coord):
    # Create a new array to store the edges of interest
    tibia_edges = np.zeros_like(label_image)

    # Process each frame
    for frame in range(label_image.shape[0]):
        # Adjust the starting coordinates for the current frame, if necessary
        current_coord = start_coord  # Assuming the point doesn't change location

        # Find the label while moving east
        east_label = move_and_find_label(label_image[frame], current_coord, direction='east')

        # Check if we need to move north
        north_label = move_and_find_label(label_image[frame], current_coord, direction='north', stop_label=east_label)

        # Set the detected label(s) in the tibia_edges array
        if east_label is not None:
            tibia_edges[frame][label_image[frame] == east_label] = east_label
            if north_label is not None:
                tibia_edges[frame][label_image[frame] == north_label] = north_label

    return tibia_edges


def find_array_with_min_n(list_of_arrays):
    template_index = np.argmin([arr.shape[0] for arr in list_of_arrays])
    print('template is frame: ', template_index)
    return template_index


def downsample_points(list_of_arrays, index=0, number=50, bone_type='tibia' ):
    zeroth_frame = sort_points_single_frame(list_of_arrays[index], bone_type)
    zeroth_nonadjusted = equidistant_points(zeroth_frame,number)
    zeroth_adjusted = adjust_downsampled_points(zeroth_nonadjusted, zeroth_frame)
    return zeroth_adjusted


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

def apply_transformations(reference_frame, transformation_matrices):
    transformed_frames = [reference_frame]
    for matrix in transformation_matrices[1:]:  # Exclude the identity matrix
        x, y, phi = matrix
        reference_frame = transform(reference_frame, x, y, phi)
        transformed_frames.append(reference_frame)

    return transformed_frames


def combined_consecutive_transform(data):
    # Select reference frame based on the shortest edge
    reference_index = find_array_with_min_n(data)
    num_frames = len(data)
    
    # Initialize lists for transformation matrices, transformed data, and costs
    transformation_matrices = [np.array([0, 0, 0])] * num_frames
    giant_list = [None] * num_frames
    cost_values = [0] * num_frames
    
    
    # Set the reference frame in the giant_list
    giant_list[reference_index] = data[reference_index]
    
    
    # Initialize the reference frame data and initial guess
    reference_data = data[reference_index]
    x0 = np.array([0, 0, 0])  # Initial guess

    # Transform preceding frames (working backwards)
    for ida in range(reference_index - 1, -1, -1):
        fr, cost = match_coords(reference_data, data[ida], x0=x0)
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        transformation_matrices[ida] = fr
        cost_values[ida] = cost
        giant_list[ida] = transformed_data

        # Update the reference data and initial guess for the next iteration
        reference_data = transformed_data
        x0 = fr

    # Reset for forward transformation
    reference_data = data[reference_index]
    x0 = np.array([0, 0, 0])

    # Transform following frames (working forwards)
    for ida in range(reference_index + 1, num_frames):
        fr, cost = match_coords(reference_data, data[ida], x0=x0)
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        transformation_matrices[ida] = fr
        cost_values[ida] = cost
        giant_list[ida] = transformed_data

        # Update the reference data and initial guess for the next iteration
        reference_data = transformed_data
        x0 = fr

    return transformation_matrices, giant_list, cost_values


def apply_transformations_new(reference_frame, transformation_matrices, reference_index):
    num_frames = len(transformation_matrices)
    transformed_frames = [None] * num_frames
    
    if reference_index < 0:
        reference_index =  num_frames + reference_index
    
    if reference_index >= num_frames or reference_index < 0:
       raise ValueError("Reference index is out of range")
    # Apply transformation for the reference frame
    transformed_frames[reference_index] = reference_frame
    
    
    # Apply transformations backwards
    current_frame = reference_frame
    for i in range(reference_index - 1, -1, -1):
        matrix = transformation_matrices[i]
        x, y, phi = matrix
        current_frame = transform(current_frame, x, y, phi)
        transformed_frames[i] = current_frame

    # Reset current_frame for forward transformations
    current_frame = reference_frame

    # Apply transformations forwards
    for i in range(reference_index + 1, num_frames):
        matrix = transformation_matrices[i]
        x, y, phi = matrix
        current_frame = transform(current_frame, x, y, phi)
        transformed_frames[i] = current_frame

    return transformed_frames



def dict_to_array(dictionary):
    points_list = [] 
    
    for key, value in dictionary[0].items():
        if (value.shape == (2,2) ):
            points_list.extend([value[0], value[1]])
        else:
            points_list.append(value)
    
    points_array = np.array(points_list)
    
    return points_array 


def reconstruct_dict(frame_number, array):
    if array.shape != (8, 2):
        raise ValueError("Array must be of shape (8, 2)")

    # Assuming the order of points in each array matches the original dictionary
    nested_dict =  {
        'points_long_axis': np.array([array[0], array[1]]),
        'U': array[2],
        'V': array[3],
        'centroid': array[4],
        'points_short_axis': np.array([array[5], array[6]]),
        'origin': array[7]
    }
    return {frame_number: nested_dict}
    
        
def sample_points_in_polygon(polygon, n_samples=1000):
    """
    Generates uniformly distributed points within the given polygon, including the frame number.

    Parameters:
    polygon (array-like): An Nx3 array where each row represents [frame, x, y].
    n_samples (int): Number of points to sample within the polygon.

    Returns:
    np.ndarray: An array of points within the polygon, including the frame number.
    """

    frame_number = polygon[0, 0]  # Extract the frame number from the first point
    x_coords = polygon[:, 1]
    y_coords = polygon[:, 2]

    min_x, min_y = np.min(x_coords), np.min(y_coords)
    max_x, max_y = np.max(x_coords), np.max(y_coords)

    random_points = np.random.uniform([min_x, min_y], [max_x, max_y], (n_samples, 2))

    path = Path(polygon[:, 1:])
    inside_points = random_points[path.contains_points(random_points)]

    # Add the frame number to the generated points
    inside_points_with_frame = np.hstack([np.full((len(inside_points), 1), frame_number), inside_points])

    return inside_points_with_frame
def get_uv_from_pca(line_points):
    A = np.array(line_points[0])
    B = np.array(line_points[1])
    AB = B - A
    mag_AB = np.linalg.norm(AB)
    U = AB / mag_AB
    V = np.array([-U[1], U[0]])
    return U, V


def fit_pca_line(coordinates, n_points=2):
    pca = PCA(n_components=1)
    pca.fit(coordinates)
    
    mean = pca.mean_
    component = pca.components_[0]
    
    projections = (coordinates - mean) @ component
    min_t, max_t = np.min(projections), np.max(projections)
    t = np.linspace(min_t, max_t, n_points)
    line_points = mean + component * t[:, np.newaxis]
      
    return line_points

def find_edges_nnew(U1, U2, V, shape_coords, num_points):
    # Parameterize long axis by the points U1 and U2
    t_values = np.linspace(0, 1, num_points)
    long_axis_points = np.array([(1-t)*U1 + t*U2 for t in t_values])

    widest_distance = 0
    widest_points = None

    for point in long_axis_points:
        # Create line segment perpendicular to the long axis
        perp_line = LineString([point - 50 * V, point + 50 * V]) # using 150 instead of 50 
        #print(perp_line)
        # Convert shape coordinates to LineString
        #shape_line = LineString(shape_coords[:, 1:])
        shape_line = LineString(shape_coords)
        #print(shape_line)
        # Find intersection
        intersection = perp_line.intersection(shape_line)
        #print(intersection)
        '''
        if isinstance(intersection, MultiPoint):
            points_list = intersection.geoms
            if len(points_list) >= 2:
                dist = points_list[0].distance(points_list[-1])
                if dist > widest_distance:
                    widest_distance = dist
                    widest_points = [list(points_list[0].coords)[0], list(points_list[-1].coords)[0]]
                    '''
        
        if isinstance(intersection, MultiPoint) and len(intersection.geoms) >= 2:
            # Check every pair of points to find the maximum distance
            for i in range(len(intersection.geoms)):
                for j in range(i+1, len(intersection.geoms)):
                    dist = intersection.geoms[i].distance(intersection.geoms[j])
                    if dist > widest_distance:
                        widest_distance = dist
                        widest_points = [list(intersection.geoms[i].coords)[0],
                                         list(intersection.geoms[j].coords)[0]]
        
    return   np.array([ [221 , 178], [228,232] ]) # np.array(widest_points) if widest_points is not None else None # np.array([ [221 , 178], [228,232] ])

def find_intersection(A, B, E, F):
    m1 = (B[1] - A[1]) / (B[0] - A[0])
    b1 = A[1] - m1 * A[0]
    m2 = (F[1] - E[1]) / (F[0] - E[0])
    b2 = E[1] - m2 * E[0]
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1
    return np.array([x_intersection, y_intersection])


def process_frame(shapes_data):
    # Process for all frames
    #frame_data = viewer.layers[1].data
    sorted_data = sorted(shapes_data, key=lambda x: x[0][0]) # was viewer.layers[1].data
    results = {}
    
    for idx, shape_coords in enumerate(sorted_data):
        print("shape coords shape: ", shape_coords.shape)
        uniform_points = sample_points_in_polygon(shape_coords)
        print("Uniform points shape:", uniform_points.shape)
        
        # Calculate PCA line points
        line_points = fit_pca_line(shape_coords[:, 1:])
        #line_points = fit_pca_line(shape_coords) # when using binary mask 
        #line_points = fit_pca_line(uniform_points[:, 1:]) # when not using binary mask 
        print(line_points, 'the line_points shape is', line_points.shape)
         
        # Get unit vectors
        U, V = get_uv_from_pca(line_points)
        
        
        # Debug Check 1: Check if U and V are perpendicular
        is_perpendicular_uv = np.abs(np.dot(U, V)) < 1e-5
        if not is_perpendicular_uv:
           print(f"Debug Check 1: For shape {idx}, U and V are not perpendicular.")
        # Compute centroid
        centroid = np.mean(shape_coords[:, 1:], axis=0)
        #centroid = np.mean(uniform_points[:, 1:], axis=0)
        # Find extreme points
        
        extreme_points = np.array(find_edges_nnew(line_points[0], line_points[1], V, shape_coords, num_points=200))
        print('These are the extreme_points: ', {extreme_points})
        # Debug check 2: check if the extreme points line is indeed perpendicualr to U  
        extreme_vector = extreme_points[1] - extreme_points[0]
        is_perpendicular_extreme = np.abs(np.dot(extreme_vector, U)) < 1e-5
        
        if not is_perpendicular_extreme:
            print(f'Debug Check 2: For shape {idx}, extreme poitns line is not perp to U ')
        #viewer.add_points(extreme_points) 
        # Compute intersection
        intersection = find_intersection(line_points[0], line_points[1], extreme_points[0], extreme_points[1])
        
        results[idx] = {
            "points_long_axis": line_points,
            "U": U,
            "V": V,
            "centroid": centroid,
            "points_short_axis": extreme_points,
            "origin": intersection
        }
    return results


def process_single_frame(binary_coords):
    # Calculate PCA line points
    line_points = fit_pca_line(binary_coords) # when using binary mask 
    #line_points = fit_pca_line(uniform_points[:, 1:]) # when not using binary mask 
    print(line_points, 'the line_points shape is', line_points.shape)
     
    # Get unit vectors
    U, V = get_uv_from_pca(line_points)
    
    
    # Debug Check 1: Check if U and V are perpendicular
    is_perpendicular_uv = np.abs(np.dot(U, V)) < 1e-5
    if not is_perpendicular_uv:
       print("Debug Check 1: For shape, U and V are not perpendicular.")
    # Compute centroid
    centroid = np.mean(binary_coords, axis=0)
    # Find extreme points
    
    extreme_points = np.array(find_edges_nnew(line_points[0], line_points[1], V, binary_coords, num_points=50))
    print(extreme_points)
    print(f'extreme_points shape is: ', {extreme_points.shape})
    # Debug check 2: check if the extreme points line is indeed perpendicualr to U  
    extreme_vector = extreme_points[1] - extreme_points[0]
    is_perpendicular_extreme = np.abs(np.dot(extreme_vector, U)) < 1e-5
    
    if not is_perpendicular_extreme:
        print('Debug Check 2: For shape, extreme poitns line is not perp to U ')
    #viewer.add_points(extreme_points) 
    # Compute intersection
    intersection = find_intersection(line_points[0], line_points[1], extreme_points[0], extreme_points[1])
    
    results = {
        "points_long_axis": line_points,
        "U": U,
        "V": V,
        "centroid": centroid,
        "points_short_axis": extreme_points,
        "origin": intersection
    }
    
    return results

def show_origin(all_frame_data, viewer):
    point_data = []
    
    for frame_index, frame_data in all_frame_data.items():
        x, y = frame_data['origin'] 
        cross = [frame_index, x , y] 
        print(cross) 
        point_data.append(cross)
    viewer.add_points(point_data, symbol='x')
 

def show_centroid(all_frame_data, viewer):
    point_data = []
    
    for frame_index, frame_data in all_frame_data.items():
        x, y = frame_data['centroid'] 
        cross = [frame_index, x , y] 
        print(cross) 
        point_data.append(cross)
    viewer.add_points(point_data, symbol='+')

def show_axis(all_frame_data, axis_name, viewer):
    lines_data = []
    
    for frame_index, frame_data in all_frame_data.items():
        point_A, point_B = frame_data[axis_name]
        
        x_A, y_A = point_A
        x_B, y_B = point_B
        
        # Constructing the line (path) for the current frame
        line = [[frame_index, x_A, y_A],  
                [frame_index, x_B, y_B]]
        lines_data.append(line)
    
    viewer.add_shapes(lines_data, shape_type='path', edge_width=2, edge_color='blue', name=f'{axis_name} line')

def show_stuff(frame_data, frame_name, viewer):
    home_directory = os.path.expanduser('~')
    show_axis(frame_data,'points_short_axis', viewer)         
    show_origin(frame_data, viewer)
    show_centroid(frame_data, viewer)
    show_axis(frame_data,'points_long_axis', viewer) 
    save_path = os.path.join(home_directory, 'Pictures', frame_name)
    np.save(save_path,frame_data)