# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 08:52:31 2023

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
#%%
# this file is just to organise. We start from tib_label. 
# the idea is to resample only 1 frame. and then find corresponding for this resampled one. 
# basically inject a pseudo label frame in the actual tib_label. 
#%%
viewer = napari.Viewer() 

viewer.add_image(tib_label, name='label_image')
#%%
base_frame = tib_label[1]

base_points = np.argwhere(base_frame)
#%%
def show_order(curve):
    plt.scatter(curve[:, 0], curve[:, 1])

    # Annotate each point with its index
    for i, (x, y) in enumerate(curve):
        plt.annotate(str(i), (x, y))

    plt.show()
#%%    
show_order(base_points)
#%%
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
#%%
base_sorted = sort_points_single_frame(base_points)
#%%
show_order(base_sorted [:-1])
base_sorted = base_sorted[:-1] # removing the last point that had an issue. 

#%%
def resample_curve3(curve, n_points=25):
    diff = np.diff(curve, axis=0)
    dists = np.sqrt((diff ** 2).sum(axis=1))
    cumulative_dists = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumulative_dists[-1]
    step = total_length / (n_points - 1)
    new_curve = [curve[0]]

    for i in range(1, n_points - 1):
        target_dist = i * step

        # Find the segment where the target distance lies
        idx = np.searchsorted(cumulative_dists, target_dist, side='right') - 1

        # Find the fraction of the way the target is between the two points of the segment
        segment_start_dist = cumulative_dists[idx]
        segment_end_dist = cumulative_dists[idx + 1]
        segment_fraction = (target_dist - segment_start_dist) / (segment_end_dist - segment_start_dist)
        
        next_point = curve[idx] + segment_fraction * (curve[idx+1] - curve[idx])
        new_curve.append(next_point)

    new_curve.append(curve[-1])
    return np.array(new_curve) 

#%%
resampled_base_sorted = resample_curve3(base_sorted, n_points=100)
#%%
viewer.add_points(resampled_base_sorted, name='resampled_base', face_color='red', size = 2)

#%%
base_boolean = np.zeros_like(tib_label[0], shape = tib_label[0].shape, dtype=bool)
#%%
for x,y in resampled_base_sorted:
    base_boolean[int(x), int(y)] = True
#%%
tib_label[1] = base_boolean 

viewer.add_image(tib_label, name='base_frame_resampled')
''' okay, so that is a success at least. albeit a bit of hand tweaking was required to remove the last point  '''
#%%
# Now we find corres using this modified stuff. 
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
all_subsets = find_corres_for_all_frames(tib_label)
#%%
viewer.add_points(points_for_napari(all_subsets), name= 'using_resampled_base' ,  face_color='orange', size=2) 
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
matrices_list = calculate_transform_matrices_procrustes(all_subsets, 1)
post_transformation = apply_transformation_all_frames(all_subsets[1], matrices_list)
viewer.add_points(points_for_napari(post_transformation), face_color='blue', size=2) 

#%%
''' here is where we are at the moment, the matrix tries its best to match, but if the assumptions themselves are not true, then it wont work. 
so, trying it one more time, but this time taking frame 9 as the base  ''' 

base_frame9 = tib_label[9]

base_points9 = np.argwhere(base_frame9)
#show_order(base_points9)
#%%
base_sorted9 = sort_points_single_frame(base_points9)
base_sorted9 = base_sorted9[:-1]
#show_order(base_sorted9)
#%%
resampled_base_sorted9 = resample_curve3(base_sorted9, n_points=100)
viewer.add_points(resampled_base_sorted9, name='resampled_base9', face_color='red', size = 2)
#%%
base_boolean = np.zeros_like(tib_label[0], shape = tib_label[0].shape, dtype=bool)
for x,y in resampled_base_sorted9:
    base_boolean[int(x), int(y)] = True
    
#%%

tib_label[9] = base_boolean 

viewer.add_image(tib_label, name='base_frame_resampled9')    
#%%
def points_for_napari(list_points):
    all_points = []

    for i, subset in enumerate(list_points):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_points.append(frame_subset)

    all_points = np.vstack(all_points)
    return all_points   
#%%
all_subsets9 = find_corres_for_all_frames(tib_label)
viewer.add_points(points_for_napari(all_subsets9), name= 'using_resampled_base9' ,  face_color='orange', size=2) 
#%%
matrices_list9 = calculate_transform_matrices_procrustes(all_subsets9, 9)
post_transformation9 = apply_transformation_all_frames(all_subsets9[9], matrices_list9)
viewer.add_points(points_for_napari(post_transformation9), face_color='blue', size=2)

''' did not help whatsoever. must resample everything before applying transformation. that was the key after all.  '''
#%%
# now resampling the whole thing 
#%%
#need to sort before resampling. 
def sort_curve_points_manually(points, frame_number):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
    starting_point = points[np.argmax(points[:, 0])]  # Highest row value
    sorted_points = [starting_point]
    remaining_points = [p for p in points.tolist() if not np.array_equal(p, starting_point)]

    while remaining_points:
        current_point = sorted_points[-1]
        distances = [np.linalg.norm(np.array(current_point) - np.array(p)) for p in remaining_points]
        next_point = remaining_points[np.argmin(distances)]
        
        # Check if the distance is much larger than average
        if len(sorted_points) > 1:
            avg_distance = np.mean([np.linalg.norm(np.array(sorted_points[i+1]) - np.array(sorted_points[i])) for i in range(len(sorted_points)-1)])
            large_jump = np.linalg.norm(np.array(next_point) - np.array(current_point)) > 5 * avg_distance
            if large_jump:
               print(f"Warning: Large distance jump detected in frame {frame_number} between point {current_point} and point {next_point}")  # Modified print statement
        
        
        sorted_points.append(next_point)
        remaining_points.remove(next_point)

    return np.array(sorted_points)

def sort_curve_for_all_frames2(frames):
    sorted_frames = []
    for frame_number, frame in enumerate(frames):  # Enumerate to get the frame number
        sorted_frame = sort_curve_points_manually(frame, frame_number)  # Pass frame number
        sorted_frames.append(sorted_frame)
    return sorted_frames

sorted9 = sort_curve_for_all_frames2(all_subsets9)
viewer.add_points(points_for_napari(sorted9), name='sorted_9', face_color='green')
#%%
show_order(sorted9[1])
#%%
all_subsets9_resampled = [resample_curve3(curve) for curve in sorted9]
viewer.add_points(points_for_napari(all_subsets9_resampled), name='resampled_napari9', size=2, face_color='brown') 

''' after this, the points look fairly uniformly distributed. Now it is time to test the efficacy. At the very least, matrices should behave as it did before. '''

#%%
matrices_list9 = calculate_transform_matrices_procrustes(sorted9, 9)
post_transformation9 = apply_transformation_all_frames(sorted9[9], matrices_list9)
viewer.add_points(points_for_napari(post_transformation9), face_color='orange', size=2)
#%%
''' the above did not work because during sorting, i removed points that were 5 times larger than the running average (i think). so, this function equalizes it. For this case, im getting lowest value of 95 and highest 100. by the way, the highest can never be greater than 100, due to the logic applied above.  '''

def equalize_array_lengths(arrays):
    """
    Equalize the lengths of (N, 2) arrays in a list to the length of the smallest array.

    Parameters:
    arrays (list of np.ndarray): List of arrays with shape (N, 2) where N can vary.

    Returns:
    list of np.ndarray: List of arrays with equalized shape (k, 2) where k is the smallest N.
    """

    # Find the smallest N
    min_n = min(arr.shape[0] for arr in arrays)

    # Initialize list to store equalized arrays
    equalized_arrays = []

    # Process each array
    for arr in arrays:
        total_remove = arr.shape[0] - min_n  # Total number of rows to remove
        remove_start = total_remove // 2  # Number of rows to remove from the start
        remove_end = total_remove - remove_start  # Number of rows to remove from the end

        # Keep the middle 'min_n' rows
        new_arr = arr[remove_start:(arr.shape[0] - remove_end), :]
        equalized_arrays.append(new_arr)

    return equalized_arrays

equalized_arrays = equalize_array_lengths(sorted9)

viewer.add_points(points_for_napari(equalized_arrays), name='sorted_9', face_color='green')
#%%
resampled_equalized =  [resample_curve3(curve, 25) for curve in equalized_arrays]
viewer.add_points(points_for_napari(resampled_equalized), name='resampled_equalized', size=2, face_color='yellow') 

#%%
matrices_list9 = calculate_transform_matrices_procrustes(resampled_equalized, 9)
post_transformation9 = apply_transformation_all_frames(resampled_equalized[9], matrices_list9)
viewer.add_points(points_for_napari(post_transformation9), face_color='orange', size=2, name='resampled_equalized')

''' not working. maybe slightly better than not resampling. ---- resampled it once more before applying the transformation. 
verdict, the whole thing isnt working too well. I suppose doing consecutive will be slightly better than this crap.   '''
#%%
''' well, instead of doing consecutive transform, lets do some more manual adjustment. and looking at the figure, only look at the vertical part. so the 15 pixels only.  '''
manual_sample = [curve[:15] for curve in resampled_equalized]
viewer.add_points(points_for_napari(manual_sample), name='manual_sampling', size=2, face_color='red') 
#%%
matrices_list9 = calculate_transform_matrices_procrustes(manual_sample, 0)
post_transformation9 = apply_transformation_all_frames(manual_sample[0], matrices_list9)
viewer.add_points(points_for_napari(post_transformation9), face_color='orange', size=2, name='manual_sampling')

''' at thevery least, the behavior of manual sampling, only the fairly vertical part has given me the best possible transformation. although it is still not perfect. But maybe it works for actual segments?? worth to look into this, to choose segments in such a way.  '''
#%%
viewer.add_image(image)
#%%

def shapes_for_napari(list_shapes):
    all_shapes = []

    for i, subset in enumerate(list_shapes):
        frame_id_column = np.full((subset.shape[0], 1), i)
        frame_subset = np.hstack([frame_id_column, subset])
        all_shapes.append(frame_subset)

    return all_shapes


ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
transformed_expanded = apply_transformation_all_frames(ref_points, matrices_list9) 
transformed_shapes = shapes_for_napari(transformed_expanded)
viewer.add_shapes(transformed_shapes, shape_type='polygon') 

''' another verdict: although the drawn shapes follows the points, it only follows the limited points. And, the limitation is restrictive, it doesnt work well for the proximal part. 
It literally follows the small edge, and following this is not really enough to follow the rest of the bone it seems.  ''' 
#%%
''' after using a very strict just a bone edge, the results are the bst so far, not for the segmentation, but for the raw points 
But i have identified a major issue. 1. the resampling DOES NOT place points at equidistance. THere is always an error. Which should not be the case. 
2. Because the points are not equidistant, the length of the curve for each frame is also not the same, then, if I try to find transformation, that is 
supposed to be rigid, it will not work, because the points themselves are morphing.  '''