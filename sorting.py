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
from scipy.interpolate import CubicSpline

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
matrices_list = calculate_transform_matrices_procrustes(all_subsets, 1)
post_transformation = apply_transformation_all_frames(all_subsets[1], matrices_list)
viewer.add_points(points_for_napari(post_transformation), face_color='blue', size=2) 

#%%
''' here is where we are at the moment, the matrix tries its best to match, but if the assumptions themselves are not true, then it wont work. 
so, trying it one more time, but this time taking frame 9 as the base  ''' 

base_frame0 = tib_label[0]

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

tib_label[0] = base_boolean 

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
tib_label = np.load('C:/Users/Aayush/Documents/thesis_files/tib_label.npy')
#%%
viewer = napari.Viewer() 
#%%
print(viewer.layers['Points'].data)
min_row = 331
max_row = 396

cropped_frame = np.array(tib_label[0])

# Set all pixels outside your vertical range to false.
cropped_frame[0:min_row, :] = False
cropped_frame[max_row+1:, :] = False
#%%
viewer.add_image(cropped_frame)
#%%
# now to replace this with the roginal array 
new_label = tib_label 
new_label[0] = cropped_frame 
viewer.add_image(new_label, name='new_label')
#%%
all_subsets = find_corres_for_all_frames(new_label)
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
viewer.add_points(points_for_napari(all_subsets), name= 'cropped_subsets' ,  face_color='orange', size=2) 
#%%
''' Ok so, manually selecting a range seems to be working okish. But the fundamental problem of resampling still remains. So, instead of find_corres directly, lets resample the base frame first 
so that we have this equidistant (very important) first. So the base is already sorted and equidistant. After that, we will find corresponding points.  '''
cropped_points = np.argwhere(cropped_frame)
resampled_base_sorted = resample_curve3(cropped_points, n_points=100)
viewer.add_points(resampled_base_sorted, name='resampled_base', face_color='red', size = 2)

#%%
def show_order(curve):
    plt.scatter(curve[:, 0], curve[:, 1])

    # Annotate each point with its index
    for i, (x, y) in enumerate(curve):
        plt.annotate(str(i), (x, y))

    plt.show()

show_order(cropped_points)
#%%
sorted_cropped = sort_points_single_frame(cropped_points)[:-1]
#%%
show_order(sorted_cropped)
#%%
''' status: i have manually selected a segment using two starta nd end points. After this, i have obtained 100 points, coincidentally. 
Now, i want to resample it , so that my curve has equidistant points. The total length of this curve, for now, is irrelevant.  '''
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
#%%
pairwise_distances(sorted_cropped)
pairwise_distances(cropped_points)
#surprisingly enough, even without resampling, the original frame is pretty stable, the distances are either 1 or 1.414. not sure why or how. 

#%%
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
#%%
new_cropped = equidistant_points(sorted_cropped, 30)

pairwise_distances(new_cropped)
show_order(new_cropped)
#%%
new_cropped_distances = pairwise_distances(new_cropped)
desired_distance = np.round ( sum(new_cropped_distances) / len(new_cropped_distances) , 3) 
#%%
from scipy.interpolate import CubicSpline

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
adjusted_points = adjust_downsampled_points(new_cropped, sorted_cropped)
#%%
viewer.add_points(new_cropped, face_color='red', size= 1)
viewer.add_points(sorted_cropped, face_color='blue', size=1)
viewer.add_points(adjusted_points, face_color='yellow', size=1)
''' at this point, i have done all i could in terms of resampling the cropped segment. the standard deviation of pairwise distance is 0.026. lets go with this.  ''' 
#%%
all_subsets = find_corres_for_all_frames(new_label)
viewer.add_points(points_for_napari(all_subsets), name= 'find_corres_on_adjusted_points')
#%%
# what is needed here is to sort the points in rest of the new_label image. (all_subsets)
def sort_all_frames(all_subsets):
    sorted_list = []
    for array in all_subsets:
        sorted_list.append(sort_points_single_frame(array))
    return sorted_list
#%%
sorted_subsets = sort_all_frames(all_subsets)
#%%
sorted_subsets = [subset[:-1] for subset in sorted_subsets] # removing the last point 
# ok the sorting is kinda ok. for some points, they are just overlapping with each other


#%%
def find_corres_other_frames3(downsampled, original_curve, desired_distances=None, fixed_distance=None):
    """
    Adjust the positions of downsampled points to specific distances 
    while ensuring they remain on the original curve.

    Parameters:
    - downsampled: np.array of shape (30,2) representing downsampled points.
    - original_curve: np.array of shape (100,2) representing the original curve.
    - desired_distances: List of distances between consecutive points.
    - fixed_distance: If desired_distances is not provided, use this as the fixed distance.

    Returns:
    - np.array of shape (30,2) representing the adjusted downsampled points.
    """
    
    # Ensure only one method is selected
    assert (desired_distances is None) != (fixed_distance is None), "Either provide desired distances or a fixed distance, not both."

    # Cubic spline interpolation of the original curve
    t = np.linspace(0, 1, len(original_curve))
    cs_x = CubicSpline(t, original_curve[:, 0])
    cs_y = CubicSpline(t, original_curve[:, 1])

    # Find the point in downsampled with the lowest x-coordinate to use as reference
    ref_index = np.argmin(downsampled[:, 0])
    reference_point = downsampled[ref_index]

    # Find the closest point in original_curve to this reference point
    distances = np.linalg.norm(original_curve - reference_point, axis=1)
    anchor_index = np.argmin(distances)
    anchor_point = original_curve[anchor_index]

    # Adjust the downsampled points starting from the new anchor point
    adjusted_points = [anchor_point]
    t_last = t[anchor_index]  # Start with the anchor point's t-value
    for i in range(1, len(downsampled)):
        desired_distance = fixed_distance if desired_distances is None else desired_distances[i-1]
        
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
corres_second_frame = find_corres_other_frames(sorted_subsets[0], sorted_subsets[1], fixed_distance=2.8)

#%%
viewer.add_points(adjusted_points, name='second', size=1, face_color='green')
#%%
# trying to do this iteratively. result doesnt look that promising. 
# this uses different base frame. 
# this code hangs the pc 

#%%
# trying to do it manually. 
corres_third_frame = find_corres_other_frames(corres_second_frame, sorted_subsets[2], fixed_distance=2.8)

#%%
viewer.add_points(corres_third_frame, name='third_', size=1, face_color='magenta')
#%%
corres_fourth_frame = find_corres_other_frames(corres_third_frame, sorted_subsets[3], fixed_distance=2.8)

#%%
viewer.add_points(corres_fourth_frame, name='fourth', size=1, face_color='brown')
#%%
# Use the function to compute all correspondences
new_subsets = compute_all_correspondences(sorted_subsets, fixed_distance=2.8)
#%%
viewer = napari.Viewer() 
viewer.add_points(points_for_napari(new_subsets), name='ultimate_destruction', size= 1, face_color='indigo')
''' by far the best result i have had so far. But need a few fundamental tweaking still: 1. make sure points are actual ON the curve. 2. first frame has double points sort of    '''
#%%
viewer.add_image(tib_label) 
#%%
def boolean_to_coords(boolean_array):
    all_coordinates = []
    ''' Input: a 3d boolean array, like a label image 
    Output: the coordinates where array is true, as a list, each list is a frame'''
    # Loop through each frame and grab the coordinates
    for frame in tib_label:
        coords = np.argwhere(frame)
        all_coordinates.append(coords)
    return all_coordinates
#%%
tib_label_chords = boolean_to_coords(tib_label)
viewer.add_points(points_for_napari(tib_label_chords), name='label_points', face_color='white', size=1)
#%%
viewer.add_points(points_for_napari(tib_label_chords), name='sorted_subsets', face_color='white', size=1)
#%%
#adding adjusted points to sorted_subsets. perhaps i did not do that before
tib_label_chords[0] = adjusted_points
#%%
# the solutiosn to the problems raised above are implemented below 
def snap_to_curve(adjusted_points, original_curve, threshold=0.5):
    """
    Snap points that are off the curve back onto it.

    Parameters:
    - adjusted_points: np.array of the points to adjust.
    - original_curve: np.array of the curve.
    - threshold: Distance threshold to snap the point back to the curve.

    Returns:
    - np.array of adjusted points.
    """
    snapped_points = []
    for point in adjusted_points:
        distances = np.linalg.norm(original_curve - point, axis=1)
        min_dist_index = np.argmin(distances)
        
        if distances[min_dist_index] > threshold:
            snapped_point = original_curve[min_dist_index]
        else:
            snapped_point = point
        
        snapped_points.append(snapped_point)
    
    return np.array(snapped_points)


def compute_all_correspondences2(sorted_subsets, fixed_distance=2.8, snap_threshold=0.5):
    """
    Compute correspondence points for all frames using iterative method and snap points to curve if off.

    Parameters:
    - sorted_subsets: List of np.array representing the frames.
    - fixed_distance: The fixed distance between points.
    - snap_threshold: Distance threshold to snap the point back to the curve.

    Returns:
    - List of np.array representing the adjusted frames.
    """
    num_frames = len(sorted_subsets)
    
    # Start with the already processed first frame
    all_adjusted_points = [sorted_subsets[0]]
    
    for i in range(1, num_frames):
        current_frame = all_adjusted_points[-1]  # Use the last adjusted frame as reference
        next_frame = sorted_subsets[i]
        
        adjusted_points = find_corres_other_frames3(current_frame, next_frame, fixed_distance=fixed_distance)
        
        # Snap points back to the curve if they're slightly off
        adjusted_points = snap_to_curve(adjusted_points, next_frame, threshold=snap_threshold)
        
        all_adjusted_points.append(adjusted_points)
    
    return all_adjusted_points

# Use the function to compute all correspondences
snapped_subsets = compute_all_correspondences2(new_list, fixed_distance=2.8, snap_threshold=0.5)

viewer.add_points(points_for_napari(snapped_subsets), name='new_list', face_color='red', size=1)
#%%
# still the weird phenomenon is occuring. probably because of sorting. how ironic. 
def sort_all_frames(all_subsets):
    sorted_list = []
    for array in all_subsets:
        sorted_list.append(sort_points_single_frame(array))
    return sorted_list
#%%
sorted_tib_label = sort_all_frames(tib_label_chords)
#%%
sorted_snapped = compute_all_correspondences(sorted_tib_label, fixed_distance=2.8, snap_threshold=0.5)
viewer.add_points(points_for_napari(sorted_snapped), name='sorted_snapped', face_color='yellow', size=1)
# allright, works now. but still, the anchor points seems off. it just looks off. 
#%%
viewer.add_points(points_for_napari(new_list), name='sorted_tib_label', face_color='white', size=1)
#%%

#%%
viewer.add_label(tib_label)
#%%
viewer0 = napari.Viewer() 
# doing it with a new approach, where the labels are not so bad. so only the first 19. 
selected_label = new_list[:19]
viewer0.add_points(points_for_napari(selected_label), name='selected_label', face_color='white', size=1)
#%%
snapped_selected = compute_all_correspondences2(selected_label, fixed_distance=2.8, snap_threshold=0.5)

viewer0.add_points(points_for_napari(snapped_selected), name='selected_snapped', face_color='red', size=1)
#%%
''' after a bit of manually selecting frames that were good, using the first anchor point as the nearest in the next frame approach works, at least in terms of all poitn being fairly equidistant, 
and also actually being on the curve itself. Now the time has come to test if thsi translates to a full fledged segment.  ''' 
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

matrices_list = calculate_transform_matrices_procrustes(snapped_selected, 0)
post_transformation = apply_transformation_all_frames(snapped_selected[0], matrices_list)

viewer0.add_points(points_for_napari(post_transformation), face_color='orange', size=1) 
#%%
''' using the new sampling selected subsets, the results are not great. this is odd because the algortihm was working well, at least for the given points. this seems to me that the 
data size is small. so below, a longer section is taken for analysis.  ''' 

tib_label = np.load('C:/Users/Aayush/Documents/thesis_files/tib_label_1.npy')
#%%
def apply_skeleton(pixelarray):
    skeletonized = np.zeros_like(pixelarray)
    for i in range(pixelarray.shape[0]):
        skel_frame = skeletonize(pixelarray[i])
        skeletonized[i] = skel_frame
    return skeletonized

label_skeleton = apply_skeleton(tib_label)
viewer.add_image(label_skeleton, name='skeleton')
#%%
# step 2: sort 
unsorted_skeleton = boolean_to_coords(label_skeleton)
#%%
def sort_points_all_frames(list_of_points):
    sorted_list = []
    for i in list_of_points:
        sorted_list.append(sort_points_single_frame(i))
    return sorted_list 
#%%
sorted_skeleton = sort_points_all_frames(unsorted_skeleton)
#%%
zeroth_unsorted = unsorted_skeleton[0]
zeroth_sorted = sort_points_single_frame(zeroth_unsorted)
#%%
# after looking at show_order for this variable, need to remove last three pixels. the number of random pixels is a crap shoot at this point. but since the whole point is to downsample just this single frame, perhaps it is not needed 
zeroth_sorted = zeroth_sorted[:-3]
#%%
downsampled_sorted = equidistant_points(zeroth_sorted, 30)
viewer.add_points(downsampled_sorted, face_color='indigo', size=1)
#%%
adjusted_downsampled = adjust_downsampled_points(downsampled_sorted, zeroth_sorted)
viewer.add_points(adjusted_downsampled, face_color='red', size=1)
#%%
# feed the adjusted downsampled, into the boolean skeleton, and run find corres. 
def coords_to_boolean(sorted_coordinates, shape=tib_label.shape):
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

unsorted_skeleton[0] = adjusted_downsampled
new_skeleton = coords_to_boolean(unsorted_skeleton)
#%%
new_new_skeleton = coords_to_boolean(sorted_skeleton)
#%%
viewer.add_image(new_new_skeleton, name='new__new_label_skeleton')
#%%
all_subsets_new = find_corres_for_all_frames(new_skeleton)
viewer.add_points(points_for_napari(all_subsets_new), name='all_subsets_new', size=2, face_color='brown')
#%%
# check transform directly here 
matrices_list = calculate_transform_matrices_procrustes(all_subsets_new, 0)
post_transformation = apply_transformation_all_frames(all_subsets_new[0], matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='orange', size=1) 

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
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(setB)
    
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
            score = abs(relative_distance - candidate_relative_distance) + abs(angle - candidate_angle)
            
            if score < best_score:
                best_score = score
                best_candidate = candidate
                
        B_corresponding.append(best_candidate)
        
    return np.array(B_corresponding)

def find_corres_for_all_frames5(fem_label):
    template_index = np.argmin([np.sum(frame) for frame in fem_label])
    print('template is frame:', template_index)
    
    template_set = fem_label[template_index]
    template_props = regionprops(template_set.astype(int))
    template_cords = template_props[0].coords

    all_subsets = []

    for i in range(len(fem_label)):
        if i == template_index:
            all_subsets.append(template_cords)
            continue

        test_set = fem_label[i]
        test_props = regionprops(test_set.astype(int))
        test_cords = test_props[0].coords

        subset_cords = find_corres_advanced(template_cords, test_cords)
        all_subsets.append(subset_cords)

    return all_subsets

all_subsets_advanced = find_corres_for_all_frames5(new_new_skeleton)
#%%
viewer.add_points(points_for_napari(all_subsets_advanced), name='all_subsets_advanced', size=2, face_color='green')
#%%
matrices_list = calculate_transform_matrices_procrustes(all_subsets_advanced, 0)
post_transformation = apply_transformation_all_frames(all_subsets_advanced[0], matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='purple', size=1) 

#%%
''' making angle and distance did not create as big of an advantage as expected. So, going back to the basics again, and trying to use fixed distance as the info.  '''

def find_corres6(setA, setB):
    if len(setA) == len(setB):
        return setB  # or return setA, since they are of equal length

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setB)
    distances, indices = nbrs.kneighbors(setA)
    B_corresponding = setB[indices.flatten()]
    return B_corresponding

def find_all6(fem_label_coords):
    template_index = 0  # Explicitly set to 0
    print('template is frame: ', template_index)

    template_cords = fem_label_coords[template_index]  # Directly use the coordinates

    all_subsets = []  # This list will hold all the subsets for each frame

    # Loop over all frames in fem_label_coords
    for i in range(len(fem_label_coords)):
        # Skip the template frame itself
        if i == template_index:
            all_subsets.append(template_cords)
            continue  # skip the rest of the loop for this iteration

        test_cords = fem_label_coords[i]  # Directly use the coordinates

        if len(template_cords) >= len(test_cords):
            all_subsets.append(test_cords)
        else:
            # Use your find_corres function to find the corresponding points
            subset_cords = find_corres6(template_cords, test_cords)

            # Add these corresponding points to all_subsets list
            all_subsets.append(subset_cords)

    return all_subsets

all_subsets_6 = find_all6(sorted_skeleton)
viewer.add_points(points_for_napari(all_subsets_6), name='all_subsets_6', size=2, face_color='green')
#%%
def pairwise_distances(points):
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    return distances

def find_corres7(setA, setB, global_target_distance):
    tolerance = 0.1  # Set some tolerance value, adjust this accordingly
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(setB)
    _, indices = nbrs.kneighbors(setA)
    B_corresponding = []
    used_points = set()

    # Use the nearest neighbor to the last template point as a starting point
    start_point = setB[indices[-1]][0]
    B_corresponding.append(start_point)
    used_points.add(tuple(start_point))

    for _ in range(1, len(setA)):
        min_diff = float('inf')
        best_point = None

        # Sort remaining points by distance to the start_point
        sorted_remaining_points = sorted(setB, key=lambda point: np.linalg.norm(start_point - np.array(point)))

        for point in sorted_remaining_points:
            if tuple(point) in used_points:
                continue

            current_distance = np.linalg.norm(start_point - np.array(point))
            diff = abs(current_distance - global_target_distance)

            # Check within tolerance
            if global_target_distance - tolerance <= current_distance <= global_target_distance + tolerance:
                if diff < min_diff:
                    min_diff = diff
                    best_point = point

        if best_point is not None:
            B_corresponding.append(best_point)
            used_points.add(tuple(best_point))
            start_point = best_point  # Update the starting point for the next iteration

    return np.array(B_corresponding)




def find_all7(fem_label_coords):
    template_index = 0
    print('template is frame: ', template_index)
    template_cords = fem_label_coords[template_index]
    all_subsets = []

    # Compute pairwise distances for template and then find the mean
    template_distances = pairwise_distances(template_cords)
    print(template_distances)
    global_target_distance = np.mean(template_distances)

    for i, test_cords in enumerate(fem_label_coords):
        if i == template_index:
            all_subsets.append(template_cords)
            continue

        if len(template_cords) >= len(test_cords):
            all_subsets.append(test_cords)
        else:
            subset_cords = find_corres7(template_cords, test_cords, global_target_distance)
            all_subsets.append(subset_cords)

    return all_subsets

all_subsets_7 = find_all7(sorted_skeleton)
viewer.add_points(points_for_napari(all_subsets_7), name='all_subsets_7', size=2, face_color='yellow')
#%%
def find_fixed_distance_subsets(arr_list, fixed_distance=8, tolerance=0.1):
    new_arr_list = []
    
    for arr in arr_list:
        subset = []
        start_point = arr[-1]  # Start at the last point
        subset.append(start_point)
        used_points = set()
        used_points.add(tuple(start_point))
        
        while True:
            best_point = None
            min_diff = float('inf')
            
            for point in arr:
                if tuple(point) in used_points:
                    continue
                
                current_distance = np.linalg.norm(start_point - point)
                
                if fixed_distance - tolerance <= current_distance <= fixed_distance + tolerance:
                    diff = abs(current_distance - fixed_distance)
                    if diff < min_diff:
                        min_diff = diff
                        best_point = point
            
            if best_point is None:
                break  # Exit if no more points can be added
            
            subset.append(best_point)
            start_point = best_point
            used_points.add(tuple(best_point))
            
        new_arr_list.append(np.array(subset))
    
    return new_arr_list

all_subsets_8 = find_fixed_distance_subsets(sorted_skeleton) 
viewer.add_points(points_for_napari(all_subsets_8), name='all_subsets_7', size=2, face_color='red')

#%%
def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def downsample_single(points, threshold=8.6):
    points = points[::-1]
    downsampled_points = [points[0]]
    i = 0
    while i < len(points) - 1:
        total_distance = 0
        j = i + 1
        while j < len(points) and total_distance < threshold:
            total_distance += distance(points[j-1], points[j])
            j += 1
        if j < len(points):
            downsampled_points.append(points[j-1])
            i = j - 1
        else:
            break
    return np.array(downsampled_points)[::-1]


def downsample_list(points_list, threshold=8.6):
    return ([downsample_single(points, threshold) for points in points_list]) 

test1_list = downsample_list(sorted_skeleton)
viewer.add_points(points_for_napari(test1_list), name='downsample_list', size=2, face_color='red')
#%%
def equalize_lengths(points_list):
    # Find the length of the smallest (n,2) array
    min_length = min([len(points) for points in points_list])

    # Trim each array in the list to have the same length as the smallest array
    equalized_list = [points[-min_length:] for points in points_list]

    return equalized_list

equalized_list = equalize_lengths(test1_list)


#%%
matrices_list = calculate_transform_matrices_procrustes(equalized_list, 0)
post_transformation = apply_transformation_all_frames(equalized_list[0], matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='purple', size=1) 
#%%
def apply_transformation2(matrix, points):
    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # Apply the transformation
    transformed_points = np.dot(homogeneous_points, matrix.T)
    
    return transformed_points[:, :2]  # Convert back to Cartesian coordinates


def procrustes2(X, Y):
    X = X.astype(float)  # Ensure X is float
    Y = Y.astype(float)
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X -= centroid_X
    Y -= centroid_Y
    U, _, Vt = np.linalg.svd(np.dot(Y.T, X))
    R = np.dot(U, Vt)
    t = centroid_Y - np.dot(R, centroid_X)
    return R, t, centroid_X, centroid_Y


def calculate_transform_matrices_procrustes2(all_coords, reference_index):
    num_frames = len(all_coords)
    transformation_matrices = []

    # Get the coordinates of the reference frame
    reference_coords = all_coords[reference_index]

    for i in range(num_frames):
        if i == reference_index:
            # Add the identity matrix for the reference frame
            transformation_matrices.append(np.array([[1.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0]]))
        else:
            # Estimate the transformation matrix using Procrustes
            R, t, centroid_X, centroid_Y = procrustes2(reference_coords, all_coords[i])
            t_final = centroid_Y - np.dot(R, centroid_X)
            transformation_matrix = np.hstack([R, t_final.reshape(-1, 1)])
            transformation_matrices.append(transformation_matrix)

    return transformation_matrices


matrices_list = calculate_transform_matrices_procrustes2(equalized_list, 0)
post_transformation = apply_transformation_all_frames(equalized_list[0], matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='purple', size=1) 
#%%
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

matrices_list = calculate_transform_matrices_procrustes_consecutive(all_subsets_advanced)
post_transformation = apply_transformation_all_frames_consecutive(all_subsets_advanced[0], matrices_list)

viewer.add_points(points_for_napari(post_transformation), face_color='red', size=1) 

#%%
ref_points = viewer.layers['expanded_shape'].data[0][:,1:]
transformed_expanded = apply_transformation_all_frames_consecutive(ref_points, matrices_list) 
transformed_shapes = shapes_for_napari(transformed_expanded)
viewer.add_shapes(transformed_shapes, shape_type='polygon') 