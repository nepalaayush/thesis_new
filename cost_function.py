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
from scipy import ndimage
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


def sort_points_single_frame(points):
    points = np.array(points, dtype=np.float32)  # Ensure it's float or int for arithmetic operations
    
    # Find starting point
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
#%%
image1 = open_nii('/data/projects/ma-nepal-segmentation/data/Singh^Udai/2023-09-11/73_MK_Radial_W_CINE_60bpm_CGA/tgv_low.nii')
image1 = normalize(image1)
image1 = np.moveaxis(image1, 1, 0)[1:]
napari.view_image(image1)
#%%
# Step 1: load the image from directory and normalize it
image = open_nii('C:/Users/Aayush/Documents/thesis_files/nepal_aayush/NW_tgv.nii')
image = normalize(image)
image = np.moveaxis(image, 1, 0)[1:]
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='NW_AN')
#%%
viewer.add_image(image, name='image')
#%%
# Step 2: apply gaussian blur to the original image and add it in napari. 
smooth_image = ndimage.gaussian_filter(image, 2)
viewer.add_image(smooth_image , name='smooth_2')

#%%
smooth_image = image # when using regularized, it is already smooth
# Step 3: take the gradient of the smooth image, both magnitude as well as direction
grad_smooth = gradify(smooth_image)[0]
grad_direction = gradify(smooth_image)[1]
viewer.add_image(grad_smooth, name='gradient_smooth')
#%%
# add the 4d image to a new viewer
viewer3 = napari.Viewer() 
#%%
# Step 4: find the best suitable low and high range for edge detection
start_time = time.time() 

def apply_canny_multiple_thresholds(pixelarray, low_range, high_range, num_steps, sigma):
    ''' Note, values less than low are ignored, values greater than high are considered edges. Pixels with a gradient magnitude between low and high are conditionally accepted if they're connected to strong edges. ''' 
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    
    # Initialize a 4D array to store results
    canny_multi_edge = np.zeros((num_steps, *pixelarray.shape), dtype=bool)
    
    for j, (low, high) in enumerate(zip(low_values, high_values)):
        canny_edge = apply_canny(pixelarray, low, high, sigma)
        canny_multi_edge[j] = canny_edge
    
    return canny_multi_edge

low_range = (5,10) # 
high_range = (11,20 ) # 
num_steps = 10
sigma = 1.5
print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(image, low_range, high_range, num_steps, sigma)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")
viewer3.add_image(canny_multi_edge, name='AN_NW_1.5')

#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[9]
viewer.add_image(tib_canny, name='after_edge_detection_sigma_1.5')

#%%
#Step 6: Use remove small objects at various ranges to find the most suitable
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
size_range = (20, 50)  # 100 min and 200 max size for femur 
num_steps = 20  # Number of steps for size parameter
connectivity = 2  # Fixed connectivity value
print(np.linspace(size_range[0],size_range[1], num_steps))
# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(tib_canny, size_range, num_steps, connectivity)


# add it to the 4d viewer
viewer3.add_image(removed_4d, name='multi_remove_small_1.5')
#%%
# pick the right index
bone_canny = removed_4d[15] 
viewer.add_image(bone_canny, name='after_remove_small_15')

#%%
# skeletonize the edge 
skeleton_bone_canny = apply_skeleton(bone_canny)
viewer.add_image(skeleton_bone_canny, name = 'skeleton_bone_canny')

#%%
label_image, features = apply_label(bone_canny)

viewer.add_labels(label_image, name='2,2,structure_label_loop')
#%%
tib_label = (label_image >= 15) & (label_image <= 21)
viewer.add_labels(tib_label, name='tib_label')

#%%
''' second time ''' 
label_image, features = apply_label(tib_label)

viewer.add_labels(label_image, name='second_iteration')
#%%

structuring_element = ndimage.generate_binary_structure(3, 3)
#%%

custom_structuring_element = np.array([
    [[True, False, True],
     [False, True, False],
     [True, False, True]],
    
    [[True, True, False],
     [False, False, False],
     [False, True, True]],
    
    [[False, True, False],
     [True, False, True],
     [False, True, False]]
])
#%%
ndlabel, features = ndimage.label(skeleton_bone_canny, structure= structuring_element, output=None)
viewer.add_labels(ndlabel, name='ndlabel_with_3,3_structure_custom')    
#print(features)
#%%
''' for some reason, doing this is actually quite benefitial.  ''' 

final_label_outer = ndlabel.copy()
final_label_outer = final_label_outer==34
viewer.add_image(final_label_outer)
#%%
tib_label_penul = ndlabel== 5 
viewer.add_labels(tib_label_penul)

#%%
final_label = viewer.layers['tib_label_penul'].data
viewer.add_image(final_label)
#%%
final_label = apply_skeleton(final_label)
viewer.add_image(final_label)
#%%
np.save('final_label_outer_AN_NW', final_label_outer)

#%%
tib_label = np.load('C:/Users/Aayush/tib_label_5e-1.npy') # continuiing on previous work 
#%%
''' third time ''' 
label_image, features = apply_label(tib_label)

viewer.add_labels(label_image, name='third_iteration')

#%%
viewer = napari.Viewer()
#%%
viewer.add_labels(tib_label)
#%%
label_image, features = apply_label(tib_label)

viewer.add_labels(label_image, name='second_iteration')
#%%
tib_label_new = label_image > 9
viewer.add_labels(tib_label_new, name='tib_label_g_than_10')

#%%
ndlabel, features = ndimage.label(tib_label_new, structure=custom_structuring_element, output=None)
viewer.add_labels(ndlabel, name='ndlabel_with_3,3_structure_custom')  
#%%
# Step 7 find labels of connected regions from the edge image
label_image, features = apply_label(tib_label_new)

viewer.add_labels(label_image, name='fourth_iteration')  


#%%
start_coord = viewer.layers['Points'].data[0][1:].astype(int) 


auto_tib_edge = find_tibia_edges(label_image, start_coord)
viewer.add_labels(auto_tib_edge) 

#%%
final_label = viewer.layers['tib_label_g_than_10'].data
viewer.add_image(final_label)
#%%
tib_coords = boolean_to_coords(final_label_outer) 
#%%
viewer.add_image(final_label)
#%%
# step 8 : take the first frame and downsample/adjust it - change the index if we want to use another refernec.e 
zeroth_frame = sort_points_single_frame(tib_coords[0])
zeroth_nonadjusted = equidistant_points(zeroth_frame,50)
zeroth_adjusted = adjust_downsampled_points(zeroth_nonadjusted, zeroth_frame)
viewer.add_points(zeroth_adjusted, face_color='orange', size =1)

#%%
def downsample_points(list_of_arrays, index=0, number=50):
    zeroth_frame = sort_points_single_frame(tib_coords[index])
    zeroth_nonadjusted = equidistant_points(zeroth_frame,number)
    zeroth_adjusted = adjust_downsampled_points(zeroth_nonadjusted, zeroth_frame)
    return zeroth_adjusted

zeroth_adjusted = downsample_points(tib_coords, 9, 100)
viewer.add_points(zeroth_adjusted, face_color='orange', size =1)
#%%
# step 8.5 try to find the shortest curve as the reference frame 
template_index = np.argmin([np.sum(frame) for frame in final_label_outer])
print('template is frame: ', template_index)
#%%
# this works when we have a 3d boolean array, if instead we have a list of arrays, this should work 

#%%
#step 9, replace this version of first frame in the original list

new_tib_coords = tib_coords.copy() 
new_tib_coords[9] = zeroth_adjusted
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

def apply_transformations(reference_frame, transformation_matrices):
    transformed_frames = [reference_frame]

    for matrix in transformation_matrices[1:]:  # Exclude the identity matrix
        x, y, phi = matrix
        reference_frame = transform(reference_frame, x, y, phi)
        transformed_frames.append(reference_frame)

    return transformed_frames
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


#%%
''' this cell works when we have the label binary image as well as sampled first frame coordinates already  ''' 
viewer = napari.Viewer() 
viewer.add_image(image, name='original_image')
viewer.add_image(tib_label)
viewer.add_points(points_for_napari(new_tib_coords), face_color='yellow', size=2)

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
viewer.add_points(points_for_napari(giant_list), size=1, face_color='red', name='consecutive_transform')

#%%
#using the last frame 
zeroth_adjusted_last = downsample_points(tib_coords, index=-1, number=50 )
#%%
# defining the coords list by replacing the smallest frame 
new_tib_coords_last = tib_coords.copy()  
new_tib_coords_last[-1] = zeroth_adjusted_last

#%%
# trying out by using the reference frame as the 'smallest' edge 
def consecutive_transform_min(data):
    # Find the reference index
    reference_index = find_array_with_min_n(data)
    print(reference_index)
    num_frames = len(data)

    # Initialize the lists
    transformation_matrices = [np.array([0, 0, 0])] * num_frames
    giant_list = [None] * num_frames
    cost_values = [0] * num_frames

    # Set the reference frame
    reference_data = data[reference_index]
    giant_list[reference_index] = reference_data  # Reference frame remains unchanged

    # Transform preceding frames (working backwards)
    for ida in range(reference_index - 1, -1, -1):
        #print(ida)
        fr, cost = match_coords(reference_data, data[ida], x0=[0, 0, 0])
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        transformation_matrices[ida] = fr
        cost_values[ida] = cost
        giant_list[ida] = transformed_data
        reference_data = transformed_data
    # Transform following frames (working forwards)
    reference_data = data[reference_index]
    for ida in range(reference_index + 1, num_frames):
        fr, cost = match_coords(reference_data, data[ida], x0=[0, 0, 0])
        transformed_data = transform(reference_data, fr[0], fr[1], fr[2])
        transformation_matrices[ida] = fr
        cost_values[ida] = cost
        giant_list[ida] = transformed_data
        reference_data = transformed_data

    return transformation_matrices, giant_list, cost_values

transformation_matrices, giant_list, cost_values_no_update = consecutive_transform_min(new_tib_coords)
viewer.add_points(points_for_napari(giant_list), size=1, face_color='indigo', name='consecutive_transform_first')


#%%
def apply_transformations_new(reference_frame, transformation_matrices, reference_index):
    num_frames = len(transformation_matrices)
    transformed_frames = [None] * num_frames

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
#%%
# to use the reference as a shape and move it around 
viewer.add_shapes(new_tib_coords[9], shape_type='polygon')
#%%
ref_points = viewer.layers['expanded_shape'].data[0]
#%%
# here ref_points is taken from the drawn shape. not shown here. 
applied_transformation = apply_transformations_new(ref_points, transformation_matrices_x, 9)    
#%%
viewer.add_shapes(shapes_for_napari(applied_transformation), shape_type='polygon', face_color='green')

#%%
disp_layer = viewer.layers["transformed_shapes"].to_labels(image.shape)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange=slice(100,410)
yrange=slice(150,400)
for ax, idi in zip(axes.flatten(), range(0,38,7)):
    ax.imshow(image[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer[idi,xrange,yrange], alpha=(disp_layer[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(tib_label[idi,xrange,yrange], alpha=(tib_label[idi,xrange,yrange] > 0).astype(float), cmap='autumn')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
    
plt.tight_layout()
plt.savefig('test.svg')



#%%
def find_min_max(list_of_matrices):
    stacked_array = np.vstack(list_of_matrices)
    min_vals = [np.min(stacked_array[:,0]), np.min(stacked_array[:,1]), np.min(stacked_array[:,2]) ]
    max_vals = [np.max(stacked_array[:,0]), np.max(stacked_array[:,1]), np.max(stacked_array[:,2]) ]
    print('The min values are ', min_vals, '\n', 'and the max values are ', max_vals)

find_min_max(transformation_matrices)    

#%%
def find_min_indices(array):
    min_arg = np.argmin(array) 
    min_indices = np.unravel_index(min_arg,array.shape)
    return (min_indices)
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

transformation_matrices_g_update, giant_list_g_update, cost_values_g_update = consecutive_transform_with_guess_update(new_tib_coords)
viewer.add_points(points_for_napari(giant_list_g_update), size=1, face_color='blue', name='consecutive_transform_with_guess_update')

#%%
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

transformation_matrices_x, giant_list_x, cost_values_x = combined_consecutive_transform(new_tib_coords)
viewer.add_points(points_for_napari(giant_list_x), size=1, face_color='blue', name='combined_consecutive_transform')

#%%
data = new_tib_coords
''' to see the behaviour of cost function  '''
dx_range = np.linspace(0, 15, 100)  # Example range for dx
dy_range = np.linspace(-15,0,100)  # Example range for dy
dphi_constant = -0.0336  # Constant value for dphi taken from the transform_matrices list, index 19 

# Create a meshgrid for dx and dy
DX, DY = np.meshgrid(dx_range, dy_range)

# Initialize an array to hold the cost function values
cost_values = np.zeros_like(DX)

# Get the coordinates for frame 20
coords1 = giant_list[20]  
coords2 = giant_list[21]

# Compute the cost function values over the grid
for i in range(DX.shape[0]):
    for j in range(DX.shape[1]):
        # Apply the transformation with the current dx, dy and the constant dphi
        transformed_coords = transform(coords1, DX[i, j], DY[i, j], dphi_constant)
        
        # Compute the cost function value
        cost_values[i, j] = coords_distance_sum(transformed_coords, coords2)

# Find the minimum cost value directly
min_cost_value = np.min(cost_values)

# Find the indices of the minimum cost value
min_cost_indices = np.where(cost_values == min_cost_value)

# Use the indices to find the corresponding dx and dy values
# Since where() returns a tuple of arrays, we take the first element of those arrays
min_dx = DX[min_cost_indices][0]
min_dy = DY[min_cost_indices][0]

# Print the results
print(f"The minimum cost function value is {min_cost_value:.4f} at dx = {min_dx:.4f} and dy = {min_dy:.4f}")

# Plot the cost function values using a colormap
plt.figure(figsize=(8, 6))
plt.contourf(DX, DY, cost_values, levels=50, cmap='viridis')
plt.colorbar(label='Cost function value')
plt.xlabel('dx')
plt.ylabel('dy')
plt.title('Cost Function Landscape for Frame 9 (dphi constant)')
plt.show()
#%%
viewer1 = napari.Viewer()
#%%
''' doing a 3d version below   ''' 
start_time = time.time() 
# Define ranges for dx, dy, and dphi
dx_range = np.linspace(0, 15, 100)  # Smaller sample for computational feasibility
dy_range = np.linspace(-15,0,100)
dphi_range = np.linspace( -0.05, 0, 100)  # Example range for dphi

# Create a 3D meshgrid for dx, dy, and dphi
DX, DY, DPHI = np.meshgrid(dx_range, dy_range, dphi_range, indexing='ij')

# Initialize a 3D array to hold the cost function values
cost_values_3d = np.zeros_like(DX)

# Get the coordinates for frame 8 and 9 
coords1 = giant_list[20]  # giant_list contains the coordinates after sampling. 
coords2 = giant_list[21]

# Compute the cost function values over the grid
for i in range(DX.shape[0]):
    for j in range(DX.shape[1]):
        for k in range(DX.shape[2]):
            transformed_coords = transform(coords1, DX[i, j, k], DY[i, j, k], DPHI[i, j, k])
            
            # Compute the cost function value
            cost_values_3d[i, j, k] = coords_distance_sum(transformed_coords, coords2) 
            
            
viewer1.add_image(np.moveaxis(cost_values_3d, -1,0) )

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")
#%%
new_array = np.array([dx_range[63],dy_range[32], dphi_range[33] ])
print(new_array)

#%%
cost_fcn = lambda x: coords_distance_sum(transform(giant_list[20], new_array[0], new_array[1], new_array[2]), giant_list[21])
print(cost_fcn([0,0,0]))
#%%
transformed_manual = transform(giant_list[20], new_array[0], new_array[1], new_array[2])
viewer.add_points(transformed_manual, size=1, face_color='indigo')
#%%
applied_transformation2 = apply_transformations(ref_points, new_trans_matrices)
viewer.add_shapes(shapes_for_napari(applied_transformation2), shape_type='polygon', face_color='blue')
    
#%%



