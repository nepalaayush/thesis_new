#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:31:24 2024

@author: aayush
"""

#%%
import os 
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import napari 
import time 
from scipy import ndimage

from utils import (open_nii, normalize, apply_canny, apply_remove, apply_skeleton, points_for_napari,
                   boolean_to_coords, apply_label, find_tibia_edges, find_array_with_min_n, downsample_points,
                   combined_consecutive_transform)

#%%
# Step 1: load the image from directory and normalize it
path = '/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/26.01.24/AN_NW_ai2_tgv_5e-2_pos.nii'
image = open_nii(path)
image = normalize(image)
image = np.moveaxis(image, 1, 0)
#%%
#add the original image to napari
viewer = napari.view_image(image,  name='NW_AN')
#%%
# add the 4d image to a new viewer
viewer3 = napari.Viewer() 
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
sigma = 2
print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(image, low_range, high_range, num_steps, sigma)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")
viewer3.add_image(canny_multi_edge, name='MM_NW')
#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[9]
viewer.add_image(tib_canny, name='after_edge_detection_sigma_2')
#%%
#Step 6: manually adjust some breaks, etc to make edge consistent 
tib_canny = viewer.layers['after_edge_detection_sigma_2'].data.astype(bool)
#%%
#Step 7: Use remove small objects at various ranges to find the most suitable
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
size_range = (150, 250)  # 100 min and 200 max size for femur 
num_steps = 20  # Number of steps for size parameter
connectivity = 2  # Fixed connectivity value
print(np.linspace(size_range[0],size_range[1], num_steps))
# Assuming smooth_image is your 3D image array
removed_4d = apply_remove_multiple_sizes(tib_canny, size_range, num_steps, connectivity)

#%%
# add it to the 4d viewer
viewer3.add_image(removed_4d, name='multi_remove_small')
#%%
# step 8 pick the right index
bone_canny = removed_4d[19] 
viewer.add_image(bone_canny, name='after_remove_small')
#%%
# step 9 skeletonize the edge 
skeleton_bone_canny = apply_skeleton(bone_canny)
viewer.add_image(skeleton_bone_canny, name = 'skeleton_bone_canny')

#%%
# step 10 label the image 
#step 10.1 works better with tibia edge 

label_image, features = apply_label(bone_canny)

viewer.add_labels(label_image, name='applying 2d label')
#%%
start_coord = (viewer.layers['Points'].data[0,1:]).astype(int)
tibia_edges = find_tibia_edges(label_image, start_coord)
#%%
#adding as labels here, can be converted to image from gui 
viewer.add_labels(tibia_edges)
#%%
# step 10.2 works better with femur 
structuring_element = ndimage.generate_binary_structure(3, 3)
ndlabel, features = ndimage.label(skeleton_bone_canny, structure= structuring_element, output=None)
viewer.add_labels(ndlabel, name='ndlabel_with_3,3_structure')    

#%%
final_label_3d = ndlabel.copy()
final_label_3d = final_label_3d==2
viewer.add_image(final_label_3d)
#%%
final_label = viewer.layers['final_label_3d'].data # or final_label_3d
#Step 11: once the final edge has been found, convert it to a list of arrays. 
tib_coords = boolean_to_coords(final_label) # use final_label_3d if that is used instead of tibia_edges
#  just finding the frame with the least number of points
find_array_with_min_n(tib_coords)
#%%
# Step 12, starting with either the first or the last frame. 
reference_frame_last = downsample_points(tib_coords, -1, 50, bone_type='femur')
new_tib_coords_last = tib_coords.copy() 
new_tib_coords_last[-1] = reference_frame_last
viewer.add_points(reference_frame_last, face_color='blue', size =1, name='reference_frame_last')
#%%
reference_frame_first = downsample_points(tib_coords, 0, 50, bone_type='femur')
new_tib_coords_first = tib_coords.copy() 
new_tib_coords_first[0] = reference_frame_first
viewer.add_points(reference_frame_first, face_color='orange', size =1, name='reference_frame_first')
#%%
#Step 13. find the transformation matrices, list of coordinates and minimized cost function values per frame 
transformation_matrices_last, giant_list_last, cost_values_last = combined_consecutive_transform(new_tib_coords_last)
viewer.add_points(points_for_napari(giant_list_last), size=1, face_color='green', name='ref_frame_last')
            
#%%
transformation_matrices_first, giant_list_first, cost_values_first = combined_consecutive_transform(new_tib_coords_first)
viewer.add_points(points_for_napari(giant_list_first), size=1, face_color='blue', name='ref_frame_first')
#%%
import pickle
with open('tib_coords_last_W_outer.pkl', 'wb') as file:
    pickle.dump(new_tib_coords_last, file)