#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:31:24 2024

@author: aayush
"""

#%%
import pickle
import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import napari 
import time 
from scipy import ndimage

from utils import (path_to_image, apply_canny, apply_remove, apply_skeleton, points_for_napari,
                   boolean_to_coords, apply_label, find_tibia_edges, find_array_with_min_n, downsample_points,
                   combined_consecutive_transform, coords_to_boolean)

    
#%%
# Step 1: load the image from directory and normalize it
path_neg = 'C:/Users/Aayush/Documents/thesis_files/data_for_thesis/MK_NW_ai2_tgv_5e-2_neg_ngn.nii'
path_pos = 'C:/Users/Aayush/Documents/thesis_files/data_for_thesis/MK_NW_ai2_tgv_5e-2_pos_ngn.nii'
#%%
image_neg = path_to_image(path_neg)[1:]
image_pos = path_to_image(path_pos)[1:]
#%%
# since our image goes from extened to flexed.. the direction means, pos is going down.. and neg is coming up 
# which means. if we want to present our data as going up then coming down .. we have to reverse the neg, put it at the first half. 
image_neg = image_neg[::-1]
#%%
full_image = np.concatenate( (image_neg, image_pos) , axis=0)

#%%
#add the original image to napari
viewer = napari.view_image(full_image,  name='MK_NW_full')

#%%
import matplotlib.pyplot as plt
image1 = full_image # added this because i directly opened this in the viiewer without path 

total_frames = len(full_image) 
desired_frames = 6

frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int)

disp_layer_tib = viewer.layers["MK_NW_tib_shape_stiched"].to_labels(image1.shape)
disp_layer_fem = viewer.layers["MK_NW_fem_shape_stiched"].to_labels(image1.shape)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(7,6), facecolor='black')
xrange = slice(80,480)
yrange=slice(120,400)
for ax, idi in zip(axes.flatten(), frame_indices):
    ax.imshow(image1[idi,xrange,yrange], cmap="gray")
    ax.imshow(disp_layer_tib[idi,xrange,yrange], alpha=(disp_layer_tib[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.imshow(disp_layer_fem[idi,xrange,yrange], alpha=(disp_layer_fem[idi,xrange,yrange] > 0).astype(float) * 0.2, cmap='brg')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Frame {idi}", color='white')
     
plt.tight_layout()

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

low_range = (0,5) # 
high_range = (5,10 ) # 
num_steps = 10
sigma = 2
print(np.linspace(low_range[0] , low_range[1], num_steps) )
print(np.linspace(high_range[0] , high_range[1], num_steps) )

canny_multi_edge = apply_canny_multiple_thresholds(full_image, low_range, high_range, num_steps, sigma)

end_time = time.time() 
print(f"Elapsed Time: {end_time - start_time} seconds")
viewer3.add_image(canny_multi_edge, name='MM_W_full')
#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[8]
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
size_range = (50, 350)  # 100 min and 200 max size for femur 
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
#%% 2
# step 10.2 works better with femur 
structuring_element = ndimage.generate_binary_structure(3, 3)
ndlabel, features = ndimage.label(skeleton_bone_canny, structure= structuring_element, output=None)
viewer.add_labels(ndlabel, name='ndlabel_with_3,3_structure')    

#%%
final_label_3d = ndlabel.copy()
#final_label_3d = (final_label_3d == 3) | (final_label_3d == 23) | (final_label_3d == 25)
final_label_3d = (final_label_3d == 3) 
viewer.add_image(final_label_3d)
#%%
#final_label = viewer.layers['tibia_edges'].data  # when using 2d labelling. 
final_label = viewer.layers['final_label_3d'].data  # or final_label_3d
#Step 11: once the final edge has been found, convert it to a list of arrays.
#%% 
tib_coords = boolean_to_coords(final_label) # use final_label_3d if that is used instead of tibia_edges
#  just finding the frame with the least number of points
find_array_with_min_n(tib_coords)
#%%
# Step 12, starting with either the first or the last frame. 
reference_frame_last = downsample_points(tib_coords, -1, 50, bone_type='tibia')
new_tib_coords_last = tib_coords.copy() 
new_tib_coords_last[-1] = reference_frame_last
viewer.add_points(reference_frame_last, face_color='blue', size =1, name='reference_frame_last')
#%%
#reference_frame_first = downsample_points(tib_coords, 0, 80, bone_type='femur')
new_tib_coords_first = tib_coords.copy() 
#new_tib_coords_first[0] = reference_frame_first
new_tib_coords_first[0] = MM_NW_ref_frame_fem
#viewer.add_points(reference_frame_first, face_color='orange', size =1, name='reference_frame_first')
viewer.add_points(MM_NW_ref_frame_fem, face_color='green', size =1, name='reference_frame_first_using_NW_fem')

#%%
#Step 13. find the transformation matrices, list of coordinates and minimized cost function values per frame 
transformation_matrices_last, giant_list_last, cost_values_last = combined_consecutive_transform(new_tib_coords_last)
viewer.add_points(points_for_napari(giant_list_last), size=1, face_color='green', name='ref_frame_last')
            

#%%
transformation_matrices_first, giant_list_first, cost_values_first = combined_consecutive_transform(new_tib_coords_first)
viewer.add_points(points_for_napari(giant_list_first), size=1, face_color='blue', name='transformed_frame_NW_stiched')
#%%

with open('MM_W_t_matrices_fem_s.pkl', 'wb') as file:
    pickle.dump(transformation_matrices_first, file)
