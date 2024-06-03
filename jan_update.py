#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:31:24 2024

@author: aayush
"""

#%%
import pickle
import os 
#os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
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
path_neg = '/data/projects/ma-nepal-segmentation/data/Maggioni^Marta_Brigid/2024-03-08/108_MK_Radial_NW_CINE_30bpm_CGA/MM_NW_ai2_tgv_5e-2_neg_ngn.nii'
path_pos ='/data/projects/ma-nepal-segmentation/data/Maggioni^Marta_Brigid/2024-03-08/108_MK_Radial_NW_CINE_30bpm_CGA/MM_NW_ai2_tgv_5e-2_pos_ngn.nii'
#%%
image_neg = path_to_image(path_neg)[2:]
image_pos = path_to_image(path_pos)[2:]
#%%
# since our image goes from extened to flexed.. the direction means, pos is going down.. and neg is coming up 
# which means. if we want to present our data as going up then coming down .. we have to reverse the neg, put it at the first half. 
image_neg = image_neg[::-1]
#%%
full_image = np.concatenate( (image_neg, image_pos) , axis=0)

#%%
#add the original image to napari
viewer = napari.view_image(full_image,  name='AN_NW_full')

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
final_label = viewer.layers['US_W_final_label_tib'].data  # or final_label_3d
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
reference_frame_first = downsample_points(tib_coords, 0, 80, bone_type='tibia')
#new_tib_coords_first = tib_coords.copy() 
tib_coords[0] = reference_frame_first
#new_tib_coords_first[0] = MM_NW_ref_frame_fem
viewer.add_points(reference_frame_first, face_color='orange', size =1, name='reference_frame_first')
#viewer.add_points(MM_NW_ref_frame_fem, face_color='green', size =1, name='reference_frame_first_using_NW_fem')

#%%
#Step 13. find the transformation matrices, list of coordinates and minimized cost function values per frame 
transformation_matrices_last, giant_list_last, cost_values_last = combined_consecutive_transform(new_tib_coords_last)
viewer.add_points(points_for_napari(giant_list_last), size=1, face_color='green', name='ref_frame_last')
            

#%%
transformation_matrices_first, giant_list_first, cost_values_first = combined_consecutive_transform(tib_coords) # new_tib_coords_first
viewer.add_points(points_for_napari(giant_list_first), size=2, face_color='green', name='transformed_frame_W_stiched')
#%%
''' below is the cost value dataframe creation and plotting routine  ''' 
with open('master_df_cost.pkl', 'wb') as file:
    pickle.dump(combined_df, file)
    
#%%
import pandas as pd
def create_cost_value(array):
    tib_coords = boolean_to_coords(array)
    reference_frame_first = downsample_points(tib_coords, 0, 80, bone_type='tibia')
    tib_coords[0] = reference_frame_first
    transformation_matrices_first, giant_list_first, cost_values_first = combined_consecutive_transform(tib_coords)
    return cost_values_first

array_names = ['MK_W_final_label_tib_stiched_other', 'MM_W_final_label_tib', 'AN_W_final_label_tib_3',  
               'MK_W_final_label_tib_s', 'AN_W_final_label_tib_5','final_label_HS_W_tib', 'JL_W_final_label_tib' ]

def create_dataframe(array_names):
    # Create an empty DataFrame
    all_data = pd.DataFrame()
    
    for i, name in enumerate(array_names):
        # Assuming the arrays are accessible as global variables, you might need to use eval() to access them
        array = eval(name)
        
        # Generate cost values for the current array
        cost_values = create_cost_value(array)
        
        # Create a temporary DataFrame
        temp_df = pd.DataFrame({
            'Dataset': [name] * len(cost_values),
            'Total Cost': cost_values,
            'Frame': list(range(len(cost_values)))
        })
        
        # Append to the main DataFrame
        all_data = pd.concat([all_data, temp_df], ignore_index=True)
    
    return all_data


cost_df_W_tib = create_dataframe(array_names)
#%%
unique_labels = cost_df_W_tib['Dataset'].unique()

# Create a dictionary mapping each unique label to an integer
label_to_int = {label: idx + 1 for idx, label in enumerate(unique_labels)}

# Replace the string labels in the DataFrame with integers
cost_df_W_tib['Dataset'] = cost_df_W_tib['Dataset'].replace(label_to_int)

#%%
cost_df_W_tib['Average Cost'] = cost_df_W_tib['Total Cost'] / 80 

#%%
# master df for all the cost values 

# Adding 'Condition' and 'Bone' columns based on the DataFrame names
cost_df_W_fem['Condition'] = 'Loaded'
cost_df_W_fem['Bone'] = 'Femur'

cost_df_W_tib['Condition'] = 'Loaded'
cost_df_W_tib['Bone'] = 'Tibia'

cost_nw_fem_all['Condition'] = 'Unloaded'
cost_nw_fem_all['Bone'] = 'Femur'

cost_nw_tib_all['Condition'] = 'Unloaded'
cost_nw_tib_all['Bone'] = 'Tibia'

# Concatenate all dataframes into one
combined_df = pd.concat([cost_df_W_fem, cost_df_W_tib, cost_nw_fem_all, cost_nw_tib_all], ignore_index=True)


#%%
# Correct the DataFrame filtering to include frames from 1 to 29
filtered_df = cost_df_50[(cost_df_50['Frame'] > 0) & (cost_df_50['Frame'] < 31)]

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a line plot of Average Cost vs Frame
plt.figure(figsize=(10, 6), dpi=300)  # Set the figure size and dpi for high resolution
plot = sns.lineplot(data=filtered_df, x='Frame', y='Average Cost', marker='o')

# Adding title and labels
plt.xlim(1, 30)
plt.xticks(np.arange(1,31))
plt.title('Average cost value per frame for all unloaded tibia datasets')
plt.xlabel('Frame Number')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.savefig('average_cost_plot.png', dpi=300)
# Display the plot
plt.show()

#%% 

# that showed a trend.. so aggregating further to see differences from datasets: 
# Calculate the mean of the 'Average Cost' for each dataset
average_of_averages = cost_df.groupby('Dataset')['Average Cost'].mean()

# Reset the index to make 'Dataset' a column again if you want to plot using seaborn directly
average_of_averages = average_of_averages.reset_index()    
    
plt.figure(figsize=(10, 6), dpi=300)  # Set figure size and dpi for high resolution
bar_plot = sns.barplot(data=average_of_averages, x='Dataset', y='Average Cost')

# Adding title and labels
plt.title('Mean alignment error per point across datasets')
plt.xlabel('Datasets')
plt.ylabel('Mean Alignment error')
#plt.savefig('avg_of_avg.png', dpi=300)
# Display the plot
plt.show()


#%%
''' adding voxel size here and also sprucing up the bar plot  '''
voxel_size = 0.48484848484848486

def plot_cost_bar(cost_df):
    # Calculate the mean and standard deviation of the 'Average Cost' for each dataset
    stats = cost_df.groupby('Dataset')['Average Cost'].agg(['mean', 'std']).reset_index()
    stats['mean_mm'] = stats['mean'] * voxel_size
    stats['std_mm'] = stats['std'] * voxel_size
    
    plt.figure(figsize=(10, 6), dpi=300)  # Set figure size and dpi for high resolution
    bar_plot = sns.barplot(
        data=stats, 
        x='Dataset', 
        y='mean_mm', 
        palette='viridis',  # Using the 'viridis' color palette
        ci=None  # Disable seaborn's built-in confidence intervals
    )
    
    # Adding error bars
    plt.errorbar(
        x=range(len(stats)), 
        y=stats['mean_mm'], 
        yerr=stats['std_mm'], 
        fmt='none', 
        c='black', 
        capsize=5  # Adding caps to the error bars
    )
    
    # Adding title and labels
    plt.title('Mean alignment error per point for all datasets')
    plt.xlabel('Dataset')
    plt.ylabel('Mean Alignment Error [mm]')
    #plt.savefig('bar_alignment_NW_fem.png', dpi=300)
    # Display the plot
    plt.show()

plot_cost_bar(cost_nw_tib_all[cost_nw_tib_all['Dataset'].isin([2,4,5,6,7])])
#%%
# this is for df_W 
'''    Dataset      mean       std   mean_mm    std_mm
0        1  0.718797  0.261929  0.348508  0.126996
1        2  0.792351  0.269295  0.384170  0.130567
2        3  1.158230  0.635632  0.561566  0.308185
3        4  0.554365  0.138189  0.268783  0.067001
4        5  0.642973  0.177957  0.311744  0.086282
5        6  0.893317  0.212606  0.433124  0.103082
6        7  1.050978  0.288532  0.509565  0.139894 ''' 


'''
stats for NW 

   Dataset      mean       std   mean_mm    std_mm
0        1  0.749045  0.266838  0.363173  0.129376
1        2  0.663258  0.233765  0.321580  0.113341
2        3  0.958464  0.379197  0.464710  0.183853
3        4  0.612775  0.227513  0.297103  0.110309
4        5  0.585517  0.130719  0.283887  0.063379
5        6  0.806982  0.178597  0.391264  0.086593
6        7  0.600224  0.120798  0.291017  0.058569


 ''' 
 
#%%
''' 
Condition: No Weight
Bin Center: -95.0005, Mean Angle: 157.5522789875202, Std Dev: 4.895974756723701
Bin Center: -85.0, Mean Angle: 161.16780614067977, Std Dev: 4.8996893031884845
Bin Center: -75.0, Mean Angle: 164.36639174346672, Std Dev: 4.711194827263206
Bin Center: -65.0, Mean Angle: 167.7939136843817, Std Dev: 4.026012020442884
Bin Center: -55.0, Mean Angle: 171.02297430248174, Std Dev: 3.962334363098029
Bin Center: -45.0, Mean Angle: 174.65886345363072, Std Dev: 4.057050876565473
Bin Center: -35.0, Mean Angle: 176.07488625179937, Std Dev: 4.086201716322172
Bin Center: -25.0, Mean Angle: 178.8816410019267, Std Dev: 4.064083980539094
Bin Center: -15.0, Mean Angle: 181.0088768054279, Std Dev: 4.259058620655808
Bin Center: -5.0, Mean Angle: 184.1754319503113, Std Dev: 4.036725630460606
Bin Center: 5.0, Mean Angle: 183.6116164045419, Std Dev: 4.019192958915905
Bin Center: 15.0, Mean Angle: 180.48648967525475, Std Dev: 3.8913841629645893
Bin Center: 25.0, Mean Angle: 178.22350881162345, Std Dev: 4.0342520825541275
Bin Center: 35.0, Mean Angle: 175.95076142289932, Std Dev: 3.8144123886283308
Bin Center: 45.0, Mean Angle: 172.7847564068593, Std Dev: 4.301310721303088
Bin Center: 55.0, Mean Angle: 170.6888378072372, Std Dev: 3.714331329880709
Bin Center: 65.0, Mean Angle: 166.936773826967, Std Dev: 3.9775494583878763
Bin Center: 75.0, Mean Angle: 163.99876562273545, Std Dev: 4.3935614483702965
Bin Center: 85.0, Mean Angle: 160.29620474232289, Std Dev: 4.75978695339811
Bin Center: 95.0, Mean Angle: 157.49515826432585, Std Dev: 4.825347990864314
Condition: Weight
Bin Center: -95.0005, Mean Angle: 158.09051108261508, Std Dev: 4.950930554393052
Bin Center: -85.0, Mean Angle: 162.44699454277614, Std Dev: 4.714744733933274
Bin Center: -75.0, Mean Angle: 165.38848468642834, Std Dev: 4.805821107177295
Bin Center: -65.0, Mean Angle: 168.769885520498, Std Dev: 4.367937219933024
Bin Center: -55.0, Mean Angle: 171.87710161071467, Std Dev: 4.090084749507679
Bin Center: -45.0, Mean Angle: 175.46788530664918, Std Dev: 3.9063772297690313
Bin Center: -35.0, Mean Angle: 176.6000394971508, Std Dev: 4.023457601364946
Bin Center: -25.0, Mean Angle: 179.89863222680725, Std Dev: 4.011374694118442
Bin Center: -15.0, Mean Angle: 181.64784503757, Std Dev: 4.028381383902172
Bin Center: -5.0, Mean Angle: 184.07309321380453, Std Dev: 3.9879246740563974
Bin Center: 5.0, Mean Angle: 182.83782679850242, Std Dev: 3.5834826359365373
Bin Center: 15.0, Mean Angle: 180.42252525694553, Std Dev: 3.567948504448165
Bin Center: 25.0, Mean Angle: 178.30544344743913, Std Dev: 3.638467262473584
Bin Center: 35.0, Mean Angle: 175.7946532682328, Std Dev: 4.082930851048263
Bin Center: 45.0, Mean Angle: 173.4483867057555, Std Dev: 4.354221464953155
Bin Center: 55.0, Mean Angle: 171.26968700554616, Std Dev: 4.048552931884515
Bin Center: 65.0, Mean Angle: 167.24693155737396, Std Dev: 4.143515135950453
Bin Center: 75.0, Mean Angle: 164.64078862496763, Std Dev: 4.787739101290026
Bin Center: 85.0, Mean Angle: 160.88658317812613, Std Dev: 4.947286374633858
Bin Center: 95.0, Mean Angle: 157.91773393462233, Std Dev: 4.730447741814311

 '''  
 