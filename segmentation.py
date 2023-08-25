# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:21:35 2023

@author: MSI
"""

import numpy as np
import napari
import nibabel as nib 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import os 
#%%
def open_nii(path):
    ''' Input: Path of nifti file (.nii) Output: pixelarray  ''' 
    nifti_img = nib.load(path)
    #pixelarray = np.transpose ( nifti_img.get_fdata(), (2,0,1)) # have to transpose because of the way nifti loads things 
    pixelarray = nifti_img.get_fdata()
    return np.moveaxis(pixelarray, -1,0)
def gradify(pixelarray):
    dy, dx = np.gradient(pixelarray, axis=(1, 2))

    # Magnitude of the gradient for each frame
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    return gradient_magnitude



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

def get_uv_from_pca(line_points):
    A = np.array(line_points[0])
    B = np.array(line_points[1])
    AB = B - A
    mag_AB = np.linalg.norm(AB)
    U = AB / mag_AB
    V = np.array([-U[1], U[0]])
    return U, V

def find_extreme_points_on_secondary_axis(centroid, shape_coords, unit_vector_perpendicular):
    relative_positions = shape_coords[:, 1:] - centroid
    scalar_projections = np.dot(relative_positions, unit_vector_perpendicular)
    
    max_index = np.argmax(scalar_projections)
    min_index = np.argmin(scalar_projections)
    
    return shape_coords[max_index, 1:], shape_coords[min_index, 1:]

def find_edges(centroid, shape_coords, unit_vector_perpendicular):
    relative_positions = shape_coords[:, 1:] - centroid
    scalar_projections = np.dot(relative_positions, unit_vector_perpendicular)
    
    max_index = np.argmax(scalar_projections)
    min_index = np.argmin(scalar_projections)
    
    max_point = shape_coords[max_index, 1:]
    min_point = shape_coords[min_index, 1:]
    
    # The direction vector of the initial line
    initial_vector = max_point - min_point
    initial_unit_vector = initial_vector / np.linalg.norm(initial_vector)
    
    # Calculate the angle between the initial unit vector and V
    angle = np.arccos(np.dot(initial_unit_vector, unit_vector_perpendicular))
    
    # Now, we rotate the points around the centroid to make the line perpendicular to U
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    rotated_max_point = np.dot(rotation_matrix, (max_point - centroid).T).T + centroid
    rotated_min_point = np.dot(rotation_matrix, (min_point - centroid).T).T + centroid
    
    return rotated_max_point, rotated_min_point


def find_intersection(A, B, E, F):
    m1 = (B[1] - A[1]) / (B[0] - A[0])
    b1 = A[1] - m1 * A[0]
    m2 = (F[1] - E[1]) / (F[0] - E[0])
    b2 = E[1] - m2 * E[0]
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1
    return np.array([x_intersection, y_intersection])


def process_frame(viewer):
    # Process for all frames
    #frame_data = viewer.layers[1].data
    sorted_data = sorted(viewer.layers[1].data, key=lambda x: x[0][0])
    results = {}
    
    for idx, shape_coords in enumerate(sorted_data):
        # Calculate PCA line points
        line_points = fit_pca_line(shape_coords[:, 1:])
        
        # Get unit vectors
        U, V = get_uv_from_pca(line_points)
        
        # Compute centroid
        centroid = np.mean(shape_coords[:, 1:], axis=0)
        
        # Find extreme points
        extreme_points = np.array(find_edges(centroid, shape_coords, V))
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


def open_napari(path):
    pixelarray = open_nii(path)
    viewer = napari.view_image(gradify(pixelarray))
    return viewer 
#%%
path = '/data/projects/ma-alieksev-cine-knee-dynamics/data/Maggioni^Marta_Brigid/2021-04-09/53_MK_Radial_NoWeight_CINE_30bpm/CINE data/data_aw3_up_4_to_31deg_12cycles.nii'
path1 = '/data/projects/ma-alieksev-cine-knee-dynamics/data/Maggioni^Marta_Brigid/2021-04-09/53_MK_Radial_NoWeight_CINE_30bpm/CINE data/data_aw3_down_4_to_31deg.nii'
#%%
#load the data and open napari viewer 
pixelarray = open_nii(path)

pix1 = open_nii(path1)
#%%
viewer = open_napari(path)


#%%
# Assuming the shapes layer is the first layer
shapes_layer = viewer.layers[1]                                                    
# Assuming you're interested in the first shape
viewer.add_shapes(shapes_layer.data) 
#%%
viewer.layers.append(shapes_layer)

#%%
fem_info = process_frame(viewer)
#%%
tib_info = process_frame(viewer)
#%%
check_info = process_frame(viewer) 
#%%
def show_origin(all_frame_data):
    point_data = []
    
    for frame_index, frame_data in all_frame_data.items():
        x, y = frame_data['origin'] 
        cross = [frame_index, x , y] 
        print(cross) 
        point_data.append(cross)
    viewer.add_points(point_data, symbol='x')
 

def show_axis(all_frame_data, axis_name):
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

def show_stuff(frame_data, frame_name):
    home_directory = os.path.expanduser('~')
    show_axis(frame_data,'points_short_axis')         
    show_origin(frame_data) 
    show_axis(frame_data,'points_long_axis') 
    save_path = os.path.join(home_directory, 'Pictures', frame_name)
    np.save(save_path,frame_data)
    
    
#%%
show_stuff(tib_info, 'CINE_Highres_tib')
#%%            
show_stuff(fem_info, 'data_aw3_up_4_to_31deg_12cycles_fem')             
                                       
#%%

def axis_check (bone_info ,ind): 
    x1, x2 = bone_info[ind]['points_short_axis']
    direction_vector = x2 - x1  
    unit_vector_line = direction_vector / np.linalg.norm(direction_vector)
    V = bone_info[ind]['V'] 
    return np.dot ( V, unit_vector_line) 
#%%

def all_axis_check(bone_info):
    all_rad = []
    for i in fem_info:
        dot = axis_check(bone_info,i)
        radians = np.arccos(dot)
        all_rad.append(radians)
    return np.degrees(all_rad)

#%%
def track_origin(all_frame_info, bone_name):
    # Extract x and y coordinates of the origin for each frame
    x_coords = [all_frame_info[frame]['origin'][0] for frame in sorted(all_frame_info)]
    y_coords = [all_frame_info[frame]['origin'][1] for frame in sorted(all_frame_info)]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_coords, x_coords, c=sorted(all_frame_info), cmap='viridis', s=50)
    plt.plot(y_coords, x_coords, '-o', markersize=5, alpha=0.6)
    plt.colorbar(label='Frame Number')
    plt.title(f'Movement of {bone_name} Origin Over Frames')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()
    
def calculate_angle(vector_a, vector_b):
    """Calculate angle in degrees between two vectors."""
    cos_theta = np.dot(vector_a, vector_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Assuming femur_info and tibia_info have the same keys (frames)
def plot_angle_vs_frame(femur_info , tibia_info, label):
    frames = sorted(femur_info.keys())
    angles = []
    
    for frame in frames:
        femur_vector = femur_info[frame]['V']
        tibia_vector = tibia_info[frame]['V']
        angle = calculate_angle(femur_vector, tibia_vector)
        angles.append(angle)
    angles = (180 - np.array(angles) ) 
    # Plot
    plt.plot(frames, angles, label=f'{label}')
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Change in Angle Between Femur and Tibia Over Frames")
    plt.grid(True)
    plt.legend()
    plt.show()

#%%



