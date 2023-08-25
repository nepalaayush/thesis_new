#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:32:11 2023

@author: aayush
"""

import numpy as np 
import napari 
import nibabel as nib 
import matplotlib.pyplot as plt 
#%%
def open_nii(path):
    ''' Input: Path of nifti file (.nii) Output: pixelarray  ''' 
    nifti_img = nib.load(path)
    #pixelarray = np.transpose ( nifti_img.get_fdata(), (2,0,1)) # have to transpose because of the way nifti loads things 
    pixelarray = nifti_img.get_fdata()
    return np.moveaxis(pixelarray, -1,0)


def normalize(pixelarray):
    
#%%
def rectangle_center(rectangle_points):
    ''' Input: An array, containing four points of the edges of a drawn shape. Output: a tuple = the center coordinates.  '''
    top_left, bottom_left, bottom_right, top_right = rectangle_points
    center_x = (top_left[1] + bottom_right[1]) / 2 # note that the index will change depending on directly gui drawing or adding shapes first
    center_y = (top_left[2] + bottom_right[2]) / 2
    frame = top_left[0] # might as well be any other point.  
    return frame, center_x, center_y

def ellipse_centers_from_shapes_layer(shapes_layer):
    ''' Inupt: A shape layer. Output: Gives the centres of all the shapes (presumably ellipses) in this shape layer as a list ''' 
    ellipse_centers = []
    for rectangle_points in shapes_layer.data:
        center = rectangle_center(rectangle_points)
        ellipse_centers.append(center)
        #swapped_ellipse_centers = [(y,x) for x,y in ellipse_centers]
    return ellipse_centers

def unit_vectors(ellipse_centers):
    '''
    Input: List of ellipse centers with frame number
    Output: Lists U and V containing the unit vectors for each frame
    '''
    # Group centers by frames
    frame_dict = {}
    for center in ellipse_centers:
        frame = center[0]
        if frame not in frame_dict:
            frame_dict[frame] = []
        frame_dict[frame].append(center[1:])  # append without frame index
    
    U_list = []
    V_list = []
    
    # For each frame, calculate U and V
    for frame, centers in frame_dict.items():
        if len(centers) != 2:  # ensures there are only 2 centers for each frame
            continue
        
        A = np.array(centers[0])
        B = np.array(centers[1])
        AB = B - A
        mag_AB = np.linalg.norm(AB)
        U = AB / mag_AB
        V = np.array([-U[1], U[0]])
        
        U_list.append(U)
        V_list.append(V)
    
    return U_list, V_list


def calculate_midpoints(points_data):
    """
    Given a list of points from the Napari points layer, calculate the midpoint for each frame.
    The points_data is expected to have two poin16ts per frame.
    Return a dictionary with frame numbers as keys and midpoints as values.
    """
    frame_midpoints = {}
    for i in range(0, len(points_data), 2):  # process two points at a time
        p1, p2 = points_data[i], points_data[i+1]
        
        # Check if both points belong to the same frame
        if p1[0] == p2[0]:
            frame = p1[0]
            mx = (p1[1] + p2[1]) / 2
            my = (p1[2] + p2[2]) / 2
            frame_midpoints[frame] = (mx, my)
    
    return frame_midpoints

def find_intersection(A, B, E, F):
    # Calculate the slope
    m1 = (B[1] - A[1]) / (B[0] - A[0])
    # Calculate the y-intercept
    b1 = A[1] - m1 * A[0]
    # Calculate the slope
    m2 = (F[1] - E[1]) / (F[0] - E[0])
    # Calculate the y-intercept
    b2 = E[1] - m2 * E[0]
    # m1 * x + b1 = m2 * x + b2
    x_intersection = (b2 - b1) / (m1 - m2)
    y_intersection = m1 * x_intersection + b1
    
    return np.array([x_intersection, y_intersection])

def calculate_secondary_axis(midpoint, V, t=50):
    Vt = t * V
    E = midpoint + Vt
    F = midpoint - Vt
    return E, F

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
    show_axis(frame_data,'points_short_axis')         
    show_origin(frame_data) 
    show_axis(frame_data,'points_long_axis') 
    np.save(f'/home/aayush/Pictures/{frame_name}',frame_data)

def get_coordinate_axes(e_center, u_vectors, v_vectors, midpoints):
    # basically converts the ellipse center variable to a usable format. can be edited for efficiency 
    primary_axis_points = [
        (np.array(e_center[i][1:]), np.array(e_center[i+1][1:]))
        for i in range(0, len(e_center), 2)
    ]
    all_frame_data = {}
    # Calculating secondary axis points and origin for each frame
    for frame, (U, V, (A, B)) in enumerate(zip(u_vectors, v_vectors, primary_axis_points)):
        
        midpoint = np.array(midpoints[frame])
        E, F = calculate_secondary_axis(midpoint, V)
        origin = find_intersection(A, B, E, F)
    
        frame_info= {
            'origin': origin,
            'points_long_axis': (A, B),
            'points_short_axis': (E, F),
            'unit_vector_long_axis': U,
            'unit_vector_short_axis': V,
        }
        all_frame_data[frame] = frame_info
    return all_frame_data


#%%
path ='/data/projects/ma-alieksev-cine-knee-dynamics/data/Koehler^Paul/2021-07-30/218_MK_Radial_Weight_CINE_60bpm_MS_Weight_CGA/data_aw3_down_8-23deg.nii'
path1 = '/data/projects/ma-alieksev-cine-knee-dynamics/data/Maggioni^Marta_Brigid/2021-06-25/235_MK_Radial_Weight_CINE_60bpm_MS_NoWeight/data_aw3_down_3_to_25deg.nii'
CINE_NormalRes_2 = '/data/projects/ma-nepal-segmentation/data/CINE_NormalRes_2.nii'
CINE_NormalRes_3_nowt = '/data/projects/ma-nepal-segmentation/data/CINE_NormalRes_3_NoWeight.nii'
CINE_NormalRes_3_wt = '/data/projects/ma-nepal-segmentation/data/CINE_NormalRes_3_Weight.nii'
#%%
#load the data and open napari viewer 
pixelarray = open_nii(CINE_NormalRes_3_wt)
viewer = napari.view_image(pixelarray) 
#%%
#after drawing stuff on the gui, store the info in a layers variable
layers = viewer.layers
shapes = layers['Shapes']
points = layers['Points'] 
#%%
#run three functions that are defined elsewhere. 
e_center = ellipse_centers_from_shapes_layer(shapes)
u_vectors, v_vectors = unit_vectors(e_center)
midpoints = calculate_midpoints(points.data)
#%%
tibia_info_CINE_NormalRes_3_wt = get_coordinate_axes(e_center,u_vectors,v_vectors, midpoints) 

#%%
show_stuff(tibia_info_CINE_NormalRes_3_wt, 'tibia_info_CINE_NormalRes_3_wt' )                      
#%%
def calculate_angle(vector_a, vector_b):
    """Calculate angle in degrees between two vectors."""
    cos_theta = np.dot(vector_a, vector_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Assuming femur_info and tibia_info have the same keys (frames)
def plot_angle_vs_frame(femur_info , tibia_info, weight):
    frames = sorted(femur_info.keys())
    angles = []
    
    for frame in frames:
        femur_vector = femur_info[frame]['unit_vector_short_axis']
        tibia_vector = tibia_info[frame]['unit_vector_short_axis']
        angle = calculate_angle(femur_vector, tibia_vector)
        angles.append(angle)
    
    # Plot
    plt.plot(frames, angles, label=f'{weight}')
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Change in Angle Between Femur and Tibia Over Frames")
    plt.grid(True)
    plt.legend()
    plt.show()
    
#%%
plot_angle_vs_frame(femur_info_CINE_NormalRes_3_nowt,  tibia_info_CINE_NormalRes_3_nowt, 'No_weight')
#%%
plot_angle_vs_frame(femur_info_CINE_NormalRes_3_wt,  tibia_info_CINE_NormalRes_3_wt, 'Weight')
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
#%%
track_origin(tibia_info_CINE_NormalRes_3_nowt, 'Tibia')
