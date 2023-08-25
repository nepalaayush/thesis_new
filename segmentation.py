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



def find_new_edges(centroid, shape_coords, unit_vector_perpendicular):
    # Create an array to store the projections and original coordinates
    projections_with_indices = []
    
    for i, point in enumerate(shape_coords):
        # Use point[1:] to get rid of the extra dimension
        actual_point = point[1:]
        
        # Calculate the vector from the centroid to the current point
        vector_to_point = actual_point - centroid
        
        # Find the scalar projection onto the unit_vector_perpendicular
        t = np.dot(vector_to_point, unit_vector_perpendicular)
        
        # Find the closest point on the line defined by the centroid and unit_vector_perpendicular
        closest_point_on_line = centroid + t * unit_vector_perpendicular
        
        # Store the projection value and index
        distance_along_line = np.dot(unit_vector_perpendicular, (closest_point_on_line - centroid))
        projections_with_indices.append((distance_along_line, i))
    
    # Sort the projections and get the indices of the furthest apart projections
    sorted_projections = sorted(projections_with_indices, key=lambda x: x[0])
    min_point_index = sorted_projections[0][1]
    max_point_index = sorted_projections[-1][1]
    
    # Retrieve the coordinates of the points with the minimum and maximum projections
    min_point = shape_coords[min_point_index, 1:]
    max_point = shape_coords[max_point_index, 1:]
    
    return max_point, min_point

from shapely.geometry import LineString, Point

def find_edges_new(centroid, shape_coords, unit_vector_perpendicular):
    # Convert shape coordinates to LineString for easier intersection checks
    shape_line = LineString(shape_coords[:, 1:])
    
    # Define a sufficiently long line segment that is perpendicular to the PCA axis
    # and passes through the centroid.
    point_A = centroid - 1000 * unit_vector_perpendicular  # A point far away from the centroid along -V
    point_B = centroid + 1000 * unit_vector_perpendicular  # A point far away from the centroid along V
    perp_line = LineString([point_A, point_B])
    
    # Find the intersection points between shape_line and perp_line
    intersection = shape_line.intersection(perp_line)
    
    if intersection.geom_type == 'MultiPoint':
        # If there are multiple intersection points, convert them to a numpy array
        return np.array([list(pt.coords)[0] for pt in intersection.geoms])
    elif intersection.geom_type == 'Point':
        # If there's only one intersection point (unlikely, but possible), return it as a single-row numpy array
        return np.array([list(intersection.coords)[0]])
    else:
        # If there's no intersection, or if the intersection is more complex (e.g., a LineString),
        # this would require special handling.
        return None



from shapely.geometry import LineString, MultiPoint

def find_edges_nnew(U1, U2, V, shape_coords, num_points=100):
    # Parameterize long axis by the points U1 and U2
    t_values = np.linspace(0, 1, num_points)
    long_axis_points = np.array([(1-t)*U1 + t*U2 for t in t_values])

    widest_distance = 0
    widest_points = None

    for point in long_axis_points:
        # Create line segment perpendicular to the long axis
        perp_line = LineString([point - 50 * V, point + 50 * V])

        # Convert shape coordinates to LineString
        shape_line = LineString(shape_coords[:, 1:])

        # Find intersection
        intersection = perp_line.intersection(shape_line)

        if isinstance(intersection, MultiPoint):
            points_list = intersection.geoms
            if len(points_list) >= 2:
                dist = points_list[0].distance(points_list[-1])
                if dist > widest_distance:
                    widest_distance = dist
                    widest_points = [list(points_list[0].coords)[0], list(points_list[-1].coords)[0]]

    return np.array(widest_points)


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
        
        
        # Debug Check 1: Check if U and V are perpendicular
        is_perpendicular_uv = np.abs(np.dot(U, V)) < 1e-5
        if not is_perpendicular_uv:
           print(f"Debug Check 1: For shape {idx}, U and V are not perpendicular.")
        # Compute centroid
        centroid = np.mean(shape_coords[:, 1:], axis=0)
        
        # Find extreme points
        #extreme_points = np.array(find_edges_new(centroid, shape_coords, V))
        extreme_points = np.array(find_edges_nnew(line_points[0], line_points[1], V, shape_coords, num_points=100))
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



# Sample usage (assuming you have centroid, shape_coords, and unit_vector_perpendicular)
# max_point, min_point = find_edges(centroid, shape_coords, unit_vector_perpendicular)

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
viewer = open_napari()


#%%

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
show_stuff(tib_info, 'tib_wt_30bpm')
#%%            
show_stuff(fem_info, 'fem_wt_30bpm')             
                                       
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
track_origin(tib_info, 'tibia_weight')
#%%
plot_angle_vs_frame(fem_info, tib_info, 'weight')
#%%
def plot_angle_vs_angle(femur_info , tibia_info, angle_list,  label):
    frames = sorted(femur_info.keys())
    angles = []
    
    for frame in frames:
        femur_vector = femur_info[frame]['V']
        tibia_vector = tibia_info[frame]['V']
        angle = calculate_angle(femur_vector, tibia_vector)
        angles.append(angle)
    angles = (180 - np.array(angles) ) 
    # Plot
    plt.plot(angle_list, angles, label=f'{label}')
    plt.xlabel("Angle from the device (degrees) ")
    plt.ylabel("Angle Between bone (degrees)")
    plt.title("Change in Angle Between Femur and Tibia Over Device angle")
    plt.grid(True)
    plt.legend()
    plt.show()

#%%
angle_list = np.arange(5,28)
plot_angle_vs_angle(fem_info, tib_info, angle_list, 'weight')
