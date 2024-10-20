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
viewer = napari.view_image(full_image,  name='ds1_NW_full')

#%%
# add the 4d image to a new viewer
viewer = napari.Viewer() 

#%%
# Step 4: find the best suitable low and high range for edge detection and also sigma

from skimage.feature import canny

def apply_canny_multiple_parameters(pixelarray, low_range, high_range, sigma_range, num_steps):
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_steps)
    
    # Initialize a 4D array to store results (num_steps x num_steps x height x width)
    canny_multi_edge = np.zeros((num_steps, num_steps, *pixelarray[0].shape), dtype=bool)
    
    for i, sigma in enumerate(sigma_values):
        for j, (low, high) in enumerate(zip(low_values, high_values)):
            canny_edge = canny(pixelarray[0], low_threshold=low, high_threshold=high, sigma=sigma)
            canny_multi_edge[i, j] = canny_edge
    
    return canny_multi_edge

# Example usage
low_range = (0, 5)
high_range = (5, 10)
sigma_range = (0.5, 3)  # Define the range for sigma values
num_steps = 25

print("Low threshold values:", np.linspace(low_range[0], low_range[1], num_steps))
print("High threshold values:", np.linspace(high_range[0], high_range[1], num_steps))
print("Sigma values:", np.linspace(sigma_range[0], sigma_range[1], num_steps))

start_time = time.time()
canny_multi_edge = apply_canny_multiple_parameters(full_image, low_range, high_range, sigma_range, num_steps)
end_time = time.time()

print(f"Elapsed Time: {end_time - start_time} seconds")

# Assuming you're using napari for visualization
viewer.add_image(canny_multi_edge, name='canny_multi_edge')

viewer.add_image(full_image[0], name='first_frame')
#%%
from PyQt5 import QtCore  # or from PySide2 import QtCore, depending on your setup

def apply_canny_multiple_parameters(pixelarray, low_range, high_range, sigma_range, num_steps, viewer):
    low_values = np.linspace(low_range[0], low_range[1], num_steps)
    high_values = np.linspace(high_range[0], high_range[1], num_steps)
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_steps)
    
    # Initialize a 4D array to store results (num_steps x num_steps x height x width)
    canny_multi_edge = np.zeros((num_steps, num_steps, *pixelarray[0].shape), dtype=bool)
    
    # Set up text overlay
    viewer.text_overlay.visible = True
    viewer.text_overlay.color = 'white'
    viewer.text_overlay.font_size = 12
    viewer.text_overlay.position = 'top_left'
    
    # Add a placeholder layer for the Canny edge detection result
    canny_layer = viewer.add_image(np.zeros_like(pixelarray[0], dtype=bool), name='canny_edge')
    
    for i, sigma in enumerate(sigma_values):
        for j, (low, high) in enumerate(zip(low_values, high_values)):
            canny_edge = canny(pixelarray[0], low_threshold=low, high_threshold=high, sigma=sigma)
            canny_multi_edge[i, j] = canny_edge
            
            # Update text overlay
            viewer.text_overlay.text = f'Sigma: {sigma:.2f}, Low: {low:.2f}, High: {high:.2f}'
            
            # Update the Canny edge layer
            canny_layer.data = canny_edge
            
            # Give napari a moment to update the display
            QtCore.QCoreApplication.processEvents()
    
    return canny_multi_edge

# Create the viewer
viewer = napari.Viewer()

# Example usage
low_range = (0, 5)
high_range = (5, 10)
sigma_range = (0.5, 3)
num_steps = 25

print("Low threshold values:", np.linspace(low_range[0], low_range[1], num_steps))
print("High threshold values:", np.linspace(high_range[0], high_range[1], num_steps))
print("Sigma values:", np.linspace(sigma_range[0], sigma_range[1], num_steps))

start_time = time.time()
canny_multi_edge = apply_canny_multiple_parameters(full_image, low_range, high_range, sigma_range, num_steps, viewer)
end_time = time.time()
print(f"Elapsed Time: {end_time - start_time} seconds")

# Add the final multi-edge image to the viewer
viewer.add_image(canny_multi_edge, name='canny_multi_edge')
viewer.add_image(full_image[0], name='first_frame')

# Connect the current_step event to update the text overlay
def update_text(event):
    time = viewer.dims.current_step[0]
    sigma, low, high = sigma_values[time], low_values[time], high_values[time]
    viewer.text_overlay.text = f'Sigma: {sigma:.2f}, Low: {low:.2f}, High: {high:.2f}'

viewer.dims.events.current_step.connect(update_text)

#%%
import numpy as np
from skimage.feature import canny
import napari
from PyQt5 import QtCore  # or from PySide2 import QtCore, depending on your setup
import time

class CannyParameters:
    def __init__(self, low_range, high_range, sigma_range, num_steps):
        self.low_values = np.linspace(low_range[0], low_range[1], num_steps)
        self.high_values = np.linspace(high_range[0], high_range[1], num_steps)
        self.sigma_values = np.linspace(sigma_range[0], sigma_range[1], num_steps)
        self.canny_multi_edge = None

def apply_canny_multiple_parameters(pixelarray, params, viewer):
    # Initialize a 4D array to store results (num_steps x num_steps x height x width)
    params.canny_multi_edge = np.zeros((len(params.sigma_values), len(params.low_values), *pixelarray[0].shape), dtype=bool)
    
    # Set up text overlay
    viewer.text_overlay.visible = True
    viewer.text_overlay.color = 'white'
    viewer.text_overlay.font_size = 24
    viewer.text_overlay.position = 'top_left'
    
    # Add a placeholder layer for the Canny edge detection result
    canny_layer = viewer.add_image(np.zeros_like(pixelarray[0], dtype=bool), name='canny_edge')
    
    for i, sigma in enumerate(params.sigma_values):
        for j, (low, high) in enumerate(zip(params.low_values, params.high_values)):
            canny_edge = canny(pixelarray[0], low_threshold=low, high_threshold=high, sigma=sigma)
            params.canny_multi_edge[i, j] = canny_edge
            
            # Update the Canny edge layer
            canny_layer.data = canny_edge
            
            # Update text overlay
            viewer.text_overlay.text = f'Sigma: {sigma:.2f}, Low: {low:.2f}, High: {high:.2f}'
            
            # Give napari a moment to update the display
            QtCore.QCoreApplication.processEvents()
    
    return params.canny_multi_edge

# Create the viewer
viewer = napari.Viewer()

# Example usage
low_range = (0, 5)
high_range = (5, 10)
sigma_range = (0.5, 3)
num_steps = 25

params = CannyParameters(low_range, high_range, sigma_range, num_steps)

print("Low threshold values:", params.low_values)
print("High threshold values:", params.high_values)
print("Sigma values:", params.sigma_values)

start_time = time.time()
canny_multi_edge = apply_canny_multiple_parameters(full_image, params, viewer)
end_time = time.time()
print(f"Elapsed Time: {end_time - start_time} seconds")

# Add the final multi-edge image to the viewer
multi_edge_layer = viewer.add_image(canny_multi_edge, name='canny_multi_edge')
viewer.add_image(full_image[0], name='first_frame')

# Function to update text overlay
def update_text(event):
    sigma_index, low_high_index = viewer.dims.current_step[:2]
    sigma = params.sigma_values[sigma_index]
    low = params.low_values[low_high_index]
    high = params.high_values[low_high_index]
    viewer.text_overlay.text = f'Sigma: {sigma:.2f}, Low: {low:.2f}, High: {high:.2f}'

# Connect the current_step event to update the text overlay
viewer.dims.events.current_step.connect(update_text)

# Initial update of text overlay
update_text(None)
#%%
#Step 5: pick the right index and add it to viewer
tib_canny = canny_multi_edge[8]
viewer.add_image(tib_canny, name='after_edge_detection_sigma_2')
#%%
# the steps that give us set R from set P 


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


def downsample_points(list_of_arrays, index=0, number=50, bone_type='tibia' ):
    zeroth_frame = sort_points_single_frame(list_of_arrays[index], bone_type)
    zeroth_nonadjusted = equidistant_points(zeroth_frame,number)
    zeroth_adjusted = adjust_downsampled_points(zeroth_nonadjusted, zeroth_frame)
    return zeroth_adjusted


#%%
# this new function directly uses the cubic spline 
def place_equidistant_points_on_spline(original_curve, n_points):
    # Fit cubic spline to original curve
    t = np.linspace(0, 1, len(original_curve))
    cs_x = CubicSpline(t, original_curve[:, 0])
    cs_y = CubicSpline(t, original_curve[:, 1])

    # Function to compute speed (magnitude of derivative)
    def speed(t):
        dx_dt = cs_x.derivative()(t)
        dy_dt = cs_y.derivative()(t)
        return np.sqrt(dx_dt**2 + dy_dt**2)

    # Compute cumulative arc length
    from scipy.integrate import quad
    arc_length = [0]
    num_samples = 1000
    t_samples = np.linspace(0, 1, num_samples)
    for i in range(1, num_samples):
        s, _ = quad(speed, t_samples[i-1], t_samples[i])
        arc_length.append(arc_length[-1] + s)
    total_length = arc_length[-1]

    # Desired arc lengths
    desired_lengths = np.linspace(0, total_length, n_points)

    # Find parameter t for each desired arc length
    t_values = []
    for s in desired_lengths:
        idx = np.searchsorted(arc_length, s)
        t1, t2 = t_samples[idx - 1], t_samples[idx]
        # Refine t using interpolation or root finding if necessary
        t_values.append((t1 + t2) / 2)

    # Evaluate spline at t_values
    points = np.column_stack((cs_x(t_values), cs_y(t_values)))
    return points
#%%

ref_spline = place_equidistant_points_on_spline(tib_coords[0], 80)

reference_frame_first = downsample_points(tib_coords, 0, 80, bone_type='tibia')
#%%

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

tib_coords[0] = reference_frame_first # here we are replacing the binary edge of first frame with our downsampled and sorted set. 

transformation_matrices_first, giant_list_first, cost_values_first = combined_consecutive_transform(tib_coords) # new_tib_coords_first
