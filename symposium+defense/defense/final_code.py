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
