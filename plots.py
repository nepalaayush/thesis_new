# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:32:54 2024

@author: Aayush
"""
import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')


import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%

with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/new_analysis_all/JL/try_JL_NW_fem_info_s.pkl', 'rb') as file:
    JL_NW_fem_info_s = pickle.load(file)    
    
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/new_analysis_all/JL/JL_NW_tib_info_s.pkl', 'rb') as file:
    JL_NW_tib_info_s = pickle.load(file)
#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/new_analysis_all/JL/JL_W_tib_info_s.pkl', 'rb') as file:
    JL_W_tib_info_s = pickle.load(file)    
#%%
with open('C:/Users/Aayush/Documents/thesis_files/thesis_new/manual_segmentation/master_df_inverted.pkl', 'rb') as file:
    master_df_inverted = pickle.load(file)
#%%
def calculate_angle_between_bones(bone1, bone2, axis='long'):
    """
    Calculate the angle between two bones based on their coordinate system.

    :param bone1: Dictionary with bone data (femur or tibia).
    :param bone2: Dictionary with bone data (femur or tibia).
    :param axis: 'long' or 'short' to choose which axis to compare.
    :return: Angle in degrees between the two bones.
    """
    def get_axis_vector(bone, axis_type):
        """Extract the vector for the specified axis from the bone data."""
        points = bone[f'points_{axis_type}_axis']
        return points[1] - points[0]  # Vector from first point to second point

    # Get vectors for the specified axis
    vector1 = get_axis_vector(bone1, axis)
    vector2 = get_axis_vector(bone2, axis)

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the angle in radians and then convert to degrees
    #using arctan to see if we get negative values and a continuous curve or not 
    angle_radians = np.arctan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))
    #angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))
    angle_degrees = np.degrees(angle_radians)
    #if angle_degrees > 90:
     #   angle_degrees =  180 - angle_degrees

    return angle_degrees

#%%

def calculate_and_plot_angles_between_bones(bone1, bone2, axis='long', name='', new_figure=True):
    """
    Calculate the angles between two bones for each frame and plot the angles.

    :param bone1: Dictionary with bone data (femur or tibia) for multiple frames.
    :param bone2: Dictionary with bone data (femur or tibia) for multiple frames.
    :param axis: 'long' or 'short' to choose which axis to compare.
    """
    angles = []
    frames = []
    
    for frame in bone1.keys():
        if frame in bone2:
            angle = calculate_angle_between_bones(bone1[frame], bone2[frame], axis)
            print(f"Frame {frame}: Angle = {angle} degrees")
            angles.append(angle)
            frames.append(frame)
    #if new_figure:
     #   plt.figure(figsize=(10, 6))
    #plt.figure(figsize=(10, 6))
    plt.plot(frames, angles, marker='o', label=name)
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.title(f'Angle between Bones over Frames ({axis.capitalize()} Axis)')
    plt.grid(True)
    plt.legend()
    if new_figure:
        plt.savefig(f'{name}_Angle_betwn_long_axes.svg')
    

    return np.array(angles)

#%%
dataset_7_NW_angles = calculate_and_plot_angles_between_bones(JL_NW_fem_info_s, JL_NW_tib_info_s, name='dataset_7_NW')


#%%
''' Maybe this is used to actually create the dataframe    ''' 
# Initialize a list to store each row of the DataFrame
data = []

# Define the range of your datasets and conditions
num_datasets = 7
conditions = ['NW', 'W']  # 'NW' for No Weight, 'W' for Weight

# Iterate over each dataset number and condition
for i in range(1, num_datasets + 1):
    for cond in conditions:
        var_name = f'dataset_{i}_{cond}_angles'
        # Access the variable by name
        if var_name in globals():
            angles = globals()[var_name]
            # Append each angle with its corresponding metadata to the data list
            for idx, angle in enumerate(angles):
                data.append({
                    'frame': idx + 1,
                    'dataset': i,
                    'condition': 'No Weight' if cond == 'NW' else 'Weight',
                    'angle': angle
                })
        else:
            print(f'Variable {var_name} not found.')

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
#%%


def tib_relative_to_fem(tib_array, fem_array):
    return tib_array - fem_array 
''' this code attempts to streamline the process that is shown below  ''' 
voxel_size = [0.7272727272727273 / 1.5, 0.7272727272727273 / 1.5]
def calc_translations(all_frame_info, point_name, label, plot_type, voxel_size):
    ''' the name of the function is a misnomer.. we are not plotting anything. just extracting the translations  ''' 
    sorted_frames = sorted(all_frame_info)
    y_coords = [all_frame_info[frame][point_name][0] for frame in sorted_frames]
    x_coords = [all_frame_info[frame][point_name][1] for frame in sorted_frames]
    translations_mm = []
    
    if plot_type == 'AP':
        ap_translations = [x - x_coords[0] for x in x_coords]
        translations_mm = [ap * voxel_size[1] for ap in ap_translations]
    elif plot_type == 'IS':
        is_translations = [y_coords[0] - y for y in y_coords]
        translations_mm = [is_ * voxel_size[0] for is_ in is_translations]
    
    # Creating a DataFrame from the translations
    data = {
        'Frame Number': sorted_frames,
        'Translations': translations_mm,
        'Type': [plot_type] * len(translations_mm),
        'Label': [label] * len(translations_mm)
    }
    return pd.DataFrame(data)


def compile_translations(fem_loaded, fem_unloaded, tib_loaded, tib_unloaded, voxel_size):
    # Initialize an empty DataFrame to hold all compiled translations
    master_df = pd.DataFrame()
    
    # Dictionary to hold the data and labels
    datasets = {
        'Femur Loaded': (fem_loaded, 'loaded'),
        'Femur Unloaded': (fem_unloaded, 'unloaded'),
        'Tibia Loaded': (tib_loaded, 'loaded'),
        'Tibia Unloaded': (tib_unloaded, 'unloaded')
    }
    
    # Iterate through the datasets and conditions to populate the DataFrame
    for label, (data, load_condition) in datasets.items():
        for plot_type in ['IS', 'AP']:
            df = calc_translations(data, 'centroid', load_condition, plot_type, voxel_size)
            df['Body Part'] = label.split()[0]  # Add body part (Femur/Tibia) as a column
            df['Condition'] = label.split()[1]  # Add condition (Loaded/Unloaded) as a column
            master_df = pd.concat([master_df, df], ignore_index=True)
    
    # Remove the 'Label' column as it's redundant with 'Condition'
    master_df = master_df.drop(columns=['Label'])
    
    # Function to calculate angles for each frame and return as DataFrame
    def calculate_angles_dataframe(femur_data, tibia_data, condition):
        angles = []
        frames = []
        
        for frame in sorted(femur_data.keys()):
            if frame in tibia_data:
                angle = calculate_angle_between_bones(femur_data[frame], tibia_data[frame], axis='long')
                angles.append(angle)
                frames.append(frame)
        
        return pd.DataFrame({
            'Frame Number': frames,
            'Translations': [np.nan] * len(angles),  # NA for Translations as it's for angles
            'Type': [np.nan] * len(angles),  # NA for Type
            'Body Part': [np.nan] * len(angles),  # NA for Body Part
            'Condition': [condition] * len(angles),
            'Angles': angles
        })

    # Calculate angles and append to the master DataFrame
    angles_loaded_df = calculate_angles_dataframe(fem_loaded, tib_loaded, 'Loaded')
    angles_unloaded_df = calculate_angles_dataframe(fem_unloaded, tib_unloaded, 'Unloaded')
    angles_df = pd.concat([angles_loaded_df, angles_unloaded_df], ignore_index=True)

    master_df = pd.concat([master_df, angles_df], ignore_index=True)
    
    return master_df



JL_master_df_no_if= compile_translations(JL_W_fem_info_s, JL_NW_fem_info_s, JL_W_tib_info_s, JL_NW_tib_info_s, voxel_size)
#%%

# when adding new dataframes: do this: 

HS_master_df['Dataset'] = '6'
JL_master_df['Dataset'] = '7'


#%%
master_df_1_7 = pd.concat([master_df_1_5, HS_master_df, JL_master_df], ignore_index=True)
#%%
# Filter the master DataFrame to get Tibia IS translations for both loaded and unloaded conditions
tibia_is_translations = master_df.loc[(master_df['Type'] == 'IS') & 
                                      (master_df['Body Part'] == 'Tibia')]

# Use Seaborn to plot
sns.lineplot(data=tibia_is_translations, x='Frame Number', y='Translations', hue='Condition')

# Adding labels and title
plt.xlabel('Frame Number')
plt.ylabel('Translations (mm)')
plt.title('Tibia IS Translations for Loaded and Unloaded Conditions')

plt.show()
#%%
filtered_df = combined_df[(combined_df['Type'] == 'IS') & (combined_df['Condition'] == 'Unloaded') & (combined_df['Body Part'] == 'Tibia')]

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_df, x='Frame Number', y='Translations', hue='Dataset', ci=None)

plt.title('IS Translations vs. Frame Number for Unloaded Condition')
plt.xlabel('Frame Number')
plt.ylabel('IS Translations')
plt.legend(title='Dataset')
plt.show()

#%%
''' A realistion that the relative translsations and the angles operate on the same columns .. so best to split them apart.  '''
# doing it for angles first, removing it from the combined df : 
angle_df = master_df[['Frame Number', 'Condition', 'Dataset', 'Angles']].copy()    
angle_df = angle_df.dropna(subset=['Angles'])
# that works 

#%%
''' the same as above, that is, to obtain the relative translations from the master df, but without using for loops ''' 
# Convert relevant columns to category types for efficiency
master_df_1_5['Dataset'] = master_df_1_5['Dataset'].astype('category')
master_df_1_5['Condition'] = master_df_1_5['Condition'].astype('category')
master_df_1_5['Type'] = master_df_1_5['Type'].astype('category')

# Separate Tibia and Femur translations
tibia_df = master_df_1_5[master_df_1_5['Body Part'] == 'Tibia'].copy()
femur_df = master_df_1_5[master_df_1_5['Body Part'] == 'Femur'].copy()
print(tibia_df.duplicated(subset=['Frame Number', 'Dataset', 'Condition', 'Type']).sum())

# Rename 'Translations' column to avoid conflict during merge
tibia_df.rename(columns={'Translations': 'Tibia_Trans'}, inplace=True)
femur_df.rename(columns={'Translations': 'Femur_Trans'}, inplace=True)

# Merge Tibia and Femur dataframes
merged_df = pd.merge(tibia_df, femur_df, on=['Frame Number', 'Dataset', 'Condition', 'Type'], suffixes=('_tibia', '_femur'))

# Calculate relative translation
merged_df['Relative Translation'] = merged_df['Tibia_Trans'] - merged_df['Femur_Trans']

# Select relevant columns
rel_trans_df = merged_df[['Frame Number', 'Dataset', 'Condition', 'Type', 'Relative Translation']]

#%%


# creating a function out of the code, also tryingf to include the angles 
def calculate_relative_translation_and_angle(master_df):
    # Convert relevant columns to category types for efficiency
    master_df['Dataset'] = master_df['Dataset'].astype('category')
    master_df['Condition'] = master_df['Condition'].astype('category')
    master_df['Type'] = master_df['Type'].astype('category')

    # Separate Tibia and Femur translations
    tibia_df = master_df[master_df['Body Part'] == 'Tibia'].copy()
    femur_df = master_df[master_df['Body Part'] == 'Femur'].copy()

    # Rename 'Translations' column to avoid conflict during merge
    tibia_df.rename(columns={'Translations': 'Tibia_Trans'}, inplace=True)
    femur_df.rename(columns={'Translations': 'Femur_Trans'}, inplace=True)
    
    print(tibia_df.duplicated(subset=['Frame Number', 'Dataset', 'Condition', 'Type']).sum())
    print(femur_df.duplicated(subset=['Frame Number', 'Dataset', 'Condition', 'Type']).sum())

    # Merge Tibia and Femur dataframes
    merged_df = pd.merge(tibia_df, femur_df, on=['Frame Number', 'Dataset', 'Condition', 'Type'], suffixes=('_tibia', '_femur'))

    # Calculate relative translation
    merged_df['Relative Translation'] = merged_df['Tibia_Trans'] - merged_df['Femur_Trans']

    # Select relevant columns for relative translation
    rel_trans_df = merged_df[['Frame Number', 'Dataset', 'Condition', 'Type', 'Relative Translation']]
    
    is_df = rel_trans_df[rel_trans_df['Type'] == 'IS']
    ap_df = rel_trans_df[rel_trans_df['Type'] == 'AP']
    
    is_df = add_percent_flexed(is_df)
    ap_df = add_percent_flexed(ap_df)

    angle_df = master_df[['Frame Number', 'Dataset', 'Condition', 'Angles']].dropna(subset=['Angles']).copy()
    
    is_df = pd.merge(is_df, angle_df, on=['Frame Number', 'Dataset', 'Condition'], how='left')
    ap_df = pd.merge(ap_df, angle_df, on=['Frame Number', 'Dataset', 'Condition'], how='left')

    
    return is_df, ap_df 
#%%
is_df_1_7 , ap_df_1_7  = calculate_relative_translation_and_angle(master_df_1_7)



#%%

ap_df_1_7 = add_percent_flexed(ap_df_1_7)


#%%
import pickle
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/angle_datasets/df_angle_bin_all.pkl', 'rb') as file:
    df_angle_bin_all =  pickle.load(file)
    
#%%
# Plotting
plt.figure(figsize=(10, 6))
#sns.scatterplot(data= angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Loaded')], x='Angles', y='Relative Translation', hue='Dataset')
#sns.regplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Loaded')], x='Angles', y='Relative Translation', scatter=False)
sns.lmplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Unloaded') & (angle_and_rel_df['Type'] == 'AP')], x='Angles', y='Relative Translation', hue='Dataset', ci=None)



plt.title('Translation vs. Angles for Unloaded Condition')
plt.xlabel('Angle (Degree)')
plt.ylabel('Translation (mm)')
#plt.axhline(0, color='r', linestyle='--')
plt.legend(title='Dataset')
plt.show()

#%%
frames_to_remove = [16, 17, 15, 18, 14, 19]

# Filter out the frames only if they are in Dataset 2 or Dataset 3 and are in the frames_to_remove list.
equal_frame_df = angle_and_rel_df[
    ~(angle_and_rel_df['Frame Number'].isin(frames_to_remove) & angle_and_rel_df['Dataset'].isin(['Dataset 2', 'Dataset 3']))
].reset_index(drop=True)

#%%
equal_frame_df['Frame Number'] = equal_frame_df.groupby(['Dataset', 'Condition']).cumcount()
#%%
import matplotlib.pyplot as plt

# Set up the subplot figure
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

conditions = equal_frame_df['Condition'].unique()

for index, condition in enumerate(conditions):
    # Filter the data for the current condition
    condition_df = equal_frame_df[equal_frame_df['Condition'] == condition]

    # Group by 'Frame Number' and calculate the mean and standard deviation of 'Relative Translation'
    stats_df = condition_df.groupby('Angles')['Relative Translation'].agg(['mean', 'std']).reset_index()

    # Plotting the mean Relative Translation
    axs[index].plot(stats_df['Angles'], stats_df['mean'], marker='o', label=f'Mean {condition}')

    # Adding the shaded error (standard deviation) area
    axs[index].fill_between(stats_df['Angles'], 
                            stats_df['mean'] - stats_df['std'], 
                            stats_df['mean'] + stats_df['std'], 
                            color='gray', alpha=0.2, label=f'Standard Deviation {condition}')

    axs[index].set_title(f'Mean Relative Translation vs. Angles ({condition})')
    axs[index].set_xlabel('Angles')
    axs[index].set_ylabel('Mean Relative Translation')
    axs[index].grid(True)
    axs[index].legend()

# Adjust the layout
plt.tight_layout()
plt.show()

#%%

with open('df_angle.pkl', 'wb') as f:
    pickle.dump(df_angle, f)   

#%%
# this code plots the average relative translation for IS for UNloaded .. along with a weird sd shading.. (because not all datasets have the same angle)

sns.lineplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Unloaded') & (angle_and_rel_df['Type'] =='IS')], x='Percent Flexed', y='Relative Translation')
#%%



''' trying to get some k space info:  '''
k_JL_actual = path_to_image ( '/data/projects/ma-nepal-segmentation/data/LIM^JONG_CHAN/2024-04-05/288_MK_Radial_NW_CINE_30bpm_CGA/JL_NW_aw2_ai2_zf_1.nii') 
#%%
real_frame = k_JL_actual[10]
#%%
# just checking how the kspace looks like, which is most probably just the magnitude info: 
# apparantly we need to do fftshift first, because it looks like our kspace is centered .. 

shifted_k = np.fft.ifftshift(k_JL)
image_reconstructed = np.fft.ifft2(shifted_k, axes=(-2,-1))  # Take the real part
image_real = image_reconstructed.real 

# doing this directly to the kspace array did not yield anything .. so could not see the spatial domain .. so trying to combine it below : 
    
#%%

k_space_complex = k_JL * (np.exp(1j * k_JL_phase))
image_reconstructed = np.fft.ifft2(k_space_complex, axes=(-2, -1))


image_absolute = np.abs(image_reconstructed)

np.power(k_frame, 0.2) # adjusting the gamma for visualization purposes 
#%%

fig, ax = plt.subplots(figsize=(10, 10))

# Apply gamma correction and display the image
ax.imshow(real_frame, cmap='gray', interpolation='none')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

# Remove the axes frame as well
ax.axis('off')

# Save the plot with a transparent background
plt.savefig('real_frame.svg', format='svg', bbox_inches='tight', transparent=True)
plt.close()  # Close the plotting window


#%% 

# trying to plot some golden angle 

import matplotlib.pyplot as plt
import numpy as np

def plot_circle_with_spokes(n_spokes):
    # Define the golden angle in radians
    golden_angle = 2 * np.pi * (1 - (1 / ((1 + np.sqrt(5)) / 2)))

    # Create a figure and a single subplot
    fig, ax = plt.subplots()

    # Draw a circle
    circle = plt.Circle((0, 0), 1, edgecolor='b', facecolor='none')
    ax.add_artist(circle)

    # Calculate the positions of each spoke end using the golden angle
    for i in range(n_spokes):
        angle = golden_angle * i
        x = np.cos(angle)
        y = np.sin(angle)
        ax.plot([0, x], [0, y], 'r')  # Red lines for the spokes

    # Set limits and equal aspect ratio to ensure the circle is not distorted
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    # Remove axes for aesthetic reasons
    ax.axis('off')

    # Show the plot
    plt.show()

# Example usage
plot_circle_with_spokes(50)  # Change the number of spokes here
#%%
def plot_radial_kspace(n_spokes, points_per_spoke):
    # Define the golden angle in radians
    golden_angle = 2 * np.pi * (1 - (1 / ((1 + np.sqrt(5)) / 2)))

    # Create a figure and a single subplot
    fig, ax = plt.subplots()

    # Draw a circle
    circle = plt.Circle((0, 0), 1, edgecolor='b', facecolor='none')
    ax.add_artist(circle)

    # Calculate the positions of each spoke end using the golden angle
    for i in range(n_spokes):
        angle = golden_angle * i
        x_end = np.cos(angle)
        y_end = np.sin(angle)

        # Plot the spoke
        ax.plot([0, x_end], [0, y_end], 'r')  # Red lines for the spokes

        # Plot points along the spoke
        for j in range(1, points_per_spoke + 1):
            x_point = np.cos(angle) * (j / points_per_spoke)
            y_point = np.sin(angle) * (j / points_per_spoke)
            ax.plot(x_point, y_point, 'go', markersize=2)  # Green points

    # Set limits and equal aspect ratio to ensure the circle is not distorted
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    # Remove axes for aesthetic reasons
    ax.axis('off')

    # Show the plot
    plt.show()

# Example usage: 276 spokes and 352 points per spoke
plot_radial_kspace(50, 10)

#%%
def plot_radial_kspace_single_spoke(n_spokes, points_per_spoke):
    # Define the golden angle in radians
    golden_angle = 2 * np.pi * (1 - (1 / ((1 + np.sqrt(5)) / 2)))

    # Create a figure and a single subplot
    fig, ax = plt.subplots()

    # Draw a circle
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none')
    ax.add_artist(circle)

    # Calculate the positions of each spoke end using the golden angle
    for i in range(n_spokes):
        angle = golden_angle * i
        x_end = np.cos(angle)
        y_end = np.sin(angle)

        # Plot the spoke
        ax.plot([0, x_end], [0, y_end], 'black')  # black lines for the spokes

        # Plot points along the spoke, only on the first spoke
        if i == 0:  # Only plot points on the first spoke
            for j in range(1, points_per_spoke + 1):
                x_point = np.cos(angle) * (j / points_per_spoke)
                y_point = np.sin(angle) * (j / points_per_spoke)
                ax.plot(x_point, y_point, 'ko', markersize=2)  # Green points

    # Set limits and equal aspect ratio to ensure the circle is not distorted
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    # Remove axes for aesthetic reasons
    ax.axis('off')

    # Show the plot
    plt.show()

# Example usage: 276 spokes and 352 points per spoke
plot_radial_kspace_single_spoke(25, 10)


#%%
def plot_radial_kspace(n_spokes, points_per_spoke, max_radius=1):
    # Define the golden angle in radians
    golden_angle = 2 * np.pi * (1 - (1 / ((1 + np.sqrt(5)) / 2)))

    # Create a figure and a single subplot
    fig, ax = plt.subplots()

    # Draw a circle
    circle = plt.Circle((0, 0), max_radius, edgecolor='black', facecolor='none')
    ax.add_artist(circle)

    # Calculate the positions of each spoke end using the golden angle
    for i in range(n_spokes):
        angle = golden_angle * i
        x_end = np.cos(angle) * max_radius
        y_end = np.sin(angle) * max_radius

        # Plot the spoke
        ax.plot([0, x_end], [0, y_end], 'black', linewidth=0.5)  # Black lines for the spokes

        # Plot points along the spoke
        for j in range(1, points_per_spoke + 1):
            radius = (j / points_per_spoke) * max_radius
            x_point = np.cos(angle) * radius
            y_point = np.sin(angle) * radius
            ax.plot(x_point, y_point, 'ko', markersize=2)  # Black points

    # Set limits and equal aspect ratio to ensure the circle is not distorted
    ax.set_xlim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_ylim(-1.1 * max_radius, 1.1 * max_radius)
    ax.set_aspect('equal')

    # Remove axes for aesthetic reasons
    ax.axis('off')

    # Show the plot
    plt.show()

# Example usage: fewer spokes and points to illustrate the concept
plot_radial_kspace(30, 20, max_radius=1)


#%%

''' trying to flip the angle curve:  '''
# Extracting subset based on Dataset and Condition
df_1_W_angles = df_angle[(df_angle['Dataset'] == 1) & (df_angle['Condition'] == 'Weight')]['angle'].reset_index(drop=True)

# Calculating the halfway point
half = len(df_1_W_angles) // 2

# Splitting the data into first half and second half
first_half = df_1_W_angles.iloc[:half]
second_half = df_1_W_angles.iloc[half:]

# Finding the index of the maximum value in the first half
max_index_first_half = first_half.idxmax()

for i in range(max_index_first_half + 1 , len(first_half) ):
    first_half[i] +=  2 * ( first_half[max_index_first_half] - first_half[i] )

# Finding the index of the maximum value in the first half
max_index_second_half = second_half.idxmax()

for i in range( half, max_index_second_half ):
    second_half[i] += 2 * ( second_half[max_index_second_half] - second_half[i] )
    
modified_angles = pd.concat([first_half, second_half], ignore_index=True)    

#%%
def flip_angle_curve(df, dataset, condition):
    # Extracting subset based on Dataset and Condition
    subset_angles = df[(df['Dataset'] == dataset) & (df['Condition'] == condition)]['angle'].reset_index(drop=True)

    # Calculating the halfway point
    half = len(subset_angles) // 2

    # Splitting the data into first half and second half
    first_half = subset_angles.iloc[:half].copy()
    second_half = subset_angles.iloc[half:].copy()

    # Finding the index of the maximum value in the first half
    max_index_first_half = first_half.idxmax()

    for i in range(max_index_first_half + 1, len(first_half)):
        first_half[i] += 2 * (first_half[max_index_first_half] - first_half[i])

    # Finding the index of the maximum value in the second half
    max_index_second_half = second_half.idxmax()

    for i in range(half, max_index_second_half):
        second_half[i] += 2 * (second_half[max_index_second_half] - second_half[i])

    # Concatenating modified first half and second half
    modified_angles = pd.concat([first_half, second_half], ignore_index=True)
    return modified_angles


def apply_modification(df):
    unique_datasets = df['Dataset'].unique()
    unique_conditions = df['Condition'].unique() 
    
    result_df = df.copy()  # Create a copy of the dataframe to hold modified values
    
    for dataset in unique_datasets:
        for condition in unique_conditions:
            # Identifying the rows that meet the current dataset and condition
            condition_mask = (df['Dataset'] == dataset) & (df['Condition'] == condition)
            
            # Extracting the indexes of rows that meet the condition
            relevant_indexes = df.index[condition_mask]
            
            # Generate modified angles using the flip_angle_curve function
            modified_angles = flip_angle_curve(df, dataset, condition)
            
            # Set the index of modified_angles to match the indexes of the rows being replaced
            modified_angles.index = relevant_indexes
            
            # Assigning the modified angles back to the result DataFrame
            result_df.loc[condition_mask, 'angle'] = modified_angles
            
            print(modified_angles)  # Optional: Check output of modified angles
            
    return result_df


modified_angle_df = apply_modification(combined_df)

#%%
# to do the relative centroid which we are not using nowadays: 

# Calculate the initial position of the tibia relative to the femur
initial_relative_position = tib_centroids[0] - fem_centroids[0]

# Calculate the relative position for each frame
relative_positions = tib_centroids - fem_centroids

# Calculate the translations relative to the initial position
is_translation = relative_positions[:, 1] - initial_relative_position[1]
ap_translation = relative_positions[:, 2] - initial_relative_position[2]

#%%
# converting the code blocks above into a function. 


def process_shape_layers(viewer, dataset):
    """
    Process shape layers and calculate translations for a single dataset.
    
    Args:
    viewer: napari viewer object
    dataset: string, dataset name (e.g., 'ds1')
    
    Returns:
    pandas DataFrame with translations for the dataset, both methods
    """
    results = []
    shape_layers = {}

    # Identify shape layers for the given dataset
    for layer in viewer.layers:
        if layer.name.startswith(f"{dataset}_"):
            parts = layer.name.split('_')
            if len(parts) == 3:
                _, bone, method = parts
                shape_layers[(bone, method)] = layer

    # Process each shape layer
    for (bone, method), layer in shape_layers.items():
        centroids = []
        for frame in range(len(layer.data)):
            points = layer.data[frame][:, 1:]  # Exclude frame number, keep only y and x
            centroid = compute_centroid(points)
            centroids.append(np.array([frame, centroid[0], centroid[1]]))
        centroids = np.array(centroids)

        # Store centroids for later use
        shape_layers[(bone, method)] = centroids

    # Calculate translations for each method
    for method in ['man', 'auto']:
        if ('tib', method) in shape_layers and ('fem', method) in shape_layers:
            tib_centroids = shape_layers[('tib', method)]
            fem_centroids = shape_layers[('fem', method)]

            # Calculate the initial position of the tibia relative to the femur
            initial_relative_position = tib_centroids[0] - fem_centroids[0]
            
            # Calculate the relative position for each frame
            relative_positions = tib_centroids - fem_centroids
            
            # Calculate the translations relative to the initial position
            is_translation = relative_positions[:, 1] - initial_relative_position[1]
            ap_translation = relative_positions[:, 2] - initial_relative_position[2]
            
            # Store results
            for frame, is_trans, ap_trans in zip(range(len(is_translation)), is_translation, ap_translation):
                results.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Frame': frame,
                    'IS_Translation': is_trans,
                    'AP_Translation': ap_trans
                })
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    return df

#%%
df3 = process_shape_layers(viewer, 'ds3')


#%%
# to calculate the  translations of the centroid for the segments .. NW comaprision manual vs auto . 


# to save shapes layer to niftis 

ds5_tib_shape_auto = viewer.layers['ds5_tib_auto']
ds5_fem_shape_auto  = viewer.layers['ds5_fem_auto']

#%%
viewer.add_shapes(ds5_tib_shape_auto.data[0], shape_type='polygon', name='ds5_tib_man')
viewer.add_shapes(ds5_fem_shape_auto.data[0], shape_type='polygon', name='ds5_fem_man')

#%%


ds1_tib_label = ds1_tib_shape_auto.to_labels((34,528,528))

frame0_tib_image = ds1_tib_label[0]
#%%
# this is an attempt to get once and for all good way to get centroid from shape points: 

# starting off with just a single frame .. so no frame info. 

frame_0_points = ds1_tib_shape_auto.data[0][:,1:] # has shape (112,2)

centroid_mean = np.mean(frame_0_points, axis=0)


def calculate_centroid(points):
    # Ensure points is a numpy array
    points = np.array(points)
    
    # Calculate the area of the polygon
    x = points[:, 0]
    y = points[:, 1]
    #area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    # Calculate centroid coordinates
    cx = np.mean(x)
    cy = np.mean(y)
    
    return cx, cy

centroid_one = calculate_centroid(frame_0_points)
viewer.add_points(centroid_one)




# Example usage with a sample array
centroid_two = compute_centroid(frame_0_points)
viewer.add_points(centroid_two, face_color='blue')


centroid_three = get_robust_centroid(frame_0_points)
viewer.add_points(centroid_three, face_color='green')


from skimage import measure

centroid_four = measure.centroid(frame0_tib_image)

viewer.add_points(centroid_four, face_color='yellow')
#%%

ds1_tib_shape_auto = viewer.layers['ds1_tib_auto']
ds1_fem_shape_auto  = viewer.layers['ds1_fem_auto']

def get_centroids(shapes_layer_tib, shapes_layer_fem):
    # Calculate centroids for each frame
    tib_centroids = []
    fem_centroids = []
    
    for frame in range(len(shapes_layer_tib.data)):
        tib_points = shapes_layer_tib.data[frame][:, 1:]  # Exclude frame number, keep only y and x
        fem_points = shapes_layer_fem.data[frame][:, 1:]  # Exclude frame number, keep only y and x
        
        tib_centroid = compute_centroid(tib_points)
        fem_centroid = compute_centroid(fem_points)
        
        # Add frame number as the first dimension
        tib_centroids.append(np.array([frame, tib_centroid[0], tib_centroid[1]]))
        fem_centroids.append(np.array([frame, fem_centroid[0], fem_centroid[1]]))
        
    return np.array(tib_centroids) , np.array ( fem_centroids ) 

ds1_tib_auto_centroid , ds1_fem_auto_centroid = get_centroids(ds1_tib_shape_auto, ds1_fem_shape_auto)



#%%
# until the function to compute centroids, we need to normalize the lengths of all the shapes. 

viewer = napari.Viewer() 

#%%

from sklearn.decomposition import PCA
from scipy.spatial import distance


def cut_bone_shape(shapes_layer, bone_type, cutoff_distance=100):
    """
    Cut the bone shape along its principal axis in the row direction.
    
    :param shapes_layer: napari Shapes layer
    :param bone_type: 'tibia' or 'femur'
    :param cutoff_distance: distance along the principal axis to make the cut
    :return: new Shapes layer with cut shapes
    """
    new_shapes = []
    
    for frame in range(len(shapes_layer.data)):
        points = shapes_layer.data[frame][:, 1:]  # Exclude frame number, keep only y and x
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(points)
        
        # Get the principal axis
        principal_axis = pca.components_[0]
        
        # Ensure the principal axis points downwards for consistency
        if principal_axis[0] < 0:
            principal_axis = -principal_axis
        
        # Project points onto the principal axis
        projected = np.dot(points - np.mean(points, axis=0), principal_axis)
        
        if bone_type == 'tibia':
            # Tibia logic (unchanged)
            extreme_index = np.argmin(points[:, 0])  # Lowest row value
            extreme_point = points[extreme_index]
            extreme_projected = projected[extreme_index]
            
            cutoff_projected = extreme_projected + cutoff_distance
            
            filtered_points = [point for point, proj in zip(points, projected) if proj <= cutoff_projected]
        
        else:  # femur
            # Femur logic (corrected)
            extreme_index = np.argmax(points[:, 0])  # Highest row value
            extreme_point = points[extreme_index]
            extreme_projected = projected[extreme_index]
            
            cutoff_projected = extreme_projected - cutoff_distance
            
            filtered_points = [point for point, proj in zip(points, projected) if proj >= cutoff_projected]
        
        # Add frame number back to the filtered points
        filtered_points_with_frame = np.column_stack((np.full(len(filtered_points), frame), filtered_points))
        
        new_shapes.append(filtered_points_with_frame)
    
    # Create a new Shapes layer with the cut shapes
    new_layer_name = f"{shapes_layer.name}_cut"
    new_layer = viewer.add_shapes(new_shapes, name=new_layer_name, shape_type='polygon')
    
    return new_layer

# Example usage:
cut_tibia_layer = cut_bone_shape(viewer.layers['ds5_tib_auto'], 'tibia')
cut_femur_layer = cut_bone_shape(viewer.layers['ds5_fem_auto'], 'femur')

#%%
# lets do manual, the autos should be the same anyways. 

tib_points = viewer.layers['ds2_tib_man_cut'].data
fem_points = viewer.layers['ds2_fem_man_cut'].data

tib_centroids = []
fem_centroids = []
for frame in range(len(tib_points)):
    tib_points_frame = tib_points[frame][:, 1:]  # Exclude frame number, keep only y and x
    fem_points_frame = fem_points[frame][:, 1:]  # Exclude frame number, keep only y and x
    
    tib_centroid = compute_centroid(tib_points_frame)
    fem_centroid = compute_centroid(fem_points_frame)
    
    tib_centroids.append(tib_centroid)
    fem_centroids.append(fem_centroid)

viewer.add_points(tib_centroids, face_color='green')
viewer.add_points(fem_centroids, face_color='green')



#%%



# adaptiong this into a function - the one that does absolute distances, the one we'll use

def compute_centroid(points):
    # this isthe one that actually computes the correct centroid by using the points 
    # Ensure the polygon is closed by appending the first point at the end
    points = np.vstack([points, points[0]])
    
    # Separate the x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate the area of the polygon
    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    
    # Calculate the centroid coordinates
    C_x = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    C_y = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    
    return C_x, C_y


def get_centroids_from_shapes(layer_tib_name, layer_fem_name):
    # Extract layers based on provided names
    shapes_layer_tib = viewer.layers[layer_tib_name]
    shapes_layer_fem = viewer.layers[layer_fem_name]
    
    # Determine the dataset number from the layer name
    dataset_number = int(layer_tib_name.split('ds')[1][0])  # e.g. ds1 -> 1
    method = 'Auto' if 'auto' in layer_tib_name else 'Manual'
    
    # Calculate centroids and translations for each frame
    data = []
    
    for frame in range(len(shapes_layer_tib.data)):
        tib_points = shapes_layer_tib.data[frame][:, 1:]  # Exclude frame number, keep only y and x
        fem_points = shapes_layer_fem.data[frame][:, 1:]  # Exclude frame number, keep only y and x
        
        tib_centroid = compute_centroid(tib_points)
        fem_centroid = compute_centroid(fem_points)
        
        # AP and IS translations
        AP_translation = fem_centroid[1] - tib_centroid[1]  # AP (x-axis)
        IS_translation = fem_centroid[0] - tib_centroid[0]  # IS (y-axis)
        
        # Append the data for this frame
        data.append([dataset_number, frame, method, fem_centroid, tib_centroid, AP_translation, IS_translation])
    
    # Create a dataframe from the collected data
    df = pd.DataFrame(data, columns=[
        'Dataset', 'Frame', 'Method', 'Femur_Centroid', 'Tibia_Centroid', 'AP_Translation', 'IS_Translation'
    ])
    
    return df

# Example usage:
#ds5_centroid_man_cut = get_centroids_from_shapes('ds5_tib_man_cut', 'ds5_fem_man_cut')

ds5_centroid_auto_cut = get_centroids_from_shapes('ds5_tib_auto_cut', 'ds5_fem_auto_cut')

#%%
ds5_cut_df = pd.concat([ds5_centroid_auto_cut, ds5_centroid_man_cut], ignore_index=True)

#%%
# once a singe dataset has been combined for both man and auto. first step is to add percent flexed by first renaming the Frame column 
#%%
def add_percent_flexed(df):
    df = df.copy()  # To avoid modifying the original dataframe
    percent_flexed_values = {}  # To store first half values for mirroring

    for dataset_id, group_data in df.groupby('Dataset'):
        total_frames = group_data['Frame Number'].max() + 1  # Total number of frames
        # Define halfway_point for the calculation, not for mirroring
        halfway_calculation_point = (total_frames // 2) - 1

        # Calculate 'Percent Flexed' for each frame
        for index, row in group_data.iterrows():
            frame = row['Frame Number']
            if frame <= halfway_calculation_point:
                # Before or at the halfway calculation point, scale from -100% to 0%
                percent_flexed = ((frame / halfway_calculation_point) * 100) - 100
                percent_flexed_values[dataset_id, frame] = percent_flexed
            else:
                # After the halfway point, mirror the value from the first half
                # Check if it's the peak frame
                if frame == halfway_calculation_point + 1:
                    percent_flexed = 0
                else:
                    # Calculate mirror frame
                    mirror_frame = halfway_calculation_point - (frame - halfway_calculation_point - 1)
                    percent_flexed = -percent_flexed_values[dataset_id, mirror_frame]
            df.at[index, 'Percent Flexed'] = percent_flexed

    return df

df5_cut = ds5_cut_df.rename(columns= {'Frame': 'Frame Number'})

df5_cut = add_percent_flexed(df5_cut)


#%%
df_cut_all = pd.concat([df1_cut, df2_cut, df3_cut, df4_cut, df5_cut], ignore_index=True)


#%%
df_cut_all.to_pickle('df_cut_all_nonscaled.pkl')

#%%
scaling_factor = 0.48484848484848486
df_cut_all['IS_Translation'] = df_cut_all['IS_Translation'] * scaling_factor
df_cut_all['AP_Translation'] = df_cut_all['AP_Translation'] * scaling_factor
#%%

def split_dataframes(df):
    # Identify unique datasets
    datasets = df['Dataset'].unique()
    
    first_half_rows = []
    second_half_rows = []
    
    for dataset in datasets:
        # Filter rows for the current dataset
        dataset_df = df[df['Dataset'] == dataset]
        
        # Get the maximum Frame Number for this dataset
        max_frame = dataset_df['Frame Number'].max()
        
        # Calculate the split point (half of the total frames)
        split_point = max_frame // 2
        
        # Split the dataset
        first_half = dataset_df[dataset_df['Frame Number'] <= split_point]
        second_half = dataset_df[dataset_df['Frame Number'] > split_point]
        
        first_half_rows.append(first_half)
        second_half_rows.append(second_half)
    
    # Combine all first halves and all second halves
    first_half_df = pd.concat(first_half_rows, ignore_index=True)
    second_half_df = pd.concat(second_half_rows, ignore_index=True)
    
    return first_half_df, second_half_df

first_half_angle_df , second_half_angle_df = split_dataframes(master_df_inverted)

first_half_angle_df['Percent Flexed'] = first_half_angle_df['Percent Flexed'] * -1 
#%%

# so far, these dataframes, first_half_df and second_half_df store the values for just 5 datasets, with dataset number 5 of JL being replaced by dataset 6 of US. 

# the first half has flexion percent starting from -100 to 0 and second half from 0 to 100. for plotting, the first half is once again converted to just 100% to 0% 
# Multiply the IS_Translation and AP_Translation columns by the given value


# Save the combined dataframe as a .pkl object
first_half_df_cut.to_pickle('first_half_df_cut.pkl')

second_half_df_cut.to_pickle('second_half_df_cut.pkl')


#%%


# Method 1: Interpolation method
def plot_interpolated_data(df, num_points=100):
    plt.figure(figsize=(12, 6))
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method].sort_values('Percent Flexed')
        
        # Create evenly spaced points between 0 and 100
        x_new = np.linspace(0, 100, num_points)
        
        # Interpolate AP_Translation values
        y_new = np.interp(x_new, method_data['Percent Flexed'], method_data['AP_Translation'])
        
        plt.plot(x_new, y_new, label=method)
    
    plt.title(f'AP Translation vs Percent Flexed (Interpolated, points={num_points})')
    plt.xlabel('Percent Flexed')
    plt.ylabel('AP Translation')
    plt.legend(title='Method')
    plt.show()
    
    
plot_interpolated_data(first_half_df, num_points=100)

#%%
def plot_binned_data(df, n_bins=20):
    # Debug: Print unique methods
    print("Unique methods:", df['Method'].unique())
    
    df['Flexion_Bin'] = pd.cut(df['Percent Flexed'], bins=n_bins, labels=False)
    binned_data = df.groupby(['Method', 'Flexion_Bin'])['AP_Translation'].agg(['mean', 'std']).reset_index()
    
    # Debug: Print binned data shape and head
    print("Binned data shape:", binned_data.shape)
    print("Binned data head:")
    print(binned_data.head())
    
    plt.figure(figsize=(12, 6))
    
    # Use Seaborn's lineplot with error bars
    sns.lineplot(data=binned_data, x='Flexion_Bin', y='mean', 
                 hue='Method', err_style='band', errorbar='sd')
    
    plt.title(f'AP Translation vs Percent Flexed (Binned, bins={n_bins})')
    plt.xlabel('Percent Flexed (Binned)')
    plt.ylabel('AP Translation')
    plt.legend(title='Method')
    plt.show()

# Usage
plot_binned_data(first_half_df, n_bins=20)
#%%

# Updated binning method with individual lines and error bars
def plot_binned_data(df, n_bins=20):
    df['Flexion_Bin'] = pd.cut(df['Percent Flexed'], bins=n_bins, labels=False)
    
    # Calculate mean and standard deviation for each bin and method
    binned_data = df.groupby(['Method', 'Flexion_Bin']).agg({
        'AP_Translation': ['mean', 'std']
    }).reset_index()
    binned_data.columns = ['Method', 'Flexion_Bin', 'AP_Translation_Mean', 'AP_Translation_Std']
    
    plt.figure(figsize=(12, 6))
    
    for method in df['Method'].unique():
        method_data = binned_data[binned_data['Method'] == method]
        plt.errorbar(method_data['Flexion_Bin'], method_data['AP_Translation_Mean'], 
                     yerr=method_data['AP_Translation_Std'], 
                     label=method, capsize=5, capthick=2, marker='o')
    
    plt.title(f'AP Translation vs Percent Flexed (Binned, bins={n_bins})')
    plt.xlabel('Percent Flexed (Binned)')
    plt.ylabel('AP Translation')
    plt.legend(title='Method')
    plt.show()
    

plot_binned_data(first_half_df, n_bins=10)


#%%

# Use seaborn's lineplot with automatic aggregation
sns.lineplot(
    data=df2_cut,
    x='Percent Flexed',
    y='IS_Translation',
    hue='Method',
    estimator='mean',  # Aggregates data points by calculating the mean
    errorbar='sd'            # Shows the standard deviation as the confidence interval
)
plt.show()

#%%
import pingouin as pg

def plot_binned_translation_data(df, translation_column, bin_width=10, figsize=(12, 8)):
    # Make a copy of the DataFrame
    df_copy = df.copy()
    
    # Define bin edges
    bin_edges = list(range(0, 101, bin_width))
    
    # Bin 'Percent Flexed' and calculate bin centers
    df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
    df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
    
    # Group by 'Method', the new 'Custom_Bin', and 'Dataset' to calculate means
    grouped = df_copy.groupby(['Method', 'Custom_Bin', 'Dataset'])[translation_column].mean().reset_index()
    grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
    
    # Replace 'Auto' with 'Semi-Auto' in the Method column
    grouped['Method'] = grouped['Method'].replace('Auto', 'Semi-Auto')
    
    # Plotting the data
    plt.figure(figsize=figsize)
    
    sns.lineplot(
        data=grouped,
        x='Bin_Center',
        y=translation_column,
        hue='Method',
        marker="o",
        ci='sd',
        err_style="band",
        err_kws={'alpha': 0.3}
    )
    
    # Ensure 100 is shown on the x-axis
    plt.xlim(0, 100)
    
    # Get current y-axis limits
    y_min, y_max = plt.ylim()

    # Add padding (10 units on each side)
    padding = 30
    plt.ylim(y_min - padding, y_max + padding)
    
    plt.xlabel("Flexion [%]", fontsize=14)
    plt.ylabel(f"{translation_column} [mm]", fontsize=14)
    plt.title(f"{translation_column} vs Flexion Percentage", fontsize=16)
    plt.grid(True)
    
    # Increase font size for tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Increase legend font size
    plt.legend(title='Method', fontsize=12, title_fontsize=14)
    
    
    
   # Count the number of frames for each dataset
    frame_counts = df.groupby('Dataset').size() /2   # This counts the number of rows (frames) per dataset
    
    # Get the minimum and maximum number of frames
    min_frames = frame_counts.min()
    max_frames = frame_counts.max()
    
    # Calculate the range of motion based on the frame counts
    min_rom = min_frames * 2  # 1 frame = 2 degrees
    max_rom = max_frames * 2  # 1 frame = 2 degrees
    
    # Add text annotation to the plot showing the range of motion
    plt.text(0.05, 0.05, f"Range of Motion: {min_rom:.0f}° to {max_rom:.0f}°", 
         transform=plt.gca().transAxes, 
         horizontalalignment='left', 
         verticalalignment='bottom',
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    
    plt.tight_layout()
    #plt.savefig('first_half_is.svg', dpi=300)
    plt.show()
    
    # Perform t-tests if there are exactly two methods
    methods = grouped['Method'].unique()
    if len(methods) == 2:
        t_test_results = []
        for name, group in grouped.groupby('Custom_Bin'):
            method1 = group[group['Method'] == methods[0]][translation_column]
            method2 = group[group['Method'] == methods[1]][translation_column]
            t_test = pg.ttest(method1, method2, paired=False)
            t_test_results.append((name, t_test['p-val'].values[0]))
        
        return pd.DataFrame(t_test_results, columns=['Bin', 'p-value'])
    else:
        print("T-tests not performed as there are not exactly two methods.")
        return None
#%%
from scipy.interpolate import interp1d
# Usage examples
def combine_translation_angle_data(df_cut, df_angle):
    combined_df = df_cut.copy()
    combined_df['Angle'] = float('nan')

    for dataset in combined_df['Dataset'].unique():
        for method in combined_df['Method'].unique():
            mask_cut = (combined_df['Dataset'] == dataset) & (combined_df['Method'] == method)
            mask_angle = (df_angle['Dataset'] == dataset) & (df_angle['Method'] == method)
            
            df_cut_subset = combined_df[mask_cut].sort_values('Percent Flexed')
            df_angle_subset = df_angle[mask_angle].sort_values('Percent Flexed')
            
            if len(df_cut_subset) == len(df_angle_subset):
                combined_df.loc[mask_cut, 'Angle'] = df_angle_subset['Angle'].values
            else:
                # Interpolate angle values
                angle_interp = interp1d(df_angle_subset['Percent Flexed'], df_angle_subset['Angle'], 
                                        kind='linear', fill_value='extrapolate')
                
                combined_df.loc[mask_cut, 'Angle'] = angle_interp(df_cut_subset['Percent Flexed'])

    return combined_df

# Use the function
first_half_combined = combine_translation_angle_data(first_half_df_cut, first_half_angle_df)
second_half_combined = combine_translation_angle_data(second_half_df_cut, second_half_angle_df)


#%%
# its flipped for some reason: 
first_half_df['Angle'] = first_half_df['Angle'] * -1 

#%%
# it is still flipped. so doing it without this minus 

def reverse_angle_column(df):
    """
    Reverses the 'Angle' column within each group of 'Dataset' and 'Method'
    while maintaining the original order of frame numbers.

    Parameters:
    df (pandas.DataFrame): Input dataframe containing 'Dataset', 'Method', 'Frame', and 'Angle' columns

    Returns:
    pandas.DataFrame: A new dataframe with the 'Angle' column reversed within each group
    """
    # Create a copy of the input dataframe to avoid modifying the original
    new_df = df.copy()

    # Sort the dataframe by Dataset, Method, and Frame number to ensure correct ordering
    new_df = new_df.sort_values(['Dataset', 'Method', 'Frame Number'])

    # Group by Dataset and Method, then reverse the Angle column within each group
    new_df['Angle'] = new_df.groupby(['Dataset', 'Method'])['Angle'].transform(lambda x: x[::-1].values)

    # Reset the index after the operation
    new_df = new_df.reset_index(drop=True)

    return new_df

# Example usage:
reversed_df = reverse_angle_column(first_half_df)
#%%

first_half_df = pd.read_pickle( '/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/first_half_trans_and_angle.pkl') 
second_half_df = pd.read_pickle('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/second_half_trans_and_angle.pkl')
#%%

def plot_six_panel_translation_and_angle(df_first_half, df_second_half, bin_width=10, figsize=(30, 20), dpi=300, y_padding=0.1, is_y_padding=0.3):
    plt.rcParams.update({'font.size': 14})  # Increase default font size
    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    
    def plot_data(df, ax, column, title, is_angle=False, is_is=False):
        df_copy = df.copy()
        bin_edges = list(range(0, 101, bin_width))
        df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
        df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
        
        grouped = df_copy.groupby(['Method', 'Custom_Bin', 'Dataset'])[column].mean().reset_index()
        grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
        grouped['Method'] = grouped['Method'].replace('Auto', 'Semi-Auto')
        
        sns.lineplot(
            data=grouped,
            x='Bin_Center',
            y=column,
            hue='Method',
            marker="o",
            ci='sd',
            err_style="band",
            err_kws={'alpha': 0.3},
            ax=ax
        )
        
        ax.set_xlim(0, 100)
        ax.set_xlabel("Flexion [%]", fontsize=16)
        ax.set_ylabel("Angle [°]" if is_angle else "Translation [mm]", fontsize=16)
        ax.set_title(title, fontsize=18, pad=20)
        #ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(title='Method', fontsize=14, title_fontsize=16)
        
        # Adjust y-axis limits with padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        padding = is_y_padding if is_is else y_padding
        ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
        
        # frame_counts = df.groupby('Dataset').size() / 2
        # min_frames, max_frames = frame_counts.min(), frame_counts.max()
        # min_rom, max_rom = min_frames * 2, max_frames * 2
        
        # ax.text(0.05, 0.95, f"Range of Motion: {min_rom:.0f}° to {max_rom:.0f}°", 
        #         transform=ax.transAxes, 
        #         horizontalalignment='left', 
        #         verticalalignment='top',
        #         fontsize=12,
        #         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # First half (Extension Phase)
    plot_data(df_first_half, axs[0, 0], 'Angle', '', is_angle=True) # leaving this blank 
    plot_data(df_first_half, axs[0, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)')
    plot_data(df_first_half, axs[0, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True)
    
    # Second half (Flexion Phase)
    plot_data(df_second_half, axs[1, 0], 'Angle', '', is_angle=True)
    plot_data(df_second_half, axs[1, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)')
    plot_data(df_second_half, axs[1, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True)
    
    # Add phase labels
    fig.text(0.5, 0.98, 'Extension Phase (Flexed to Extended)', ha='center', va='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.51, 'Flexion Phase (Extended to Flexed)', ha='center', va='center', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    return fig

# Call the function with the combined dataframes and save the figure
fig = plot_six_panel_translation_and_angle(reversed_df, second_half_df, bin_width=12, figsize=(30, 20), dpi=300, y_padding=0.0, is_y_padding=0.9)
fig.savefig('six_panels_no_grid.png', dpi=300, bbox_inches='tight')

#%%
def plot_six_panel_translation_and_angle(df_first_half, df_second_half, bin_width=10, figsize=(30, 20), dpi=300, y_padding=0.1, is_y_padding=0.3):
    plt.rcParams.update({'font.size': 24})  # Increase default font size
    fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    
    def plot_data(df, ax, column, title, is_angle=False, is_is=False):
        df_copy = df.copy()
        bin_edges = list(range(0, 101, bin_width))
        df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
        df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
        
        grouped = df_copy.groupby(['Method', 'Custom_Bin', 'Dataset'])[column].mean().reset_index()
        grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
        grouped['Method'] = grouped['Method'].replace('Auto', 'Semi-Auto')
        
        sns.lineplot(
            data=grouped,
            x='Bin_Center',
            y=column,
            hue='Method',
            marker="o",
            ci='sd',
            err_style="band",
            err_kws={'alpha': 0.3},
            ax=ax
        )
        
        ax.set_xlim(0, 100)
        ax.set_xlabel("Flexion [%]", fontsize=24)
        ax.set_ylabel("Angle [°]" if is_angle else "Translation [mm]", fontsize=24)
        ax.set_title(title, fontsize=24, pad=20)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='--', alpha=0.4)
        ax.tick_params(axis='both', which='major', labelsize=24)        
        ax.legend(title='Method', fontsize=24, title_fontsize=24)
        
        # Adjust y-axis limits with padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        padding = is_y_padding if is_is else y_padding
        ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
        
        if is_angle:
            ax.set_ylim([-25,35])
            ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 6))
            ax.set_yticks(np.linspace(-20, 30, 6))
        elif is_is:
            ax.set_ylim([-63,-51])
            ax.set_yticks(np.linspace(-62, -52, 6))
        else:
            ax.set_ylim([-32.5,-2.5])        
            ax.set_yticks(np.linspace(-30, -5, 6))
        
        # Calculate values and std at x=0 and x=100 for each method
        results = {}
        for method in ['Manual', 'Semi-Auto']:
            method_data = grouped[grouped['Method'] == method]
            
            # For x=0
            start_data = method_data[method_data['Bin_Center'] == method_data['Bin_Center'].min()]
            start_value = start_data[column].mean()
            start_std = start_data[column].std()
            
            # For x=100
            end_data = method_data[method_data['Bin_Center'] == method_data['Bin_Center'].max()]
            end_value = end_data[column].mean()
            end_std = end_data[column].std()
            
            # Calculate change and propagate error
            change = end_value - start_value
            change_std = np.sqrt(start_std**2 + end_std**2)
            
            results[method] = {
                'start': (start_value, start_std),
                'end': (end_value, end_std),
                'change': (change, change_std)
            }
        
        return results
    
    changes = []
    
    # First half (Extension Phase)
    changes.append(plot_data(df_first_half, axs[0, 0], 'Angle', '', is_angle=True))
    changes.append(plot_data(df_first_half, axs[0, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)'))
    changes.append(plot_data(df_first_half, axs[0, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True))
    
    # Second half (Flexion Phase)
    changes.append(plot_data(df_second_half, axs[1, 0], 'Angle', '', is_angle=True))
    changes.append(plot_data(df_second_half, axs[1, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)'))
    changes.append(plot_data(df_second_half, axs[1, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True))
    
    # Add phase labels
    fig.text(0.5, 0.98, 'Extension Phase (Flexed to Extended)', ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.51, 'Flexion Phase (Extended to Flexed)', ha='center', va='center', fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    
    # Print out the changes with standard deviations
    print("Changes from minimum to maximum flexion (with standard deviations):")
    titles = ["Extension Phase - Angle", "Extension Phase - AP Translation", "Extension Phase - IS Translation",
              "Flexion Phase - Angle", "Flexion Phase - AP Translation", "Flexion Phase - IS Translation"]
    for i, (title, result) in enumerate(zip(titles, changes)):
        print(f"{i+1}. {title}:")
        for method in ['Manual', 'Semi-Auto']:
            start_val, start_std = result[method]['start']
            end_val, end_std = result[method]['end']
            change_val, change_std = result[method]['change']
            unit = '°' if 'Angle' in title else 'mm'
            print(f"   {method}:")
            print(f"     Start: {start_val:.2f} ± {start_std:.2f} {unit}")
            print(f"     End: {end_val:.2f} ± {end_std:.2f} {unit}")
            print(f"     Change: {change_val:.2f} ± {change_std:.2f} {unit}")
        print()
    
    return fig

# Call the function with the combined dataframes and save the figure
fig = plot_six_panel_translation_and_angle(adjusted_first_half_df, adjusted_second_half_df, bin_width=10, figsize=(30, 20), dpi=300, y_padding=0.3, is_y_padding=0.9)
#fig.savefig('six_panels_with_grid.svg', dpi=300, bbox_inches='tight')
#%%
def plot_four_panel_translation(df_first_half, df_second_half, bin_width=10, figsize=(20, 16), y_padding=0.1):
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    def plot_translation_data(df, ax, translation_column, title):
        df_copy = df.copy()
        bin_edges = list(range(0, 101, bin_width))
        df_copy['Custom_Bin'] = pd.cut(df_copy['Percent Flexed'], bins=bin_edges, include_lowest=True)
        df_copy['Bin_Center'] = df_copy['Custom_Bin'].apply(lambda x: x.mid)
        
        grouped = df_copy.groupby(['Method', 'Custom_Bin', 'Dataset'])[translation_column].mean().reset_index()
        grouped['Bin_Center'] = grouped['Custom_Bin'].apply(lambda x: x.mid)
        grouped['Method'] = grouped['Method'].replace('Auto', 'Semi-Auto')
        
        sns.lineplot(
            data=grouped,
            x='Bin_Center',
            y=translation_column,
            hue='Method',
            marker="o",
            ci='sd',
            err_style="band",
            err_kws={'alpha': 0.3},
            ax=ax
        )
        
        ax.set_xlim(0, 100)
        ax.set_xlabel("Flexion [%]", fontsize=12)
        ax.set_ylabel("Translation [mm]", fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(title='Method', fontsize=10, title_fontsize=12)
        
        # Adjust y-axis limits with padding
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * y_padding, y_max + y_range * y_padding)
        
        frame_counts = df.groupby('Dataset').size() / 2
        min_frames, max_frames = frame_counts.min(), frame_counts.max()
        min_rom, max_rom = min_frames * 2, max_frames * 2
        
        ax.text(0.05, 0.05, f"Range of Motion: {min_rom:.0f}° to {max_rom:.0f}°", 
                transform=ax.transAxes, 
                horizontalalignment='left', 
                verticalalignment='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plot_translation_data(df_first_half, axs[0, 0], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)')
    plot_translation_data(df_first_half, axs[0, 1], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)')
    plot_translation_data(df_second_half, axs[1, 0], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)')
    plot_translation_data(df_second_half, axs[1, 1], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)')
    
    # Add phase labels
    fig.text(0.5, 0.98, 'Extension Phase (Flexed to Extended)', ha='center', va='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.51, 'Flexion Phase (Extended to Flexed)', ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

plot_four_panel_translation(first_half_df_cut, second_half_df_cut, bin_width=10, figsize=(20, 16), y_padding=1)
#%%
# If you want to see the t-test results (if applicable)
if result_ap is not None:
    print("T-test results for AP Translation:")
    print(result_ap)

if result_is is not None:
    print("T-test results for IS Translation:")
    print(result_is)
    
#%%

def plot_dataset_specific_translations(df):
    # Function to create a single subplot
    def create_subplot(ax, data, translation_column, dataset):
        sns.lineplot(
            data=data,
            x='Percent Flexed',
            y=translation_column,
            hue='Method',
            marker="o",
            ax=ax
        )
        ax.set_title(f"{dataset}")
        ax.set_xlabel("Flexion [%]")
        ax.set_ylabel(f"{translation_column} [mm]")
        ax.grid(True)
        
        # Ensure 100 is shown on the x-axis
        ax.set_xlim(0, 100)
        
        # Add padding to y-axis (30 units on each side)
        y_min, y_max = ax.get_ylim()
        #ax.set_ylim(y_min - 30, y_max + 30)
        
        # Replace 'Auto' with 'Semi-Auto' in the legend
        handles, labels = ax.get_legend_handles_labels()
        labels = ['Semi-Auto' if label == 'Auto' else label for label in labels]
        ax.legend(handles, labels, title='Method')

    # Create two figures, one for AP and one for IS
    fig_ap, axes_ap = plt.subplots(2, 3, figsize=(20, 12))
    fig_is, axes_is = plt.subplots(2, 3, figsize=(20, 12))

    # Flatten the axes arrays for easier iteration
    axes_ap = axes_ap.flatten()
    axes_is = axes_is.flatten()

    # Get unique datasets
    datasets = df['Dataset'].unique()

    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset]
        
        # AP Translation
        create_subplot(axes_ap[i], dataset_df, 'AP_Translation', dataset)
        
        # IS Translation
        create_subplot(axes_is[i], dataset_df, 'IS_Translation', dataset)

    # Remove extra subplots if any
    for i in range(len(datasets), 6):
        axes_ap[i].remove()
        axes_is[i].remove()

    # Adjust layout and add super titles
    fig_ap.suptitle("AP Translation vs Flexion Percentage for Each Dataset", fontsize=16)
    fig_is.suptitle("IS Translation vs Flexion Percentage for Each Dataset", fontsize=16)
    
    fig_ap.tight_layout()
    fig_is.tight_layout()

    # Show plots
    plt.show()

# Usage example:
plot_dataset_specific_translations(adjusted_second_half_df)

#%%

def adjust_translations_at_flexion(df, is_first_half):
    """
    Adjusts the IS_Translation values for the Manual method to match the Auto method
    at specific flexion points depending on whether it's the first or second half of the data.

    Parameters:
    df (pandas.DataFrame): Input dataframe containing 'Dataset', 'Method', 'Percent Flexed', and 'IS_Translation' columns
    is_first_half (bool): True if adjusting first half (match at 100% flexion), False if second half (match at 0% flexion)

    Returns:
    pandas.DataFrame: A new dataframe with adjusted IS_Translation values
    """
    # Create a copy of the dataframe to avoid modifying the original
    adjusted_df = df.copy()
    
    # Get unique datasets
    datasets = adjusted_df['Dataset'].unique()
    
    # Set the flexion point based on whether it's the first or second half
    flexion_point = 100 if is_first_half else 0
    
    for dataset in datasets:
        # Filter for the current dataset and the specific flexion point
        mask = (adjusted_df['Dataset'] == dataset) & (adjusted_df['Percent Flexed'] == flexion_point)
        
        # Get the auto value for IS translation at the specific flexion point
        auto_is = adjusted_df.loc[mask & (adjusted_df['Method'] == 'Auto'), 'IS_Translation'].values[0]
        
        # Update the manual value to match the auto value at the specific flexion point
        adjusted_df.loc[mask & (adjusted_df['Method'] == 'Manual'), 'IS_Translation'] = auto_is
    
    return adjusted_df

# Usage example:
adjusted_first_half_df = adjust_translations_at_flexion(reversed_df, is_first_half=True)
adjusted_second_half_df = adjust_translations_at_flexion(second_half_df, is_first_half=False)

#%%




df = first_half_df.copy()  # Create a copy to avoid modifying the original dataframe
df['Dataset'] = df['Dataset'].replace(6, 5)

#%%

# Task 1: Rename Dataset 6 to 5
def rename_dataset(df):
    df.loc[df['Dataset'] == 6, 'Dataset'] = 5
    return df

first_half_df = rename_dataset(first_half_df)
second_half_df = rename_dataset(second_half_df)

# Task 2: Replace Dataset 3 with new data
def replace_dataset_3(main_df, new_df):
    # Remove existing Dataset 3 from main dataframe
    main_df = main_df[main_df['Dataset'] != 3]
    
    # Concatenate the main dataframe with the new Dataset 3
    result_df = pd.concat([main_df, new_df], ignore_index=True)
    
    # Sort the result by Dataset and Frame Number
    result_df = result_df.sort_values(['Dataset', 'Frame Number'])
    
    return result_df

# Apply the function to both halves
first_half_df = replace_dataset_3(first_half_df, first_half_df3)
second_half_df = replace_dataset_3(second_half_df, second_half_df3)

#%%
# Step 1: Replace 'Unloaded' with 'Auto' in the 'Condition' column
master_df_inverted['Condition'] = master_df_inverted['Condition'].replace('Unloaded', 'Auto')

# Step 2: Remove all rows where the 'Condition' column is 'Loaded'
master_df_inverted = master_df_inverted[master_df_inverted['Condition'] != 'Loaded']
