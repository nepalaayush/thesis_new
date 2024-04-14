# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:32:54 2024

@author: Aayush
"""
import os 
os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
#os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')


import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%

with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/JL/JL_NW_tib_info_s.pkl', 'rb') as file:
    JL_NW_tib_info_s = pickle.load(file)    


with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/JL/JL_W_tib_info_s.pkl', 'rb') as file:
    JL_W_tib_info_s = pickle.load(file)
    
    
with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/JL/try_JL_NW_fem_info_s.pkl', 'rb') as file:
    JL_NW_fem_info_s = pickle.load(file)
    

with open('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new/new_analysis_all/JL/JL_W_fem_info_s.pkl', 'rb') as file:
    JL_W_fem_info_s = pickle.load(file)
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
with open('/data/projects/ma-nepal-segmentation/data/data_20_03/angle_and_rel_df.pkl', 'rb') as file:
    angle_and_rel_df =  pickle.load(file)
    
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

with open('ap_df_1_7.pkl', 'wb') as f:
    pickle.dump(ap_df_1_7, f)   

#%%
# this code plots the average relative translation for IS for UNloaded .. along with a weird sd shading.. (because not all datasets have the same angle)

sns.lineplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Unloaded') & (angle_and_rel_df['Type'] =='IS')], x='Percent Flexed', y='Relative Translation')
