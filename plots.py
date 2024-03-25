# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 16:32:54 2024

@author: Aayush
"""
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%




master_df = MK_translations_new # is named MK_W_master_df.pkl in the saved file. 

# when adding new dataframes: do this: 
# For Dataset 1
#master_df['Dataset'] = 'Dataset1'

# For Dataset 2
#new_master_df['Dataset'] = 'Dataset2'

# combined_df = pd.concat([master_df, new_master_df], ignore_index=True)
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
# Add a 'Dataset' column to distinguish between them
MK_master_df['Dataset'] = 'Dataset 1'
MM_master_df['Dataset'] = 'Dataset 2'
AN_master_df['Dataset'] = 'Dataset 3'

combined_df = pd.concat([MK_master_df, MM_master_df, AN_master_df])

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
angle_df = combined_df[['Frame Number', 'Condition', 'Dataset', 'Angles']].copy()    
angle_df = angle_df.dropna(subset=['Angles'])
# that works 
#%%
# now trying to extract the relative translation and store it as its own df 
# Initialize a list to store the relative translation data
rel_trans_data = []

# Iterate over each dataset, condition, and type
for dataset in combined_df['Dataset'].unique():
    for condition in combined_df['Condition'].unique():
        for translation_type in combined_df['Type'].unique():
            # Get the frame numbers available for this dataset, condition, and type
            frames = combined_df[(combined_df['Dataset'] == dataset) & 
                                 (combined_df['Condition'] == condition) & 
                                 (combined_df['Type'] == translation_type)]['Frame Number'].unique()
            
            for frame in frames:
                # Get translations for Tibia and Femur for the current frame, dataset, condition, and type
                tibia_trans = combined_df[(combined_df['Dataset'] == dataset) & 
                                          (combined_df['Condition'] == condition) & 
                                          (combined_df['Frame Number'] == frame) & 
                                          (combined_df['Type'] == translation_type) & 
                                          (combined_df['Body Part'] == 'Tibia')]['Translations'].values
                
                femur_trans = combined_df[(combined_df['Dataset'] == dataset) & 
                                          (combined_df['Condition'] == condition) & 
                                          (combined_df['Frame Number'] == frame) & 
                                          (combined_df['Type'] == translation_type) & 
                                          (combined_df['Body Part'] == 'Femur')]['Translations'].values

                # Check if translations are available for both Tibia and Femur
                if tibia_trans.size > 0 and femur_trans.size > 0:
                    # Calculate relative translation and store the result
                    rel_trans = tibia_trans[0] - femur_trans[0]
                    rel_trans_data.append({'Frame Number': frame, 'Dataset': dataset, 'Condition': condition, 'Type': translation_type, 'Relative Translation': rel_trans})

# Create a new dataframe from the relative translation data
rel_trans_df = pd.DataFrame(rel_trans_data)
# got rel trans for is as well as ap translations 
#%%

rel_trans_is = rel_trans_df[(rel_trans_df['Type'] == 'IS')]
 
#%%



#angle_and_rel_df = pd.concat([angle_df, rel_trans_df]) # this does not work because concat does things vertically , and we want horizontal stacking 

angle_and_rel_df = pd.merge(angle_df, rel_trans_is, 
                            on=['Frame Number', 'Condition', 'Dataset'], 
                            how='left')


#%%
import pickle
with open('/data/projects/ma-nepal-segmentation/data/data_20_03/angle_and_rel_df.pkl', 'rb') as file:
    angle_and_rel_df =  pickle.load(file)
    
#%%
# Plotting
plt.figure(figsize=(10, 6))
#sns.scatterplot(data= angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Loaded')], x='Angles', y='Relative Translation', hue='Dataset')
#sns.regplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Loaded')], x='Angles', y='Relative Translation', scatter=False)
sns.lmplot(data=angle_and_rel_df[(angle_and_rel_df['Condition'] == 'Unloaded')], x='Angles', y='Relative Translation', hue='Dataset', ci=None)



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


