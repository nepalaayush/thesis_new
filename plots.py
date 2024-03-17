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
# Step 1: Organize the data
# Assuming your arrays are named ap_w, is_w, ap_nw, is_nw, and angles, and each has the same length
df_loaded_MM = pd.DataFrame({
    'Dataset': 'Dataset 1',
    'Frame': range(1, len(MM_angles_W) + 1),
    'Condition': 'Loaded',
    'AP_Tibia': MM_W_tib_rel_ap,
    'IS_Tibia': MM_W_tib_rel_is,
    'Angle': MM_angles_W
})

df_unloaded_MM = pd.DataFrame({
    'Dataset': 'Dataset 1',
    'Frame': range(1, len(MM_angles_NW) + 1),
    'Condition': 'Unloaded',
    'AP_Tibia': MM_NW_tib_rel_ap,
    'IS_Tibia': MM_NW_tib_rel_is,
    'Angle': MM_angles_NW
})

df_loaded_AN = pd.DataFrame({
    'Dataset': 'Dataset 2',
    'Frame': range(1, len(AN_angles_W) + 1),
    'Condition': 'Loaded',
    'AP_Tibia': AN_W_ap_tib_rel,
    'IS_Tibia': AN_W_is_tib_rel,
    'Angle': AN_angles_W
})

df_unloaded_AN = pd.DataFrame({
    'Dataset': 'Dataset 2',
    'Frame': range(1, len(AN_angles_NW) + 1),
    'Condition': 'Unloaded',
    'AP_Tibia': AN_NW_ap_tib_rel,
    'IS_Tibia': AN_NW_is_tib_rel,
    'Angle': AN_angles_NW
})

df_loaded_MK = pd.DataFrame({
    'Dataset': 'Dataset 3',
    'Frame': range(1, len(MK_angles_W) + 1),
    'Condition': 'Loaded',
    'AP_Tibia': MK_ap_W_tib_rel,
    'IS_Tibia': MK_is_W_tib_rel,
    'Angle': MK_angles_W
})

df_unloaded_MK = pd.DataFrame({
    'Dataset': 'Dataset 3',
    'Frame': range(1, len(MK_angles_NW) + 1),
    'Condition': 'Unloaded',
    'AP_Tibia': MK_ap_NW_tib_rel,
    'IS_Tibia': MK_is_NW_tib_rel,
    'Angle': MK_angles_NW
})




df_combined = pd.concat([df_loaded_MM, df_unloaded_MM, df_loaded_AN, df_unloaded_AN, df_loaded_MK, df_unloaded_MK], ignore_index=True)
#%%
# Find the minimum frame length across all datasets
min_frame_length = df_combined.groupby('Dataset')['Frame'].nunique().min()

# Filter the dataframe to include only rows where the Frame number is less than or equal to the minimum frame length
df_combined_equal_frames = df_combined[df_combined['Frame'] <= min_frame_length].copy()
#%%
### the side by side plot 
# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for loaded condition
sns.lineplot(x='Frame', y='IS_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Loaded'], ax=axs[0], hue='Dataset', ci=None)
axs[0].set_title('Inferior (-ve) / Superior (+ve) Translation for Loaded Condition')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Translation [mm]')

# Plot for unloaded condition
sns.lineplot(x='Frame', y='IS_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Unloaded'], ax=axs[1],hue='Dataset', ci=None)
axs[1].set_title('Inferior (-ve) /   Superior (+ve) Translation for Unloaded Condition')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Translation [mm]')

plt.tight_layout()
plt.show()
plt.show()

#%%
sns.set(style="whitegrid")

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for loaded condition
sns.lineplot(x='Frame', y='AP_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Loaded'], ax=axs[0], hue='Dataset', ci=None)
axs[0].set_title('Anterior (-ve) / Posterior (+ve) Translation for Loaded Condition')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Translation [mm]')

# Plot for unloaded condition
sns.lineplot(x='Frame', y='AP_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Unloaded'], ax=axs[1],hue='Dataset', ci=None)
axs[1].set_title('ANterior (-ve) /   Posterior (+ve) Translation for Unloaded Condition')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Translation [mm]')

plt.tight_layout()
plt.show()
plt.show()

#%%
### on the same plot 
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(10, 6))

# Calculate the mean IS translation for each frame and condition, then plot them
sns.lineplot(x='Frame', y='IS_Tibia', hue='Condition', data=df_combined_equal_frames, estimator='mean', ci=None)

# Set the title and labels
plt.title('Mean IS Translation Across Frames for Loaded and Unloaded Conditions')
plt.xlabel('Frame')
plt.ylabel('Translation [mm]')
plt.legend(title='Condition')
plt.tight_layout()
plt.show()

#%%
# with sd 
sns.set(style="white")

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for loaded condition with the standard deviation as a shaded region
sns.lineplot(x='Frame', y='IS_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Loaded'], ax=axs[0], estimator='mean', ci='sd')
axs[0].set_title('Mean IS Translation for Loaded Condition')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Translation [mm]')
axs[0].set_ylim(-15, 15) 

# Plot for unloaded condition with the standard deviation as a shaded region
sns.lineplot(x='Frame', y='IS_Tibia', data=df_combined_equal_frames[df_combined_equal_frames['Condition'] == 'Unloaded'], ax=axs[1], estimator='mean', ci='sd')
axs[1].set_title('Mean IS Translation for Unloaded Condition')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Translation [mm]')
axs[1].set_ylim(-15, 15)


plt.tight_layout()
plt.show()
#%%

# Concatenate the dataframes for loaded and unloaded conditions
df_loaded_all = pd.concat([df_loaded_MM, df_loaded_AN, df_loaded_MK])
df_unloaded_all = pd.concat([df_unloaded_MM, df_unloaded_AN, df_unloaded_MK])

# Function to compute mean and standard deviation
def compute_stats(df, column):
    return df.groupby('Frame')[column].mean(), df.groupby('Frame')[column].std()

# Compute stats for loaded condition
mean_loaded_AP, std_loaded_AP = compute_stats(df_loaded_all, 'AP_Tibia')
mean_loaded_IS, std_loaded_IS = compute_stats(df_loaded_all, 'IS_Tibia')

# Compute stats for unloaded condition
mean_unloaded_AP, std_unloaded_AP = compute_stats(df_unloaded_all, 'AP_Tibia')
mean_unloaded_IS, std_unloaded_IS = compute_stats(df_unloaded_all, 'IS_Tibia')

# Creating subplots
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# AP Loaded
ax[0, 0].errorbar(mean_loaded_AP.index, mean_loaded_AP, yerr=std_loaded_AP, fmt='-o')
ax[0, 0].set_title('AP Tibia Translation - Loaded')
ax[0, 0].set_xlabel('Frame')
ax[0, 0].set_ylabel('Translation')

# AP Unloaded
ax[0, 1].errorbar(mean_unloaded_AP.index, mean_unloaded_AP, yerr=std_unloaded_AP, fmt='-o')
ax[0, 1].set_title('AP Tibia Translation - Unloaded')
ax[0, 1].set_xlabel('Frame')
ax[0, 1].set_ylabel('Translation')

# IS Loaded
ax[1, 0].errorbar(mean_loaded_IS.index, mean_loaded_IS, yerr=std_loaded_IS, fmt='-o')
ax[1, 0].set_title('IS Tibia Translation - Loaded')
ax[1, 0].set_xlabel('Frame')
ax[1, 0].set_ylabel('Translation')

# IS Unloaded
ax[1, 1].errorbar(mean_unloaded_IS.index, mean_unloaded_IS, yerr=std_unloaded_IS, fmt='-o')
ax[1, 1].set_title('IS Tibia Translation - Unloaded')
ax[1, 1].set_xlabel('Frame')
ax[1, 1].set_ylabel('Translation')

plt.tight_layout()
plt.show()
#%%
# Step 2: Define plotting functions

def plot_translations_vs_angles(df, x_axis='Frame', parameter='AP_Tibia'):
    title_addition = "Posterior (+ve) / Anterior (-ve)" if "AP" in parameter else "Superior (+ve) / Inferior (-ve)"
    sns.lineplot(x=x_axis, y=parameter, hue='Condition', style='Dataset', data=df, markers=True, dashes=False) 
    plt.xlabel(f'{x_axis} ({"degrees" if x_axis == "Angle" else "Frame"})')
    plt.ylabel('Translations [mm]')
    plt.title(f'Translations vs. {x_axis} ({title_addition})')
    plt.legend(title='Condition & Dataset', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.grid(True)
    plt.show()

def plot_boxplots(df, parameter='AP_Tibia'):
    title_addition = "Posterior (+ve) / Anterior (-ve)" if "AP" in parameter else "Superior (+ve) / Inferior (-ve)"
    sns.boxplot(x='Condition', y=parameter, data=df)
    plt.title(f'Translations Boxplot ({title_addition})')
    plt.ylabel('Translations [mm]')
    plt.show()
    
#%%
# Group by Frame and Condition, then calculate mean and SD
def plot_translation(df_combined, parameter):
    # Aggregate the data using string representation for mean and std
    grouped = df_combined.groupby(['Frame', 'Condition']).agg(
        Mean_Translation=(parameter, 'mean'),
        SD_Translation=(parameter, 'std')
    ).reset_index()    

    # Filter for Loaded and Unloaded
    grouped_loaded = grouped[grouped['Condition'] == 'Loaded']
    grouped_unloaded = grouped[grouped['Condition'] == 'Unloaded']

    # Find common frame range
    common_frames = set(grouped_loaded['Frame']).intersection(grouped_unloaded['Frame'])
    grouped_loaded = grouped_loaded[grouped_loaded['Frame'].isin(common_frames)]
    grouped_unloaded = grouped_unloaded[grouped_unloaded['Frame'].isin(common_frames)]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting for 'Loaded'
    ax1.plot(grouped_loaded['Frame'], grouped_loaded['Mean_Translation'], label='Loaded', marker='o', color='red')
    ax1.fill_between(grouped_loaded['Frame'], 
                     grouped_loaded['Mean_Translation'] - grouped_loaded['SD_Translation'], 
                     grouped_loaded['Mean_Translation'] + grouped_loaded['SD_Translation'], 
                     color='red', alpha=0.3)
    ax1.set_title('Loaded Condition')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel(f'Mean {parameter} Translation [mm]')
    ax1.grid(True)
    ax1.legend()

    # Plotting for 'Unloaded'
    ax2.plot(grouped_unloaded['Frame'], grouped_unloaded['Mean_Translation'], label='Unloaded', marker='o', color='blue')
    ax2.fill_between(grouped_unloaded['Frame'], 
                     grouped_unloaded['Mean_Translation'] - grouped_unloaded['SD_Translation'], 
                     grouped_unloaded['Mean_Translation'] + grouped_unloaded['SD_Translation'], 
                     color='blue', alpha=0.3)
    ax2.set_title('Unloaded Condition')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel(f'Mean {parameter} Translation [mm]')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# To plot IS_Tibia translations
plot_translation(df_combined, 'IS_Tibia')
# To plot AP_Tibia translations, simply call the function with 'AP_Tibia'
# plot_translation(df_combined, 'AP_Tibia')


#%%
# Find the minimum frame length for each condition across all datasets
max_frame_loaded = min(df_combined[df_combined['Condition'] == 'Loaded'].groupby('Dataset')['Frame'].max())
max_frame_unloaded = min(df_combined[df_combined['Condition'] == 'Unloaded'].groupby('Dataset')['Frame'].max())

# Filter the data up to the minimum frame for each condition
filtered_loaded = df_combined[(df_combined['Condition'] == 'Loaded') & (df_combined['Frame'] <= max_frame_loaded)]
filtered_unloaded = df_combined[(df_combined['Condition'] == 'Unloaded') & (df_combined['Frame'] <= max_frame_unloaded)]

# Calculate the mean IS_Tibia for each frame
mean_loaded = filtered_loaded.groupby('Frame')['IS_Tibia'].mean()
mean_unloaded = filtered_unloaded.groupby('Frame')['IS_Tibia'].mean()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for loaded condition
mean_loaded.plot(ax=axes[0], marker='o', title='Inferior (-ve) / Superior (+ve) Tibia Translations - Loaded')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Mean IS Tibia')

# Plot for unloaded condition
mean_unloaded.plot(ax=axes[1], marker='o', title='Inferior (-ve) / Superior (+ve) Tibia Translations - Unloaded')
axes[1].set_xlabel('Frame')

plt.tight_layout()
plt.show()

#%%
# Calculating max frames for 'Loaded' and 'Unloaded' conditions
max_frame_loaded = min(df_combined[df_combined['Condition'] == 'Loaded'].groupby('Dataset')['Frame'].max())
max_frame_unloaded = min(df_combined[df_combined['Condition'] == 'Unloaded'].groupby('Dataset')['Frame'].max())

# Filtering data up to the minimum frame for each condition
filtered_loaded = df_combined[(df_combined['Condition'] == 'Loaded') & (df_combined['Frame'] <= max_frame_loaded)]
filtered_unloaded = df_combined[(df_combined['Condition'] == 'Unloaded') & (df_combined['Frame'] <= max_frame_unloaded)]

# Calculating mean and standard deviation for IS_Tibia for each frame
mean_loaded = filtered_loaded.groupby('Frame')['IS_Tibia'].mean()
std_loaded = filtered_loaded.groupby('Frame')['IS_Tibia'].std()

mean_unloaded = filtered_unloaded.groupby('Frame')['IS_Tibia'].mean()
std_unloaded = filtered_unloaded.groupby('Frame')['IS_Tibia'].std()

# Plotting with standard deviation as shaded regions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot for loaded condition with standard deviation
axes[0].plot(mean_loaded.index, mean_loaded, marker='o', label='Mean Loaded')
axes[0].fill_between(mean_loaded.index, mean_loaded - std_loaded, mean_loaded + std_loaded, color='blue', alpha=0.2)
axes[0].set_title('Inferior (-ve) / Superior (+ve) Tibia Translations - Loaded')
axes[0].set_xlabel('Frame')
axes[0].set_ylabel('Mean IS Tibia')

# Plot for unloaded condition with standard deviation
axes[1].plot(mean_unloaded.index, mean_unloaded, marker='o', label='Mean Unloaded')
axes[1].fill_between(mean_unloaded.index, mean_unloaded - std_unloaded, mean_unloaded + std_unloaded, color='red', alpha=0.2)
axes[1].set_title('Inferior (-ve) / Superior (+ve) Tibia Translations - Unloaded')
axes[1].set_xlabel('Frame')

plt.tight_layout()
plt.grid()
plt.show()
#%%
# Calculate the minimum frame length for each condition across all datasets
min_frame_loaded = df_combined[df_combined['Condition'] == 'Loaded'].groupby('Dataset')['Frame'].max().min()
min_frame_unloaded = df_combined[df_combined['Condition'] == 'Unloaded'].groupby('Dataset')['Frame'].max().min()

# Filter the data up to the minimum frame for each condition
filtered_loaded = df_combined[(df_combined['Condition'] == 'Loaded') & (df_combined['Frame'] <= min_frame_loaded)]
filtered_unloaded = df_combined[(df_combined['Condition'] == 'Unloaded') & (df_combined['Frame'] <= min_frame_unloaded)]

# Concatenate the filtered data back together
filtered_data = pd.concat([filtered_loaded, filtered_unloaded])

# Calculate the mean and standard deviation for IS_Tibia for each frame and condition
mean_data = filtered_data.groupby(['Condition', 'Frame'])['IS_Tibia'].mean()
std_data = filtered_data.groupby(['Condition', 'Frame'])['IS_Tibia'].std()

# Plotting the mean with standard deviation as shaded regions
plt.figure(figsize=(10, 6))

conditions = ['Loaded', 'Unloaded']
colors = ['blue', 'red']  # Just to differentiate the plots

for condition, color in zip(conditions, colors):
    condition_mean = mean_data[condition]
    condition_std = std_data[condition]
    plt.plot(condition_mean.index, condition_mean, marker='o', color=color, label=f'Mean {condition}')
    plt.fill_between(condition_mean.index, condition_mean - condition_std, condition_mean + condition_std, color=color, alpha=0.2)

plt.title('Mean IS Tibia Translations with Standard Deviation')
plt.xlabel('Frame')
plt.ylabel('Mean IS Tibia')
plt.legend()
plt.show()

#%%
# Plotting translations against frame number
plot_translations_vs_angles(df_combined, 'Frame', 'AP_Tibia')

# Plotting translations against angle
plot_translations_vs_angles(df_combined, 'Angle', 'AP_Tibia')

# Plotting IS translations against frame number
plot_translations_vs_angles(df_combined, 'Frame', 'IS_Tibia')

plot_translations_vs_angles(df_combined, 'Angle', 'IS_Tibia')

# Generating a boxplot for IS translations
plot_boxplots(df_combined, 'IS_Tibia')

#%%
combined_is_loaded = np.vstack([AN_W_is_tib_rel, MK_is_W_tib_rel[:10], MM_W_tib_rel_is[:10]])
combined_is_loaded_mean = np.mean(combined_is_loaded, axis=0)
combined_is_loaded_std = np.std(combined_is_loaded, axis=0)

combined_is_unloaded = np.vstack([AN_NW_is_tib_rel, MK_is_NW_tib_rel[:10], MM_NW_tib_rel_is[:10]])
combined_is_unloaded_mean = np.mean(combined_is_unloaded, axis=0)
combined_is_unloaded_std = np.std(combined_is_unloaded, axis=0)