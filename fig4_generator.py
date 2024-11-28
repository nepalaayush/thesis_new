# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:45:45 2024

@author: MSI
"""


import pickle
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#%%
first_half_df = pd.read_pickle('first_half_trans_and_angle.pkl') 
second_half_df = pd.read_pickle('second_half_trans_and_angle.pkl')

#%%
def plot_six_panel_translation_and_angle(df_first_half, df_second_half, bin_width=10, figsize=(30, 20), dpi=300, y_padding=0.1, is_y_padding=0.3):
    # Reset matplotlib params to default at the start
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Set the font family to a common system font
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Explicitly set DPI for the figure and display
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    
    # Adjust font sizes based on screen DPI
    base_size = 12  # Base font size
    dpi_scale = dpi / 100.0  # Scale factor based on DPI
    
    TITLE_SIZE = int(base_size * 1.3 * dpi_scale)
    LABEL_SIZE = int(base_size * 1.2 * dpi_scale)
    SUBTITLE_SIZE = int(base_size * 1.0 * dpi_scale)  # Reduced subtitle size
    TICK_SIZE = int(base_size * dpi_scale)
    LEGEND_SIZE = int(base_size * dpi_scale)
    
    # Create figure with adjusted size
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    
    def plot_data(df, ax, column, title, is_angle=False, is_is=False, show_legend=False):
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
        ax.set_xlabel("Flexion [%]", fontsize=LABEL_SIZE, fontfamily='DejaVu Sans')
        ax.set_ylabel("Angle [°]" if is_angle else "Translation [mm]", 
                     fontsize=LABEL_SIZE, fontfamily='DejaVu Sans')
        ax.set_title(title, fontsize=SUBTITLE_SIZE, pad=10, fontfamily='DejaVu Sans')
        
        # Set major and minor ticks for x-axis
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))    # Major ticks every 20
        ax.xaxis.set_minor_locator(plt.MultipleLocator(5))     # Minor ticks every 5
        
        # Set major and minor ticks for y-axis based on plot type
        if is_angle:
            ax.yaxis.set_major_locator(plt.MultipleLocator(10))  # Major ticks every 10
            ax.yaxis.set_minor_locator(plt.MultipleLocator(2.5)) # Minor ticks every 2.5
        elif is_is:
            ax.yaxis.set_major_locator(plt.MultipleLocator(2))   # Major ticks every 2
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5)) # Minor ticks every 0.5
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))   # Major ticks every 5
            ax.yaxis.set_minor_locator(plt.MultipleLocator(1.25))# Minor ticks every 1.25
        
        # Grid settings - show both major and minor grid with same style
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.4)  # Show major grid
        ax.grid(True, which='minor', linestyle='-', linewidth=0.5, alpha=0.4)  # Show minor grid
        
        # Make minor ticks more visible
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.tick_params(axis='both', which='minor', length=4, width=1, labelsize=TICK_SIZE)
        
        # Place legend inside the plot
        if show_legend:
            ax.legend(title='Method', fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE,
                     loc='lower right', bbox_to_anchor=(0.95, 0.05))
        else:
            ax.legend().remove()
        
        # Y-axis limits
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        padding = is_y_padding if is_is else y_padding
        ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
        
        if is_angle:
            ax.set_ylim([-25,35])
            ax.set_yticks(np.linspace(-20, 30, 6))
        elif is_is:
            ax.set_ylim([-63,-51])
            ax.set_yticks(np.linspace(-62, -52, 6))
        else:
            ax.set_ylim([-32.5,-2.5])        
            ax.set_yticks(np.linspace(-30, -5, 6))
        
        # Calculate statistics
        results = {}
        for method in ['Manual', 'Semi-Auto']:
            method_data = grouped[grouped['Method'] == method]
            start_data = method_data[method_data['Bin_Center'] == method_data['Bin_Center'].min()]
            start_value = start_data[column].mean()
            start_std = start_data[column].std()
            end_data = method_data[method_data['Bin_Center'] == method_data['Bin_Center'].max()]
            end_value = end_data[column].mean()
            end_std = end_data[column].std()
            change = end_value - start_value
            change_std = np.sqrt(start_std**2 + end_std**2)
            results[method] = {
                'start': (start_value, start_std),
                'end': (end_value, end_std),
                'change': (change, change_std)
            }
        return results
    
    changes = []
    
    # Plot the panels - show legend in top-middle panel
    changes.append(plot_data(df_first_half, axs[0, 0], 'Angle', '', is_angle=True))
    changes.append(plot_data(df_first_half, axs[0, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)', show_legend=True))
    changes.append(plot_data(df_first_half, axs[0, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True))
    changes.append(plot_data(df_second_half, axs[1, 0], 'Angle', '', is_angle=True))
    changes.append(plot_data(df_second_half, axs[1, 1], 'AP_Translation', 'Anterior (+ve) / Posterior (-ve)'))
    changes.append(plot_data(df_second_half, axs[1, 2], 'IS_Translation', 'Superior (+ve) / Inferior (-ve)', is_is=True))
    
    # Adjust layout first
    plt.tight_layout()
    
    # Add phase labels with adjusted positions
    fig.text(0.5, 0.95, 'Extension Phase (Flexed to Extended)', 
             ha='center', va='center', fontsize=TITLE_SIZE, 
             fontweight='bold', fontfamily='DejaVu Sans')
    fig.text(0.5, 0.48, 'Flexion Phase (Extended to Flexed)', 
             ha='center', va='center', fontsize=TITLE_SIZE, 
             fontweight='bold', fontfamily='DejaVu Sans')
    
    # Final layout adjustments
    plt.subplots_adjust(top=0.90, bottom=0.07, left=0.05, right=0.90, hspace=0.4, wspace=0.3)
    
    # Print statistics
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

#%%
fig = plot_six_panel_translation_and_angle(first_half_df, second_half_df, bin_width=10, figsize=(30, 20), dpi=300, y_padding=0.3, is_y_padding=0.9)
fig.savefig('six_panels_with_grid_test.svg', bbox_inches='tight')

#%%
cost_df = pd.read_pickle('C:/Users/MSI/Documents/thesis_new/tib_fem_cost_dfs.pkl')

#%%
tibia_avg = cost_df[cost_df['Bone'] == 'Tibia']['Average Cost'].mean()
femur_avg = cost_df[cost_df['Bone'] == 'Femur']['Average Cost'].mean()

print(f"Tibia average cost: {tibia_avg:.2f}")
print(f"Femur average cost: {femur_avg:.2f}")

#%%
means = cost_df.groupby(['Dataset', 'Bone'])['Average Cost'].mean() * 0.48484848484848486
print(means)