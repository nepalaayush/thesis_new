#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:55:40 2024

@author: aayush
"""

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("talk")

with open('/data/projects/ma-nepal-segmentation/data/data_20_03/angle_and_rel_df.pkl', 'rb') as file:
    angle_and_rel_df =  pickle.load(file)
angle_and_rel_df.head()
#%%
angle_and_rel_df["Dataset"] = pd.Categorical(angle_and_rel_df["Dataset"].apply(lambda x: x.split(" ")[1]))
angle_and_rel_df["Condition"] = pd.Categorical(angle_and_rel_df["Condition"])
angle_and_rel_df["Type"] = pd.Categorical(angle_and_rel_df["Type"])
angle_and_rel_df.head()
#%%
fg = sns.lmplot(data=angle_and_rel_df, x='Angles', y='Relative Translation', hue='Dataset', ci=None, col="Condition")
fg.fig.suptitle('Translation vs. Angles for Unloaded Condition', y=1.05)
for ax in fg.axes.flat:
    ax.set_xlabel('Angle (Degree)')
fg.axes[0,0].set_ylabel('Translation (mm)')
#%%

frames_to_remove = [16, 17, 15, 18, 14, 19]
equal_frame_df = angle_and_rel_df[
    ~(angle_and_rel_df['Frame Number'].isin(frames_to_remove) & angle_and_rel_df['Dataset'].isin(["2", "3"]))
].reset_index(drop=True)
equal_frame_df['Frame Number'] = equal_frame_df.groupby(['Dataset', 'Condition']).cumcount()

#%%
''' this is a working code to plot the IS translation for all datasets w.r.t frame or percent flexed. the mean can be adjusted by commenting out hue 
'''  
fg = sns.relplot(
    is_df_1_7, 
    x="Percent Flexed",
    #x = 'Frame Number',
    y="Relative Translation", 
    col="Condition", 
    hue="Dataset", 
    kind="line", 
    #ci='sd', # one sd shows a bit more variation than the default 95% confidence interval 
#    facet_kws={"sharey":False}
)
fg.refline(y=0)


fg.fig.suptitle('Inferior(+ve)/Superior(-ve) translation of the tibia relative to femur', fontsize=16, ha='center')
fg.fig.subplots_adjust(top=0.86)


#%%

# Define narrow bin edges
narrow_bin_edges = range(-100, 101, 7)  # Bins from -100 to 100 with a step of 1

# Assign each 'Percent Flexed' value to a narrow bin
is_df['Narrow_Bin'] = pd.cut(is_df['Percent Flexed'], bins=narrow_bin_edges, include_lowest=True)

# Make sure to include 'Condition' in the groupby
narrow_binned_means = is_df.groupby(['Condition', 'Narrow_Bin', 'Dataset'])['Relative Translation'].mean().reset_index()

# Calculate the bin centers
narrow_binned_means['Bin_Center'] = narrow_binned_means['Narrow_Bin'].apply(lambda x: x.mid)

# Plotting
fg = sns.relplot(
    data=narrow_binned_means, 
    x="Bin_Center", 
    y="Relative Translation", 
    col="Condition", 
    #hue="Dataset", 
    kind="line"
)

# Add a reference line at y=0
fg.refline(y=0)

# Adjust the layout and display the plot
plt.subplots_adjust(top=0.9)
plt.show()
# %%
fg = sns.relplot(
    angle_and_rel_df[ (angle_and_rel_df['Type'] =='IS')], 
    x="Percent Flexed", 
    y="Relative Translation", 
    col="Condition", 
    #hue="Dataset", 
    kind="line",
    marker="o",
    ci=None,
#    facet_kws={"sharey":False}
)
fg.refline(y=0)
# %% 

sns.relplot(
    data=angle_and_rel_df, 
    x='Angles', 
    y='Relative Translation', 
    col='Condition', 
    kind='line', 
    hue="Dataset"
)

#%%
 
NBIN = 10
# equal_frame_df["Angles binned"] = pd.cut(equal_frame_df["Angles"], NBIN)

#%%
equal_frame_df_agg = (
 equal_frame_df
 .assign(Angles_binned=pd.cut(equal_frame_df["Angles"], NBIN))
 .groupby(["Dataset", "Condition", "Angles_binned"])
 [["Relative Translation", "Angles"]]
 .agg({"Angles": "mean", "Relative Translation" : ["mean", "std"]})
 .dropna()
)
# %%
fg = sns.relplot(
    data=equal_frame_df_agg, 
    x=("Angles", "mean"), 
    y=("Relative Translation", "mean"), 
    hue="Dataset", 
    row="Condition"
)
for (ds, cond), df_ in equal_frame_df_agg.groupby(["Dataset", "Condition"]):
    fg.axes_dict[cond].fill_between(
        df_[("Angles", "mean")], 
        df_[("Relative Translation", "mean")] - df_[("Relative Translation", "std")], 
        df_[("Relative Translation", "mean")] + df_[("Relative Translation", "std")], 
        alpha=0.5,
        color=f"C{int(ds)-1}"
    )
#%%
equal_frame_df_agg = (
 equal_frame_df
 .assign(Angles_binned=pd.cut(equal_frame_df["Angles"], NBIN))
 .groupby(["Condition", "Angles_binned"])
 [["Relative Translation", "Angles"]]
 .agg({"Angles": "mean", "Relative Translation" : ["mean", "std"]})
 .dropna()
)
# %%
fg = sns.relplot(
    data=equal_frame_df_agg, 
    x=("Angles", "mean"), 
    y=("Relative Translation", "mean"), 
    row="Condition"
)
for cond, df_ in equal_frame_df_agg.groupby(["Condition"]):
    fg.axes_dict[cond].fill_between(
        df_[("Angles", "mean")], 
        df_[("Relative Translation", "mean")] - df_[("Relative Translation", "std")], 
        df_[("Relative Translation", "mean")] + df_[("Relative Translation", "std")], 
        alpha=0.5,
    )