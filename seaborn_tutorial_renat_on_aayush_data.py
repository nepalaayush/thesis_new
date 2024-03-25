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

# %% 
fg = sns.relplot(
    equal_frame_df, 
    x="Frame Number", 
    y="Relative Translation", 
    col="Condition", 
    hue="Dataset", 
    kind="line", 
#    facet_kws={"sharey":False}
)
fg.refline(y=0)

# %%
fg = sns.relplot(
    equal_frame_df, 
    x="Frame Number", 
    y="Relative Translation", 
    col="Condition", 
    #hue="Dataset", 
    kind="line",
    marker="o"
#    facet_kws={"sharey":False}
)
fg.refline(y=0)
# %% 

sns.relplot(
    data=equal_frame_df, 
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