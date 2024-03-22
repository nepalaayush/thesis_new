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

