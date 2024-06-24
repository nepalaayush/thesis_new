#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:22:06 2024

@author: aayush
"""

import pickle
import os 
#os.chdir('C:/Users/Aayush/Documents/thesis_files/thesis_new')
os.chdir('/data/projects/ma-nepal-segmentation/scripts/git/thesis_new')
#%%
import numpy as np 
import napari 
import time 
from scipy import ndimage

from utils import (path_to_image, apply_canny, apply_remove, apply_skeleton, points_for_napari,
                   boolean_to_coords, apply_label, find_tibia_edges, find_array_with_min_n, downsample_points,
                   combined_consecutive_transform, coords_to_boolean)

    
#%%
# Step 1: load the image from directory and normalize it
path_neg = '/data/projects/ma-nepal-segmentation/data/Maggioni^Marta_Brigid/2024-03-08/108_MK_Radial_NW_CINE_30bpm_CGA/MM_NW_ai2_tgv_5e-2_neg_ngn.nii'
path_pos ='/data/projects/ma-nepal-segmentation/data/Maggioni^Marta_Brigid/2024-03-08/108_MK_Radial_NW_CINE_30bpm_CGA/MM_NW_ai2_tgv_5e-2_pos_ngn.nii'
#%%
image_neg = path_to_image(path_neg)[2:]
image_pos = path_to_image(path_pos)[2:]
#%%
# since our image goes from extened to flexed.. the direction means, pos is going down.. and neg is coming up 
# which means. if we want to present our data as going up then coming down .. we have to reverse the neg, put it at the first half. 
image_neg = image_neg[::-1]
#%%
full_image = np.concatenate( (image_neg, image_pos) , axis=0)

#%%
#add the original image to napari
viewer = napari.view_image(full_image,  name='MM_NW_full')