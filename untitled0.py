# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:41:31 2023

@author: Aayush
"""

import pydicom as dicom 
import napari
import numpy as np 
import os 
#%%

im1 = dicom.dcmread('C:/Users/Aayush/Downloads/0003.DCM')
napari.view_image(im1.pixel_array)
#%%
file_path = "im0.txt"

with open(file_path, 'w') as f:
    f.write(str(im0))
    
#%%
def get_pixelArray(dicomlist):
    ''' 
    Input: A list where each item in the list is an individual dicom file (individual dataset). 
    Output: Gives an array of shape n * x * y where n is the number of dicom files. so basically, given a folder of datasets, this function stores this folder as an array. 
    ''' 
    pixelArray = []
    for i in dicomlist:
        i.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian # depends on the file 
        pixelArray.append(i.pixel_array)

    return np.asarray(pixelArray)

def get_dicom_list(path):
    ''' Input: the path of a folder that contains exclusively dicom file format images 
    Output: a list that stores each dicom file. 
    ''' 
    os.chdir(path)
    dicom_list = []
    for file in os.listdir(path):
        dicom_list.append(dicom.dcmread(file, force=True)  )
    return dicom_list

dicomlist = get_dicom_list('C:/Users/Aayush/Downloads/2/dicom')
pdata = get_pixelArray(dicomlist)
#%%
napari.view_image(pdata)