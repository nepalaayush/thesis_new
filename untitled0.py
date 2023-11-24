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

im1 = dicom.dcmread('C:/Users/Aayush/Documents/thesis_files/internship_stuff/0011175460 Phantom Internship Gadovistreihe/0047274091 Medphys Internship/MR se_32TE/MR000000.dcm')
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
def dir_check(p):
    ''' checks if the directory contains folder or a file. Returns if folder, doesnt return a file.  ''' 
    for entry in os.scandir(p):
        if entry.is_dir():
            return(entry)


def delete_firstindex(array):
    array = np.delete(array, [0], axis = 0)
    return array
#%%
path_t2 = r'C:\Users\Aayush\Documents\thesis_files\internship_stuff\0011175460 Phantom Internship Gadovistreihe\0047274091 Medphys Internship\MR se_32TE'
#%%
dicomlist = get_dicom_list(r'C:\Users\Aayush\Documents\thesis_files\internship_stuff\0011175460 Phantom Internship Gadovistreihe\0047274091 Medphys Internship\MR se_32TE')
#%%
pdata = get_pixelArray(dicomlist)
#%%
napari.view_image(pdata)
''' contrary to my initial belief, actually, the dicom that is new, is still able to load and show the image. the problem then must lie in echo times stuff or perhaps the string literal stuff  ''' 
#%%


#%%
def save_dicom_to_text(dicom_dataset, text_file_path):
    """
    Save the DICOM dataset to a text file.

    :param dicom_dataset: pydicom FileDataset object
    :param text_file_path: Path to the text file where the dataset will be saved
    """
    with open(text_file_path, 'w') as file:
        for tag in dicom_dataset.keys():
            data_element = dicom_dataset[tag]
            try:
                file.write(f"{str(tag)} {data_element.name}: {data_element.value}\n")
            except:
                file.write(f"{str(tag)} {data_element.name}: [Error retrieving value]\n")

# Example usage
# Replace 'your_text_file_path.txt' with the path where you want to save the file
save_dicom_to_text(dicomlist[0], 'first_echo.txt')


#%%
def extract_echo_time(dicom_dataset):
    """
    Extract the effective echo time from a DICOM dataset, considering multi-frame structure.

    :param dicom_dataset: pydicom FileDataset object
    :return: Effective echo time or None if not found
    """
    try:
        # Accessing the multi-frame functional groups sequence
        multi_frame_sequence = dicom_dataset.get((0x5200, 0x9230), None)
        if multi_frame_sequence:
            # Accessing the MR Echo Sequence
            mr_echo_sequence = multi_frame_sequence[0].get((0x0018, 0x9114), None)
            if mr_echo_sequence:
                # Retrieving the Effective Echo Time
                effective_echo_time = mr_echo_sequence[0].get('EffectiveEchoTime', None)
                if effective_echo_time is not None:
                    return effective_echo_time  # directly return the value, which might be a float
    except Exception as e:
        print(f"Error while extracting echo time: {e}")

    return None

# Test the function
echo_time = extract_echo_time(dicomlist[0])
if echo_time is not None:
    print(f"Echo Time: {echo_time}")
else:
    print("Echo Time not found in the dataset.")

#%% 
def get_t2data_new(path):
    dlist = get_dicom_list(path)

    pixel_data_per_echo = []
    echo_times = []

    for dicom_file in dlist:
        # Extract echo time from each DICOM file
        echo_time = extract_echo_time(dicom_file)
        echo_times.append(echo_time)

        # Extract pixel data (slices) from each DICOM file
        pixel_data = dicom_file.pixel_array
        pixel_data_per_echo.append(pixel_data)

    # Combine pixel data for all echos into a single 4D array
    pixelarray = np.stack(pixel_data_per_echo, axis=0)

    return pixelarray, np.array(echo_times), dlist


try_pixel , try_echo , try_dlist = get_t2data_new(path_t2)

''' by the way this cell actually works. so do not delete  ''' 

#%%
def get_t2data_new(directories):
    all_pixel_data = []
    all_echo_times = []
    all_dlists = []

    for directory in directories:
        dlist = get_dicom_list(directory)
        pixel_data_per_echo = []
        echo_times = []

        for dicom_file in dlist:
            # Extract echo time from each DICOM file
            echo_time = extract_echo_time(dicom_file)
            echo_times.append(echo_time)

            # Extract pixel data (slices) from each DICOM file
            pixel_data = dicom_file.pixel_array
            pixel_data_per_echo.append(pixel_data)

        # Combine pixel data for all echos into a single 4D array for this directory
        pixelarray = np.stack(pixel_data_per_echo, axis=0)

        # Accumulate results from this directory
        all_pixel_data.append(pixelarray)
        all_echo_times.append(echo_times)
        all_dlists.append(dlist)

    # Depending on your requirements, you can return combined results
    # or handle them as needed for further processing
    return all_pixel_data, all_echo_times, all_dlists