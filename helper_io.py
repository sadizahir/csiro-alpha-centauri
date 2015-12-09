# -*- coding: utf-8 -*-
"""
Helper functions for reading files.
Created on Thu Dec 03 10:05:27 2015

@author: ZAH010
"""

from __future__ import print_function

# Basic Packages
import nibabel as nib
import numpy as np

# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 0
SHOW_IMG = 1
TEST = 1  

"""
Extracts the data for single image from a NIFTI-encoded medical image.
Returns the raw image data and the affine transformation as a tuple.

get_nifti_slice(str, int) --> (np.ndarray, np.ndarray)
"""
def get_nifti_slice(filename, slice_no):
    # Load the data
    image = nib.load(filename)
    image_data = image.get_data()
    orientation = image.get_affine()
    
    if DEBUG:
        print("Debug is ", DEBUG)
        print("NIFTI Shape: ", image_data.shape)
        
    # Extract the slice from the 3D volume
    if len(image_data.shape) == 4:
        image_slice = image_data[:, :, slice_no, 0]
    else:
        image_slice = image_data[:, :, slice_no]
    
    # Everything is backwards when you display it
    image_slice = np.fliplr(image_slice).T
    image_slice = np.flipud(image_slice)
    orientation_slice = np.fliplr(orientation).T
    orientation_slice = np.flipud(orientation)
    
    if DEBUG:
        print("Slice Shape: ", image_slice.shape)
        
    return (image_slice, orientation_slice)

"""
Loads the NRRD data, which is a label for ROIs in an associated NIFTI.

get_nrrd_data(str, int) --> (np.ndarray, np.ndarray)
"""
def get_nrrd_data(filename_label, slice_no):
    # Load the data
    label = nib.load(filename_label)
    label_data = label.get_data()
    orientation = label.get_affine()
    
    if DEBUG:
        print("Label (Volume) Shape: ", label_data.shape)
    
    # Extract the slices from the 3D Volume
    label_slice = label_data[:, :, slice_no]
    label_slice = np.fliplr(label_slice).T
    label_slice = np.flipud(label_slice)
    orientation_label = np.fliplr(orientation).T
    orientation_label = np.flipud(orientation)

    if DEBUG:
        print("Label (Slice) Shape: ", label_slice.shape)
    
    return (label_slice, orientation_label)
    
"""
Determine the slice number with the highest amount of label.
"""
def find_biggest_slice(filename_label):
    label = nib.load(filename_label)
    label_data = label.get_data()
    slice_count = 0 # counts the number of nonzero pixels in the biggest slice so far
    slice_no = 0 # the biggest slice so far    
    
    # Go through each slice of the label volume in order to determine the
    # slice with the largest label
    for i in range(label_data.shape[2]): # 2 will be the number of slices
        label_slice = label_data[:, :, i]
        current_count = np.count_nonzero(label_slice)
        #print("Slice {}, Count {}".format(i, current_count))
        if current_count > slice_count:
            slice_count = current_count
            slice_no = i
    
    if DEBUG:
        print("Largest slice for ", filename_label, " is #", slice_no)
    
    return slice_no