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
DEBUG = 1
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
        print("NIFTI Shape: ", image_data.shape)
        
    # Extract the slice from the 3D volume
    image_slice = image_data[:, :, image_data.shape[2]-slice_no, 0]
    
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
    label_slice = label_data[:, :, label_data.shape[2]-slice_no]
    label_slice = np.fliplr(label_slice).T
    label_slice = np.flipud(label_slice)
    orientation_label = np.fliplr(orientation).T
    orientation_label = np.flipud(orientation)

    if DEBUG:
        print("Label (Slice) Shape: ", label_slice.shape)
    
    return (label_slice, orientation_label) 