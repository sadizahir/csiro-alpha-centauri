# -*- coding: utf-8 -*-
"""
Helper functions to read medical images, extract patches, perform PCA,
apply Gabor filters and plot the results.
Created on Wed Dec 02 08:00:05 2015

@author: ZAH010
"""

from __future__ import print_function

# Basic Packages
import numpy as np
import matplotlib.pyplot as plt

# Helpers
from helper_io import get_nifti_slice, get_nrrd_data
from helper_eigen import extract_roi_patches, get_eigenpatches
from helper_eigen import show_eigenpatches
from helper_gabor import generate_kernels, power, compute_feats
from helper_gabor import compute_powers, match, plot_gabor

# Patch Generation Constants
PATCH_SIZE = 20
MAX_EIGEN = 25

# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 1
SHOW_IMG = 1
TEST = 1  
  
"""
This is where I test!
"""
def test_routine():
    pass

if TEST:
    test_routine()
    path = ""
    filename = path + "anon_mr_150420_30sec.nii.gz"
    filename_label = path + "anon_mr_150420_30sec.nii-label.nrrd.nii.gz"
    slice_no = 51
    print("Example: ", filename, "\n")
    
    # Grab the image
    image_slice, orientation_slice = get_nifti_slice(filename, slice_no)
    if SHOW_IMG:
        plt.imshow(image_slice, cmap = plt.cm.gray)
        plt.show()
    
    # Grab the labels
    label_slice, orientation_label = get_nrrd_data(filename_label, slice_no)
    if SHOW_IMG:
        plt.imshow(label_slice, cmap=plt.cm.gray)
        plt.show()

    # Show the mask
    if SHOW_IMG:
        print("Masked version: ")
        mask = np.where(label_slice == 0, label_slice, image_slice)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.show()  
    
    # Extract patches in ROI
    patches_mask = extract_roi_patches(image_slice, label_slice, PATCH_SIZE)
    
    # Get the decomposed patches
    eigens = get_eigenpatches(patches_mask, PATCH_SIZE, MAX_EIGEN)
    
    # Show the eigens, if you want
    if SHOW_IMG:
        show_eigenpatches(eigens)
        
    # Generate Gabor Kernels
    kernels = generate_kernels()
    
    # Show the Gabors
    plot_gabor(eigens)

    # Store all the features and Gabor responses    
    all_features = []
    all_powers = []
    for eigen in eigens:
        all_features.append(compute_feats(eigen, kernels))
        all_powers.append(compute_powers(eigen, kernels))
    
    
    
    
    
    
    
    