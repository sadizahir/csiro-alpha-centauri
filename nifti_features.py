# -*- coding: utf-8 -*-
"""
Helper functions to read medical images, extract patches, perform PCA,
apply Gabor filters and plot the results.
Created on Wed Dec 02 08:00:05 2015

@author: ZAH010
"""

from __future__ import print_function
from time import time

# Basic Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Scientific
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA

# Patch Generation Constants
PATCH_SIZE = 20
MAX_EIGEN = 25

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
    
"""
Given an MRI slice and a matching label for that slice, extract patches from
the area of interest (1 in the binary label image) and return an array of the
patches.

extract_roi_patches(np.ndarray, np.ndarray, int) --> list(np.ndarray)
"""
def extract_roi_patches(image, label, patch_size):
    # Extract patches
    psize = (patch_size, patch_size)
    patches_original = extract_patches_2d(image, psize)
    patches_label = extract_patches_2d(label, psize)
    patches_mask = []
    
    for i, patch in enumerate(patches_original):
        if 0 not in patches_label[i]:
            patches_mask.append(patch)
            
    if DEBUG:
        print("Number of ROI patches: ", len(patches_mask))
            
    return patches_mask

"""
Generates important component patches given a set of normally-extracted
patches by using PCA.

get_eigenpatches(np.ndarray, int) --> np.ndarray
"""
def get_eigenpatches(patches, no_components):
    # Configure PCA    
    pca = PCA(n_components=no_components)
    
    # Grab and reshape the patches
    patches = np.array(patches)
    patches = patches.reshape(patches.shape[0], -1)
    #patches_mask -= np.mean(patches_mask, axis=0)
    #patches_mask /= np.std(patches_mask, axis=0)
    
    # Decompose
    t0 = time()
    eigens = pca.fit(patches).components_
    dt = time() - t0
    
    if DEBUG:
        print("Decomposed in %.2fs." % dt)
    
    return eigens
    
"""
Given some eigenpatches, plot them on the screen. This should work for
all patches in general.

show_eigenpatches_(np.ndarray) --> None
"""
def show_eigenpatches(eigens, patch_size):
    psize = (patch_size, patch_size)
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(eigens):
        plt.subplot(5, 5, i + 1)
        plt.imshow(comp.reshape(psize), cmap=plt.cm.gray_r, 
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('Eigen-decomposition of Patches in ROI\n', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()
    
"""
This is where I test!
"""
def test_routine():
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
    eigens = get_eigenpatches(patches_mask, MAX_EIGEN)
    
    # Show the eigens, if you want
    if SHOW_IMG:
        show_eigenpatches(eigens, PATCH_SIZE)

if TEST:
    test_routine()
    
    
    
    
    