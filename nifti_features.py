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

# Scientific
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

# Helpers
from helper_io import get_nifti_slice, get_nrrd_data
from helper_eigen import extract_roi_patches, get_eigenpatches, show_eigenpatches

# Patch Generation Constants
PATCH_SIZE = 20
MAX_EIGEN = 25

# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 1
SHOW_IMG = 1
TEST = 1  


"""
Generates Gabor kernels.
"""
def generate_kernels():
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

"""
Image normalisation from the Gabor example.
"""
def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

"""
From a bunch of kernels and an image, compute a set of features of the image.
"""    
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats
    
"""
From a kernel and an image, compute all the powers of an image.
"""
def compute_powers(image, kernels):
    powers = []
    for kernel in kernels:
        powers.append(power(image, kernel))
    return powers

"""
Compares features against reference features.
"""
def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i
    
"""
Plot a selection of filter banks and their Gabor responses.
"""
def plot_gabor(images): #, patch_size):
    
    #images = images.reshape(images.shape[0], patch_size, patch_size)
    
    results = []
    kernel_params = []
    for theta in (0, 1):
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, 
                                                    frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel) for img in images]))
    
    fig, axes = plt.subplots(nrows=5, ncols=11, figsize=(11, 6))
    plt.gray()
    
    fig.suptitle('Image responses for (a selection of) Gabor filter kernels', 
                 fontsize=16)
    
    axes[0][0].axis('off')
    
    # Plot original images
    for label, img, ax in zip([str(i) for i in range(0, len(images))], 
                               images, axes[0][1:]):
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
        # Plot Gabor kernel
        ax = ax_row[0]
        ax.imshow(np.real(kernel), cmap=plt.cm.gray, interpolation='nearest')
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
        # Plot Gabor responses with the contrast normalized for each filter
        vmin = np.min(powers)
        vmax = np.max(powers)
        for patch, ax in zip(powers, ax_row[1:]):
            ax.imshow(patch, vmin=vmin, vmax=vmax, cmap=plt.cm.gray,
                      interpolation='nearest')
            ax.axis('off')
    
    plt.show()

  
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
    
    
    
    
    
    
    
    