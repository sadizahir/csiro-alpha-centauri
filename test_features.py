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
import os
import dill

# Sci
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages

# Helpers
from helper_io import get_nifti_slice, get_nrrd_data, find_biggest_slice
from helper_eigen import extract_roi_patches, get_eigenpatches
from helper_eigen import show_eigenpatches
from helper_gabor import generate_kernels, power, compute_feats
from helper_gabor import compute_powers, match, plot_gabor


# Patch Generation Constants
PATCH_SIZE = 20
MAX_EIGEN = 10

# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 1
SHOW_IMG = 0
GENERATE = 1
REPORTS = 0
CLASSIFY = 0

class SliceInfo():
    def __init__(self, filename, slice_no, image_slice, orientation_slice, 
                 label_slice, orientation_label, kernels,
                 patches_mask, eigens_mask,
                 all_features_mask, all_powers_mask,
                 patches_nonmask, eigens_nonmask,
                 all_features_nonmask, all_powers_nonmask):
        
        self.filename = filename
        self.slice_no = slice_no
        
        self.image_slice = image_slice
        self.orientation_slice = orientation_slice
        
        self.label_slice = label_slice
        self.orientation_label = orientation_label

        self.kernels = kernels        
        
        #self.patches_mask = patches_mask
        #self.patches_nonmask = patches_nonmask
        
        self.eigens_mask = eigens_mask
        self.eigens_nonmask = eigens_nonmask
        
        self.all_features_mask = all_features_mask
        self.all_powers_mask = all_powers_mask
        self.all_features_nonmask = all_features_nonmask
        self.all_powers_nonmask = all_powers_nonmask
    
    def shape_info(self):
        print("Image: ", np.array(self.image_slice).shape)
        print("Orientation: ", np.array(self.orientation_slice).shape)
        print("Label: ", np.array(self.label_slice).shape)
        print("Label Orientation: ", np.array(self.orientation_label).shape)
        print("Kernels: ", np.array(self.kernels).shape)
        #print("Masked Patches: ", np.array(self.patches_mask).shape)
        print("Masked Eigens: ", np.array(self.eigens_mask).shape)
        print("Masked Features: ", np.array(self.all_features_mask).shape)
        print("Masked Powers: ", np.array(self.all_powers_mask).shape)
        #print("Unmasked Patches: ", np.array(self.patches_nonmask).shape)
        print("Unmasked Eigens: ", np.array(self.eigens_nonmask).shape)
        print("Unmasked Features: ", np.array(self.all_features_nonmask).shape)
        print("Unmasked Powers: ", np.array(self.all_powers_nonmask).shape)
        
    def display_all(self):
        plt.imshow(self.image_slice, cmap = plt.cm.gray)
        plt.show()
        
        plt.imshow(self.label_slice, cmap=plt.cm.gray)
        plt.show()
        
        print("Masked version: ")
        mask = np.where(self.label_slice == 0, self.label_slice, 
                        self.image_slice)
        plt.imshow(mask, cmap=plt.cm.gray)
        plt.show()
        
        show_eigenpatches(self.eigens_mask)
        
        plot_gabor(self.eigens_mask)
        
    def save_report(self, path):
        save_fn = path + os.path.basename(self.filename)
        eigen_fn = show_eigenpatches(self.eigens_mask, save_fn)   
        gabor_fn = plot_gabor(self.eigens_mask, save_fn)
        
        # base pdf
        plt.figure(figsize=(20, 28))
        
        plt.suptitle('Report: Slice ' + str(self.slice_no) + 
        ' for ' + self.filename + '\n', fontsize=48)
        
        # prepare the mask
        mask = np.where(self.label_slice == 0, self.label_slice, 
                        self.image_slice)
        
        # Saves a summary of the slice information as pdf 
        with PdfPages(save_fn + '.pdf') as pdf:
            plt.subplot(2, 3, 1)
            plt.imshow(self.image_slice, cmap = plt.cm.gray)
            plt.xticks(())
            plt.yticks(()) 
            plt.title('Original Image', fontsize=24)
            
            plt.subplot(2, 3, 2)
            plt.imshow(mask, cmap = plt.cm.gray)
            plt.xticks(())
            plt.yticks(())
            plt.title('Mask', fontsize=24)
            
            plt.subplot(2, 3, 3)
            plt.imshow(plt.imread(eigen_fn))
            plt.xticks(())
            plt.yticks(())
            plt.title('Eigenpatches', fontsize=24)
            
            plt.subplot(2, 1, 2)
            plt.imshow(plt.imread(gabor_fn))
            plt.xticks(())
            plt.yticks(())
            plt.title('Gabor responses', fontsize=24)
            
            pdf.savefig()
            plt.close()

"""
Given a filename, open the volume, extract a slice and associated slice
from a label file, extract patches from a region of interest, decompose the
patches, and apply Gabor filters.
"""
def process(filename, filename_label, slice_no):
    
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
    patches_mask, patches_nonmask = extract_roi_patches(image_slice, 
                                                        label_slice, 
                                                        PATCH_SIZE)
    
    # Get the decomposed patches
    eigens_mask = get_eigenpatches(patches_mask, PATCH_SIZE, MAX_EIGEN)
    eigens_nonmask = get_eigenpatches(patches_nonmask, PATCH_SIZE, MAX_EIGEN)
    
    # Show the eigens, if you want
    if SHOW_IMG:
        show_eigenpatches(eigens_mask)
        
    # Generate Gabor Kernels
    kernels = generate_kernels()
    
    # Show the Gabors
    if SHOW_IMG:
        plot_gabor(eigens_mask)

    # Store all the features and Gabor responses    
    all_features_mask = []
    all_powers_mask = []
    complete_features_mask = []
    
    all_features_nonmask = []
    all_powers_nonmask = []
    complete_features_nonmask = []
    
    for eigen in eigens_mask:
        all_features_mask.append(compute_feats(eigen, kernels))
        all_powers_mask.append(compute_powers(eigen, kernels))
    
    for eigen in eigens_nonmask:
        all_features_nonmask.append(compute_feats(eigen, kernels))
        all_powers_nonmask.append(compute_powers(eigen, kernels))
        
#    for patch in patches_mask:
#        complete_features_mask.append(compute_feats(patch, kernels))
#    
#    for patch in patches_nonmask:
#        complete_features_nonmask.append(compute_feats(patch, kernels))
        
        
    return SliceInfo(filename, slice_no, image_slice, orientation_slice, 
                     label_slice, orientation_label, kernels,
                     patches_mask, eigens_mask,
                     all_features_mask, all_powers_mask,
                     patches_nonmask, eigens_nonmask,
                     all_features_nonmask, all_powers_nonmask)
""" 
Given a set of features of associated labels, trains a random forest
classifier.
"""
def train_rf_classifier(features, labels, no_trees):
    rf = RandomForestClassifier(n_estimators=no_trees, n_jobs=-1)
    rf.fit(features, labels)

"""
This is where I test!
"""
def test_routine():
    pass

if GENERATE:
    test_routine()
    path = "ct/"

    filenames = []
    filenames_label = []

    for fn in os.listdir(path):
        if "CTV" in fn:
            filenames_label.append(path + fn)
        else:
            filenames.append(path + fn)
    
    #filenames = [path + "anon_mr_150420_30sec.nii.gz"]
    #filenames_label = [path + "anon_mr_150420_30sec.nii-label.nrrd.nii.gz"]
    #slice_no = 128-105
    
    slice_infos = []
    
    for i, fn in enumerate(filenames):
        slice_no = find_biggest_slice(filenames_label[i])
        if DEBUG:
            print("Processing: ", fn, ", Slice ", slice_no)
        slice_info = process(fn, filenames_label[i], slice_no)
        slice_infos.append(slice_info)
        
    if REPORTS:
        for sl in slice_infos:
            sl.save_report("reports/")

else:
    with open('slice_infos.pkl', 'rb') as f:
        slice_infos = dill.load(f)
        
#if CLASSIFY:
#    # we need to go through each slice and collect the masked features and
#    # unmasked features into separate arrays
#    masked_features = []
#    unmasked_features = []
#    for slice_info in slice_infos:
#        masked_features.append(slice_info.all_features_mask)
#        unmasked_features.append(slice_info.all_features_nonmask)