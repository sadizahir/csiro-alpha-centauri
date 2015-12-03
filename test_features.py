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

# Sci
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages

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

class SliceInfo():
    def __init__(self, filename, slice_no, image_slice, orientation_slice, 
                 label_slice, 
                 orientation_label, patches_mask, eigens, kernels,
                 all_features, all_powers):
        
        self.filename = filename
        self.slice_no = slice_no
        self.image_slice = image_slice
        self.orientation_slice = orientation_slice
        self.label_slice = label_slice
        self.orientation_label = orientation_label
        
        self.patches_mask = patches_mask
        
        self.eigens = eigens
        
        self.kernels = kernels
        self.all_features = all_features
        self.all_powers = all_powers
        
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
        
        show_eigenpatches(self.eigens)
        
        plot_gabor(self.eigens)
        
    def save_report(self):
        eigen_fn = show_eigenpatches(self.eigens, self.filename)   
        gabor_fn = plot_gabor(self.eigens, self.filename)
        
        # base pdf
        plt.figure(figsize=(20, 28))
        
        plt.suptitle('Report: Slice ' + str(self.slice_no) + 
        ' for ' + self.filename + '\n', fontsize=48)
        
        # prepare the mask
        mask = np.where(self.label_slice == 0, self.label_slice, 
                        self.image_slice)
        
        # Saves a summary of the slice information as pdf 
        with PdfPages(self.filename + '.pdf') as pdf:
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
        
    return SliceInfo(filename, slice_no, image_slice, orientation_slice, 
                     label_slice,
                     orientation_label, patches_mask, eigens, kernels,
                     all_features, all_powers)
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

if TEST:
    test_routine()
    path = ""
    filenames = [path + "anon_mr_150420_30sec.nii.gz"]
    filenames_label = [path + "anon_mr_150420_30sec.nii-label.nrrd.nii.gz"]
    slice_no = 51
    
    slice_infos = []
    
    for i, fn in enumerate(filenames):
        if DEBUG:
            print("Processing: ", fn, ", Slice ", slice_no)
        slice_info = process(fn, filenames_label[i], slice_no)
        slice_infos.append(slice_info)
        
    for sl in slice_infos:
        #sl.save_report()
        pass