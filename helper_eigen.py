# -*- coding: utf-8 -*-
"""
Helper functions for generating patches and eigenpatches.
Created on Thu Dec 03 10:15:50 2015

@author: ZAH010
"""

from __future__ import print_function
from time import time

# Basic Packages
import numpy as np
import matplotlib.pyplot as plt

# Scientific
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA

# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 1
SHOW_IMG = 1
TEST = 1  
    
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
    patches_nonmask = []
    
    for i, patch in enumerate(patches_original):
        if 0 not in patches_label[i]:
            patches_mask.append(patch)
        else:
            patches_nonmask.append(patch)
            
    if DEBUG:
        print("Number of ROI patches: ", len(patches_mask))
            
    return patches_mask, patches_nonmask

"""
Generates important component patches given a set of normally-extracted
patches by using PCA.

get_eigenpatches(np.ndarray, int) --> np.ndarray
"""
def get_eigenpatches(patches, patch_size, no_components):    
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
    
    # Reshape back to normal
    eigens = eigens.reshape(eigens.shape[0], patch_size, patch_size)
    
    return eigens

"""
Generates a selection of max_patches random patches from the input patches.
"""
def get_randoms(patches, max_patches):
    #print(len(patches)-1, max_patches)
    select = np.random.random_integers(0, len(patches)-1, max_patches)
    randoms = []
    for i in select:
        randoms.append(patches[i])
    return np.array(randoms)

"""
Given some eigenpatches, plot them on the screen. This should work for
all patches in general.

show_eigenpatches_(np.ndarray) --> None
"""
def show_eigenpatches(eigens, filename=None): #, patch_size):
    #psize = (patch_size, patch_size)

    columns = 5
    rows = int(len(eigens) / columns)
    if rows < 1:
        rows = 1
    #subplot_adj = (0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    #height = int(rows / 2)

    plt.figure(figsize=(5, rows + 1))
    for i, comp in enumerate(eigens):
        plt.subplot(rows, columns, i + 1)
        #plt.imshow(comp.reshape(psize), cmap=plt.cm.gray, 
        #           interpolation='nearest')
        plt.imshow(comp, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    if not filename:
        plt.suptitle('Eigen-decomposition of Patches in ROI\n', fontsize=16)
    #plt.subplots_adjust(*subplot_adj)
    if filename:
        save_to = filename + '.eigen.png'
        plt.savefig(save_to)
        return save_to
    else:
        #plt.show()
        pass