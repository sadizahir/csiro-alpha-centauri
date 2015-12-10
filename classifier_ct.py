# -*- coding: utf-8 -*-
"""
Flexible implementation of the hard-coded test_features_ct.py script.

Created on Thu Dec 10 11:41:03 2015

@author: Sadi
"""

from __future__ import print_function

# Basic Packages
import numpy as np
import os
import dill

# Sci
from sklearn.ensemble import RandomForestClassifier

# Helpers
from helper_io import get_nifti_slice, get_nrrd_data, find_biggest_slice
from helper_io import get_filenames
from helper_eigen import extract_roi_patches
from helper_eigen import show_eigenpatches
from helper_gabor import generate_kernels, compute_feats
from helper_gabor import compute_powers, plot_gabor
    
if __name__ == "__main__": # only run if it's the main module
    path = "ct/" # this will be the path to the CTs and the associated labels
    label_filter = "CTV" # this string will be used to select the labels

    images_fn, labels_fn = get_filenames(path, label_filter)
    
    