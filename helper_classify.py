# -*- coding: utf-8 -*-
"""
Functions to help with the classification of pieces of images.
Created on Fri Jan 29 09:33:14 2016

@author: Sadi
"""

from __future__ import print_function

from helper_io import get_filenames
from helper_io import find_biggest_slice
from helper_io import get_nifti_slice

from helper_eigen import extract_roi_patches_w
from helper_eigen import get_randoms_w

from helper_gabor import generate_kernels
from helper_gabor import compute_feats
from helper_gabor import compute_hogs
from helper_gabor import compute_intens

from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from joblib import Parallel, delayed
from scipy import stats

import sys
import getopt

import dill
import joblib

import numpy as np
import matplotlib.pyplot as plt


"""
Given a handle to a (trained) Random Forest Classifier, a handle to a set of
kernels for generating Gabor features, and a handle to a group of patches,
this function will apply the RF to each patch in sequence and return the
results.

The file containing the patches is serialised to disk with a pickle using the
dill library. When unpacked, the structure looks like this:
[(a, b, c), patches_a, patches_r]

a and b are integers which represent in what bounds of the full set of patches
does patches_a and patches_r represent. For example, if you are classifying
patches 1001-2000 of the original set, a = 1001 and b = 2000. patches_a
is an array containing a susbset of the patches from the original image, and
patches_r is an array containing the associated patches of the ROI mask
determined by a previous Atlas registration. c is the length of the entire set.

This function generally can't be used independently because it's a helper
function to the main reconstruction process in another file, and the files
that it expects are temporary files that are destroyed once the reconstruction
is complete.
"""
def classify_patch_group(fn_rf, fn_kern, fn_patch_info):
    t0 = time()
    RF = dill.load(open(fn_rf, 'rb')) # unpack the RF classifier
    kernels = dill.load(open(fn_kern, 'rb')) # unpack the kernels
    patch_info = dill.load(open(fn_patch_info, 'rb')) # unpack the patch info
    
    a, b, c = patch_info[0] # get the bounds of the set
    patches_a = patch_info[1] # grab the set to classify
    patches_r = patch_info[2] # grab the set to compare with (atlas mask)
    
    
    results = []
    
    for i, patch in enumerate(patches_a): # go through each patch
        if np.all(patches_r[i]): # if the patch is entirely masked
            feat = compute_feats(patch, kernels).flatten().reshape(1, -1)
            intens = np.array(compute_intens(patch)).flatten().reshape(1, -1)
            feat = np.concatenate((feat, intens), axis=1)            
            prediction = RF.predict(feat)
            #print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
            results.append(np.full(patch.shape, prediction))
        else: # the associated ROI patch is totally zero
            results.append(np.zeros(patch.shape))
    dt = time() - t0
    print("Classified group {}-{}/{} in {:.2f} time".format(a, b, c, dt))
    return results