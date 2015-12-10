# -*- coding: utf-8 -*-
"""
Flexible implementation of the hard-coded test_features_ct.py script.

Created on Thu Dec 10 11:41:03 2015

@author: Sadi
"""

from __future__ import print_function

# Basic Packages
import numpy as np
import dill
from time import time

# Sci
from sklearn.ensemble import RandomForestClassifier

# Helpers
from helper_io import get_filenames, find_biggest_slice, get_nifti_slice
from helper_eigen import extract_roi_patches, get_randoms, get_eigenpatches
from helper_gabor import generate_kernels, compute_feats

# CONSTANTS
path = "ct/" # path to the CTs and the associated labels
lb_name = "CTV" # string used to select the labels
psize = 10 # patch "radius", so the patch dimensions are psize x psize
# how many "principal component" patches to generate per slice per class
# for example, 10 will result in 10 PC patches for masked, 10 PC patches
# for non-masked regions.
pc_max = 50
ct_cap_min = -1000 # minimum CT brightness
ct_cap_max = 2000 # maximum CT brightness (set to 0 for no cap)
ct_monte = 1 # 1 will use random patches; 0 will use PCA patches
pickle = "slice_infos.pkl" # path and filename to save the SliceInfos
generate = 0 # toggle generation of new SliceInfos
classify = 1 # test classification engine
no_trees = 10 # number of trees to use for classifier

class SliceInfo():
    def __init__(self, filename, slice_no,
                 slice_im, slice_im_or,
                 slice_lb, slice_lb_or,
                 patches_m_pc, patches_n_pc,
                 feats_m, feats_n):
        self.filename = filename
        self.slice_no = slice_no
        
        self.slice_im = slice_im
        self.slice_im_or = slice_im_or
        self.slice_lb = slice_lb
        self.slice_lb_or = slice_lb_or
        
        self.patches_m_pc = patches_m_pc
        self.patches_n_pc = patches_n_pc
        self.feats_m = feats_m
        self.feats_n = feats_n

    def get_info(self):
        print("Filename: ", self.filename)
        print("Slice No.: ", self.slice_no)
        print("Image: ", np.array(self.slice_im).shape)
        print("Orientation: ", np.array(self.slice_im_or).shape)
        print("Label: ", np.array(self.slice_lb).shape)
        print("Label Orientation: ", np.array(self.slice_lb_or).shape)
        print("Masked Patches: ", np.array(self.patches_m_pc).shape)
        print("Unmasked Patches: ", np.array(self.patches_n_pc).shape)
        print("Masked Features: ", np.array(self.feats_m).shape)
        print("Unmasked Features: ", np.array(self.feats_n).shape)
        

def create_pc_patches(slice_im, slice_lb):
    # for CT, cap the image
    if ct_cap_max:
        slice_im = np.clip(slice_im, ct_cap_min, ct_cap_max)
    
    # extract all of the patches
    # _m means "masked" patches, i.e. patches that are in the ROI
    # _n means "non-masked" patches, i.e. patches that aren't in the ROI
    patches_m, patches_n = extract_roi_patches(slice_im, slice_lb, psize)

    # use random patches to represent an image
    if ct_monte:
        # _m_pc means "masked" "principal components"
        # _n_pc means "non-masked" "principal components"
        patches_m_pc = get_randoms(patches_m, pc_max)
        patches_n_pc = get_randoms(patches_n, pc_max)
        
    # generate eigenpatches using PCA
    else:
        patches_m_pc = get_eigenpatches(patches_m, psize, pc_max)
        patches_n_pc = get_eigenpatches(patches_n, psize, pc_max)
    
    return patches_m_pc, patches_n_pc
    
def create_sliceinfos(images_fn, labels_fn):
    kernels = generate_kernels() # create gabor kernels
    slice_infos = []    
    
    for i in range(len(images_fn)):        
        # figure out the biggest slice
        slice_no = find_biggest_slice(path + labels_fn[i])
        
        # get the slice, label, and associated orientations
        slice_im, slice_im_or = get_nifti_slice(path + images_fn[i], slice_no)
        slice_lb, slice_lb_or = get_nifti_slice(path + labels_fn[i], slice_no)
        
        # figure out the principal patches
        pc_payload = (slice_im, slice_lb)
        patches_m_pc, patches_n_pc = create_pc_patches(*pc_payload)
        
        # compute gabor features for the patches
        feats_m = []
        feats_n = []
        for patch in patches_m_pc:
            feats_m.append(compute_feats(patch, kernels))
        for patch in patches_n_pc:
            feats_n.append(compute_feats(patch, kernels))
        
        # package it into a SliceInfo object
        si_payload = (images_fn[i], slice_no, 
                      slice_im, slice_im_or,
                      slice_lb, slice_lb_or,
                      patches_m_pc, patches_n_pc,
                      feats_m, feats_n)
        
        slice_infos.append(SliceInfo(*si_payload))
    
    return slice_infos

"""
Gives you flattened training features from every slice except for the ith
slice, which will be the test slice.
"""
def generate_training_feats(slice_infos, i):
    train_m = []
    train_n = []    
    
    # go through each slice
    for j in range(len(slice_infos)):
        # if it's the ith slice, don't include it in the training
        if j == i:
            continue
        train_m.append(slice_infos[j].feats_m)
        train_n.append(slice_infos[j].feats_n)
    
    train_m = np.array(train_m)
    train_n = np.array(train_n)
    
    tms = train_m.shape
    tns = train_n.shape
    
    train_m = train_m.reshape(tms[0] * tms[1], tms[2] * tms[3])
    train_n = train_n.reshape(tns[0] * tns[1], tns[2] * tns[3])
    
    samples = np.concatenate((train_m, train_n))
    labels = ['M' for k in train_m] + ['N' for p in train_n]

    return samples, labels


def train_rf_classifier(features, labels, no_trees):
    rf = RandomForestClassifier(n_estimators=no_trees, n_jobs=-1)
    rf.fit(features, labels)
    return rf
    
"""
Tests the RF classifier on the pre-computed features for a particular
slice.
"""
def test_rf_feats(slice_infos, i):
    print("Testing Case {}/{}: ".format(i+1, len(slice_infos)), end="")
    
    # We need to train on masked and unmasked features from every
    # slice except for this one
    feats, labels = generate_training_feats(slice_infos, i)

    # Train the classifier
    RF = train_rf_classifier(feats, labels, no_trees)
    
    # Test the RF on each of the feats from the test slice
    test_sl = slice_infos[i]
    successes = 0
    trials = 0
    total = len(test_sl.feats_m) + len(test_sl.feats_n)
    for feat_m in test_sl.feats_m:
        # flatten the feature so it can be tested
        feat_m = feat_m.flatten().reshape(1, -1)
        if RF.predict(feat_m) == 'M':
            successes += 1
        trials += 1
        print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
        i+1, len(slice_infos), successes, trials, total), end="")
    for feat_n in test_sl.feats_n:
        # flatten the feature so it can be tested
        feat_n = feat_n.flatten().reshape(1, -1)
        if RF.predict(feat_n) == 'N':
            successes += 1
        trials += 1
        print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
        i+1, len(slice_infos), successes, trials, total), end="")
    
    print("")
    
    return successes, trials
    
def run():
    if generate:
        # get lists of files to process
        images_fn, labels_fn = get_filenames(path, lb_name)    
        
        # generate sliceinfos for all those images
        slice_infos = create_sliceinfos(images_fn, labels_fn)
        
        # save the slice info
        with open(pickle, 'wb') as f:
            dill.dump(slice_infos, f)
    
    else:
        with open(pickle, 'rb') as f:
            slice_infos = dill.load(f)
        
    total_successes = 0
    total_trials = 0
    
    if classify:
        # Go through each slice and attempt classification

        t0 = time()
        
        for i in range(len(slice_infos)):
            successes, trials = test_rf_feats(slice_infos, i)
            total_successes += successes
            total_trials += trials
        
        print("Rate: {}/{} = {:.2f}".format(total_successes, total_trials,
              float(total_successes)/float(total_trials)*100))
        
        dt = time() - t0
        print("Took %.2f seconds." % dt)

        
if __name__ == "__main__": # only run if it's the main module
    run()