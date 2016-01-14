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
from joblib import Parallel, delayed
import joblib
import sys
import matplotlib.pyplot as plt
import os

# Sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.feature import hog

# Helpers
from helper_io import get_filenames, find_biggest_slice, get_nifti_slice
from helper_eigen import extract_roi_patches_w, get_randoms_w
from helper_gabor import generate_kernels, compute_feats

# CONSTANTS
path = "ct/" # path to the CTs and the associated labels
im_name = "CT.nii.gz" # string used to select the images
lb_name = "CT_CTV.nii.gz" # string used to select the labels
psize = 12 # patch "radius", so the patch dimensions are psize x psize
# how many "principal component" patches to generate per slice per class
# for example, 10 will result in 10 PC patches for masked, 10 PC patches
# for non-masked regions.
pc_max = 500
ct_cap_min = -1000 # minimum CT brightness (set ct_cap_max for no clipping)
ct_cap_max = 2000 # maximum CT brightness (set to 0 for no clipping)
ct_monte = 1 # 1 will use random patches; 0 will use PCA patches
pickle = "slice_infos_12.pkl" # path and filename to save the SliceInfos
recons = "recons_12_3.pkl" # path and filename to store the patches of reconstruction
generate = 0 # toggle generation of new SliceInfos
classify = 0 # test classification engine
no_trees = 10 # number of trees to use for classifier
fullspec_i = 3
crop = (128, 128)
np.seterr(all='ignore')

# BIGTEST CONSTANTS
psizes = [12]

"""
Class to contain all the training data from a slice extracted from a 3D MRI.
Consists of the filename of the slice, the "number" of the slice (i.e. the 
index of the slice in the 3D MRI), the original slice image, the associated
label image, their orientations, masked/unmasked: patches, gabor features,
HOG features, and "percentage values"; the percentage of label in that
particular patch.
"""
class SliceInfo():
    def __init__(self, filename, slice_no,
                 slice_im, slice_im_or,
                 slice_lb, slice_lb_or,
                 patches_m_pc, patches_n_pc,
                 feats_m, feats_n,
                 vals_m=None, vals_n=None,
                 hogs_m=None, hogs_n=None):
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
        
        self.hogs_m = hogs_m
        self.hogs_n = hogs_n
        
        self.vals_m = vals_m
        self.vals_n = vals_n

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
        print("Masked Values Shape: ", np.array(self.vals_m).shape)
        print("Unmasked Values Shape: ", np.array(self.vals_n).shape)
        if self.hogs_m:
            print("Masked HOGs: ", np.array(self.hogs_m).shape)
        if self.hogs_n:
            print("Unmasked HOGs: ", np.array(self.hogs_n).shape)
        if self.vals_m:
            print("Masked Values: ", self.vals_m)
        if self.vals_n:
            print("Unmasked Values: ", self.vals_n)
        
"""
Deprecated.
"""
#def create_pc_patches(slice_im, slice_lb):
#    # for CT, cap the image
#    if ct_cap_max:
#        slice_im = np.clip(slice_im, ct_cap_min, ct_cap_max)
#    
#    # extract all of the patches
#    # _m means "masked" patches, i.e. patches that are in the ROI
#    # _n means "non-masked" patches, i.e. patches that aren't in the ROI
#    patches_m, patches_n = extract_roi_patches(slice_im, slice_lb, psize)
#
#    # use random patches to represent an image
#    if ct_monte:
#        # _m_pc means "masked" "principal components"
#        # _n_pc means "non-masked" "principal components"
#        patches_m_pc = get_randoms(patches_m, pc_max)
#        patches_n_pc = get_randoms(patches_n, pc_max)
#        
#    # generate eigenpatches using PCA
#    else:
#        patches_m_pc = get_eigenpatches(patches_m, psize, pc_max)
#        patches_n_pc = get_eigenpatches(patches_n, psize, pc_max)
#    
#    return patches_m_pc, patches_n_pc

"""
Deprecated.
"""
#def create_sliceinfo(images_fn, labels_fn, kernels, i):
#    # figure out the biggest slice
#    slice_no = find_biggest_slice(path + labels_fn[i])
#    
#    # get the slice, label, and associated orientations
#    slice_im, slice_im_or = get_nifti_slice(path + images_fn[i], slice_no)
#    slice_lb, slice_lb_or = get_nifti_slice(path + labels_fn[i], slice_no)
#    
#    # if crop, we crop the image down
#    if crop:    
#        crop_x, crop_y = crop
#        start_x = slice_im.shape[0] / 2 - crop_x / 2
#        end_x = start_x + crop_x
#        start_y = slice_im.shape[1] / 2 - crop_y / 2
#        end_y = start_y + crop_y
#    
#        slice_im = slice_im[start_x:end_x, start_y:end_y]
#        slice_lb = slice_lb[start_x:end_x, start_y:end_y]
#            
#    # figure out the principal patches
#    pc_payload = (slice_im, slice_lb)
#    patches_m_pc, patches_n_pc = create_pc_patches(*pc_payload)
#    
#    # compute gabor features for the patches
#    feats_m = []
#    feats_n = []
#    for patch in patches_m_pc:
#        feats_m.append(compute_feats(patch, kernels))
#    for patch in patches_n_pc:
#        feats_n.append(compute_feats(patch, kernels))
#    
#    # package it into a SliceInfo object
#    si_payload = (images_fn[i], slice_no, 
#                  slice_im, slice_im_or,
#                  slice_lb, slice_lb_or,
#                  patches_m_pc, patches_n_pc,
#                  feats_m, feats_n)
#    
#    return SliceInfo(*si_payload)

"""
Deprecated.
"""
#def create_sliceinfos(images_fn, labels_fn):
#    kernels = generate_kernels() # create gabor kernels
#    slice_infos = []    
#    
#    if len(sys.argv) < 2:
#        for i in range(len(images_fn)):                
#            slice_infos.append(create_sliceinfo(images_fn, labels_fn, kernels, i))
#    
#    else:
#        slice_infos = Parallel(n_jobs=int(sys.argv[1]))(delayed(create_sliceinfo)(images_fn, labels_fn, kernels, i) for i in range(len(images_fn)))
#    
#    return slice_infos

"""
Deprecated.
"""
#def classify_patch(RF, kernels, patches, i):
#    # compute the feats of the patch
#    patch = patches[i]
#    feat = compute_feats(patch, kernels)
#    feat = feat.flatten().reshape(1, -1)
#    prediction = RF.predict(feat)
#    print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
#    if prediction == 'M':
#        return np.ones(patch.shape)
#    elif prediction == 'N':
#        return np.zeros(patch.shape)
#    else:
#        return np.full(patch.shape, prediction)

"""
Given an image (most likely a small patch), generates HOG features for that
image. Returns a list of numbers.

In this release, the HOG features have fixed parameters. In the future, they
will be modifiable through global constants.
"""
def compute_hogs(patch):
    fd, hog_im = hog(patch, orientations=8, pixels_per_cell=(4, 4), 
                     cells_per_block=(1, 1), visualise=True)
    return fd

"""
Given an image and associated label, generate the "principal component patches"
of the image separated into two separate groups: masked and unmasked PCPs.

A tuple of four sets is returned: masked PCPs, unmasked PCPs, masked values,
and unmasked values. The explanations for "PCP" and "value" are described
below.

The image is (optionally) preprocessed if ct_cap_max is set to clip the
brightness values of the pixels to the range [ct_cap_min, ct_cap_max]. This
is generally done to filter out extremely bright seeds that appear in
prostate CT scans.

The masked and unmasked patches are created from a function call (see
"extract_roi_patches_w"). Masked patches are described as patches of the
image which have at least one pixel that is "pixel of interest". The pixel
of interest is defined as a pixel whose associated pixel in the label image is
high/bright/1. In addition, each patch also has a "value", which describes
the percentage of pixels of interest in that patch as a decimal (i.e. between
0 and 1).

All the masked and unmasked patches are reduced into a "principal component"
set, which is a random selection of the existing masked and unmasked patches.
The number of random patches selected is defined by the constant pc_max. The
number of random patches is guaranteed and thus the number of unmasked and
masked PCPs will always be equal. If ct_max exceeds the number of patches 
available in the selection pool of all masked/unmasked patches, patches will be
duplicated to fill out the required ct_max PCPs. So if ct_max = 500 and you
only have 30 masked patches available, the 30 will be approximately repeated
across the 500 PCPs produced.

By extension, the PCPs may have duplicates even if the available patches exceed
ct_max. This is because the RNG doesn't check if an existing patch has already
been added to the PCP record, so the RNG may produce the same patch multiple
times.
"""
def create_pc_patches_w(slice_im, slice_lb):
    # for CT, cap the image
    if ct_cap_max:
        slice_im = np.clip(slice_im, ct_cap_min, ct_cap_max)
    
    # extract all of the patches
    # _m means "masked" patches, i.e. patches that are in the ROI
    # _n means "non-masked" patches, i.e. patches that aren't in the ROI
    patches_m, patches_n, vals_m, vals_n = extract_roi_patches_w(slice_im, 
                                                               slice_lb, 
                                                               psize)

    # use random patches to represent an image
    if ct_monte:
        # _m_pc means "masked" "principal components"
        # _n_pc means "non-masked" "principal components"
        patches_m_pc, vals_m_pc = get_randoms_w(patches_m, vals_m, pc_max)
        patches_n_pc, vals_n_pc = get_randoms_w(patches_n, vals_n, pc_max)
        
    # PCA is deprecated in this implementation
#    else:
#        patches_m_pc = get_eigenpatches(patches_m, psize, pc_max)
#        patches_n_pc = get_eigenpatches(patches_n, psize, pc_max)
    
    return patches_m_pc, patches_n_pc, vals_m_pc, vals_n_pc

"""
Given a list of image filenames, a list of label filenames, a set of Gabor
kernels, and the index of the image in the list of images, produces a
"SliceInfo" object from the data. These filenames reference existing 3DMRI
Nifti volumes that are in a folder defined by the "path" constant at the top.
Images are defined with the extension stored in "im_name" and the label
extension stored in "lb_name", both constants defined at the top.

This is done by first finding the largest label in the 3D volume. The largest
label is the label image whose contains the most amount of bright pixels. This
is referred to as the "slice no." for a particular SliceInfo object.

The slice_no slice of each 3DMRI and associated 3D Label is extracted and
stored as raw image data (see "get_nifti_slice" in helper_io).

The slice is optionally cropped down to a fixed size around the center as
defined by the crop constant defined at the top. If the crop constant is None
or 0, the slice isn't cropped. This cropping applies to both the image and
the label; they need to be associated with each other.

The principal patches of the slice are computed with "create_pc_patches_w" (see
above). Each principal patch is described in terms of features (see 
"compute_feats" and "compute_hogs"). The combination of the slice information,
the patches and the features are stored in a SliceInfo object which is
finally returned.
"""
def create_sliceinfo_w(images_fn, labels_fn, kernels, i):
    # figure out the biggest slice
    print("Creating Weighted Slice Info... {}/{}: {}".format(i+1, len(images_fn), labels_fn[i]))
    slice_no = find_biggest_slice(path + labels_fn[i])
    
    # get the slice, label, and associated orientations
    slice_im, slice_im_or = get_nifti_slice(path + images_fn[i], slice_no)
    slice_lb, slice_lb_or = get_nifti_slice(path + labels_fn[i], slice_no)
    
    # if crop, we crop the image down
    if crop:    
        crop_x, crop_y = crop
        start_x = slice_im.shape[0] / 2 - crop_x / 2
        end_x = start_x + crop_x
        start_y = slice_im.shape[1] / 2 - crop_y / 2
        end_y = start_y + crop_y
    
        slice_im = slice_im[start_x:end_x, start_y:end_y]
        slice_lb = slice_lb[start_x:end_x, start_y:end_y]
            
    # figure out the principal patches
    pc_payload = (slice_im, slice_lb)
    patches_m_pc, patches_n_pc, vals_m, vals_n = create_pc_patches_w(*pc_payload)
    
    # compute gabor features for the patches
    feats_m = []
    feats_n = []
    hogs_m = []
    hogs_n = []
    for patch in patches_m_pc:
        feats_m.append(compute_feats(patch, kernels))
        hogs_m.append(compute_hogs(patch))
    for patch in patches_n_pc:
        feats_n.append(compute_feats(patch, kernels))
        hogs_n.append(compute_hogs(patch))
    
    # package it into a SliceInfo object
    si_payload = (images_fn[i], slice_no, 
                  slice_im, slice_im_or,
                  slice_lb, slice_lb_or,
                  patches_m_pc, patches_n_pc,
                  feats_m, feats_n,
                  vals_m, vals_n,
                  hogs_m, hogs_n)
    
    
    return SliceInfo(*si_payload)

"""
Given a list of filenames of 3DMRI Nifti images and a list of filenames of
of their associated labels, creates a list of SliceInfo objects, each of which
containing the slice information, patches and features associated to a
particular Nifti filename in the input arguments.

Each SliceInfo object is created on a separate process, with the maximum
number of processes defined as the first commandline argument. If there is no
commandline argument or if the number of processes is 1, the SliceInfos are
generated serially.

Returns the list of the SliceInfo objects.
"""
def create_sliceinfos_w(images_fn, labels_fn):
    print("Creating weighted Slice Infos...")
    kernels = generate_kernels() # create gabor kernels
    slice_infos = []    
    
    if len(sys.argv) < 2:
        for i in range(len(images_fn)):                
            slice_infos.append(create_sliceinfo_w(images_fn, labels_fn, kernels, i))
    
    else:
        slice_infos = Parallel(n_jobs=int(sys.argv[1]))(delayed(create_sliceinfo_w)(images_fn, labels_fn, kernels, i) for i in range(len(images_fn)))
    
    return slice_infos

"""
Given access to an entire set of SliceInfo objects and given an index which
will be the index of the "test slice", creates a set of "features" and
associated "labels" which will be used to train the random forest classifier.
The sets are reshaped signficantly so that the classifier can fit to them.

Returns a tuple containing the set of samples and set of labels for an RF
to fit directly against them. The sample/label set could also be used for
any other machine learning application which supports inputs of this shape
(2D features; 1D labels). The training features leave-n-out, where n = 1 and
the left out data corresponds to the ith SliceInfo object.

There is some deprecated code in here. Originally, patches and features were
labelled in a binary manner. That is, a patch was described as being a
"masked" patch or an "unmasked" patch, where patches containing at least one
pixel of interest were labelled as "masked". Now, the SliceInfos generally
use patch "values" instead, which describe what percentage of the patch or
associated feature describing the patch contains bright pixels in the
associated label patch.
"""
def generate_training_feats(slice_infos, i):
    train_m = []
    train_n = []

    labels_m = []    
    labels_n = []
    
    hogs_m = []
    hogs_n = []
    
    # go through each slice
    for j in range(len(slice_infos)):
        # if it's the ith slice, don't include it in the training
        if j == i:
            continue
        train_m.append(slice_infos[j].feats_m)
        train_n.append(slice_infos[j].feats_n)
        if slice_infos[j].vals_m != None:
            labels_m.extend(slice_infos[j].vals_m)
            labels_n.extend(slice_infos[j].vals_n)
        if slice_infos[j].hogs_m != None:
            hogs_m.append(slice_infos[j].hogs_m)
            hogs_n.append(slice_infos[j].hogs_n)
    
    train_m = np.array(train_m)
    train_n = np.array(train_n) 
    hogs_m = np.array(hogs_m)
    hogs_n = np.array(hogs_n)    
    
    tms = train_m.shape
    tns = train_n.shape
    hms = hogs_m.shape
    hns = hogs_n.shape
    
    print(tms, tns)
    print(hms, hns)
    
    train_m = train_m.reshape(tms[0] * tms[1], tms[2] * tms[3])
    train_n = train_n.reshape(tns[0] * tns[1], tns[2] * tns[3])
    hogs_m = hogs_m.reshape(hms[0] * hms[1], hms[2])
    hogs_n = hogs_n.reshape(hns[0] * hns[1], hns[2])
    
    train_m = np.concatenate((train_m, hogs_m), axis=1)
    train_n = np.concatenate((train_n, hogs_n), axis=1)
    
    print(train_m.shape, train_n.shape)
    #print(hogs_m.shape, hogs_n.shape)    
    
    #raise Exception
    
    samples = np.concatenate((train_m, train_n))
    if slice_infos[0].vals_m == None:
        labels = ['M' for k in train_m] + ['N' for p in train_n]
    else:
        labels = labels_m + labels_n

    return samples, labels

"""
Given a set of features, a set of associated labels, and the number of trees
that the classifier should use, trains a random forest classifier on the data
and returns a reference to the classifier for use in predicting the classes
of patches.
"""
def train_rf_classifier(features, labels, no_trees):
    rf = RandomForestClassifier(n_estimators=no_trees, n_jobs=-1)
    rf.fit(features, labels)
    return rf
    
"""
Given access to the full set of existing SliceInfo objects, and an index which
represents a "test slice", generates training data using all the SliceInfo
objects and their features (except for the ith SliceInfo object) and trains
a random forest classifier on that data (see "generate_training_feats" and
"train_rf_classifier"). Then, asks the trained classifier to predict the
classes of the patches stored inside the test slice (the ith SliceInfo object).
Records whether the random forest correctly classified each slice, and then
displays the results. The predictions are only on the existing features stored
for training in the SliceInfo object, so it doesn't check every single patch.
Essentially, this means that it only tests on the PCPs of the ith SliceInfo.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
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
        if(len(sys.argv) < 2):
            print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
            i+1, len(slice_infos), successes, trials, total), end="")
    for feat_n in test_sl.feats_n:
        # flatten the feature so it can be tested
        feat_n = feat_n.flatten().reshape(1, -1)
        if RF.predict(feat_n) == 'N':
            successes += 1
        trials += 1
        if(len(sys.argv) < 2):
            print("\rTesting Case {}/{}: {}/{}/{} successes.".format(
            i+1, len(slice_infos), successes, trials, total), end="")
    
    print("")
    print(np.asarray(slice_infos[i].feats_m).shape)
    
    return successes, trials


"""
Works like test_rf_feats, but instead of testing the classifier on the PCPs
of each SliceInfo object, tests the classifier on *every* patch that is
generated from the slice image.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
"""
def test_rf_feats_fullspec(slice_infos, i):
    feats, labels = generate_training_feats(slice_infos, i)
    RF = train_rf_classifier(feats, labels, no_trees)
    
    test_sl = slice_infos[i]
    image = test_sl.slice_im
    label = test_sl.slice_lb
    patches_m, patches_n = extract_roi_patches(image, label, psize)
    patches_n = patches_n
    plabels_m = ['M' for m in patches_m]
    plabels_n = ['N' for n in patches_n]
    kernels = generate_kernels()
    
    tot = len(patches_m) + len(patches_n)
    
    res1 = []
    res2 = []
    
    t0 = time()
    
    if len(sys.argv) >= 2:
        res1 = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_classify)(RF, kernels, p, plabels_m[i], i, tot) for i, p in enumerate(patches_m))
        res2 = Parallel(n_jobs=int(sys.argv[1]))(delayed(check_classify)(RF, kernels, p, plabels_n[i], i, tot) for i, p in enumerate(patches_n))
    else:
        for i,p in enumerate(patches_m): # go through each patch, and classify it!
            res1.append(check_classify(RF, kernels, p, plabels_m[i], i, tot))
        pass

    dt = time() - t0    
    
    print(res1.count(True) + res2.count(True))
    print("Finished in {:.2f} seconds.".format(dt))

"""
Helper function to check the prediction for a single patch, used in
test_rf_feats_fullspec.

**This function is not updated for the HOG featureset that is now added by
default to the SliceInfo class. Therefore, you'll get an error and the
program will crash if you try to call this function right now.
"""
def check_classify(RF, kernels, patch, patch_label, i, tot):        
    # kernel the patch
    print("Checking Classifying patch {}/{}".format(i, tot))
    feat = compute_feats(patch, kernels)
    feat = feat.flatten().reshape(1, -1)
    prediction = RF.predict(feat)
    if prediction == patch_label:
        return True
    else:
        return False        
      
def classify_patch_w(fn, kernels, patches, i):
    RF = joblib.load(fn)
    patch = patches[i]
    feat = compute_feats(patch, kernels)
    feat = feat.flatten().reshape(1, -1)
    hogs = compute_hogs(patch)
    hogs = hogs.flatten().reshape(1, -1)
    feat = np.concatenate((feat, hogs), axis=1)
    
    prediction = RF.predict(feat)
    print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
    if prediction == 'M':
        return np.ones(patch.shape)
    elif prediction == 'N':
        return np.zeros(patch.shape)
    else:
        return np.full(patch.shape, prediction)

"""
Generate labels using Random Forest on a particular
"""
def rf_reconstruct(slice_infos, i):
    feats, labels = generate_training_feats(slice_infos, i)
    labels = np.array(labels)
    RF = train_rf_classifier(feats, labels, no_trees)
    
    test_sl = slice_infos[i]
    image = test_sl.slice_im
    
    kernels = generate_kernels()
    
    # break the image into patches; all of these will be classified
    patch_size = (psize, psize)
    # _a stands for "all"
    patches_a = extract_patches_2d(image, patch_size)
    # _p stands for "predict"
    
    # dump the RF
    fn_rf = 'rf.joblib'
    joblib.dump(RF, fn_rf)
        
    # check each patch
    if len(sys.argv) >= 2:
        #patches_p = Parallel(n_jobs=int(sys.argv[1]))(delayed(classify_patch)(RF, kernels, patches_a, i) for i in range(len(patches_a)))
        patches_p = Parallel(n_jobs=int(sys.argv[1]))(delayed(classify_patch_w)(fn_rf, kernels, patches_a, i) for i in range(len(patches_a)))
            
    else:
        patches_p = []
        for i in range(len(patches_a)):
            patches_p.append(classify_patch(RF, kernels, patches_a, i))
    
    # reconstruct based on the patch
    recons_im = reconstruct_from_patches_2d(np.asarray(patches_p), image.shape)
    print(recons_im.shape)
    
    # save patches_p to the drive, because it took so much work to make!
    with open(recons, 'wb') as f:
        dill.dump(recons_im, f)

"""
Threshold a reconstructed image by pushing numbers below value down to 0 and
numbers equal to or above value up to 1.
"""
def threshold(image, value, round=None):
    threshold = np.array(image) # make a copy
    threshold[threshold < value] = 0
    if round:
        threshold[threshold >= value] = 1
    return threshold

"""
Uses the label to mask out the image.
"""
def mask_out(image, label):
    mask = np.where(label == 0, -1000, image)
    return mask

def compare_im(recons_im, real_lb, display=False):
    if display:
        plt.imshow(recons_im)
        plt.show()
        
        plt.imshow(real_lb)
        plt.show()
    
    recons_dice = recons_im.astype(int)
    label_dice = real_lb.astype(int)
    dice = float(2*np.count_nonzero(recons_dice & label_dice))/(np.count_nonzero(recons_dice)
    + np.count_nonzero(label_dice))

    return dice
    
def run():
    if generate:
        t0 = time()
        # get lists of files to process
        images_fn, labels_fn = get_filenames(path, im_name, lb_name)    
        
        # generate sliceinfos for all those images
        if generate == 1:
            slice_infos = create_sliceinfos(images_fn, labels_fn)
        elif generate == 2:
            slice_infos = create_sliceinfos_w(images_fn, labels_fn)
            print(slice_infos[0].vals_m[:30])

        
        dt = time() - t0
        print("Finished in %.2f seconds." % dt)
        
        # save the slice info
        with open(pickle, 'wb') as f:
            dill.dump(slice_infos, f)        
    
    else:
        with open(pickle, 'rb') as f:
            slice_infos = dill.load(f)
        
    total_successes = 0
    total_trials = 0
    
    
    if classify == 1: # Go through each slice and classify PCPs
    # PCP = principal component patches

        t0 = time()
        
        if len(sys.argv) >= 2:        
            res = Parallel(n_jobs=int(sys.argv[1]))(delayed(test_rf_feats)(slice_infos, i) for i in range(len(slice_infos)))
            for successes, trials in res:
                total_successes += successes
                total_trials += trials

        else:
            for i in range(len(slice_infos)):
                successes, trials = test_rf_feats(slice_infos, i)
                total_successes += successes
                total_trials += trials        
            
        print("Rate: {}/{} = {:.2f}".format(total_successes, total_trials,
              float(total_successes)/float(total_trials)*100))
        
        dt = time() - t0
        print("Took %.2f seconds." % dt)
        
    elif classify == 2: # check classification of all patches for one slice
        test_rf_feats_fullspec(slice_infos, fullspec_i)
        pass
    
    elif classify == 3: # fully reconstruct for one slice
        rf_reconstruct(slice_infos, fullspec_i)
        
    elif classify == 4: # show the reconstruct for one slice
        with open(recons, 'rb') as f:
            recons_im = dill.load(f)

        real_lb = slice_infos[fullspec_i].slice_lb        
        
#        for threshval in range(0, 101):
#            threshval = threshval / 100.0
#            recons_th = threshold(recons_im, threshval, True)
#            print("%.2f" % threshval, "%.2f" % compare_im(recons_th, real_lb))
        
        recons_th = threshold(recons_im, 0.45, True)
        compare_im(recons_th, real_lb, True)

def plot_save_comparisons():
    with open(pickle, 'rb') as f:
        slice_infos = dill.load(f)
        
    threshv = 0.50
    
    # go through each case
    for i in range(0, 35):
        with open("recons_ctv/recons_12_" + str(i) + ".pkl", 'rb') as f:
            recons_im = dill.load(f)
            recons_th = threshold(recons_im, threshv, True)
            recons_tv = threshold(recons_im, threshv)
        real_lb = slice_infos[i].slice_lb
        if ct_cap_max:
            slice_im = np.clip(slice_infos[i].slice_im, ct_cap_min, ct_cap_max)
        # plot each
        plt.figure(figsize=(8,12))
        #plt.subplot(4, 1, 1)
        plt.suptitle("Prostate Reconstructions with HOG at Threshold {}, Case {}".format(threshv,i))
        plt.axis('off')
        plt.subplot(3, 2, 1)
        plt.axis('off')
        plt.imshow(slice_im, cmap=plt.cm.gray)
        plt.subplot(3, 2, 2)
        plt.axis('off')

        plt.imshow(slice_im, cmap=plt.cm.gray)
        plt.subplot(3, 2, 3)
        plt.axis('off')

        plt.imshow(recons_tv)
        plt.subplot(3, 2, 4)
        plt.axis('off')

        plt.imshow(real_lb)
        plt.subplot(3, 2, 5)
        plt.axis('off')

        plt.imshow(mask_out(slice_im, recons_th), cmap=plt.cm.gray)
        plt.subplot(3, 2, 6)
        plt.axis('off')
        plt.imshow(mask_out(slice_im, real_lb), cmap=plt.cm.gray)
        plt.savefig('compares_ctv/case_' + str(i) + '.png')
                

def bigtest():
    # really long test to generate SliceInfos at all radiuses
    
    images_fn, labels_fn = get_filenames(path, im_name, lb_name)    
    
    # go through all candidate psizes
    for p in psizes:
        global psize
        psize = p
        t0 = time()
        slice_infos = create_sliceinfos_w(images_fn, labels_fn)
        dt = time() - t0
        print("Finished in %.2f seconds." % dt)
         
        # save the slice info
        with open("slice_infos_" + str(psize) + ".pkl", 'wb') as f:
            dill.dump(slice_infos, f)   
            
def bigtest2():
    
    for radius in [12]: # go through each SliceInfo
        global fullspec_i
        global recons
        with open("slice_infos_" + str(radius) + ".pkl", 'rb') as f:
            slice_infos = dill.load(f)
        for x in range(0, 35):
            fullspec_i = x
            recons = "recons_" + str(radius) + "_" + str(fullspec_i) + ".pkl"
            rf_reconstruct(slice_infos, fullspec_i)

def get_all_similarities(repath):
    similarities = {}
    for fn in sorted(os.listdir(repath)):
        sep = fn.split('.')[0].split('_')
        radius = int(sep[1])
        if similarities.get(radius) == None:
            similarities[radius] = []    
        with open(repath + fn, 'rb') as f:
            similarities[radius].append(dill.load(f))
    
    #plt.imshow(similarities[10][0])

    rad_sims = []    
    
    # generate similarity index for each image
    for radius in sorted(similarities.keys()):
        
        print("Radius: ", radius)
        
        with open("slice_infos_" + str(radius) + ".pkl", 'rb') as f:
            slice_infos = dill.load(f)

        slice_sims = []        
        
        for index, recons_im in enumerate(similarities[radius]):
            # load the real label
            real_lb = slice_infos[index].slice_lb
            
            print("Slice: ", index)

            thresh_sims = []                        
            # go through all possible thresholds
            for threshval in range(0, 101):
                threshval = threshval / 100.0
                
                # mask the generated recons_im to a specific threshval
                recons_th = threshold(recons_im, threshval, True)
                
                # Compare them
                compare_val = compare_im(recons_th, real_lb)
                thresh_sims.append(compare_val)
            
            slice_sims.append(thresh_sims)
        
        rad_sims.append(slice_sims)
    
    # pull the means
    emp = []
    for i in range(len(rad_sims)):
        emp.append(np.array(rad_sims)[i].mean(axis=0))
    
    with open("similarities.pkl", 'wb') as f:
        dill.dump(rad_sims, f)
        
    np.savetxt("emp.csv", np.array(emp), delimiter=",")
        
    
def bulk_rename(repath):
    for fn in sorted(os.listdir(repath)):
        if len(fn) == 15:
            os.rename(repath + "/" + fn, repath + "/" + fn[:10] + '0' + fn[10:])
    
        
if __name__ == "__main__": # only run if it's the main module
    #get_all_similarities("recons_bladder/")
    #bigtest2()
    #run()
    #plot_save_comparisons()
    #bigtest()
    pass