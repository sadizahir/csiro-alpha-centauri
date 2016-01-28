# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:16:35 2016

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

# CONSTANTS
path = "ct/" # path to the CTs and the associated labels
im_name = "CT.nii.gz" # string used to select the images
lb_name = "CT_CTV.nii.gz" # string used to select the labels
ro_name = "ATL_CTV_DILATE.nii.gz" # string used to select regions of interest
pickle = "training_data_CTV_12_atl_512_int.pkl" # path and filename to save the SliceInfos
recons = "recons_12_3.pkl" # path and filename to store the patches of reconstruction
psize = 12 # patch "radius", so the patch dimensions are psize x psize
# how many "principal component" patches to generate per slice per class
# for example, 10 will result in 10 PC patches for masked, 10 PC patches
# for non-masked regions.
pc_max = 500
ct_cap_min = -1000 # minimum CT brightness (set ct_cap_max for no clipping)
ct_cap_max = 1000 # maximum CT brightness (set to 0 for no clipping)
ct_monte = 1 # 1 will use random patches; 0 will use PCA patches
no_trees = 10 # number of trees to use for classifier
fullspec_i = 3
crop = None
np.seterr(all='ignore')

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
                 slice_ro, slice_ro_or,
                 patches_m_pc, patches_n_pc,
                 feats_m, feats_n,
                 intens_m, intens_n,
                 vals_m=None, vals_n=None):
        self.filename = filename
        self.slice_no = slice_no
        
        self.slice_im = slice_im
        self.slice_im_or = slice_im_or
        self.slice_lb = slice_lb
        self.slice_lb_or = slice_lb_or
        self.slice_ro = slice_ro
        self.slice_ro_or = slice_ro_or
                
        self.patches_m_pc = patches_m_pc
        self.patches_n_pc = patches_n_pc
        self.feats_m = feats_m
        self.feats_n = feats_n
        self.intens_m = intens_m
        self.intens_n = intens_n
        
        self.vals_m = vals_m
        self.vals_n = vals_n

    def get_info(self):
        print("Filename: ", self.filename)
        print("Slice No.: ", self.slice_no)
        print("Image: ", np.array(self.slice_im).shape)
        print("Orientation: ", np.array(self.slice_im_or).shape)
        print("Label: ", np.array(self.slice_lb).shape)
        print("Label Orientation: ", np.array(self.slice_lb_or).shape)
        print("Atlas ROI: ", np.array(self.slice_ro).shape)
        print("Atlas ROI Orientation: ", np.array(self.slice_ro_or).shape)
        print("Masked Patches: ", np.array(self.patches_m_pc).shape)
        print("Unmasked Patches: ", np.array(self.patches_n_pc).shape)
        print("Masked Features: ", np.array(self.feats_m).shape)
        print("Unmasked Features: ", np.array(self.feats_n).shape)
        print("Masked I. Features: ", np.array(self.intens_m).shape)
        print("Unmasked I. Features: ", np.array(self.intens_n).shape)
        if self.vals_m:
            #print("Masked Values: ", self.vals_m)
            print("Masked Values Shape: ", np.array(self.vals_m).shape)
        if self.vals_n:
            #print("Unmasked Values: ", self.vals_n)
            print("Unmasked Values Shape: ", np.array(self.vals_n).shape)

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
Compute the intensity features of a particular patch. Specifically, the mean,
variance, skewness and kurtosis of the intensity values of the patch.
"""
def compute_intens(patch):
    feats = []
    #patch = patch.flatten().reshape(1, -1)
    feats.append(patch.mean())
    feats.append(patch.var())
    feats.extend(stats.skew(patch))
    feats.extend(stats.kurtosis(patch))
    return feats
    
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
def create_sliceinfo_w(images_fn, labels_fn, regint_fn, kernels, i):
    # figure out the biggest slice
    print("Creating Weighted Slice Info... {}/{}: {}".format(i+1, len(images_fn), labels_fn[i]))
    slice_no = find_biggest_slice(path + labels_fn[i])
    
    # get the slice, label, and associated orientations
    slice_im, slice_im_or = get_nifti_slice(path + images_fn[i], slice_no)
    slice_lb, slice_lb_or = get_nifti_slice(path + labels_fn[i], slice_no)
    slice_ro, slice_ro_or = get_nifti_slice(path + regint_fn[i], slice_no)
    
    # if crop, we crop the image down
    if crop:    
        crop_x, crop_y = crop
        start_x = slice_im.shape[0] / 2 - crop_x / 2
        end_x = start_x + crop_x
        start_y = slice_im.shape[1] / 2 - crop_y / 2
        end_y = start_y + crop_y
    
        slice_im = slice_im[start_x:end_x, start_y:end_y]
        slice_lb = slice_lb[start_x:end_x, start_y:end_y]
        slice_ro = slice_ro[start_x:end_x, start_y:end_y]
            
    # figure out the principal patches
    pc_payload = (slice_im, slice_lb)
    patches_m_pc, patches_n_pc, vals_m, vals_n = create_pc_patches_w(*pc_payload)
    
    # compute gabor features for the patches
    feats_m = []
    feats_n = []
    intens_m = []
    intens_n = []
    for patch in patches_m_pc:
        feats_m.append(compute_feats(patch, kernels))
        intens_m.append(compute_intens(patch))
    for patch in patches_n_pc:
        feats_n.append(compute_feats(patch, kernels))
        intens_n.append(compute_intens(patch))
    
    # package it into a SliceInfo object
    si_payload = (images_fn[i], slice_no, 
                  slice_im, slice_im_or,
                  slice_lb, slice_lb_or,
                  slice_ro, slice_ro_or,
                  patches_m_pc, patches_n_pc,
                  feats_m, feats_n,
                  intens_m, intens_n,
                  vals_m, vals_n) 
                  
    return SliceInfo(*si_payload)

"""
Creates all the SliceInfo objects from a set of 3D volume MRI and label
filenames (see "create_sliceinfo_w") and saves them to a *.pkl file described
in the "pickle" constant at the top.
"""
def generate_training_data(jobs):
    print("Generating Training Data...")
    slice_infos = []
    
    images_fn, labels_fn, regint_fn = get_filenames(path, im_name, lb_name, ro_name) 
    kernels = generate_kernels()
        
    slice_infos = Parallel(n_jobs=jobs)(delayed(create_sliceinfo_w)(images_fn, 
                           labels_fn, regint_fn, kernels, i) for i in range(len(images_fn)))

    with open(pickle, 'wb') as f:
        dill.dump(slice_infos, f)

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
    intens_m = []
    intens_n = []

    labels_m = []    
    labels_n = []
    
    # go through each slice
    for j in range(len(slice_infos)):
        # if it's the ith slice, don't include it in the training
        if j == i:
            continue
        train_m.append(slice_infos[j].feats_m)
        intens_m.append(slice_infos[j].intens_m)
        train_n.append(slice_infos[j].feats_n)
        intens_n.append(slice_infos[j].intens_n)
        if slice_infos[j].vals_m != None:
            labels_m.extend(slice_infos[j].vals_m)
            labels_n.extend(slice_infos[j].vals_n)    
    
    train_m = np.array(train_m)
    train_n = np.array(train_n) 
    intens_m = np.array(intens_m)
    intens_n = np.array(intens_n)
    
    tms = train_m.shape
    tns = train_n.shape
    ims = intens_m.shape
    ins = intens_n.shape
    
    print(tms, tns)
    print(ims, ins)
    
    
    train_m = train_m.reshape(tms[0] * tms[1], tms[2] * tms[3])
    train_n = train_n.reshape(tns[0] * tns[1], tns[2] * tns[3])
    intens_m = intens_m.reshape(ims[0] * ims[1], ims[2])
    intens_n = intens_n.reshape(ins[0] * ins[1], ins[2])
    
    print(train_m.shape, train_n.shape)
    print(intens_m.shape, intens_n.shape)
    
    train_m = np.concatenate((train_m, intens_m), axis=1)
    train_n = np.concatenate((train_n, intens_n), axis=1)

    #train_m = np.concatenate((train_m), axis=1)
    #train_n = np.concatenate((train_n), axis=1)    
            
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

Optimally you would use the features and labels created by "generate_training_
feats" but any features, labels set would suffice for the classifier.
"""
def train_rf_classifier(features, labels, no_trees):
    rf = RandomForestClassifier(n_estimators=no_trees, n_jobs=-1)
    rf.fit(features, labels)
    return rf

"""
Classify a patch using a supplied RandomForestClassifier.
"""
def classify_patch_w(fn, kernels, patches, i):
    RF = joblib.load(fn)
    patch = patches[i]
    feat = compute_feats(patch, kernels)
    feat = feat.flatten().reshape(1, -1)
    hogs = compute_hogs(patch)
    hogs = hogs.flatten().reshape(1, -1)
    pixels = patch.flatten().reshape(1, -1)
    feat = np.concatenate((feat, hogs, pixels), axis=1)
    
    prediction = RF.predict(feat)
    #print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
    if prediction == 'M':
        return np.ones(patch.shape)
    elif prediction == 'N':
        return np.zeros(patch.shape)
    else:
        return np.full(patch.shape, prediction)

"""
Classifies multiple patches.
"""
def classify_patch_p(fn, kernels, patches_a, patches_r, a, b):
    print("Classifying group {}-{}/{}".format(a, b, len(patches_a)))
    RF = joblib.load(fn)
    res = []
    for i in range(a, b):
        if i >= len(patches_a):
            break
        patch = patches_a[i]
        if np.all(patches_r[i]): # if there's any nonzero
            feat = compute_feats(patch, kernels)
            feat = feat.flatten().reshape(1, -1)
            intens = np.array(compute_intens(patch))
            intens = intens.flatten().reshape(1, -1)
            #hogs = compute_hogs(patch)
            #hogs = hogs.flatten().reshape(1, -1)
            #pixels = patch.flatten().reshape(1, -1)
            feat = np.concatenate((feat, intens), axis=1)
            
            prediction = RF.predict(feat)
            #print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
            if prediction == 'M':
                res.append(np.ones(patch.shape))
            elif prediction == 'N':
                res.append(np.zeros(patch.shape))
            else:
                res.append(np.full(patch.shape, prediction))
        else: # the associated ROI patch is totally zero
            res.append(np.zeros(patch.shape))
    return res
    
"""
Classifies multiple patches like above, but does so by reading specific
sections from a temp directory.
"""
def classify_patch_f(RF_pkl, kernels_pkl, patches_a_pkl, patches_r_pkl, a, b):
    RF = dill.load(open(RF_pkl, 'rb'))
    kernels = dill.load(open(kernels_pkl, 'rb'))
    patches_a = dill.load(open(patches_a_pkl, 'rb'))
    patches_r = dill.load(open(patches_r_pkl, 'rb'))
    print("Classifying group {}-{}/{}".format(a, b, len(patches_a)))
    res = []
    for i in range(a, b):
        if i >= len(patches_a):
            break
        patch = patches_a[i]
        if np.all(patches_r[i]): # if there's any nonzero
            feat = compute_feats(patch, kernels)
            feat = feat.flatten().reshape(1, -1)
            intens = np.array(compute_intens(patch))
            intens = intens.flatten().reshape(1, -1)
            #hogs = compute_hogs(patch)
            #hogs = hogs.flatten().reshape(1, -1)
            #pixels = patch.flatten().reshape(1, -1)
            feat = np.concatenate((feat, intens), axis=1)
            
            prediction = RF.predict(feat)
            #print("Classifying patch {}/{}: {}".format(i, len(patches), prediction))
            if prediction == 'M':
                res.append(np.ones(patch.shape))
            elif prediction == 'N':
                res.append(np.zeros(patch.shape))
            else:
                res.append(np.full(patch.shape, prediction))
        else: # the associated ROI patch is totally zero
            res.append(np.zeros(patch.shape))
    return res

"""
Generate labels using Random Forest.
"""
def rf_reconstruct(jobs, slice_infos, i):
    t0 = time()
    feats, labels = generate_training_feats(slice_infos, i)
    labels = np.array(labels)
    print(feats.shape, labels.shape)
    RF = train_rf_classifier(feats, labels, no_trees)

    dt1 = time() - t0   
    t0 = time()
    
    test_sl = slice_infos[i]
    image = test_sl.slice_im
    rimage = test_sl.slice_ro
    
    kernels = generate_kernels()
    
    dt2 = time() - t0
    t0 = time()
    
    # break the image into patches; all of these will be classified
    patch_size = (psize, psize)
    # _a stands for "all"
    patches_a = extract_patches_2d(image, patch_size)
    # _p stands for "predict"
    
    patches_r = extract_patches_2d(rimage, patch_size)
    # _r stands for "registered"
    
    dt3 = time() - t0
    t0 = time()
    
    # dump the RF
    fn_rf = 'rf.joblib'
    joblib.dump(RF, fn_rf)
    
    dt4 = time() - t0
    t0 = time()
    
    chunk_size = len(patches_a) / float(jobs)
    
    rf_pkl = "temp/rf.pkl"
    kernels_pkl = "temp/kernels.pkl"
    patches_a_pkl = "temp/patches_a.pkl"
    patches_r_pkl = "temp/patches_r.pkl"
    
    dill.dump(RF, open(rf_pkl, 'wb'))
    dill.dump(kernels, open(kernels_pkl, 'wb'))
    dill.dump(patches_a, open(patches_a_pkl, 'wb'))
    dill.dump(patches_r, open(patches_r_pkl, 'wb'))
    
    # check each patch
    if len(sys.argv) >= 2:
        #patches_p = Parallel(n_jobs=jobs)(delayed(classify_patch)(RF, kernels, patches_a, i) for i in range(len(patches_a)))
        #patches_p = Parallel(n_jobs=jobs)(delayed(classify_patch_w)(fn_rf, kernels, patches_a, i) for i in range(len(patches_a)))
        patches_x = Parallel(n_jobs=jobs)(delayed(classify_patch_p)(fn_rf, kernels, patches_a, patches_r, i, i+int(chunk_size)) for i in range(0, len(patches_a), int(chunk_size)))    
        #patches_x = Parallel(n_jobs=jobs)(delayed(classify_patch_f)(rf_pkl, kernels_pkl, patches_a_pkl, patches_r_pkl, i, i+int(chunk_size)) for i in range(0, len(patches_a), int(chunk_size)))            
        patches_p = []        
        for group in patches_x:
            patches_p.extend(group)
    else:
        patches_p = []
        for i in range(len(patches_a)):
            patches_p.append(classify_patch_w(RF, kernels, patches_a, i))
        
            
    dt5 = time() - t0
    t0 = time()
    
    # reconstruct based on the patch
    recons_im = reconstruct_from_patches_2d(np.asarray(patches_p), image.shape)
    
    dt6 = time() - t0
    t0 = time()
    
    print("Completed Reconstruction {}/{}: {} DT: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(i, len(slice_infos), recons, dt1, dt2, dt3, dt4, dt5, dt6))
    
    # save reconstruction!
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

def min_max(sl):
    masked = mask_out(sl.slice_im, sl.slice_lb).flatten()
    # cap out the seeds (for CT)
    masked[masked >= ct_cap_max] = -1000
    local_max = masked.max()
    masked[masked == -1000] = 1000 # eliminate the -1000 pixels so we get true min
    local_min = masked.min()    
    return (local_min, local_max)

"""
Find the minimum and maximum of each masked image and present them as an
array of tuples. The minimums don't include the -1000 mask value.
"""
def min_maxs(slice_infos):
    mm = []
    for sl in slice_infos:
        mm.append(min_max(sl))
    return mm

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

def generate_labels(jobs):
    print("Generating Labels...")
    global fullspec_i
    global recons
    with open(pickle, 'rb') as f:
        slice_infos = dill.load(f)
    for x in range(0, 30):
        fullspec_i = x
        recons = "recons_" + str(psize) + "_" + str(fullspec_i) + ".pkl"
        print("Working on...", recons)
        rf_reconstruct(jobs, slice_infos, fullspec_i)

def generate_reports():
    with open(pickle, 'rb') as f:
        slice_infos = dill.load(f)
        
    threshv = 0.50
    allbestsims = []
    allsims = []
    
    # go through each case
    for i in range(0, 35):
        allsims.append([])
        with open("recons_ctv_atl_512_int/recons_12_" + str(i) + ".pkl", 'rb') as f:
            recons_im = dill.load(f)
            recons_th = threshold(recons_im, threshv, True)
            recons_tv = threshold(recons_im, threshv)
        real_lb = slice_infos[i].slice_lb
        real_ro = slice_infos[i].slice_ro
        if ct_cap_max:
            slice_im = np.clip(slice_infos[i].slice_im, ct_cap_min, ct_cap_max)

        best_sim = 0
        best_thresh = 0.5

        for a in range(0, 100):
            a = a/100.0
            recons_th = threshold(recons_im, a, True)
            sim = compare_im(recons_th, real_lb)
            allsims[i].append(sim)
            if sim > best_sim:
                best_sim = sim
                best_thresh = a
        
        allbestsims.append(best_sim)
        recons_th = threshold(recons_im, best_thresh, True)
        recons_tv = threshold(recons_im, best_thresh)
        
        # plot each
        plt.figure(figsize=(8,12))
        #plt.subplot(4, 1, 1)
        
        plt.axis('off')
        plt.subplot(3, 2, 1)
        plt.axis('off')
        plt.imshow(real_ro, cmap=plt.cm.gray)
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
        plt.suptitle("CTV Reconstructions with ATL/INT at Threshold {}, Case {}".format(best_thresh,i) + "\nSimilarity: {} | Range: {}".format(compare_im(recons_th, real_lb), min_max(slice_infos[i])))
        plt.savefig('compares_ctv_atl_512_int/case_' + str(i) + '.png')
    
    with open('compares_ctv_atl_512/sims.pkl', 'wb') as f:
        dill.dump(allsims, f)
    
    print("Average DSC: ", np.mean(allbestsims))

"""
Commandline arguments for classification routine:

-t:
Generate (t)raining data using the MRI volumes inside the "path" directory. The
training data will be stored as a *.pkl file which can be re-used for testing
and segmentation reconstruction efforts.

-l:
Reconstruct all the (l)abels found in the training data *.pkl file generated
above and put the reconstructions in a directory.

-r:
Go through all the reconstructions and make a (r)eport comparing the
reconstructions with the ground truth labels stored in the training data *.pkl
file generated above and save the reports in a directory.
"""
def main(argv):
    supported_opts = "t:l:r"

    try:
        opts, args = getopt.getopt(argv[1:], supported_opts)
    except:
        print (argv[0] + " -" + supported_opts)
        sys.exit(2)
    
    for opt, arg in opts:
       
        if "-t" == opt: # generate training data
            try:
                jobs = int(arg)
            except:
                print("Defaulting to 1 job for training...")
                jobs = 1
            generate_training_data(jobs)
        
        if "-l" == opt: # generate labels
            try:
                jobs = int(arg)
            except:
                print("Defaulting to 1 job for training...")
                jobs = 1
            generate_labels(jobs)
        
        if "-r" == opt: # generate reports
            generate_reports()
    

if __name__ == "__main__":
    main(sys.argv)