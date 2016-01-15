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
from skimage.feature import hog
from joblib import Parallel, delayed

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
psize = 12 # patch "radius", so the patch dimensions are psize x psize
# how many "principal component" patches to generate per slice per class
# for example, 10 will result in 10 PC patches for masked, 10 PC patches
# for non-masked regions.
pc_max = 500
ct_cap_min = -1000 # minimum CT brightness (set ct_cap_max for no clipping)
ct_cap_max = 2000 # maximum CT brightness (set to 0 for no clipping)
ct_monte = 1 # 1 will use random patches; 0 will use PCA patches
pickle = "training_data_CTV_12.pkl" # path and filename to save the SliceInfos
recons = "recons_12_3.pkl" # path and filename to store the patches of reconstruction
generate = 0 # toggle generation of new SliceInfos
classify = 0 # test classification engine
no_trees = 10 # number of trees to use for classifier
fullspec_i = 3
crop = (128, 128)
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

def generate_training_data(jobs):
    print("Generating Training Data...")
    slice_infos = []
    
    images_fn, labels_fn = get_filenames(path, im_name, lb_name) 
    kernels = generate_kernels()
        
    slice_infos = Parallel(n_jobs=jobs)(delayed(create_sliceinfo_w)(images_fn, labels_fn, kernels, i) for i in range(len(images_fn)))

    with open(pickle, 'wb') as f:
        dill.dump(slice_infos, f)
        
def generate_labels(jobs):
    print("Generating Labels...")

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
    supported_opts = "t:lr"

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
            print("You chose L.")
        
        if "-r" == opt: # generate reports
            print("You chose R.")
    

if __name__ == "__main__":
    main(sys.argv)