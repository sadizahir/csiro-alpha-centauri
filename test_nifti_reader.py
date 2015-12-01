# -*- coding: utf-8 -*-
"""
Tests the Nibabel Nifti reader. Opens a file and displays it.
Masks the Nifti with an nrrd label.
Created on Tue Dec 01 09:17:45 2015

@author: ZAH010
"""
# The good print
from __future__ import print_function

# Timing
from time import time

# Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as pl

# Filter stuff
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA

PATCH_SIZE = 20
MAX_EIGEN = 25

# Path to the file we're interested in
#path = "../Crohns/converted/anon_1181403/anon/nifti/"
path = ""
filename = path + "anon_mr_150420_30sec.nii.gz"
filename_label = path + "anon_mr_150420_30sec.nii-label.nrrd.nii.gz"

# Load a nifti file
image = nib.load(filename)

# Get the raw data
image_data = image.get_data()
orientation = image.get_affine()
print(image_data.shape)

# Extract a relevant slice from the 3D volume
image_slice = image_data[:, :, image_data.shape[2]-51, 0]
image_slice = np.fliplr(image_slice).T
image_slice = np.flipud(image_slice)

pl.imshow(image_slice, cmap=pl.cm.gray)
pl.show()
print(image_slice.shape)

# Load the NRRD information. For now, it loads NIFTI-converted NRRD from
# SMILI, but later we'll read the NRRD natively.

image_label = nib.load(filename_label)
data_label = image_label.get_data()
orientation_label = image_label.get_affine()
label_slice = data_label[:, :, data_label.shape[2]-51] # the label is only coronal?
label_slice = np.fliplr(label_slice).T
label_slice = np.flipud(label_slice)

pl.imshow(label_slice, cmap=pl.cm.gray)
pl.show()
print(label_slice.shape)

# Mask the label_slice onto the image slice
mask = np.where(label_slice == 0, label_slice, image_slice)
pl.imshow(mask, cmap=pl.cm.gray)
pl.show()

# Extract patches
psize = (PATCH_SIZE, PATCH_SIZE)
patches_original = extract_patches_2d(image_slice, psize)
patches_label = extract_patches_2d(label_slice, psize)
patches_mask = []

for k in patches_label:
    if 0 not in k:
        break

for i, patch in enumerate(patches_original):
    if 0 not in patches_label[i]:
        patches_mask.append(patch)

# We still have like 6000+ patches. Reduce with PCA
pca = PCA(n_components=MAX_EIGEN)
patches_mask = np.array(patches_mask)
patches_mask2 = patches_mask
patches_mask = patches_mask.reshape(patches_mask.shape[0], -1)
#patches_mask -= np.mean(patches_mask, axis=0)
#patches_mask /= np.std(patches_mask, axis=0)

# Decompose
t0 = time()
eigens = pca.fit(patches_mask).components_
dt = time() - t0
print("Decomposed in %.2fs." % dt)

#plot dictionary
pl.figure(figsize=(4.2, 4))
for i, comp in enumerate(eigens):
    pl.subplot(5, 5, i + 1)
    pl.imshow(comp.reshape(psize), cmap=pl.cm.gray_r, interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.suptitle('Eigen-decomposition of Patches in ROI\n' 
+ 'Train time %.1fs on %d patches' % (dt, MAX_EIGEN), fontsize=16)
pl.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
pl.show()
