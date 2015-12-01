# -*- coding: utf-8 -*-
"""
Tests the Nibabel Nifti reader. Opens a file and displays it.
Masks the Nifti with an nrrd label.
Created on Tue Dec 01 09:17:45 2015

@author: ZAH010
"""
# The good print
from __future__ import print_function

# Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as pl

# Filter stuff
from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel

# Path to the file we're interested in
path = "../Crohns/converted/anon_1181403/anon/nifti/"
filename = path + "anon_mr_150420_30sec.nii.gz"
filename_label = path + "anon_mr_150420_30sec.nii-label.nrrd.nii.gz"

# Load a nifti file
image = nib.load(filename)

# Get the raw data
data = image.get_data()
orientation = image.get_affine()
print(data.shape)

# Extract a relevant slice from the 3D volume
image_slice = data[:, :, data.shape[2]-51, 0]
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
label_slice = data_label[:, :, data.shape[2]-51] # the label is only coronal?
label_slice = np.fliplr(label_slice).T
label_slice = np.flipud(label_slice)

pl.imshow(label_slice, cmap=pl.cm.gray)
pl.show()
print(label_slice.shape)

# Mask the label_slice onto the image slice
mask = np.where(label_slice == 0, label_slice, image_slice)
pl.imshow(mask, cmap=pl.cm.gray)
pl.show()

# Apply a Gabor filter to the masked image
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
            
shrink = (slice(0, None, 3), slice(0, None, 3))
brick = mask
grass = mask
wall = mask
image_names = ('brick', 'grass', 'wall')
images = (brick, grass, wall)

# prepare reference features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)
ref_feats[1, :, :] = compute_feats(grass, kernels)
ref_feats[2, :, :] = compute_feats(wall, kernels)

print('Rotated images matched against references using Gabor filter banks:')

print('original: brick, rotated: 30deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: brick, rotated: 70deg, match result: ', end='')
feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])

print('original: grass, rotated: 145deg, match result: ', end='')
feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
print(image_names[match(feats, ref_feats)])


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (0, 1):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel), interpolation='nearest')
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()