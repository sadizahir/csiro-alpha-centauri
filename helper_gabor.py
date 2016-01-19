# -*- coding: utf-8 -*-
"""
Helper functions for Gabor kernels and the like.
Created on Thu Dec 03 10:19:12 2015

@author: ZAH010
"""

from __future__ import print_function

# Basic Packages
import numpy as np
import matplotlib.pyplot as plt

# Scientific
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from scipy import stats
from skimage.feature import hog


# Debugging stuff
# Set to 0 if you don't want image-size messages, timing, etc.
DEBUG = 1
SHOW_IMG = 1
TEST = 1  

"""
Generates Gabor kernels.
"""
def generate_kernels():
    kernels = []
    for theta in range(16):
        theta = theta / 16. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    return kernels

"""
Image normalisation from the Gabor example.
"""
def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

"""
From a bunch of kernels and an image, compute a set of features of the image.
"""    
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 4), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
        feats[k, 2] = stats.skew(filtered.flatten())
        feats[k, 3] = stats.kurtosis(filtered.flatten())
    return feats
    
"""
From a kernel and an image, compute all the powers of an image.
"""
def compute_powers(image, kernels):
    powers = []
    for kernel in kernels:
        powers.append(power(image, kernel))
    return powers

"""
Compares features against reference features.
"""
def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i

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
Plot a selection of filter banks and their Gabor responses.
"""
def plot_gabor(images, filename=None): #, patch_size):
    
    #images = images.reshape(images.shape[0], patch_size, patch_size)
    
    results = []
    kernel_params = []
    thetas = [0, 1, 2, 3]
    for theta in thetas:
        theta = theta / 4. * np.pi
        for frequency in (0.1, 0.4):
            kernel = gabor_kernel(frequency, theta=theta)
            params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, 
                                                    frequency)
            kernel_params.append(params)
            # Save kernel and the power image for each image
            results.append((kernel, [power(img, kernel) for img in images]))
    
    fig, axes = plt.subplots(nrows=2*len(thetas)+1, ncols=11, figsize=(11, 6))
    plt.gray()
    
    if not filename:
        fig.suptitle('Image responses for (a selection of) Gabor filter kernels', 
                     fontsize=16)
    
    axes[0][0].axis('off')
    
    # Plot original images
    for label, img, ax in zip([str(i) for i in range(0, len(images))], 
                               images, axes[0][1:]):
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
        # Plot Gabor kernel
        ax = ax_row[0]
        ax.imshow(np.real(kernel), cmap=plt.cm.gray, interpolation='nearest')
        ax.set_ylabel(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
        # Plot Gabor responses with the contrast normalized for each filter
        vmin = np.min(powers)
        vmax = np.max(powers)
        for patch, ax in zip(powers, ax_row[1:]):
            ax.imshow(patch, vmin=vmin, vmax=vmax, cmap=plt.cm.gray,
                      interpolation='nearest')
            ax.axis('off')
    
    if filename:
        save_to = filename + '.gabor.png'
        plt.savefig(save_to)
        return save_to
    else:
        plt.show()