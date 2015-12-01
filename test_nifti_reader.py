# -*- coding: utf-8 -*-
"""
Tests the Nibabel Nifti reader. Opens a file and displays it.
Created on Tue Dec 01 09:17:45 2015

@author: ZAH010
"""

# Packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as pl

# Path to the file we're interested in
path = "../Crohns/converted/anon_1181403/anon/nifti/"
filename = path + "anon_mr_150420_30sec.nii.gz"

# Load a nifti file
image = nib.load(filename)

# Get the raw data
data = image.get_data()
orientation = image.get_affine()
print(data.shape)

# Extract a relevant slice from the 3D volume
image_slice = data[:, :,  data.shape[2]/2-15, 0]
image_slice = np.fliplr(image_slice).T
image_slice = np.flipud(image_slice)

print image_slice.shape

# Show the image slice
pl.imshow(image_slice, cmap=pl.cm.gray)
pl.show()

# Comments:
# The number of slices here (72) are less than the ones in Slicer (like 200).
# How do I see the other slices?